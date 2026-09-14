"""Benchmark: surrogate vs pycbc MCMC parameter estimation across many events.

Runs the same matched-filter MCMC with the FLARE surrogate and with live pycbc
as the template engine, through identical conditioning / prior / warm-start /
seed, on a list of real GWOSC events. Writes one directory per event (posterior
summary + corner overlay) plus a summary table. Resumable: events with an
existing result are skipped, so a long run survives interruption.

    python scripts/pe_benchmark.py --project checkpoints/<seobnrv4-model> \
        --approximant SEOBNRv4 --nwalkers 24 --nsteps 300 --outdir pe_benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import WaveformGenerator  # noqa: E402
from src.inference.estimator import GWEventEstimator, Prior, WaveformBank  # noqa: E402
from src.inference.gwosc import fetch_event_data  # noqa: E402
from src.inference.predictor import WaveformPrediction, WaveformPredictor  # noqa: E402

DEFAULT_EVENTS = [
    "GW150914", "GW170104", "GW170729", "GW170809", "GW170814", "GW170818",
    "GW170823", "GW190521_074359", "GW190630_185205", "GW190828_063405",
    "GW190519_153544", "GW190602_175927",
]


class PyCBCPredictor:
    def __init__(self, approximant, L, delta_t, f_lower=20.0, merger_frac=0.9):
        self.delta_t = delta_t
        self.meta = {"td_fixed_window": True, "merger_frac": merger_frac}
        self.reduced_order = True
        self._g = WaveformGenerator(waveform_length=L, delta_t=delta_t, f_lower=f_lower,
                                    approximant=approximant, fixed_window=True)

    def predict(self, m1, m2, s1z=0.0, s2z=0.0, inc=0.0, ecc=0.0, sigma_level=None):
        h = self._g._generate_single_waveform(np.array([m1, m2, s1z, s2z, inc, ecc]), clean=True)
        return WaveformPrediction(np.asarray(h, float), sample_rate=self.delta_t), \
            WaveformPrediction(np.zeros_like(h), sample_rate=self.delta_t)

    def batch_predict(self, full, batch_size=256):
        hp = [self.predict(*p)[0] for p in np.asarray(full)]
        return hp, [WaveformPrediction(np.zeros_like(w.data), sample_rate=self.delta_t) for w in hp]


def summarise(res):
    out, s = {}, res.samples
    idx = {n: i for i, n in enumerate(res.param_names)}
    for n in res.param_names:
        lo, md, hi = np.percentile(s[:, idx[n]], [5, 50, 95])
        out[n] = [float(md), float(lo), float(hi)]
    m1, m2 = s[:, idx["m1"]], s[:, idx["m2"]]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    out["chirp_mass"] = [float(np.median(mc)), *[float(x) for x in np.percentile(mc, [5, 95])]]
    return out


def corner_overlay(rs, rp, path, event):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import corner
    labels = rs.param_names
    rng = [(min(a.min(), b.min()), max(a.max(), b.max())) for a, b in zip(rs.samples.T, rp.samples.T)]
    fig = corner.corner(rp.samples, labels=labels, color="C1", range=rng, hist_kwargs={"density": True})
    corner.corner(rs.samples, fig=fig, labels=labels, color="C0", range=rng, hist_kwargs={"density": True})
    fig.suptitle(f"{event}: surrogate (blue) vs pycbc (orange)", y=1.01)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def run_event(event, surrogate, pycbc_engine, prior, args, edir):
    data = fetch_event_data(event, detectors=list(args.detectors), duration=args.duration,
                            sample_rate=1.0 / surrogate.delta_t, f_lower=surrogate.f_lower)
    nested = args.sampler == "nested"
    # emcee warm-starts walkers from a surrogate bank near the peak; nested
    # sampling draws live points from the prior, so no warm-start is needed.
    p0 = None
    if not nested:
        bank = WaveformBank(surrogate, prior=prior); bank.build(n_samples=args.bank_size, seed=args.seed)
        est_s = GWEventEstimator(surrogate, data, prior=prior, likelihood="matched_filter")
        p0 = est_s.initialize_walkers(args.nwalkers, bank=bank, seed=args.seed)
    row = {"event": event, "sampler": args.sampler}
    res_store = {}
    rfile = os.path.join(edir, "result.json")
    for tag, eng in (("surrogate", surrogate), ("pycbc", pycbc_engine)):
        est = GWEventEstimator(eng, data, prior=prior, likelihood="matched_filter")
        t0 = time.time()
        if nested:
            # GPU surrogate stays single-process (never fork a CUDA context); the
            # CPU-bound pycbc reference pools across cores.
            nproc = args.pool if tag == "pycbc" else 1
            res = est.run_nested(nlive=args.nlive, dlogz=args.dlogz,
                                 pool_processes=nproc, progress=False, seed=args.seed)
        else:
            res = est.run_mcmc(nwalkers=args.nwalkers, nsteps=args.nsteps, p0=p0.copy(),
                               progress=False, seed=args.seed)
        secs = round(time.time() - t0, 1)
        entry = {"seconds": secs,
                 "acceptance": round(float(res.acceptance_fraction), 3),
                 "summary": summarise(res)}
        if res.log_evidence is not None:
            entry["log_evidence"] = round(float(res.log_evidence), 3)
            entry["log_evidence_err"] = round(float(res.log_evidence_err or 0.0), 3)
        row[tag] = entry
        res_store[tag] = res
        mc = entry["summary"]["chirp_mass"][0]
        lnz = f" lnZ={entry.get('log_evidence')}" if "log_evidence" in entry else ""
        print(f"    [{tag:9s}] {secs:7.0f}s  Mc={mc:6.2f}{lnz}", flush=True)
        # write a partial result after each engine so progress is always visible
        json.dump(row, open(rfile, "w"), indent=2)
    row["speedup"] = round(row["pycbc"]["seconds"] / max(row["surrogate"]["seconds"], 1e-9), 2)
    # chirp-mass agreement (the tightly-constrained quantity)
    smc, pmc = row["surrogate"]["summary"]["chirp_mass"], row["pycbc"]["summary"]["chirp_mass"]
    row["mc_surrogate"], row["mc_pycbc"] = smc[0], pmc[0]
    row["mc_abs_diff"] = round(abs(smc[0] - pmc[0]), 3)
    corner_overlay(res_store["surrogate"], res_store["pycbc"],
                   os.path.join(edir, "corner.png"), event)
    json.dump(row, open(os.path.join(edir, "result.json"), "w"), indent=2)
    return row


def write_summary(outdir, rows):
    rows = sorted(rows, key=lambda r: r["mc_pycbc"])
    sampler = rows[0].get("sampler", "mcmc") if rows else "mcmc"
    method = ("dynesty nested sampling" if sampler == "nested"
              else "matched-filter MCMC · warm-started")
    lines = ["# Surrogate vs pycbc PE benchmark\n",
             f"{len(rows)} events · {method} · identical conditioning/prior.\n",
             "| event | Mc surrogate | Mc pycbc | |ΔMc| | m1 surr | m1 pycbc | speedup | acc(s/p) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, p = r["surrogate"]["summary"], r["pycbc"]["summary"]
        lines.append(
            f"| {r['event']} | {s['chirp_mass'][0]:.2f} | {p['chirp_mass'][0]:.2f} | "
            f"{r['mc_abs_diff']:.2f} | {s['m1'][0]:.1f} | {p['m1'][0]:.1f} | "
            f"{r['speedup']:.1f}× | {r['surrogate']['acceptance']:.2f}/{r['pycbc']['acceptance']:.2f} |")
    diffs = [r["mc_abs_diff"] for r in rows]
    lines.append(f"\nMedian |ΔMc| = {np.median(diffs):.3f} M☉ · "
                 f"max |ΔMc| = {np.max(diffs):.3f} M☉ · mean speedup {np.mean([r['speedup'] for r in rows]):.1f}×")
    open(os.path.join(outdir, "summary.md"), "w").write("\n".join(lines))
    json.dump(rows, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--approximant", default="SEOBNRv4")
    ap.add_argument("--events", nargs="+", default=DEFAULT_EVENTS)
    ap.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    ap.add_argument("--sampler", choices=["mcmc", "nested"], default="mcmc",
                    help="mcmc = emcee ensemble (default); nested = dynesty nested sampling "
                         "(better for the broad, multimodal high-mass posteriors)")
    ap.add_argument("--nwalkers", type=int, default=24)
    ap.add_argument("--nsteps", type=int, default=300)
    ap.add_argument("--nlive", type=int, default=500, help="nested: number of live points")
    ap.add_argument("--dlogz", type=float, default=0.1, help="nested: ln-evidence stop tolerance")
    ap.add_argument("--pool", type=int, default=1,
                    help="nested: CPU worker processes for the pycbc reference "
                         "(surrogate stays single-process on GPU)")
    ap.add_argument("--bank-size", type=int, default=1500)
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="pe_benchmark")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    surrogate = WaveformPredictor(args.project, device=args.device)
    surrogate.f_lower = float(surrogate.meta.get("f_lower", 20.0))
    hp, _ = surrogate.predict(m1=40, m2=32)
    L = len(hp.data)
    pycbc_engine = PyCBCPredictor(args.approximant, L, surrogate.delta_t,
                                  f_lower=surrogate.f_lower,
                                  merger_frac=surrogate.meta.get("merger_frac", 0.9))
    prior = Prior(bounds={"m1": (15.0, 100.0), "m2": (15.0, 100.0),
                          "s1z": (-0.9, 0.9), "s2z": (-0.9, 0.9)}, fixed={"inc": 0.0, "ecc": 0.0})

    rows = []
    for event in args.events:
        edir = os.path.join(args.outdir, event)
        rfile = os.path.join(edir, "result.json")
        if os.path.exists(rfile):
            rows.append(json.load(open(rfile)))
            print(f"[skip] {event} (already done)", flush=True)
            continue
        os.makedirs(edir, exist_ok=True)
        print(f"[run ] {event} ...", flush=True)
        try:
            row = run_event(event, surrogate, pycbc_engine, prior, args, edir)
            rows.append(row)
            print(f"[done] {event}: Mc surr {row['mc_surrogate']:.2f} vs pycbc {row['mc_pycbc']:.2f} "
                  f"(|Δ|={row['mc_abs_diff']:.2f}), {row['speedup']:.1f}×", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {event}: {type(exc).__name__}: {exc}", flush=True)
        if rows:
            write_summary(args.outdir, rows)
    print(f"\nBenchmark complete: {len(rows)} events.")


if __name__ == "__main__":
    main()
