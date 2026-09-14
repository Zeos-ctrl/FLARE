"""Higher-mode parameter estimation on a real GWOSC event: surrogate vs pycbc.

Runs the extended-range SEOBNRv4HM surrogate and live pycbc SEOBNRv4HM as template
engines through the identical estimator, prior and warm start on real detector
data (default GW190412 -- a strong higher-mode, high-mass-ratio event now inside
the 5-100 Msun range). Inclination is free so the higher modes contribute.

    python scripts/hm_pe_event.py --surrogate checkpoints/seobnrv4hm_5mode_wide \
        --event GW190412 --nwalkers 24 --nsteps 300 --outdir docs/images/hm_pe_GW190412
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
from src.inference.predictor import WaveformPrediction  # noqa: E402
from src.inference.surrogate import load_surrogate  # noqa: E402


class PyCBCEngine:
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
    out = {}
    for i, n in enumerate(res.param_names):
        lo, md, hi = np.percentile(res.samples[:, i], [5, 50, 95])
        out[n] = [float(md), float(lo), float(hi)]
    # chirp mass
    idx = {n: i for i, n in enumerate(res.param_names)}
    if "m1" in idx and "m2" in idx:
        m1, m2 = res.samples[:, idx["m1"]], res.samples[:, idx["m2"]]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        out["chirp_mass"] = [float(np.median(mc)), *[float(x) for x in np.percentile(mc, [5, 95])]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--event", default="GW190412")
    ap.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    ap.add_argument("--nwalkers", type=int, default=24)
    ap.add_argument("--nsteps", type=int, default=300)
    ap.add_argument("--bank-size", type=int, default=1500)
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engines", nargs="+", default=["surrogate", "pycbc"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    surrogate = load_surrogate(args.surrogate, device=args.device)
    hp, _ = surrogate.predict(m1=35, m2=9)
    L = len(hp.data)
    pycbc_engine = PyCBCEngine("SEOBNRv4HM", L, surrogate.delta_t, f_lower=surrogate.f_lower,
                               merger_frac=surrogate.merger_frac)
    engines = {"surrogate": surrogate, "pycbc": pycbc_engine}

    data = fetch_event_data(args.event, detectors=list(args.detectors), duration=args.duration,
                            sample_rate=1.0 / surrogate.delta_t, f_lower=surrogate.f_lower)

    # box kept above the total-mass floor (m1>=25, m2>=5 => total>=30, HM (5,5)
    # stays under Nyquist); covers GW190412 (~35, ~9). inclination free.
    prior = Prior(bounds={"m1": (25.0, 55.0), "m2": (5.0, 25.0), "s1z": (-0.9, 0.9),
                          "s2z": (-0.9, 0.9), "inc": (0.0, np.pi)}, fixed={"ecc": 0.0})

    out = {"event": args.event}
    p0 = None
    for tag in args.engines:
        est = GWEventEstimator(engines[tag], data, prior=prior, likelihood="matched_filter")
        if p0 is None:
            bank = WaveformBank(surrogate, prior=prior); bank.build(n_samples=args.bank_size, seed=args.seed)
            p0 = est.initialize_walkers(args.nwalkers, bank=bank, seed=args.seed)
        t0 = time.time()
        res = est.run_mcmc(nwalkers=args.nwalkers, nsteps=args.nsteps, p0=p0.copy(),
                           progress=False, seed=args.seed)
        dt = time.time() - t0
        sm = summarise(res)
        out[tag] = {"seconds": round(dt, 1), "acceptance": round(float(res.acceptance_fraction), 3),
                    "summary": sm}
        out[f"_res_{tag}"] = res
        print(f"{tag:9s} {dt:6.1f}s acc={res.acceptance_fraction:.2f}  "
              f"Mc={sm['chirp_mass'][0]:.2f} m1={sm['m1'][0]:.1f} m2={sm['m2'][0]:.1f} "
              f"s1z={sm['s1z'][0]:+.2f}", flush=True)

    if "surrogate" in args.engines and "pycbc" in args.engines:
        out["speedup"] = round(out["pycbc"]["seconds"] / max(out["surrogate"]["seconds"], 1e-9), 2)
        try:
            import matplotlib; matplotlib.use("Agg"); import corner
            rs, rp = out.pop("_res_surrogate"), out.pop("_res_pycbc")
            labels = rs.param_names
            rng = [(min(a.min(), b.min()), max(a.max(), b.max())) for a, b in zip(rs.samples.T, rp.samples.T)]
            fig = corner.corner(rp.samples, labels=labels, color="C1", range=rng, hist_kwargs={"density": True})
            corner.corner(rs.samples, fig=fig, labels=labels, color="C0", range=rng, hist_kwargs={"density": True})
            fig.suptitle(f"{args.event} higher-mode PE: surrogate (blue) vs pycbc (orange)", y=1.01)
            fig.savefig(os.path.join(args.outdir, f"corner_{args.event}.png"), dpi=150, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            print("corner failed:", exc)
    out = {k: v for k, v in out.items() if not k.startswith("_res")}
    json.dump(out, open(os.path.join(args.outdir, f"hm_pe_{args.event}.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
