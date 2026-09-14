#!/usr/bin/env python
"""Compare surrogate vs live-pycbc parameter estimation on real GWOSC events.

The question this answers is not "does the surrogate look like the approximant"
(that is ``mean_match``) but "does swapping the template engine move the
POSTERIOR". So everything except the engine is held identical: the same
conditioned data (fetched once and shared), the same prior, the same
``GWEventEstimator`` and matched-filter likelihood, the same warm-start walker
positions, and the same seed. The only difference is who generates templates.

The pycbc side is a drop-in ``PyCBCPredictor`` exposing the three things
``GWEventEstimator`` actually touches -- ``delta_t``, ``meta``, and
``predict(...)`` -- and returning live ``WaveformGenerator`` output on the same
fixed window. Note ``likelihood="matched_filter"`` is required: it is the only
path that passes s1z/s2z through to the template (estimator.py:325-329); the
gaussian path silently drops spins and would compare two spin-agnostic runs.

Wall-clock per run is recorded too: the surrogate's speed win only shows up for
expensive TD approximants (SEOBNRv4 ~250 ms/template) where generation, not the
matched filter (~13 ms), dominates.

Example:
    python scripts/pe_compare.py --project checkpoints/<model> \
        --events GW150914 GW170814 GW170729 --nwalkers 24 --nsteps 1000 \
        --outdir docs/images/pe_compare_seobnr_20k
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import WaveformGenerator  # noqa: E402
from src.inference.estimator import GWEventEstimator, Prior, WaveformBank  # noqa: E402
from src.inference.gwosc import fetch_event_data  # noqa: E402
from src.inference.predictor import WaveformPredictor  # noqa: E402

logger = logging.getLogger("pe_compare")


@dataclass
class _Wave:
    """Minimal stand-in for WaveformPrediction (the estimator only reads .data)."""
    data: np.ndarray


class PyCBCPredictor:
    """WaveformPredictor-compatible engine that generates live pycbc templates.

    Mirrors the surrogate's fixed-window convention exactly (same L, delta_t,
    f_lower, merger at ``merger_frac*L``) so the two engines are interchangeable
    inside GWEventEstimator.
    """

    def __init__(self, approximant: str, waveform_length: int, delta_t: float,
                 f_lower: float = 20.0, merger_frac: float = 0.9):
        self.delta_t = delta_t
        self.meta = {"td_fixed_window": True, "merger_frac": merger_frac}
        self.reduced_order = True
        self._gen = WaveformGenerator(
            waveform_length=waveform_length, delta_t=delta_t, f_lower=f_lower,
            approximant=approximant, fixed_window=True)

    def predict(self, m1, m2, s1z=0.0, s2z=0.0, inc=0.0, ecc=0.0, sigma_level=None):
        h = self._gen._generate_single_waveform(
            np.array([m1, m2, s1z, s2z, inc, ecc], dtype=float), clean=True)
        return _Wave(np.asarray(h, dtype=np.float64)), _Wave(np.zeros_like(h))

    def batch_predict(self, full_params, batch_size: int = 256):
        hp = [self.predict(*p)[0] for p in np.asarray(full_params)]
        return hp, [_Wave(np.zeros_like(w.data)) for w in hp]


def _summarise(result, prior) -> dict:
    """Median + 90% CI per free parameter, plus derived Mc and chi_eff."""
    out = {}
    names = result.param_names
    s = result.samples
    for i, n in enumerate(names):
        col = s[:, i]
        lo, hi = np.percentile(col, [5, 95])
        out[n] = {"median": float(np.median(col)), "lower_90": float(lo),
                  "upper_90": float(hi)}
    idx = {n: i for i, n in enumerate(names)}
    if "m1" in idx and "m2" in idx:
        m1, m2 = s[:, idx["m1"]], s[:, idx["m2"]]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        lo, hi = np.percentile(mc, [5, 95])
        out["chirp_mass"] = {"median": float(np.median(mc)), "lower_90": float(lo),
                             "upper_90": float(hi)}
        if "s1z" in idx and "s2z" in idx:
            chi = (m1 * s[:, idx["s1z"]] + m2 * s[:, idx["s2z"]]) / (m1 + m2)
            lo, hi = np.percentile(chi, [5, 95])
            out["chi_eff"] = {"median": float(np.median(chi)), "lower_90": float(lo),
                              "upper_90": float(hi)}
    return out


def _corner_overlay(res_surr, res_pycbc, path: str, event: str):
    import corner
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = res_surr.param_names
    rng = [(min(a.min(), b.min()), max(a.max(), b.max()))
           for a, b in zip(res_surr.samples.T, res_pycbc.samples.T)]
    fig = corner.corner(res_pycbc.samples, labels=labels, color="C1",
                        range=rng, quantiles=[0.05, 0.5, 0.95], show_titles=False,
                        hist_kwargs={"density": True})
    corner.corner(res_surr.samples, fig=fig, labels=labels, color="C0",
                  range=rng, quantiles=[0.05, 0.5, 0.95], show_titles=False,
                  hist_kwargs={"density": True})
    fig.suptitle(f"{event}: FLARE surrogate (blue) vs live pycbc (orange)", y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="checkpoint dir of the trained surrogate")
    p.add_argument("--events", nargs="+", default=["GW150914", "GW170814", "GW170729"])
    p.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    p.add_argument("--approximant", default="SEOBNRv4", help="engine for the pycbc side")
    p.add_argument("--nwalkers", type=int, default=24)
    p.add_argument("--nsteps", type=int, default=1000)
    p.add_argument("--bank-size", type=int, default=2000)
    p.add_argument("--duration", type=float, default=32.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(args.outdir, exist_ok=True)

    surrogate = WaveformPredictor(args.project, device=args.device)
    L = len(surrogate.t_norm_array) if hasattr(surrogate, "t_norm_array") else None
    meta = getattr(surrogate, "meta", {}) or {}
    if not meta.get("td_fixed_window"):
        print("ERROR: model is not fixed-window; PE requires td_fixed_window=True",
              file=sys.stderr)
        return 2
    merger_frac = float(meta.get("merger_frac", 0.9))
    hp, _ = surrogate.predict(m1=40.0, m2=32.0)
    L = len(hp.data)
    pycbc_engine = PyCBCPredictor(args.approximant, L, surrogate.delta_t,
                                  merger_frac=merger_frac)

    prior = Prior.default()
    all_out = {}

    for event in args.events:
        logger.info("=== %s ===", event)
        # Fetch ONCE and share, so both engines see byte-identical data/PSD.
        data = fetch_event_data(event, detectors=list(args.detectors),
                                duration=args.duration,
                                sample_rate=1.0 / surrogate.delta_t, f_lower=20.0)

        est_s = GWEventEstimator(surrogate, data, prior=prior,
                                 likelihood="matched_filter")
        est_p = GWEventEstimator(pycbc_engine, data, prior=prior,
                                 likelihood="matched_filter")

        # One warm-start, shared: isolates the template engine as the only variable.
        bank = WaveformBank(surrogate, prior=prior)
        bank.build(n_samples=args.bank_size, seed=args.seed)
        p0 = est_s.initialize_walkers(args.nwalkers, bank=bank, seed=args.seed)

        row = {}
        for tag, est in (("surrogate", est_s), ("pycbc", est_p)):
            t0 = time.time()
            res = est.run_mcmc(nwalkers=args.nwalkers, nsteps=args.nsteps,
                               p0=p0.copy(), progress=False, seed=args.seed)
            dt = time.time() - t0
            row[tag] = {"seconds": round(dt, 1),
                        "acceptance": round(float(res.acceptance_fraction), 3),
                        "summary": _summarise(res, prior)}
            row[f"_res_{tag}"] = res
            logger.info("%-9s %6.1f s  acc=%.2f  Mc=%.2f m1=%.1f",
                        tag, dt, res.acceptance_fraction,
                        row[tag]["summary"]["chirp_mass"]["median"],
                        row[tag]["summary"]["m1"]["median"])

        _corner_overlay(row.pop("_res_surrogate"), row.pop("_res_pycbc"),
                        os.path.join(args.outdir, f"corner_compare_{event}.png"), event)
        row["speedup"] = round(row["pycbc"]["seconds"] / max(row["surrogate"]["seconds"], 1e-9), 2)
        all_out[event] = row
        logger.info("%s speedup x%.2f", event, row["speedup"])

    with open(os.path.join(args.outdir, "pe_compare.json"), "w") as f:
        json.dump(all_out, f, indent=2)
    print(json.dumps(all_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
