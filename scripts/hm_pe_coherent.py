"""Coherent (antenna-response) HM parameter estimation on a 2-detector injection.

Uses the new coherent likelihood: both surrogate polarizations projected through
each detector's antenna pattern with the geometric time delay, phase marginalised,
distance + sky sampled. This is what makes inclination and distance measurable
(they stop being a pure amplitude degeneracy) -- the payoff of h_cross + a network.

A true SEOBNRv4HM signal is injected into H1+L1 (projected with the true sky/
orientation) and recovered with the surrogate. ``--check`` just evaluates the
likelihood at the truth vs perturbations (fast sanity); otherwise it runs MCMC.

    python scripts/hm_pe_coherent.py --surrogate checkpoints/seobnrv4hm_5mode_20k --check
    python scripts/hm_pe_coherent.py --surrogate checkpoints/seobnrv4hm_5mode_20k \
        --nwalkers 40 --nsteps 1200 --outdir docs/images/hm_pe_coherent
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.estimator import GWEventEstimator, Prior, WaveformBank  # noqa: E402
from src.inference.gwosc import DetectorData, EventData  # noqa: E402
from src.inference.surrogate import load_surrogate  # noqa: E402

GPS = 1234567890.0


def inject_2det(truth, snr_net, sample_rate, duration, f_lower, detectors, seed):
    """Project a SEOBNRv4HM signal into each detector and add coloured noise."""
    from pycbc.waveform import get_td_waveform
    from pycbc.types import TimeSeries
    from pycbc.psd import aLIGOZeroDetHighPower, inverse_spectrum_truncation
    from pycbc.noise import noise_from_psd
    from pycbc.detector import Detector
    from pycbc.filter import sigma

    dt = 1.0 / sample_rate
    N = int(duration * sample_rate)
    flen = N // 2 + 1
    df = 1.0 / duration
    psd = aLIGOZeroDetHighPower(flen, df, f_lower)

    hp, hc = get_td_waveform(approximant="SEOBNRv4HM", mass1=truth["m1"], mass2=truth["m2"],
                             spin1z=truth["s1z"], spin2z=truth["s2z"], inclination=truth["inc"],
                             coa_phase=truth["coa_phase"], distance=truth["distance"],
                             delta_t=dt, f_lower=f_lower)
    hp = np.asarray(hp, float); hc = np.asarray(hc, float)
    pk = int(np.abs(hp + 1j * hc).argmax())
    rng = np.random.default_rng(seed)
    dets = {}
    snrs = []
    for name in detectors:
        det = Detector(name)
        fp, fc = det.antenna_pattern(truth["ra"], truth["dec"], truth["psi"], GPS)
        tshift = det.time_delay_from_earth_center(truth["ra"], truth["dec"], GPS)
        proj = fp * hp + fc * hc
        sig = np.zeros(N)
        c = N // 2 + int(round(tshift / dt))
        lo = c - pk
        seg = proj[max(0, -lo):]
        lo = max(0, lo); seg = seg[:N - lo]
        sig[lo:lo + len(seg)] = seg
        snrs.append(float(sigma(TimeSeries(sig, delta_t=dt), psd=psd, low_frequency_cutoff=f_lower)))
        noise = np.asarray(noise_from_psd(N, dt, psd, seed=int(rng.integers(1 << 31))), float)
        strain = TimeSeries(sig + noise, delta_t=dt, epoch=GPS - duration / 2)
        psd_ist = inverse_spectrum_truncation(psd, int(4 * sample_rate),
                                              low_frequency_cutoff=f_lower, trunc_method="hann")
        dets[name] = DetectorData(detector=name, strain=strain, psd=psd_ist, delta_t=dt,
                                  epoch=float(strain.start_time))
    net = float(np.sqrt(np.sum(np.square(snrs))))
    # rescale to target network SNR
    scale = snr_net / net
    for name in detectors:
        d = dets[name]
        s = np.asarray(d.strain, float)
        # can't cleanly rescale signal after adding noise; rebuild with scaled distance instead
    return dets, net


def build_event(truth, snr_net, sample_rate, duration, f_lower, detectors, seed):
    # scale distance so the network SNR hits the target (SNR ~ 1/distance)
    dets, net0 = inject_2det(truth, snr_net, sample_rate, duration, f_lower, detectors, seed)
    truth2 = dict(truth); truth2["distance"] = truth["distance"] * (net0 / snr_net)
    dets, net = inject_2det(truth2, snr_net, sample_rate, duration, f_lower, detectors, seed)
    ev = EventData(event_name="INJ_HM_COH", gps_time=GPS, detectors=dets, f_lower=f_lower)
    return ev, truth2, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--sample-rate", type=float, default=4096.0)
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--nwalkers", type=int, default=40)
    ap.add_argument("--nsteps", type=int, default=1200)
    ap.add_argument("--bank-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fix-sky", action="store_true",
                    help="hold sky (ra,dec,psi) at truth and sample only distance + intrinsic + inc "
                         "(the sky is localised independently by the network; this isolates the "
                         "inclination/distance measurement the antenna response unlocks)")
    ap.add_argument("--outdir", default="docs/images/hm_pe_coherent")
    args = ap.parse_args()

    truth = dict(m1=80.0, m2=25.0, s1z=0.3, s2z=-0.1, inc=1.0, coa_phase=0.3,
                 distance=800.0, ra=1.7, dec=-0.5, psi=0.9)
    surrogate = load_surrogate(args.surrogate, device=args.device)
    ev, truth2, net = build_event(truth, args.snr, args.sample_rate, args.duration,
                                  surrogate.f_lower, args.detectors, args.seed)
    print(f"injection: net SNR {net:.1f}, distance {truth2['distance']:.0f} Mpc, "
          f"inc {truth['inc']:.2f}, sky (ra={truth['ra']:.2f}, dec={truth['dec']:.2f})")

    prior = Prior(bounds={"m1": (20, 100), "m2": (20, 100), "s1z": (-0.9, 0.9),
                          "s2z": (-0.9, 0.9), "inc": (0.0, np.pi)}, fixed={"ecc": 0.0})
    est = GWEventEstimator(surrogate, ev, prior=prior, likelihood="coherent",
                           distance_bounds=(100.0, 5000.0))
    if args.fix_sky:
        est.fixed_sky = (truth["ra"], truth["dec"], truth["psi"])

    tvec = np.array([truth["m1"], truth["m2"], truth["s1z"], truth["s2z"], truth["inc"],
                     truth2["distance"], truth["ra"], truth["dec"], truth["psi"]])
    if args.check:
        ll_t = est.coherent_log_posterior(tvec)
        print(f"\nlogL @ truth = {ll_t:.1f}")
        for name, idx, val in [("wrong m1(50)", 0, 50), ("wrong inc(0)", 4, 0.0),
                               ("2x distance", 5, tvec[5] * 2), ("wrong dec(+0.5)", 7, 0.5),
                               ("wrong ra(3.5)", 6, 3.5)]:
            p = tvec.copy(); p[idx] = val
            print(f"logL @ {name:16s} = {est.coherent_log_posterior(p):.1f}  (dL={ll_t - est.coherent_log_posterior(p):+.1f})")
        return

    os.makedirs(args.outdir, exist_ok=True)
    bank = WaveformBank(surrogate, prior=prior); bank.build(n_samples=args.bank_size, seed=args.seed)
    p0 = None
    t0 = time.time()
    res = est.run_mcmc(nwalkers=args.nwalkers, nsteps=args.nsteps, bank=bank, p0=p0,
                       progress=False, seed=args.seed)
    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min, acc={res.acceptance_fraction:.2f}")
    s = res.samples; names = res.param_names
    out = {"truth": truth2, "snr": net, "seconds": round(dt, 1),
           "acceptance": float(res.acceptance_fraction), "summary": {}}
    print(f"{'param':>8} {'truth':>8} {'median[5,95]':>22}")
    tmap = {"m1": truth["m1"], "m2": truth["m2"], "s1z": truth["s1z"], "s2z": truth["s2z"],
            "inc": truth["inc"], "distance": truth2["distance"], "ra": truth["ra"],
            "dec": truth["dec"], "psi": truth["psi"]}
    for i, n in enumerate(names):
        lo, md, hi = np.percentile(s[:, i], [5, 50, 95])
        out["summary"][n] = [float(md), float(lo), float(hi)]
        print(f"{n:>8} {tmap.get(n, np.nan):8.2f} {md:8.2f} [{lo:7.2f},{hi:7.2f}]")
    try:
        import matplotlib; matplotlib.use("Agg"); import corner
        truths = [tmap.get(n) for n in names]
        fig = corner.corner(s, labels=names, truths=truths, truth_color="C3",
                            quantiles=[0.05, 0.5, 0.95], show_titles=True, title_fmt=".2f")
        fig.suptitle("Coherent HM PE (antenna response): surrogate recovery, truth in red", y=1.01)
        fig.savefig(os.path.join(args.outdir, "corner_coherent.png"), dpi=140, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001
        print("corner failed:", exc)
    json.dump(out, open(os.path.join(args.outdir, "hm_pe_coherent.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
