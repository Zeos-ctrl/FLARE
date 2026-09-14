"""Coherent higher-mode PE across a CATALOGUE of injections: surrogate vs pycbc vs truth.

Injects a set of SEOBNRv4HM signals with known parameters spanning the mass /
spin / inclination space into H1+L1 (projected with each injection's true sky and
orientation, coloured noise added), then recovers each with BOTH the FLARE HM
surrogate and live pycbc SEOBNRv4HM through the identical coherent antenna-response
likelihood. Every corner overlays the surrogate posterior (blue), the pycbc
posterior (orange) and the injected truth (red crosshairs), in the style of
``docs/images/hm_pe_coherent/corner_coherent.png``.

Because the coherent likelihood uses BOTH polarizations, the pycbc engine here
returns a real h_cross (unlike the single-polarization engines in the other
scripts). Sky is held at truth (localised independently) so the run isolates the
inclination + distance measurement the antenna response unlocks.

    python scripts/hm_pe_catalogue.py --surrogate models/FLARE-SEOBNRv4HM \
        --nwalkers 32 --nsteps 600 --outdir docs/images/hm_pe_catalogue
    python scripts/hm_pe_catalogue.py --surrogate models/FLARE-SEOBNRv4HM --smoke
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
from src.inference.predictor import WaveformPrediction  # noqa: E402
from src.inference.surrogate import load_surrogate  # noqa: E402

GPS = 1234567890.0

# Injected signals spanning the 20-100 Msun, |chi|<0.9 space. Asymmetric masses
# and non-face-on inclinations so the higher modes carry real information; sky
# positions varied so the antenna projection differs per injection.
TRUTHS = [
    dict(m1=60.0, m2=30.0, s1z=0.40, s2z=0.10, inc=1.00, coa_phase=0.3, ra=1.7, dec=-0.5, psi=0.9),
    dict(m1=45.0, m2=40.0, s1z=-0.20, s2z=0.30, inc=0.55, coa_phase=1.1, ra=3.1, dec=0.4, psi=1.6),
    dict(m1=80.0, m2=25.0, s1z=0.30, s2z=-0.10, inc=1.30, coa_phase=0.7, ra=0.6, dec=-1.0, psi=0.4),
    dict(m1=35.0, m2=22.0, s1z=0.50, s2z=0.20, inc=0.90, coa_phase=2.4, ra=4.5, dec=0.9, psi=2.2),
    dict(m1=90.0, m2=55.0, s1z=-0.50, s2z=-0.30, inc=2.00, coa_phase=1.8, ra=2.2, dec=0.1, psi=0.7),
    dict(m1=55.0, m2=26.0, s1z=0.10, s2z=0.00, inc=2.35, coa_phase=0.2, ra=5.3, dec=-0.3, psi=1.2),
]


class CoherentPyCBCEngine:
    """Live SEOBNRv4HM on the fixed window, returning BOTH polarizations.

    Mirrors the surrogate's ``predict(m1,m2,s1z,s2z,inc)`` -> (h_plus, h_cross)
    interface so it drops straight into the coherent likelihood. Generates the
    waveform at the model reference distance, pins the merger (peak of |h+ + i h×|)
    at ``merger_frac*L`` and truncates to length L, exactly as the surrogate's
    fixed-window modes are aligned.
    """

    def __init__(self, L, delta_t, f_lower, merger_frac, ref_distance_mpc):
        self.delta_t = float(delta_t)
        self.L = int(L)
        self.merger_frac = float(merger_frac)
        self.f_lower = float(f_lower)
        self.ref_distance_mpc = float(ref_distance_mpc)
        self.reduced_order = True
        self.meta = {"td_fixed_window": True, "merger_frac": self.merger_frac}

    def _place(self, arr):
        L, ms = self.L, int(self.merger_frac * self.L)
        pk = int(np.abs(arr).argmax()) if len(arr) else 0
        out = np.zeros(L, dtype=np.float64)
        lo = ms - pk
        a0 = max(0, -lo)
        o0 = max(0, lo)
        n = min(len(arr) - a0, L - o0)
        if n > 0:
            out[o0:o0 + n] = arr[a0:a0 + n]
        return out

    def predict(self, m1, m2, s1z=0.0, s2z=0.0, inc=0.0, ecc=0.0, sigma_level=None):
        from pycbc.waveform import get_td_waveform
        hp, hc = get_td_waveform(approximant="SEOBNRv4HM", mass1=m1, mass2=m2,
                                 spin1z=s1z, spin2z=s2z, inclination=inc, coa_phase=0.0,
                                 distance=self.ref_distance_mpc, delta_t=self.delta_t,
                                 f_lower=self.f_lower)
        hp = np.asarray(hp, dtype=np.float64)
        hc = np.asarray(hc, dtype=np.float64)
        # align both polarizations by the merger of the analytic amplitude
        analytic = np.abs(hp + 1j * hc)
        ms = int(self.merger_frac * self.L)
        pk = int(analytic.argmax()) if len(analytic) else 0

        def place(arr):
            out = np.zeros(self.L, dtype=np.float64)
            lo = ms - pk
            a0 = max(0, -lo); o0 = max(0, lo)
            n = min(len(arr) - a0, self.L - o0)
            if n > 0:
                out[o0:o0 + n] = arr[a0:a0 + n]
            return out

        return (WaveformPrediction(place(hp), sample_rate=self.delta_t),
                WaveformPrediction(place(hc), sample_rate=self.delta_t))

    def batch_predict(self, full, batch_size=256):
        hp, hc = [], []
        for p in np.asarray(full):
            a, b = self.predict(*p[:5])
            hp.append(a); hc.append(b)
        return hp, hc


def inject_2det(truth, sample_rate, duration, f_lower, detectors, seed):
    """Project a true SEOBNRv4HM signal into each detector and add coloured noise."""
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
    dets, snrs = {}, []
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
    return dets, net


def build_event(truth, snr_net, sample_rate, duration, f_lower, detectors, seed):
    """Scale distance so the network SNR hits the target, then inject."""
    truth = dict(truth); truth.setdefault("distance", 1000.0)  # nominal; rescaled below
    dets, net0 = inject_2det(truth, sample_rate, duration, f_lower, detectors, seed)
    truth2 = dict(truth); truth2["distance"] = truth["distance"] * (net0 / snr_net)
    dets, net = inject_2det(truth2, sample_rate, duration, f_lower, detectors, seed)
    ev = EventData(event_name="INJ_HM_CAT", gps_time=GPS, detectors=dets, f_lower=f_lower)
    return ev, truth2, net


def plot_corner(rs, rp, truths, labels, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import corner
    cols = list(zip(rs.samples.T, rp.samples.T))
    rng = []
    for j, (a, b) in enumerate(cols):
        lo = min(a.min(), b.min(), truths[j]); hi = max(a.max(), b.max(), truths[j])
        pad = 0.05 * (hi - lo + 1e-9)
        rng.append((lo - pad, hi + pad))
    fig = corner.corner(rp.samples, labels=labels, color="C1", range=rng,
                        hist_kwargs={"density": True})
    corner.corner(rs.samples, fig=fig, labels=labels, color="C0", range=rng,
                  truths=truths, truth_color="C3", quantiles=[0.05, 0.5, 0.95],
                  show_titles=True, title_fmt=".2f", hist_kwargs={"density": True})
    fig.suptitle(title, y=1.01)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--sample-rate", type=float, default=4096.0)
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=600)
    ap.add_argument("--bank-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engines", nargs="+", default=["surrogate", "pycbc"])
    ap.add_argument("--pycbc-idx", nargs="+", type=int, default=None,
                    help="injection indices that get the (slow) pycbc cross-check; "
                         "default = all. Surrogate always runs on every injection.")
    ap.add_argument("--smoke", action="store_true",
                    help="one injection, tiny chain, no pycbc -- validates the pipeline fast")
    ap.add_argument("--outdir", default="docs/images/hm_pe_catalogue")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    truths = TRUTHS[:1] if args.smoke else TRUTHS
    engines_wanted = ["surrogate"] if args.smoke else args.engines
    nsteps = 60 if args.smoke else args.nsteps
    nwalkers = 16 if args.smoke else args.nwalkers

    surrogate = load_surrogate(args.surrogate, device=args.device)
    ref = float(getattr(surrogate, "ref_distance_mpc", 1.0))
    pycbc_engine = CoherentPyCBCEngine(surrogate.L, surrogate.delta_t, surrogate.f_lower,
                                       surrogate.merger_frac, ref)
    engines = {"surrogate": surrogate, "pycbc": pycbc_engine}

    pycbc_idx = set(range(len(truths))) if args.pycbc_idx is None else set(args.pycbc_idx)
    tkeys = ["m1", "m2", "s1z", "s2z", "inc", "distance"]
    catalogue = []

    for i, base in enumerate(truths):
        tag = f"inj{i:02d}"
        edir = os.path.join(args.outdir, tag)
        os.makedirs(edir, exist_ok=True)
        want = [e for e in engines_wanted if e != "pycbc" or i in pycbc_idx]

        rfile = os.path.join(edir, "result.json")
        if os.path.exists(rfile):
            prev = json.load(open(rfile))
            if all(e in prev for e in want):
                catalogue.append(prev); write_summary(args.outdir, catalogue)
                print(f"[skip] {tag} (already done: {', '.join(want)})", flush=True)
                continue

        ev, truth2, net = build_event(base, args.snr, args.sample_rate, args.duration,
                                      surrogate.f_lower, args.detectors, args.seed + i)
        print(f"\n[{tag}] net SNR {net:.1f}  m1={base['m1']:.0f} m2={base['m2']:.0f} "
              f"q={base['m1']/base['m2']:.1f} inc={base['inc']:.2f} dL={truth2['distance']:.0f}Mpc",
              flush=True)

        prior = Prior(bounds={"m1": (20, 100), "m2": (20, 100), "s1z": (-0.9, 0.9),
                              "s2z": (-0.9, 0.9), "inc": (0.0, np.pi)}, fixed={"ecc": 0.0})
        truth_vec = [truth2[k] for k in tkeys]
        row = {"injection": tag, "truth": {k: truth2[k] for k in tkeys}, "snr": round(net, 2)}
        res_store = {}
        for eng_tag in want:
            est = GWEventEstimator(engines[eng_tag], ev, prior=prior, likelihood="coherent",
                                   distance_bounds=(50.0, 5000.0))
            est.fixed_sky = (base["ra"], base["dec"], base["psi"])
            bank = WaveformBank(surrogate, prior=prior)
            bank.build(n_samples=args.bank_size, seed=args.seed + i)
            t0 = time.time()
            res = est.run_mcmc(nwalkers=nwalkers, nsteps=nsteps, bank=bank, p0=None,
                               progress=False, seed=args.seed + i)
            dt = time.time() - t0
            summ = {}
            for j, n in enumerate(res.param_names):
                lo, md, hi = np.percentile(res.samples[:, j], [5, 50, 95])
                summ[n] = [float(md), float(lo), float(hi)]
            row[eng_tag] = {"seconds": round(dt, 1),
                            "acceptance": round(float(res.acceptance_fraction), 3),
                            "summary": summ}
            res_store[eng_tag] = res
            print(f"    {eng_tag:9s} {dt:6.1f}s acc={res.acceptance_fraction:.2f}  "
                  + "  ".join(f"{n}={summ[n][0]:.2f}" for n in ["m1", "m2", "inc", "distance"]
                              if n in summ), flush=True)

        if "surrogate" in res_store and "pycbc" in res_store:
            row["speedup"] = round(row["pycbc"]["seconds"] / max(row["surrogate"]["seconds"], 1e-9), 2)
            order = res_store["surrogate"].param_names
            tv = [truth2[k] for k in order]
            plot_corner(res_store["surrogate"], res_store["pycbc"], tv, order,
                        os.path.join(edir, "corner.png"),
                        f"{tag}: surrogate (blue) vs pycbc (orange), truth (red)  "
                        f"q={base['m1']/base['m2']:.1f}, inc={base['inc']:.2f}")
        elif "surrogate" in res_store:
            order = res_store["surrogate"].param_names
            tv = [truth2[k] for k in order]
            plot_corner(res_store["surrogate"], res_store["surrogate"], tv, order,
                        os.path.join(edir, "corner.png"), f"{tag}: surrogate vs truth (red)")
        json.dump(row, open(os.path.join(edir, "result.json"), "w"), indent=2)
        catalogue.append(row)
        write_summary(args.outdir, catalogue)

    print(f"\nCatalogue complete: {len(catalogue)} injections.")


def write_summary(outdir, rows):
    lines = ["# Coherent HM injection catalogue: surrogate vs pycbc vs truth\n",
             f"{len(rows)} injections · coherent antenna-response likelihood · sky fixed at truth.\n",
             "| inj | truth (m1,m2,inc,dL) | surr (m1,inc,dL) | pycbc (m1,inc,dL) | SNR | speedup |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        t = r["truth"]
        s = r.get("surrogate", {}).get("summary", {})
        p = r.get("pycbc", {}).get("summary", {})

        def g(d, k):
            return f"{d[k][0]:.1f}" if k in d else "—"
        lines.append(
            f"| {r['injection']} | {t['m1']:.0f},{t['m2']:.0f},{t['inc']:.2f},{t['distance']:.0f} "
            f"| {g(s,'m1')},{g(s,'inc')},{g(s,'distance')} "
            f"| {g(p,'m1')},{g(p,'inc')},{g(p,'distance')} "
            f"| {r['snr']:.1f} | {r.get('speedup','—')}× |")
    open(os.path.join(outdir, "summary.md"), "w").write("\n".join(lines))
    json.dump(rows, open(os.path.join(outdir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
