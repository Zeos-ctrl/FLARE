#!/usr/bin/env python
"""Build a merger-aligned waveform bank (.npz) offline.

Generation is the only genuinely slow step in the pipeline (SEOBNRv4 is ~250 ms
per waveform, ~100x IMRPhenomD), while reduced-order training runs in seconds.
Building the bank ONCE and reusing it via ``ProjectConfig.waveform_bank`` lets
every subsequent experiment -- new SVD rank, new loss, new head scheme -- re-run
without paying generation again.

Why offline rather than in-process: the reduced-order build (Hilbert ->
complex128, float64 SVD) already peaks ~10 GB at N=10k, L=16384. Doing generation
in the same process pushes it over. Here the strain is streamed straight to a
float32 .npy memmap, so this stage's RAM stays flat regardless of N and a crash
at 90% doesn't lose the work (rerun resumes from the last finished chunk).

Bank format (consumed by ``WaveformGenerator._load_bank``):
    strain (M, L) float32 -- already merger-aligned/resized to L
    params (M, 6) float64 -- [m1, m2, s1z, s2z, inclination, eccentricity]

Example -- the 20k fixed-window aligned-spin SEOBNRv4 bank:
    python scripts/build_bank.py --approximant SEOBNRv4 --n 20000 \
        --length 16384 --sample-rate 4096 --fixed-window \
        --mass-min 20 --mass-max 100 --spin-min -0.9 --spin-max 0.9 \
        --out data_cache/seobnrv4_spin_fixedwin_20k_L16384.npz
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import WaveformGenerator  # noqa: E402

logger = logging.getLogger("build_bank")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--approximant", default="SEOBNRv4")
    p.add_argument("--n", type=int, required=True, help="number of waveforms")
    p.add_argument("--length", type=int, required=True,
                   help="output samples L (fixed-window: L*delta_t = window seconds)")
    p.add_argument("--sample-rate", type=float, default=4096.0)
    p.add_argument("--f-lower", type=float, default=20.0)
    p.add_argument("--mass-min", type=float, default=20.0)
    p.add_argument("--mass-max", type=float, default=100.0)
    p.add_argument("--spin-min", type=float, default=-0.9)
    p.add_argument("--spin-max", type=float, default=0.9)
    p.add_argument("--fixed-window", action="store_true",
                   help="PE-faithful fixed physical grid (merger at 0.9L, no warp/trim)")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = leave FLARE_GEN_WORKERS / generator default alone")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.workers:
        os.environ["FLARE_GEN_WORKERS"] = str(args.workers)

    delta_t = 1.0 / args.sample_rate
    gen = WaveformGenerator(waveform_length=args.length, delta_t=delta_t,
                            f_lower=args.f_lower, approximant=args.approximant,
                            fixed_window=args.fixed_window)

    # inc/ecc pinned to 0: inclination is a pure amplitude scale (the match
    # normalises it out) and IMRPhenomD/SEOBNRv4 ignore eccentricity. A range of
    # low==high is pinned to the constant by sample_parameters.
    params = gen.sample_parameters(
        args.n, method="lhs",
        mass_range=(args.mass_min, args.mass_max),
        spin_range=(args.spin_min, args.spin_max),
        incl_range=(0.0, 0.0), ecc_range=(0.0, 0.0),
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    strain_path = args.out + ".strain.npy"
    params_path = args.out + ".params.npy"
    done_path = args.out + ".done"

    # Resume support: the params memmap is the source of truth for the sampling,
    # so an interrupted run must reuse the SAME parameters or rows and params
    # would desynchronise.
    start = 0
    if os.path.exists(strain_path) and os.path.exists(params_path) and os.path.exists(done_path):
        prev = np.load(params_path, mmap_mode="r")
        if prev.shape == params.shape and np.allclose(prev, params):
            start = int(np.loadtxt(done_path))
            logger.info("Resuming from row %d", start)
        else:
            logger.warning("Existing partial bank has different params; starting over")

    strain = np.lib.format.open_memmap(
        strain_path, mode="r+" if start else "w+",
        dtype=np.float32, shape=(args.n, args.length))
    if not start:
        np.save(params_path, params)

    t0 = time.time()
    for lo in range(start, args.n, args.chunk):
        hi = min(lo + args.chunk, args.n)
        waveforms = gen._generate_all(params[lo:hi], clean=True)
        strain[lo:hi] = np.asarray(waveforms, dtype=np.float32)
        strain.flush()
        with open(done_path, "w") as f:
            f.write(str(hi))

        elapsed = time.time() - t0
        rate = (hi - start) / max(elapsed, 1e-9)
        eta = (args.n - hi) / max(rate, 1e-9)
        logger.info("%d/%d  %.1f wf/s  elapsed %.1f min  eta %.1f min",
                    hi, args.n, rate, elapsed / 60, eta / 60)

    logger.info("Generation done in %.1f min; writing %s", (time.time() - t0) / 60, args.out)
    np.savez(args.out, strain=np.asarray(strain), params=params)

    for tmp in (strain_path, params_path, done_path):
        os.remove(tmp)

    size_gb = os.path.getsize(args.out) / 1e9
    logger.info("Wrote %s (%.2f GB, N=%d, L=%d)", args.out, size_gb, args.n, args.length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
