#!/usr/bin/env python
"""End-to-end smoke test: prove the FLARE pipeline works on this machine.

Generates a handful of SEOBNRv4 waveforms, trains a tiny surrogate for a couple of
epochs on CPU, then loads the checkpoint back and predicts a waveform. It exercises
the whole path — waveform generation (PyCBC/LALSuite), the reduced-order + network
training, checkpointing, and inference — in ~1-2 minutes, without a GPU or any
pre-trained weights.

    python scripts/smoke_test.py

Exits non-zero if any stage fails. Output goes to checkpoints/smoke_test/ (git-ignored).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import ProjectConfig
from src.core.trainer import Trainer
from src.inference.predictor import WaveformPredictor

PROJECT = "smoke_test"


def main() -> int:
    print("[1/3] Training a tiny surrogate (60 samples, 2 epochs, CPU)...")
    config = ProjectConfig(
        project_name=PROJECT,
        num_samples=60,
        waveform="SEOBNRv4",
        num_epochs=2,
        device="cpu",
    )
    Trainer(config).run_training()

    print("[2/3] Loading the checkpoint and predicting a waveform...")
    predictor = WaveformPredictor(config.project_path, device="cpu")
    h_plus, h_cross = predictor.predict(m1=35, m2=30, s1z=0.1, s2z=-0.2, inc=0.5)

    print("[3/3] Checking the output...")
    n = len(h_plus.data)
    finite = bool((h_plus.data == h_plus.data).all())  # no NaNs
    assert n > 0, "predicted waveform is empty"
    assert finite, "predicted waveform contains NaNs"

    print(f"\nOK - trained, saved to {config.project_path}, and predicted a "
          f"{n}-sample waveform.")
    print("FLARE is working end-to-end on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
