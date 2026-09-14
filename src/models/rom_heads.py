"""Registry of reduced-order (ROM) *head architectures*.

A ROM surrogate predicts a waveform as ``h = A(t)*cos(phi(t))`` from a few SVD
coefficients. HOW the phase is split across networks is the main architectural
knob, so -- in the spirit of the pipeline's swappable ``custom_models/`` -- the
choices live here as named entries you select with ``ProjectConfig.rom_heads``
(or ``SettingsModel.rom_heads`` from the dashboard). Add a new scheme by adding a
row; the trainer/predictor read the resolved flags, so nothing else changes.

Each scheme resolves to two low-level flags:
  * ``scale_factor``   -- split phase into scale * shape and SVD only the shape
                          (needed for long low-mass waveforms; raw-phase SVD caps
                          ~0.65 because the dominant coeff *is* the total phase).
  * ``separate_scale`` -- give the (precision-critical) log-scale its own network
                          instead of sharing the phase head.
"""
from __future__ import annotations

ROM_HEADS = {
    # 2 heads: amplitude net + raw-phase SVD net. The original scheme; simple,
    # but the phase net must predict the total accumulated phase to ~1e-4 -- fine
    # for short (high-mass) waveforms, poor for long low-mass ones.
    "amp_phase": {"scale_factor": False, "separate_scale": False,
                  "heads": ("amp", "phase"),
                  "doc": "2 heads: amplitude + raw-phase SVD (classic)."},

    # 2 heads: amplitude net + one phase net that predicts [shape coeffs, log-scale]
    # together. Scale-factored (handles low mass) but the scale shares capacity.
    "amp_shape_scale_2h": {"scale_factor": True, "separate_scale": False,
                           "heads": ("amp", "phase"),
                           "doc": "2 heads: amplitude + shared [shape, log-scale]."},

    # 3 heads: amplitude + shape + a dedicated scale net. The precision-critical
    # log-scale gets its own network -- highest match on the low-mass benchmark.
    "amp_shape_scale": {"scale_factor": True, "separate_scale": True,
                        "heads": ("amp", "shape", "scale"),
                        "doc": "3 heads: amplitude + shape + dedicated scale (best)."},
}

DEFAULT = "amp_shape_scale"


def resolve(name: str) -> dict:
    """Return the flag dict for a named head scheme (raises on unknown name)."""
    if name not in ROM_HEADS:
        raise ValueError(
            f"unknown rom_heads '{name}'; options: {sorted(ROM_HEADS)}")
    return ROM_HEADS[name]
