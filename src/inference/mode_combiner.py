"""Spin-weighted spherical-harmonic mode combiner, calibrated to LAL/pycbc.

Turns a set of (l, m) modes h_lm(t) into observed polarizations at a given
inclination, coalescence phase and distance:

    h+ - i h× = Σ_{l, m>0} [ Y(l, m)·h_lm + Y(l,-m)·(-1)^l·conj(h_lm) ]
    Y(l, m)   = SpinWeightedSphericalHarmonic(inclination, π/2 - coa_phase, -2, l, m)
    h+ =  Re(Σ),   h× = -Im(Σ)

The azimuth (π/2 - coa_phase) and the -Im sign were fixed against
pycbc.get_td_waveform('SEOBNRv4HM'), reproducing the full waveform to a match of
1.00000 across inclination, coa_phase and mass ratio. The m<0 partners use the
aligned-spin symmetry h_{l,-m} = (-1)^l conj(h_lm).

This is the combiner for ALIGNED-spin models (inertial frame == co-precessing
frame). A precessing model rotates its co-precessing modes by the time-dependent
Euler angles BEFORE calling this; that hook belongs in the model, not here.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def spherical_weights(modes_lm, inclination: float, coa_phase: float = 0.0):
    """{(l,m): (Y_pos, Y_neg·(-1)^l)} complex weights for a fixed geometry.

    Geometry-only (independent of the waveform), so they can be precomputed once
    and reused / moved to the GPU for a differentiable combiner.
    """
    import lal
    az = np.pi / 2.0 - coa_phase
    out = {}
    for (l, m) in modes_lm:
        y_pos = complex(lal.SpinWeightedSphericalHarmonic(inclination, az, -2, l, m))
        y_neg = complex(lal.SpinWeightedSphericalHarmonic(inclination, az, -2, l, -m)) * ((-1) ** l)
        out[(l, m)] = (y_pos, y_neg)
    return out


def combine_modes(modes: Dict[Tuple[int, int], np.ndarray], inclination: float,
                  coa_phase: float = 0.0, distance_mpc: float = 1.0,
                  ref_distance_mpc: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Combine {(l,m): complex ndarray} → (h_plus, h_cross) real arrays.

    ``modes`` holds the m>0 modes on a common time grid. Distance scales the
    output linearly from ``ref_distance_mpc`` (where the modes were generated) to
    ``distance_mpc``.
    """
    w = spherical_weights(list(modes.keys()), inclination, coa_phase)
    L = max(len(v) for v in modes.values())
    out = np.zeros(L, dtype=complex)
    for lm, h in modes.items():
        hm = np.zeros(L, dtype=complex)
        hm[:len(h)] = h
        y_pos, y_neg = w[lm]
        out += y_pos * hm + y_neg * np.conj(hm)
    scale = ref_distance_mpc / distance_mpc
    return scale * out.real, -scale * out.imag
