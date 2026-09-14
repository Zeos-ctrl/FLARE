"""Reconstruct PHYSICAL-time waveforms from the merger-aligned surrogate.

The surrogate predicts amplitude ``A(u)`` and phase ``phi(u)`` on the
merger-aligned NORMALISED grid ``u in [0, 1]`` (merger pinned at
``MERGER_FRAC``), because the generator resizes every training waveform onto that
grid with a monotone (PCHIP) time warp -- great for the training match metric,
but it *removes the absolute time/frequency scale*. To use the surrogate as a
matched-filter template against real detector data we must put it back on a
physical time axis.

Given the physical inspiral+merger duration ``T_pre`` (start->merger) and
ringdown duration ``T_post`` (merger->end), the physical time of each normalised
node is the inverse of the resize warp::

    t(u) = PCHIP{(0, 0), (MERGER_FRAC, T_pre), (1, T_pre + T_post)}(u)

We then resample ``A`` and ``phi`` onto a uniform ``delta_t`` grid and form
``h = A cos(phi)``. Round-tripping a pycbc waveform through the forward resize and
this inverse (with the true durations) reproduces it to match ~1.0.

``DurationModel`` predicts ``(T_pre, T_post)`` from masses in ~0.2 ms via pycbc's
analytic ``get_imr_duration`` plus a small polynomial correction (coefficients in
``data_cache/duration_model_<approx>.npz``). NOTE: matched-filter PE is extremely
sensitive to this duration (a chirp of N cycles de-phases by ~2*pi*N*dT/T), so the
model's ~0.4-2% accuracy is adequate for waveform reproduction and coarse
searches but NOT for high-precision low-mass PE -- see the module tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

MERGER_FRAC = 0.9
_DEFAULT_MODEL = "data_cache/duration_model_imrphenomd.npz"


def _imr_eta_deg4_features(m1: np.ndarray, m2: np.ndarray, f_lower: float,
                           approximant: str) -> np.ndarray:
    """Design matrix used by the saved duration model: polynomial in
    (log get_imr_duration, log eta)."""
    from pycbc.pnutils import get_imr_duration
    m1 = np.atleast_1d(m1).astype(float); m2 = np.atleast_1d(m2).astype(float)
    eta = (m1 * m2) / (m1 + m2) ** 2
    imr = np.array([get_imr_duration(a, b, 0.0, 0.0, f_lower, approximant)
                    for a, b in zip(m1, m2)])
    x, y = np.log(imr), np.log(eta)
    cols = []
    for i in range(5):
        for j in range(4 - i if i < 4 else 1):
            cols.append(x ** i * y ** j)
    return np.stack(cols, axis=1)          # (N, 11)


@dataclass
class DurationModel:
    """Predict physical (T_pre, T_post) durations from component masses."""
    cpre: np.ndarray
    cpost: np.ndarray
    f_lower: float = 20.0
    approximant: str = "IMRPhenomD"

    @classmethod
    def load(cls, path: str = _DEFAULT_MODEL) -> "DurationModel":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"duration model not found: {path}. Fit one with "
                f"scratchpad/pe_calib4.py.")
        d = np.load(path)
        return cls(cpre=d["cpre"], cpost=d["cpost"],
                   f_lower=float(d["f_lower"]), approximant=str(d["approximant"]))

    def predict(self, m1, m2) -> Tuple[np.ndarray, np.ndarray]:
        X = _imr_eta_deg4_features(m1, m2, self.f_lower, self.approximant)
        return np.exp(X @ self.cpre), np.exp(X @ self.cpost)


def warp_to_physical(amp_u: np.ndarray, phase_u: np.ndarray,
                     t_pre: float, t_post: float, delta_t: float,
                     merger_frac: float = MERGER_FRAC,
                     taper_frac: float = 0.05) -> np.ndarray:
    """Inverse of the merger-aligned resize: place A(u), phi(u) on a uniform
    physical ``delta_t`` grid and return ``h = A cos(phi)`` (length depends on the
    total physical duration).

    ``taper_frac`` applies a cosine (Hann) ramp over the leading ``taper_frac`` of
    the samples. The surrogate's envelope starts at ~1e-3 of peak (the resize trim
    threshold), so an untapered start is a step discontinuity that leaks broadband
    power and produces spurious matched-filter SNR against real data -- the taper
    removes it (matched pycbc waveforms are likewise tapered)."""
    from scipy.interpolate import PchipInterpolator

    L = len(amp_u)
    u = np.linspace(0.0, 1.0, L)
    t_of_u = PchipInterpolator([0.0, merger_frac, 1.0],
                               [0.0, float(t_pre), float(t_pre + t_post)])(u)
    total = float(t_pre + t_post)
    n = max(2, int(np.ceil(total / delta_t)))
    t_uni = np.arange(n) * delta_t
    amp_p = np.interp(t_uni, t_of_u, amp_u)
    phase_p = np.interp(t_uni, t_of_u, phase_u)
    h = amp_p * np.cos(phase_p)
    if taper_frac > 0:
        k = max(2, int(taper_frac * n))
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, k)))
        h[:k] *= ramp
    return h


def reconstruct_physical(amp_u: np.ndarray, phase_u: np.ndarray,
                         m1: float, m2: float, delta_t: float,
                         duration_model: Optional[DurationModel] = None,
                         durations: Optional[Tuple[float, float]] = None,
                         merger_frac: float = MERGER_FRAC) -> np.ndarray:
    """Physical-time strain from surrogate A(u), phi(u) for one (m1, m2).

    Supply ``durations=(T_pre, T_post)`` to use exact/known durations, otherwise a
    ``DurationModel`` (defaults to the saved one) predicts them from the masses.
    """
    if durations is not None:
        t_pre, t_post = durations
    else:
        dm = duration_model or DurationModel.load()
        tp, tq = dm.predict(m1, m2)
        t_pre, t_post = float(tp[0]), float(tq[0])
    return warp_to_physical(amp_u, phase_u, t_pre, t_post, delta_t, merger_frac)
