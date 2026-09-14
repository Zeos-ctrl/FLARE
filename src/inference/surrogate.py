"""Uniform surrogate-waveform interface.

One inference API across model families. Each concrete surrogate declares its
intrinsic ``param_names``, its ``modes``, and its ``frame`` ("aligned",
"precessing", "eccentric"); the base class turns predicted modes into observed
polarizations via the calibrated spherical-harmonic combiner and exposes a
pycbc-style ``get_td_waveform`` plus a ``predict``/``batch_predict`` shim that
plugs directly into :class:`~src.inference.estimator.GWEventEstimator`.

    AlignedModeSurrogate  -- aligned-spin higher-mode models (SEOBNRv4HM):
                             {m1, m2, s1z, s2z}, inertial-frame Ylm sum.
    (future) PrecessingModeSurrogate -- + in-plane spins, co-precessing modes
                             rotated by the Euler angles before the sum.
    (future) EccentricModeSurrogate  -- + eccentricity / anomaly.

The point of the shared base is that PE and any downstream consumer target the
same ``get_td_waveform(**params, inclination, coa_phase, distance)`` regardless
of which physics the model carries.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.data.features import FeatureExtractor
from src.inference.mode_combiner import combine_modes
from src.inference.predictor import WaveformPrediction
from src.models.rom import CoeffMLP

# Canonical full parameter vector shared with the estimator.
PARAM_ORDER = ["m1", "m2", "s1z", "s2z", "inc", "ecc"]


class SurrogateWaveform:
    """Base class: modes -> polarizations -> pycbc-style waveforms."""

    param_names: List[str] = []
    modes: List[Tuple[int, int]] = []
    frame: str = "aligned"

    # set by subclasses
    delta_t: float = 1.0 / 4096
    L: int = 16384
    merger_frac: float = 0.9
    ref_distance_mpc: float = 1.0

    def generate_modes(self, **intrinsic) -> Dict[Tuple[int, int], np.ndarray]:
        """Return {(l,m): complex TD mode} for one binary (intrinsic params)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _combine(self, modes, inclination, coa_phase, distance):
        return combine_modes(modes, inclination, coa_phase,
                             distance_mpc=distance, ref_distance_mpc=self.ref_distance_mpc)

    def get_td_waveform(self, inclination: float = 0.0, coa_phase: float = 0.0,
                        distance: float = 1.0, **intrinsic):
        """pycbc-style: return (h_plus, h_cross) as pycbc TimeSeries.

        ``intrinsic`` are this model's ``param_names`` (e.g. mass1, mass2,
        spin1z, spin2z). The merger sits at ``merger_frac*L``; the epoch places it
        at t=0 as pycbc does.
        """
        from pycbc.types import TimeSeries
        modes = self.generate_modes(**intrinsic)
        hp, hc = self._combine(modes, inclination, coa_phase, distance)
        epoch = -int(round(self.merger_frac * self.L)) * self.delta_t
        return (TimeSeries(hp, delta_t=self.delta_t, epoch=epoch),
                TimeSeries(hc, delta_t=self.delta_t, epoch=epoch))

    # --- estimator compatibility shim (mirrors WaveformPredictor) -------
    @property
    def meta(self) -> dict:
        return {"td_fixed_window": True, "merger_frac": self.merger_frac}

    def predict(self, m1, m2, s1z=0.0, s2z=0.0, inc=0.0, ecc=0.0, sigma_level=None):
        modes = self.generate_modes(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z)
        hp, hc = self._combine(modes, inc, 0.0, self.ref_distance_mpc)
        return (WaveformPrediction(hp, sample_rate=self.delta_t),
                WaveformPrediction(hc, sample_rate=self.delta_t))

    def batch_predict(self, full_params, batch_size: int = 256):
        p = np.asarray(full_params, dtype=float)
        mb = self._generate_modes_batch(p[:, :4])   # (N,L) complex per mode
        hp_list, hc_list = [], []
        for i in range(len(p)):
            modes = {lm: mb[lm][i] for lm in self.modes}
            hp, hc = self._combine(modes, p[i, 4], 0.0, self.ref_distance_mpc)
            hp_list.append(WaveformPrediction(hp, sample_rate=self.delta_t))
            hc_list.append(WaveformPrediction(hc, sample_rate=self.delta_t))
        return hp_list, hc_list


class AlignedModeSurrogate(SurrogateWaveform):
    """Aligned-spin higher-mode surrogate (e.g. SEOBNRv4HM).

    Loads a checkpoint written by ``scratch/hm/train_hm_match.py``: per-mode
    amplitude/phase ``CoeffMLP`` nets + SVD bases + normalisation, plus meta.
    """

    frame = "aligned"

    def __init__(self, path: str, device: str = "cuda"):
        self.device = device
        meta = json.load(open(os.path.join(path, "meta.json")))
        self.approximant = meta["approximant"]
        self.modes = [tuple(m) for m in meta["modes"]]
        self.features = meta["features"]
        self.feat_mean = np.asarray(meta["feat_mean"], dtype=np.float32)
        self.feat_std = np.asarray(meta["feat_std"], dtype=np.float32)
        self.L = int(meta["L"]); self.delta_t = float(meta["delta_t"])
        self.merger_frac = float(meta["merger_frac"])
        self.ref_distance_mpc = float(meta.get("ref_distance_mpc", 1.0))
        self.f_lower = float(meta.get("f_lower", 20.0))
        self.reduced_order = True
        self.param_names = ["mass1", "mass2", "spin1z", "spin2z"]
        hidden, depth = int(meta["hidden"]), int(meta["depth"])
        in_dim = len(self.features)

        arr = np.load(os.path.join(path, "arrays.npz"))
        self._m = {}
        for lm in self.modes:
            key = f"{lm[0]}_{lm[1]}"
            ab = torch.from_numpy(arr[f"{key}_ab"]).to(device)
            pb = torch.from_numpy(arr[f"{key}_pb"]).to(device)
            anet = CoeffMLP(in_dim, ab.shape[1], hidden=hidden, depth=depth, fourier_bands=0).to(device)
            pnet = CoeffMLP(in_dim, pb.shape[1], hidden=hidden, depth=depth, fourier_bands=0).to(device)
            anet.load_state_dict(torch.load(os.path.join(path, f"amp_{key}.pt"),
                                            map_location=device, weights_only=True))
            pnet.load_state_dict(torch.load(os.path.join(path, f"ph_{key}.pt"),
                                            map_location=device, weights_only=True))
            anet.eval(); pnet.eval()
            self._m[lm] = dict(
                ab=ab, pb=pb, anet=anet, pnet=pnet,
                acm=torch.tensor(arr[f"{key}_acm"], device=device),
                acs=torch.tensor(arr[f"{key}_acs"], device=device),
                pcm=torch.tensor(arr[f"{key}_pcm"], device=device),
                pcs=torch.tensor(arr[f"{key}_pcs"], device=device),
                ascale=float(arr[f"{key}_ascale"]))

    def _theta(self, params6: np.ndarray) -> torch.Tensor:
        feats = FeatureExtractor.compute_features(np.asarray(params6, dtype=float), self.features)
        norm = (feats - self.feat_mean) / self.feat_std
        return torch.from_numpy(norm.astype(np.float32)).to(self.device)

    def _generate_modes_batch(self, params4: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        p = np.asarray(params4, dtype=float)
        p6 = np.zeros((len(p), 6)); p6[:, :4] = p
        theta = self._theta(p6)
        out = {}
        with torch.no_grad():
            for lm, m in self._m.items():
                amp = torch.clamp((m["anet"](theta) * m["acs"] + m["acm"]) @ m["ab"].T, min=0) * m["ascale"]
                ph = (m["pnet"](theta) * m["pcs"] + m["pcm"]) @ m["pb"].T
                out[lm] = (amp * torch.exp(1j * ph)).cpu().numpy()
        return out

    def generate_modes(self, mass1, mass2, spin1z=0.0, spin2z=0.0):
        mb = self._generate_modes_batch(np.array([[mass1, mass2, spin1z, spin2z]]))
        return {lm: v[0] for lm, v in mb.items()}


def load_surrogate(path: str, device: str = "cuda") -> SurrogateWaveform:
    """Load a surrogate, dispatching on the checkpoint's declared frame."""
    meta = json.load(open(os.path.join(path, "meta.json")))
    frame = meta.get("frame", "aligned")
    if frame == "aligned":
        return AlignedModeSurrogate(path, device=device)
    raise NotImplementedError(f"frame '{frame}' not yet implemented")
