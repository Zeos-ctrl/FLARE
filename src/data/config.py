"""
Global runtime configuration shared across model construction.

The refactored model builders (``src/models/*``) expect a handful of
module-level constants (device, model type, Fourier-feature settings). This
module is the single source of truth for them: it reads defaults from the
project-level ``config.yaml`` when present and otherwise falls back to sensible
values, so that ``import``-ing the models never fails on a fresh checkout.

Override the config path with the ``FLARE_CONFIG`` environment variable.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import torch
import yaml

logger = logging.getLogger(__name__)

# Project root is two levels up from this file: <root>/src/data/config.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.environ.get("FLARE_CONFIG", os.path.join(_PROJECT_ROOT, "config.yaml"))


def _load_yaml(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to parse %s (%s); using defaults", path, exc)
    return {}


_CFG = _load_yaml(_CONFIG_PATH)
_MODEL_CFG: Dict[str, Any] = _CFG.get("model", {})


def _resolve_device(preference: str) -> torch.device:
    """Honour the requested device but fall back to CPU when CUDA is absent."""
    if preference.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(preference)


# --- Device --------------------------------------------------------------
DEVICE = _resolve_device(str(_CFG.get("device", "cuda")))

# --- Model selection -----------------------------------------------------
MODEL_TYPE: str = _MODEL_CFG.get("model_type", "mlp")

# --- Fourier-feature settings -------------------------------------------
# The YAML carries a single set of Fourier settings under ``model``; we expose
# them separately for the amplitude and phase networks so each can be tuned
# independently later without breaking the builder API.
_FOURIER_BANDS = _MODEL_CFG.get("fourier_bands", 16)
_FOURIER_MAX_FREQ = _MODEL_CFG.get("fourier_max_freq", 10.0)
_FOURIER_LEARNABLE = _MODEL_CFG.get("fourier_learnable", False)

AMP_FOURIER_BANDS: int = _MODEL_CFG.get("amp_fourier_bands", _FOURIER_BANDS)
AMP_FOURIER_MAX_FREQ: float = _MODEL_CFG.get("amp_fourier_max_freq", _FOURIER_MAX_FREQ)
AMP_FOURIER_LEARNABLE: bool = _MODEL_CFG.get("amp_fourier_learnable", _FOURIER_LEARNABLE)

PHASE_FOURIER_BANDS: int = _MODEL_CFG.get("phase_fourier_bands", _FOURIER_BANDS)
PHASE_FOURIER_MAX_FREQ: float = _MODEL_CFG.get("phase_fourier_max_freq", _FOURIER_MAX_FREQ)
PHASE_FOURIER_LEARNABLE: bool = _MODEL_CFG.get("phase_fourier_learnable", _FOURIER_LEARNABLE)


__all__ = [
    "DEVICE",
    "MODEL_TYPE",
    "AMP_FOURIER_BANDS",
    "AMP_FOURIER_MAX_FREQ",
    "AMP_FOURIER_LEARNABLE",
    "PHASE_FOURIER_BANDS",
    "PHASE_FOURIER_MAX_FREQ",
    "PHASE_FOURIER_LEARNABLE",
]
