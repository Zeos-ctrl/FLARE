from __future__ import annotations

import torch.nn as nn

from src.data.config import *
from src.models.custom import load_custom_model
from src.models.mlp import AmplitudeDNN_Full
from src.models.mlp import PhaseDNN_Full


def make_amp_model(
    in_param_dim: int,
    params,
) -> nn.Module:
    """
    Returns the amplitude model specified by MODEL.amp_model_type,
    instantiated with the hyperparameters in MODEL.*.
    """
    if params.get("model_kind") == "custom":
        return load_custom_model(params.get("amp_module", ""), in_param_dim)
    mtype = MODEL_TYPE
    if mtype == 'mlp':
        return AmplitudeDNN_Full(
            in_param_dim=in_param_dim,
            time_dim=1,
            fourier_bands=params.get("fourier_bands", AMP_FOURIER_BANDS),
            fourier_max_freq=params.get("fourier_max_freq", AMP_FOURIER_MAX_FREQ),
            fourier_learnable=params.get("fourier_learnable", AMP_FOURIER_LEARNABLE),
            amp_hidden=[params["amp_hidden_size"]] * params["layers"],
            N_banks=params["banks"],
            dropout=params["dropout"],
        )
    else:
        raise ValueError(f"Unknown amp_model_type '{MODEL_TYPE}'")


def make_phase_model(
    param_dim: int,
    params,
) -> nn.Module:
    """
    Returns the phase model specified by MODEL.phase_model_type,
    instantiated with the hyperparameters in MODEL.*.
    """
    if params.get("model_kind") == "custom":
        return load_custom_model(params.get("phase_module", ""), param_dim)
    mtype = MODEL_TYPE
    if mtype == 'mlp':
        return PhaseDNN_Full(
            param_dim=param_dim,
            time_dim=1,
            fourier_bands=params.get("fourier_bands", PHASE_FOURIER_BANDS),
            fourier_max_freq=params.get("fourier_max_freq", PHASE_FOURIER_MAX_FREQ),
            fourier_learnable=params.get("fourier_learnable", PHASE_FOURIER_LEARNABLE),
            phase_hidden=[params["phase_hidden_size"]] * params["layers"],
            N_banks=params["banks"],
            dropout=params["dropout"],
        )
    else:
        raise ValueError(f"Unknown phase_model_type '{MODEL_TYPE}'")
