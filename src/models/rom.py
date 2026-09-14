"""Reduced-order coefficient network: theta -> SVD coefficients.

In reduced-order mode the surrogate does not predict amplitude/phase at every
time sample; it predicts the handful of SVD basis coefficients for a waveform,
which are then expanded against the stored basis. The map ``theta -> coeffs`` is
smooth and low-dimensional, so a small Fourier-feature MLP suffices and trains in
seconds on N rows (one row per waveform).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mlp import FourierFeature


class CoeffMLP(nn.Module):
    """Maps standardized features ``theta`` (B, d) to ``k`` SVD coefficients."""

    def __init__(self, in_param_dim: int, out_dim: int,
                 hidden: int = 256, depth: int = 4,
                 fourier_bands: int = 16, fourier_max_freq: float = 10.0,
                 dropout: float = 0.0):
        super().__init__()
        self.ff = FourierFeature(in_param_dim, num_bands=fourier_bands,
                                 max_freq=fourier_max_freq, learnable=False)
        # Concatenate the RAW features with their Fourier embedding: some targets
        # are near-linear in a feature (e.g. the phase log-scale ~ log(chirp_mass))
        # and a sinusoid-only embedding cannot represent a linear trend precisely.
        in_dim = in_param_dim + in_param_dim * fourier_bands * 2
        layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(torch.cat([theta, self.ff(theta)], dim=-1)))
