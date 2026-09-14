"""Load operator-authored custom models from Python files on disk.

Custom architectures are supplied as ``.py`` files the operator writes into
``custom_models/`` (override with the ``FLARE_CUSTOM_MODELS`` env var). Each
file must define::

    def build_model(in_param_dim, time_dim=1):
        return SomeModule(...)   # a torch.nn.Module

whose ``forward(t_norm, theta)`` accepts a ``(B, time_dim)`` time tensor and a
``(B, in_param_dim)`` parameter tensor and returns ``(B, 1)`` — matching the
built-in amplitude/phase networks and the training loop.

The dashboard only ever *names* one of these files; it never executes code
pasted over HTTP. Loading a module is an explicit, operator-controlled import
of a file already present on disk — the same trust level as the rest of the
installed source.
"""
from __future__ import annotations

import importlib.util
import os
from typing import List

CUSTOM_MODELS_DIR = os.environ.get("FLARE_CUSTOM_MODELS", "custom_models")


def list_custom_models() -> List[str]:
    """Return the names (file stems) of available custom-model files."""
    if not os.path.isdir(CUSTOM_MODELS_DIR):
        return []
    return sorted(
        f[:-3]
        for f in os.listdir(CUSTOM_MODELS_DIR)
        if f.endswith(".py") and not f.startswith("_")
    )


def load_custom_model(name: str, in_param_dim: int, time_dim: int = 1):
    """Import ``custom_models/<name>.py`` and build its model.

    Args:
        name: File stem (no ``.py``) of a file in ``CUSTOM_MODELS_DIR``.
        in_param_dim: Number of input parameters (features).
        time_dim: Time-input dimensionality (1 for the built-in loop).

    Returns:
        The ``torch.nn.Module`` returned by the file's ``build_model``.
    """
    import torch.nn as nn

    if not name:
        raise ValueError("Custom model selected but no model file was chosen.")
    path = os.path.join(CUSTOM_MODELS_DIR, f"{name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Custom model file not found: {path}. Place a .py file defining "
            f"build_model(in_param_dim, time_dim) in {CUSTOM_MODELS_DIR}/."
        )

    spec = importlib.util.spec_from_file_location(f"flare_custom_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    builder = getattr(module, "build_model", None)
    if builder is None:
        raise ValueError(
            f"{path} must define build_model(in_param_dim, time_dim)."
        )
    model = builder(in_param_dim, time_dim)
    if not isinstance(model, nn.Module):
        raise ValueError(f"{path}: build_model(...) must return a torch.nn.Module.")
    return model
