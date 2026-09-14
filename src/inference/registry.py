"""Model registry: load released FLARE surrogates by canonical name.

The curated model library lives under ``models/`` — one directory per released
model (canonical name), holding the inference files plus a ``card.json``. The
tracked ``models/registry.json`` catalogues every model with its family, frame,
parameter space and a ``loader`` key; the weights themselves are gitignored (the
standard ML pattern: config in git, large artifacts out).

``load_model(name)`` reads the registry and dispatches on the loader key, so a
new physics family plugs in by (1) dropping its checkpoint under ``models/`` with
a card, (2) adding a registry entry, and (3) if it needs new reconstruction, a
loader branch here. The ``frame`` field is the extension hook the uniform
interface already uses (aligned / precessing / eccentric).

    from src.inference.registry import list_models, load_model
    model = load_model("FLARE-SEOBNRv4HM")
    hp, hc = model.get_td_waveform(mass1=60, mass2=25, spin1z=0.3, spin2z=-0.1,
                                   inclination=1.0, distance=100)
"""
from __future__ import annotations

import json
import os
from typing import List

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_PATH = os.path.join(_REPO, "models", "registry.json")


def _registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def list_models(include_planned: bool = False) -> List[dict]:
    """Return the catalogue of released models (optionally including planned)."""
    reg = _registry()
    models = list(reg.get("models", []))
    if include_planned:
        models += reg.get("planned", [])
    return models


def get_model_card(name: str) -> dict:
    for m in list_models(include_planned=True):
        if m["name"] == name:
            return m
    raise KeyError(f"model '{name}' not in registry; available: "
                   f"{[m['name'] for m in list_models()]}")


def load_model(name: str, device: str = "cuda"):
    """Load a released surrogate by canonical name, dispatching on its loader.

    Returns a WaveformPredictor (loader='predictor') or a SurrogateWaveform
    subclass (loader='aligned_modes', ...). Both expose ``predict`` /
    ``batch_predict`` so they drop into GWEventEstimator; the mode-based models
    also expose ``get_td_waveform``.
    """
    card = get_model_card(name)
    if card.get("status") == "planned":
        raise NotImplementedError(
            f"'{name}' is a planned model ({card.get('family')}); not yet trained.")
    path = os.path.join(_REPO, card["path"])
    loader = card["loader"]
    if loader == "predictor":
        from src.inference.predictor import WaveformPredictor
        m = WaveformPredictor(path, device=device)
        m.f_lower = float(m.meta.get("f_lower", 20.0))
        return m
    if loader == "aligned_modes":
        from src.inference.surrogate import AlignedModeSurrogate
        return AlignedModeSurrogate(path, device=device)
    raise NotImplementedError(
        f"loader '{loader}' for model '{name}' is not implemented yet "
        f"(it is the extension point for the {card.get('family')} family).")
