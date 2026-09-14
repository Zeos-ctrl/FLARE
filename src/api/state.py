"""
Persistence for dashboard settings and model designs.

The UI itself stores nothing in the browser. Everything the dashboard saves
lives on the server filesystem:

* Global system/data settings -> ``<state>/settings.json``
* Named model architecture designs -> ``<state>/models/<name>.json``

where ``<state>`` defaults to ``flare_state/`` and can be overridden with the
``FLARE_STATE`` environment variable. (Trained checkpoints, datasets and
results are stored separately under ``checkpoints/<project>/``.)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from src.api.schemas import ModelDesignModel, SettingsModel

STATE_DIR = os.environ.get("FLARE_STATE", "flare_state")
SETTINGS_PATH = os.path.join(STATE_DIR, "settings.json")
MODELS_DIR = os.path.join(STATE_DIR, "models")


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_settings() -> SettingsModel:
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH) as f:
            return SettingsModel(**json.load(f))
    return SettingsModel()


def save_settings(settings: SettingsModel) -> SettingsModel:
    _ensure_dirs()
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings.model_dump(), f, indent=2)
    return settings


def list_model_designs() -> List[ModelDesignModel]:
    if not os.path.isdir(MODELS_DIR):
        return [ModelDesignModel()]  # a sensible default until one is saved
    designs = []
    for fname in sorted(os.listdir(MODELS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(MODELS_DIR, fname)) as f:
                designs.append(ModelDesignModel(**json.load(f)))
    return designs or [ModelDesignModel()]


def load_model_design(name: str) -> ModelDesignModel:
    path = os.path.join(MODELS_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return ModelDesignModel(**json.load(f))
    if name == "default":
        return ModelDesignModel()
    raise FileNotFoundError(name)


def save_model_design(design: ModelDesignModel) -> ModelDesignModel:
    _ensure_dirs()
    path = os.path.join(MODELS_DIR, f"{design.name}.json")
    with open(path, "w") as f:
        json.dump(design.model_dump(), f, indent=2)
    return design


def delete_model_design(name: str) -> bool:
    path = os.path.join(MODELS_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
