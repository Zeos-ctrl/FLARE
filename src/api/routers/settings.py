"""CRUD for global settings and named model designs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api import state
from src.api.schemas import ModelDesignModel, SettingsModel

router = APIRouter(prefix="/api", tags=["settings"])


# --- Settings -----------------------------------------------------------
@router.get("/settings", response_model=SettingsModel)
def get_settings():
    return state.load_settings()


@router.put("/settings", response_model=SettingsModel)
def put_settings(settings: SettingsModel):
    return state.save_settings(settings)


# --- Model designs ------------------------------------------------------
@router.get("/models")
def list_models():
    return state.list_model_designs()


@router.get("/models/{name}", response_model=ModelDesignModel)
def get_model(name: str):
    try:
        return state.load_model_design(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model design not found")


@router.put("/models/{name}", response_model=ModelDesignModel)
def put_model(name: str, design: ModelDesignModel):
    design.name = name
    return state.save_model_design(design)


@router.delete("/models/{name}")
def delete_model(name: str):
    if not state.delete_model_design(name):
        raise HTTPException(status_code=404, detail="Model design not found")
    return {"deleted": name}


# --- Custom model files -------------------------------------------------
@router.get("/custom-models")
def list_custom_models():
    """Names of operator-authored model files in custom_models/."""
    from src.models.custom import CUSTOM_MODELS_DIR, list_custom_models

    return {"dir": CUSTOM_MODELS_DIR, "models": list_custom_models()}
