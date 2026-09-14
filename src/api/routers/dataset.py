"""Preview generated waveforms and inspect cached datasets."""
from __future__ import annotations

import os

import numpy as np
from fastapi import APIRouter, HTTPException

from src.api.schemas import DatasetPreviewRequest

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

CHECKPOINT_DIR = os.environ.get("FLARE_CHECKPOINTS", "checkpoints")
PARAM_NAMES = ["m1", "m2", "s1z", "s2z", "inc", "ecc"]

# Cap preview cost: waveform generation is the slow part.
MAX_PREVIEW = 16
MAX_POINTS = 512  # per-waveform points returned to the client


def _downsample(arr: np.ndarray, n: int = MAX_POINTS) -> list:
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].tolist()


@router.post("/preview")
def preview_dataset(req: DatasetPreviewRequest):
    """Generate a handful of waveforms so the user can eyeball them first."""
    from src.data.generator import WaveformGenerator

    n = min(req.n_samples, MAX_PREVIEW)
    gen = WaveformGenerator(waveform_length=req.waveform_length,
                            delta_t=1.0 / req.sample_rate,
                            f_lower=req.f_lower, approximant=req.waveform,
                            fixed_window=req.td_fixed_window)
    params = gen.sample_parameters(n, mass_range=tuple(req.mass_range),
                                   spin_range=tuple(req.spin_range),
                                   incl_range=tuple(req.incl_range),
                                   ecc_range=tuple(req.ecc_range), seed=0)

    waveforms = []
    for p in params:
        try:
            h = gen._generate_single_waveform(p, clean=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Waveform generation failed: {exc}")
        waveforms.append({
            "parameters": dict(zip(PARAM_NAMES, p.tolist())),
            "strain": _downsample(h),
        })

    return {
        "waveform": req.waveform,
        "n_samples": n,
        "waveform_length": req.waveform_length,
        "parameters": [dict(zip(PARAM_NAMES, p.tolist())) for p in params],
        "waveforms": waveforms,
    }


@router.get("/{project_name}")
def dataset_info(project_name: str):
    """Summary of a project's cached training dataset."""
    path = os.path.join(CHECKPOINT_DIR, project_name, "dataset.npz")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No cached dataset")

    d = np.load(path, allow_pickle=False)
    params = d["parameters"]
    return {
        "project_name": project_name,
        "n_samples": int(params.shape[0]),
        "amplitude_scale": float(d["amplitude_scale"]),
        "parameters": {
            name: {
                "min": float(params[:, i].min()),
                "max": float(params[:, i].max()),
                "mean": float(params[:, i].mean()),
                "values": params[:, i].tolist(),
            }
            for i, name in enumerate(PARAM_NAMES)
        },
    }
