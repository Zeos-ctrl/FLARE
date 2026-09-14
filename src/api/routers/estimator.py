"""Launch GWOSC parameter-estimation jobs and stream MCMC progress."""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException

from src.api.jobs import JobHandle, job_manager
from src.api.schemas import EstimateRequest

router = APIRouter(prefix="/api/estimation", tags=["estimation"])

CHECKPOINT_DIR = os.environ.get("FLARE_CHECKPOINTS", "checkpoints")


def _estimate_target(handle: JobHandle, req: EstimateRequest) -> dict:
    from src.inference.estimator import GWEventEstimator, Prior, WaveformBank
    from src.inference.predictor import WaveformPredictor

    path = os.path.join(CHECKPOINT_DIR, req.project_name)
    handle.emit({"phase": "load", "message": f"Loading model {req.project_name}"})
    predictor = WaveformPredictor(path, device=req.device)

    prior = None
    if req.free_params:
        prior = Prior(bounds={k: tuple(v) for k, v in req.free_params.items()},
                      fixed=req.fixed_params or {})

    handle.emit({"phase": "fetch", "message": f"Fetching {req.event} data"})
    estimator = GWEventEstimator(
        predictor, req.event, detectors=req.detectors,
        prior=prior, duration=req.duration,
        likelihood=req.likelihood, offsource_psd=req.offsource_psd,
        distance_bounds=(req.distance_min, req.distance_max),
        snr_window_s=req.snr_window_s,
    )

    handle.emit({"phase": "bank", "message": f"Building bank ({req.bank_size} templates)"})
    bank = WaveformBank(predictor, estimator.prior)
    bank.build(n_samples=req.bank_size)

    def on_step(info: dict):
        handle.check_cancel()
        handle.emit({"phase": "mcmc", **info})

    handle.emit({"phase": "mcmc", "message": "Running MCMC"})
    result = estimator.run_mcmc(
        nwalkers=req.nwalkers, nsteps=req.nsteps, bank=bank,
        progress=False, on_step=on_step,
    )

    out_dir = os.path.join(path, "estimation", req.event)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "posterior_summary.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    result.corner_plot(path=os.path.join(out_dir, "corner.png"))
    # Time-domain reconstruction diagnostic (fixed-window models only).
    try:
        if estimator.reconstruction_plot(
                result, path=os.path.join(out_dir, "reconstruction.png")) is not None:
            handle.emit({"phase": "mcmc", "message": "Saved reconstruction plot"})
    except Exception as exc:  # noqa: BLE001 - never fail the job on a plot
        handle.emit({"phase": "mcmc", "message": f"Reconstruction plot skipped: {exc}"})
    # Persist a thinned sample set for client-side plotting.
    thin = max(1, len(result.samples) // 4000)
    samples = result.samples[::thin]
    np_path = os.path.join(out_dir, "samples.json")
    with open(np_path, "w") as f:
        json.dump({"param_names": result.param_names,
                   "samples": samples.tolist()}, f)

    handle.emit({"phase": "done", "message": "Estimation complete"})
    return result.to_dict()


@router.post("")
def start_estimation(req: EstimateRequest):
    path = os.path.join(CHECKPOINT_DIR, req.project_name)
    if not os.path.exists(os.path.join(path, "amp_model.pt")):
        raise HTTPException(status_code=404, detail="Project has no trained model")
    job = job_manager.submit(
        "estimation", _estimate_target,
        params={"project_name": req.project_name, "event": req.event}, req=req,
    )
    return job.to_dict()


@router.get("/events")
def list_events(catalog: str = "GWTC-1-confident"):
    """List a catalog of GWOSC events for the dashboard dropdown."""
    from src.inference.gwosc import list_events as _list
    try:
        return _list(catalog=catalog)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"GWOSC query failed: {exc}")


@router.get("/{project_name}/{event}/samples")
def get_samples(project_name: str, event: str):
    path = os.path.join(CHECKPOINT_DIR, project_name, "estimation", event, "samples.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No samples found")
    with open(path) as f:
        return json.load(f)
