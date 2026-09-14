"""
FLARE dashboard API.

A FastAPI backend that wraps the training, tuning, evaluation and GWOSC
parameter-estimation pipelines behind a job-based HTTP interface, so the React
dashboard can drive long-running work and stream live progress.

Run with:  uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import (
    agent,
    dataset,
    estimator,
    evaluation,
    experiments,
    jobs,
    projects,
    settings,
    training,
    tuning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="FLARE Dashboard API",
    description="Train, tune, evaluate and run parameter estimation with the "
                "FLARE gravitational-wave surrogate.",
    version="0.1.0",
)

# The dev frontend runs on a different origin (Vite on :5173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(settings.router)
app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(tuning.router)
app.include_router(evaluation.router)
app.include_router(estimator.router)
app.include_router(experiments.router)
app.include_router(agent.router)
app.include_router(jobs.router)


@app.get("/api/health")
def health():
    import torch
    return {"status": "ok", "cuda_available": torch.cuda.is_available()}
