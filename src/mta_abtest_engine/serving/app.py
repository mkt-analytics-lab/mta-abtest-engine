"""Prediction API.

Run locally:   uv run poe serve        ->  http://localhost:8000/docs
The model is loaded once at startup from ``APP_MODEL_PATH``. If no model
exists yet, the API still starts (``/health`` reports ``model_loaded=false``)
and ``/predict`` returns 503 instead of crashing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from .. import __version__
from ..config import get_settings
from ..features.build import build_features
from ..training.train import metadata_path
from .schemas import HealthResponse, PredictRequest, PredictResponse

state: dict[str, Any] = {"model": None, "model_version": None}


def load_model() -> None:
    model_path = get_settings().model_path
    if not model_path.exists():
        logger.warning("No model found at {}; /predict will return 503", model_path)
        state.update(model=None, model_version=None)
        return
    state["model"] = joblib.load(model_path)
    meta_file = metadata_path(model_path)
    if meta_file.exists():
        state["model_version"] = json.loads(meta_file.read_text())["mlflow_run_id"]
    logger.info("Loaded model from {}", model_path)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(title="Prediction API", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=state["model"] is not None,
        model_version=state["model_version"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    raw = pd.DataFrame([record.model_dump() for record in request.records])
    features = build_features(raw)  # same code path as training
    probabilities = state["model"].predict_proba(features)[:, 1]
    return PredictResponse(
        probabilities=[round(float(p), 6) for p in probabilities],
        model_version=state["model_version"],
    )
