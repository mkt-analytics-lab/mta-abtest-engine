"""Session propensity API (ADR 0018).

Run locally:   uv run poe serve        ->  http://localhost:8000/docs
POST /score with a user's sessions (time, channel group, funnel stage); the API
returns the propensity of the last session with the model baked into the image
(`models/model.joblib`, exported from the registry champion). If no model is
present the API still starts: /health reports it and /score returns 503.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from loguru import logger

from .. import __version__
from ..config import get_settings
from ..features.scoring import FEATURE_COLUMNS
from .journey import last_session_features
from .schemas import HealthResponse, ScoreRequest, ScoreResponse

state: dict[str, Any] = {"model": None, "model_version": None}


def metadata_path(model_path):
    return model_path.with_suffix(".meta.json")


def load_model() -> None:
    model_path = get_settings().model_path
    state.update(model=None, model_version=None)
    if not model_path.exists():
        logger.warning("No model found at {}; /score will return 503", model_path)
        return
    meta = json.loads(metadata_path(model_path).read_text())
    if meta["feature_columns"] != FEATURE_COLUMNS:  # training-serving skew guard
        raise RuntimeError("Model was trained with different features than this code")
    state.update(model=joblib.load(model_path), model_version=meta["model_version"])
    logger.info("Loaded {} v{}", meta["registered_model"], meta["model_version"])


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(title="Session propensity API", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=state["model"] is not None,
        model_version=state["model_version"],
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = last_session_features([s.model_dump() for s in request.sessions])
    propensity = float(state["model"].predict_proba(features)[:, 1][0])
    return ScoreResponse(
        propensity=round(propensity, 6),
        n_sessions=len(request.sessions),
        model_version=state["model_version"],
    )
