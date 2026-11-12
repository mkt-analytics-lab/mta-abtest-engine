"""API contracts (Pydantic). Invalid requests are rejected with HTTP 422."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    """One observation. Fields must match ``data.ingest.FEATURE_COLUMNS``
    (a parity test in ``tests/unit/test_parity.py`` enforces it)."""

    model_config = ConfigDict(extra="forbid")

    x1: float
    x2: float
    x3: float
    x4: float


class PredictRequest(BaseModel):
    records: list[Record] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    probabilities: list[float]
    model_version: str | None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
