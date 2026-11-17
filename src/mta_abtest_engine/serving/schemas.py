"""API contracts (Pydantic). Invalid requests are rejected with HTTP 422."""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from ..data.contracts import CHANNEL_GROUPS
from .journey import FUNNEL_STEPS


class Session(BaseModel):
    """One session of the user's journey (no purchase in it)."""

    model_config = ConfigDict(extra="forbid")

    session_start_ts: AwareDatetime  # must carry a timezone, e.g. ...Z
    channel_group: str
    funnel_stage: str = Field(description="browse | product | cart | checkout")

    @field_validator("channel_group")
    @classmethod
    def _known_channel(cls, value: str) -> str:
        if value not in CHANNEL_GROUPS:
            raise ValueError(f"unknown channel group; expected one of {CHANNEL_GROUPS}")
        return value

    @field_validator("funnel_stage")
    @classmethod
    def _known_stage(cls, value: str) -> str:
        if value not in FUNNEL_STEPS:
            raise ValueError(
                f"unknown funnel stage; expected one of {list(FUNNEL_STEPS)}"
            )
        return value


class ScoreRequest(BaseModel):
    sessions: list[Session] = Field(min_length=1, max_length=500)


class ScoreResponse(BaseModel):
    propensity: float = Field(description="P(purchase within 14 days), uncalibrated")
    n_sessions: int
    model_version: str | None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
