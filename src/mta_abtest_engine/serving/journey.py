"""Turn a request (a user's sessions) into the feature row of its last session.

Shared by the API and its parity tests: the API never computes features itself,
it calls the same ``build_scoring_features`` as training and batch scoring.
"""

from __future__ import annotations

import pandas as pd

from ..features.scoring import build_scoring_features

FUNNEL_STEPS = {"browse": 0, "product": 1, "cart": 2, "checkout": 3}


def journey_frame(sessions: list[dict]) -> pd.DataFrame:
    """Touchpoints of one anonymous user, ordered by time."""
    df = pd.DataFrame(sessions).sort_values("session_start_ts", ignore_index=True)
    return pd.DataFrame(
        {
            "session_key": [f"s-{i}" for i in range(len(df))],
            "user_pseudo_id": "request",
            "touch_position": range(1, len(df) + 1),
            "session_start_ts": pd.to_datetime(df["session_start_ts"], utc=True),
            "channel_group": df["channel_group"],
            "funnel_step": df["funnel_stage"].map(FUNNEL_STEPS),
        }
    )


def last_session_features(sessions: list[dict]) -> pd.DataFrame:
    journey = journey_frame(sessions)
    return build_scoring_features(journey, journey["session_key"].iloc[[-1]])
