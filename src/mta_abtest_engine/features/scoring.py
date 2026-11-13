"""Features of a scoring point: what the user did up to and including a session.

Used by training AND by batch scoring (ADR 0010, 0011): one function, so the
model sees the same inputs in both (no training-serving skew). Every feature is
a running (cumulative) value over the user's touchpoints ordered by position,
so a feature can never use a later session (no leakage). Pure functions.
"""

from __future__ import annotations

import re

import pandas as pd

from ..data.contracts import CHANNEL_GROUPS


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


CHANNEL_SLUGS: dict[str, str] = {name: _slug(name) for name in CHANNEL_GROUPS}
STAGE_THRESHOLDS: dict[str, int] = {"product": 1, "cart": 2, "checkout": 3}

FEATURE_COLUMNS: list[str] = [
    # history up to and including the scoring session
    "n_sessions",
    "n_channels",
    "days_since_first_touch",
    "hours_since_prev_session",
    "max_funnel_step",
    *[f"n_sessions_{stage}_plus" for stage in STAGE_THRESHOLDS],
    *[f"n_sessions_{slug}" for slug in CHANNEL_SLUGS.values()],
    # the scoring session itself
    "cur_funnel_step",
    *[f"cur_channel_{slug}" for slug in CHANNEL_SLUGS.values()],
]


def build_scoring_features(
    touchpoints: pd.DataFrame, session_keys: pd.Series
) -> pd.DataFrame:
    """Feature matrix for ``session_keys`` (index = session_key, fixed column order).

    ``touchpoints`` must contain every earlier touchpoint of those users; later
    ones may be present and are ignored by construction.
    """
    tp = touchpoints.sort_values(["user_pseudo_id", "touch_position"]).copy()
    by_user = tp.groupby("user_pseudo_id", sort=False)
    out = pd.DataFrame(index=tp.index)

    out["n_sessions"] = by_user.cumcount() + 1
    first_seen = ~tp.duplicated(["user_pseudo_id", "channel_group"])
    out["n_channels"] = first_seen.groupby(tp["user_pseudo_id"]).cumsum()
    start = by_user["session_start_ts"].transform("min")
    out["days_since_first_touch"] = (
        tp["session_start_ts"] - start
    ).dt.total_seconds() / 86400
    prev_gap = by_user["session_start_ts"].diff().dt.total_seconds() / 3600
    out["hours_since_prev_session"] = prev_gap.fillna(-1.0)  # -1 = first session
    out["max_funnel_step"] = by_user["funnel_step"].cummax()

    for stage, step in STAGE_THRESHOLDS.items():
        reached = (tp["funnel_step"] >= step).astype(int)
        out[f"n_sessions_{stage}_plus"] = reached.groupby(tp["user_pseudo_id"]).cumsum()
    for name, slug in CHANNEL_SLUGS.items():
        is_channel = (tp["channel_group"] == name).astype(int)
        out[f"n_sessions_{slug}"] = is_channel.groupby(tp["user_pseudo_id"]).cumsum()
        out[f"cur_channel_{slug}"] = is_channel
    out["cur_funnel_step"] = tp["funnel_step"]

    out.index = tp["session_key"]
    missing = set(session_keys) - set(out.index)
    if missing:
        raise ValueError(f"{len(missing)} session keys not found in touchpoints")
    return out.loc[list(session_keys), FEATURE_COLUMNS].astype(float)
