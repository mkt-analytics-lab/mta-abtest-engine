"""Journey windowing, labels and temporal split (ADR 0009).

A journey starts at the user's first touchpoint. The label is "purchase within
``horizon_days``"; touchpoints after the horizon are dropped because they are
future information relative to the label. The same windowing must be applied
when scoring a live journey, so it lives in ``features/`` (no training-serving
skew). Pure functions: DataFrame in, DataFrame out.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

SPLITS = ("train", "test", "excluded")


def apply_horizon(touchpoints: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Keep touchpoints in [journey start, journey start + horizon)."""
    tp = touchpoints.copy()
    start = tp.groupby("user_pseudo_id")["session_start_ts"].transform("min")
    tp["journey_start_ts"] = start
    in_window = tp["session_start_ts"] < start + pd.Timedelta(days=horizon_days)
    tp = tp[in_window].copy()
    # Dropping a suffix of the journey keeps positions valid; counts change.
    tp["n_touchpoints"] = tp.groupby("user_pseudo_id")["session_key"].transform("size")
    return tp.reset_index(drop=True)


def label_journeys(windowed: pd.DataFrame) -> pd.DataFrame:
    """One row per user: journey start, touchpoints in window, label, revenue."""
    return (
        windowed.groupby("user_pseudo_id", as_index=False)
        .agg(
            journey_start_ts=("journey_start_ts", "first"),
            n_touchpoints=("session_key", "size"),
            converted=("is_conversion_touch", "any"),
            revenue_usd=("purchase_revenue_usd", "sum"),
        )
        .sort_values("user_pseudo_id", ignore_index=True)
    )


def assign_split(
    journey_start: pd.Series,
    horizon_days: int,
    test_start: dt.date,
    data_end: dt.date,
) -> pd.Series:
    """train / test / excluded so that every label is fully observed.

    - train: the whole label window ends before the test period starts.
    - test: starts in the test period and its window ends before data_end.
    - excluded: label window crosses test_start or data_end (censored).
    """
    horizon = pd.Timedelta(days=horizon_days)
    test_ts = pd.Timestamp(test_start, tz="UTC")
    end_ts = pd.Timestamp(data_end, tz="UTC")
    window_end = journey_start + horizon
    split = pd.Series("excluded", index=journey_start.index)
    split[window_end <= test_ts] = "train"
    split[(journey_start >= test_ts) & (window_end <= end_ts)] = "test"
    return split


def build_journey_dataset(
    touchpoints: pd.DataFrame,
    horizon_days: int,
    test_start: dt.date,
    data_end: dt.date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Windowed touchpoints and labelled journeys, both with a ``split`` column."""
    windowed = apply_horizon(touchpoints, horizon_days)
    journeys = label_journeys(windowed)
    journeys["split"] = assign_split(
        journeys["journey_start_ts"], horizon_days, test_start, data_end
    )
    windowed = windowed.merge(
        journeys[["user_pseudo_id", "split"]], on="user_pseudo_id", how="left"
    )
    return windowed, journeys
