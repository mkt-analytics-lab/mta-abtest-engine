"""Horizon windowing, labels and temporal split on the hand-written fixture.

Fixture (tests/fixtures/README.md): u2 converts 4.5 days after its first touch,
u4 after ~1 day; u5 has a second touch 5 days after the first.
"""

import datetime as dt

import pandas as pd
import pytest

from mta_abtest_engine.data.contracts import validate_mart
from mta_abtest_engine.features.journeys import (
    apply_horizon,
    assign_split,
    build_journey_dataset,
)

TEST_START = dt.date(2021, 1, 1)
DATA_END = dt.date(2021, 2, 1)


@pytest.fixture
def tp(touchpoints_df):
    return validate_mart("fct_touchpoints", touchpoints_df)


def _labels(journeys: pd.DataFrame) -> dict[str, bool]:
    return dict(zip(journeys["user_pseudo_id"], journeys["converted"], strict=True))


def test_long_horizon_keeps_every_touchpoint(tp):
    windowed, journeys = build_journey_dataset(tp, 14, TEST_START, DATA_END)
    assert len(windowed) == len(tp)
    assert _labels(journeys) == {
        "u1": False,
        "u2": True,
        "u3": False,
        "u4": True,
        "u5": False,
    }


def test_short_horizon_drops_late_touches_and_conversions(tp):
    # 2 days: u2's purchase (day 4.5) and u5's second touch (day 5) fall outside.
    windowed, journeys = build_journey_dataset(tp, 2, TEST_START, DATA_END)
    assert _labels(journeys)["u2"] is False
    assert _labels(journeys)["u4"] is True  # converts within ~1 day
    u5 = windowed[windowed["user_pseudo_id"] == "u5"]
    assert len(u5) == 1 and (u5["n_touchpoints"] == 1).all()


def test_windowed_touchpoints_never_exceed_horizon(tp):
    windowed = apply_horizon(tp, 3)
    age = windowed["session_start_ts"] - windowed["journey_start_ts"]
    assert (age < pd.Timedelta(days=3)).all()


def test_split_follows_journey_start(tp):
    _, journeys = build_journey_dataset(tp, 14, TEST_START, DATA_END)
    split = dict(zip(journeys["user_pseudo_id"], journeys["split"], strict=True))
    assert split == {
        "u1": "train",
        "u2": "train",
        "u3": "train",
        "u4": "train",
        "u5": "test",
    }


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ("2020-12-10", "train"),  # window ends Dec 24 < Jan 1
        ("2020-12-25", "excluded"),  # window crosses test_start
        ("2021-01-10", "test"),
        ("2021-01-25", "excluded"),  # window crosses data_end (censored)
    ],
)
def test_split_boundaries(start, expected):
    starts = pd.Series([pd.Timestamp(start, tz="UTC")])
    assert assign_split(starts, 14, TEST_START, DATA_END).iloc[0] == expected
