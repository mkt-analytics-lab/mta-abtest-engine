"""Scoring-point features on the hand-written fixture (tests/fixtures/README.md)."""

import pandas as pd
import pytest

from mta_abtest_engine.data.contracts import validate_mart
from mta_abtest_engine.features.scoring import FEATURE_COLUMNS, build_scoring_features


@pytest.fixture
def tp(touchpoints_df):
    return validate_mart("fct_touchpoints", touchpoints_df)


def test_columns_and_order_are_fixed(tp):
    features = build_scoring_features(tp, tp["session_key"])
    assert list(features.columns) == FEATURE_COLUMNS
    assert list(features.index) == list(tp["session_key"])


def test_history_features_for_second_touch(tp):
    # u2: Paid Search browse (Nov 3 09:00) -> Referral product (Nov 5 18:30)
    row = build_scoring_features(tp, pd.Series(["u2-2002"])).iloc[0]
    assert row["n_sessions"] == 2
    assert row["n_channels"] == 2
    assert row["hours_since_prev_session"] == pytest.approx(57.5)
    assert row["days_since_first_touch"] == pytest.approx(57.5 / 24)
    assert row["max_funnel_step"] == 1
    assert row["n_sessions_product_plus"] == 1
    assert row["n_sessions_paid_search"] == 1
    assert row["n_sessions_referral"] == 1
    assert row["cur_channel_referral"] == 1
    assert row["cur_channel_paid_search"] == 0


def test_first_session_has_no_previous_gap(tp):
    row = build_scoring_features(tp, pd.Series(["u1-1001"])).iloc[0]
    assert row["hours_since_prev_session"] == -1
    assert row["n_sessions"] == 1


def test_repeated_channel_counts_once_in_n_channels(tp):
    row = build_scoring_features(tp, pd.Series(["u5-5002"])).iloc[0]
    assert row["n_sessions"] == 2
    assert row["n_channels"] == 1
    assert row["n_sessions_email"] == 2


def test_no_leakage_from_later_sessions(tp):
    # Features of u2's first touch must not change when its later sessions exist.
    full = build_scoring_features(tp, pd.Series(["u2-2001"]))
    truncated = tp[~tp["session_key"].isin(["u2-2002", "u2-2003"])]
    alone = build_scoring_features(truncated, pd.Series(["u2-2001"]))
    pd.testing.assert_frame_equal(full, alone)


def test_unknown_session_key_fails_fast(tp):
    with pytest.raises(ValueError, match="not found"):
        build_scoring_features(tp, pd.Series(["nope"]))
