"""Journey mart contracts: accept the fixtures, reject each kind of bad data."""

import pandas as pd
import pandera.errors as pa_errors
import pytest

from mta_abtest_engine.data.contracts import validate_mart


def test_touchpoints_fixture_passes(touchpoints_df):
    validated = validate_mart("fct_touchpoints", touchpoints_df)
    assert len(validated) == len(touchpoints_df)
    assert str(validated["session_start_ts"].dt.tz) == "UTC"


def test_journeys_fixture_passes(journeys_df):
    assert len(validate_mart("fct_user_journeys", journeys_df)) == len(journeys_df)


def test_missing_campaign_stays_null(touchpoints_df):
    validated = validate_mart("fct_touchpoints", touchpoints_df)
    assert (
        validated.loc[validated["user_pseudo_id"] == "u3", "session_campaign"]
        .isna()
        .all()
    )


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("channel_group", "TikTok"),  # unmapped channel
        ("funnel_stage_pre_purchase", "purchase"),  # label leak
        ("funnel_step", 7),
        ("purchase_revenue_usd", -1.0),
    ],
)
def test_touchpoints_bad_values_rejected(touchpoints_df, column, bad_value):
    bad = touchpoints_df.copy()
    bad.loc[0, column] = bad_value
    with pytest.raises(pa_errors.SchemaErrors):
        validate_mart("fct_touchpoints", bad)


def test_touch_after_purchase_rejected(touchpoints_df):
    bad = touchpoints_df.copy()
    # u2 converts on touch 3 of 3; pretend the journey had a 4th touch.
    bad.loc[bad["user_pseudo_id"] == "u2", "n_touchpoints"] = 4
    with pytest.raises(pa_errors.SchemaErrors):
        validate_mart("fct_touchpoints", bad)


def test_two_conversions_in_one_journey_rejected(touchpoints_df):
    bad = touchpoints_df.copy()
    bad.loc[bad["session_key"] == "u2-2002", "is_conversion_touch"] = True
    with pytest.raises(pa_errors.SchemaErrors):
        validate_mart("fct_touchpoints", bad)


def test_duplicate_session_rejected(touchpoints_df):
    with pytest.raises(pa_errors.SchemaErrors):
        validate_mart("fct_touchpoints", touchpoints_df.iloc[[0, 0]])


def test_journey_ending_before_start_rejected(journeys_df):
    bad = journeys_df.copy()
    bad.loc[1, "journey_end_ts"] = bad.loc[1, "journey_start_ts"] - pd.Timedelta(days=1)
    with pytest.raises(pa_errors.SchemaErrors):
        validate_mart("fct_user_journeys", bad)


def test_unexpected_column_rejected(journeys_df):
    with pytest.raises(pa_errors.SchemaErrors):
        validate_mart("fct_user_journeys", journeys_df.assign(extra=1))
