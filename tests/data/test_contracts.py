"""Data contract tests: the schema accepts good data and rejects bad data."""

import pandera.errors as pa_errors
import pytest

from mta_abtest_engine.data.validation import validate_dataframe


def test_valid_data_passes(raw_df):
    assert len(validate_dataframe(raw_df)) == len(raw_df)


def test_nulls_are_rejected(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "x1"] = None
    with pytest.raises(pa_errors.SchemaErrors):
        validate_dataframe(bad)


def test_invalid_target_is_rejected(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "target"] = 2
    with pytest.raises(pa_errors.SchemaErrors):
        validate_dataframe(bad)


def test_unexpected_columns_are_rejected(raw_df):
    with pytest.raises(pa_errors.SchemaErrors):
        validate_dataframe(raw_df.assign(extra=1.0))
