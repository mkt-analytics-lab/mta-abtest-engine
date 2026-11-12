import pandas as pd
import pytest

from mta_abtest_engine.features.build import MODEL_FEATURES, build_features


def test_build_features_returns_model_features_in_order(raw_df):
    features = build_features(raw_df)
    assert list(features.columns) == MODEL_FEATURES
    assert len(features) == len(raw_df)


def test_build_features_is_deterministic(raw_df):
    pd.testing.assert_frame_equal(build_features(raw_df), build_features(raw_df))


def test_build_features_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="Missing input columns"):
        build_features(raw_df.drop(columns=["x1"]))
