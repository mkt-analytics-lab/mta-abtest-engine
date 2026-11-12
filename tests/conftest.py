"""Shared fixtures. Test layout mirrors the quality phase of the blueprint:

tests/unit/         pure functions (config, features, parity)
tests/data/         data contracts (schema validation)
tests/model/        behavioural model tests (MFT / INV / DIR)
tests/integration/  several components together (API, full pipeline)
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mta_abtest_engine.data.ingest import TARGET_COLUMN, generate_dataset
from mta_abtest_engine.features.build import build_features
from mta_abtest_engine.training.train import train_model


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return generate_dataset(n_samples=2000, seed=0)


@pytest.fixture(scope="session")
def trained_model(raw_df: pd.DataFrame) -> Pipeline:
    X = build_features(raw_df)
    y = raw_df[TARGET_COLUMN]
    return train_model(X, y, "logistic_regression", {"max_iter": 1000}, seed=0)
