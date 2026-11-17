"""Shared fixtures. Test layout mirrors the quality phase of the blueprint:

tests/unit/         pure functions (config, features, metrics, attribution, parity)
tests/data/         data contracts of the journey marts (pandera)
tests/model/        behavioural model tests (MFT / INV / DIR)
tests/integration/  several components together (pipelines, registry, API)

Fixtures in tests/fixtures/ are hand-written versions of the dbt marts.
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def touchpoints_df() -> pd.DataFrame:
    """Hand-written fct_touchpoints (see tests/fixtures/README.md)."""
    return pd.read_csv(
        FIXTURES / "fct_touchpoints.csv", parse_dates=["session_start_ts"]
    )


@pytest.fixture
def journeys_df() -> pd.DataFrame:
    """Hand-written fct_user_journeys (see tests/fixtures/README.md)."""
    return pd.read_csv(
        FIXTURES / "fct_user_journeys.csv",
        parse_dates=["journey_start_ts", "journey_end_ts"],
    )


@pytest.fixture(autouse=True)
def _collect_sqlite_connections():
    """MLflow and Optuna leave temporary SQLite connections (in tmp_path) open.
    Collect them after each test with their ResourceWarning silenced, so the
    output stays clean and real warnings remain visible."""
    yield
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        gc.collect()


@pytest.fixture(autouse=True)
def _artifacts_in_tmp(tmp_path, monkeypatch):
    """MLflow artifacts written by tests go to tmp_path, never into the repo."""
    monkeypatch.setenv("APP_MLFLOW_ARTIFACT_LOCATION", str(tmp_path / "artifacts"))
