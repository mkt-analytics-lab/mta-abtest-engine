"""Optuna tuning on synthetic scoring points (no BigQuery, small budget)."""

import datetime as dt
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from mta_abtest_engine.config import ParamSpec, Settings, load_config
from mta_abtest_engine.features.journeys import assign_split
from mta_abtest_engine.features.scoring import FEATURE_COLUMNS
from mta_abtest_engine.training import tuning

pytestmark = pytest.mark.integration


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(Path("configs/config.yaml"))
    cfg.tuning.n_trials = 3
    cfg.tuning.timeout_seconds = 120
    cfg.tuning.storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    cfg.tuning.search_space["max_iter"] = ParamSpec(type="int", low=50, high=100)
    cfg.propensity.bootstrap_samples = 50
    cfg.propensity.report_path = tmp_path / "report.csv"
    return cfg


@pytest.fixture
def synthetic(cfg):
    rng = np.random.default_rng(0)
    n = 4000
    start = pd.Timestamp("2020-11-01", tz="UTC")
    ts = start + pd.to_timedelta(rng.uniform(0, 92, n), unit="D")
    step = rng.integers(0, 4, n)
    points = pd.DataFrame(
        {
            "session_key": [f"s{i}" for i in range(n)],
            "user_pseudo_id": rng.integers(0, 1500, n).astype(str),
            "scoring_ts": ts,
            "label": rng.binomial(1, 0.02 + 0.1 * step),
        }
    )
    j = cfg.journey
    points["split"] = assign_split(
        points["scoring_ts"], j.horizon_days, j.test_start, j.data_end
    )
    features = pd.DataFrame(0.0, index=range(n), columns=FEATURE_COLUMNS)
    features["max_funnel_step"] = step
    features["n_sessions"] = rng.integers(1, 5, n)
    return points, features


def test_validation_masks_never_touch_test_or_overlap(cfg, synthetic):
    points, _ = synthetic
    sub, val = tuning.temporal_validation_masks(
        points, 14, dt.date(2020, 12, 1), cfg.journey.test_start
    )
    assert not (sub & val).any()
    assert (points.loc[sub | val, "split"] == "train").all()
    horizon = pd.Timedelta(days=14)
    assert (
        points.loc[sub, "scoring_ts"] + horizon <= pd.Timestamp("2020-12-01", tz="UTC")
    ).all()
    assert (points.loc[val, "scoring_ts"] >= pd.Timestamp("2020-12-01", tz="UTC")).all()


def test_suggest_params_respects_types(cfg):
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 0.05,
            "max_iter": 80,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 30,
            "l2_regularization": 0.1,
        }
    )
    params = tuning.suggest_params(trial, cfg.tuning.search_space)
    assert isinstance(params["max_iter"], int)
    assert isinstance(params["learning_rate"], float)


def test_study_finds_params_and_is_resumable(cfg, synthetic):
    points, features = synthetic
    logged = []
    study = tuning.run_study(
        cfg,
        points,
        features,
        cfg.tuning.storage,
        log_trial=lambda trial, params, loss: logged.append(loss),
    )
    assert set(study.best_params) == set(cfg.tuning.search_space)
    assert np.isfinite(study.best_value)
    again = tuning.run_study(cfg, points, features, cfg.tuning.storage)
    assert len(again.trials) == 2 * cfg.tuning.n_trials  # resumed, not restarted


def test_run_gates_tuned_model_against_current_gbm(
    tmp_path, cfg, synthetic, monkeypatch
):
    points, features = synthetic
    data = tmp_path / "tp.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(data)
    cfg.data.touchpoints_path = data
    monkeypatch.setattr(tuning, "build_dataset", lambda tp, c: synthetic)
    settings = Settings(
        _env_file=None, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    gate = tuning.run(cfg, settings)
    assert gate.candidate == "gbm_tuned" and gate.reference == "gbm"
    assert (tmp_path / "tuning_best_params.json").exists()
