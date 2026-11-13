"""Propensity pipeline pieces without BigQuery: dataset on fixtures, evaluation
and MLflow logging on synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mta_abtest_engine.config import Settings, load_config
from mta_abtest_engine.data.contracts import validate_mart
from mta_abtest_engine.features.scoring import FEATURE_COLUMNS
from mta_abtest_engine.pipelines.propensity import (
    build_dataset,
    candidate_models,
    evaluate_models,
    log_runs,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def cfg():
    return load_config(Path("configs/config.yaml"))


@pytest.fixture
def synthetic():
    """Scoring points where deeper funnel steps buy more often."""
    rng = np.random.default_rng(0)
    n = 3000
    step = rng.integers(0, 4, n)
    label = rng.binomial(1, 0.01 + 0.08 * step)
    points = pd.DataFrame(
        {
            "session_key": [f"s{i}" for i in range(n)],
            "user_pseudo_id": rng.integers(0, 800, n).astype(str),
            "label": label,
            "split": np.where(np.arange(n) < 2000, "train", "test"),
        }
    )
    features = pd.DataFrame(0.0, index=range(n), columns=FEATURE_COLUMNS)
    features["max_funnel_step"] = step
    return points, features


def test_dataset_on_fixture_is_row_aligned(cfg, touchpoints_df):
    tp = validate_mart("fct_touchpoints", touchpoints_df)
    points, features = build_dataset(tp, cfg)
    assert len(points) == len(features) == 7
    assert list(features.columns) == FEATURE_COLUMNS


def test_funnel_rule_beats_base_rate(cfg, synthetic):
    points, features = synthetic
    summary, _ = evaluate_models(candidate_models(cfg), points, features, 50, 0)
    loss = summary[summary["metric"] == "log_loss"].set_index("model")["value"]
    assert loss["funnel_rule"] < loss["base_rate"]


def test_runs_are_logged_to_mlflow(tmp_path, cfg, synthetic):
    import mlflow

    points, features = synthetic
    summary, _ = evaluate_models(candidate_models(cfg), points, features, 50, 0)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(_env_file=None, mlflow_tracking_uri=uri)
    log_runs(summary, points, cfg, settings, {"git_commit": "abc", "data_sha256": "x"})

    runs = mlflow.search_runs(experiment_names=[cfg.propensity.experiment_name])
    assert set(runs["tags.model"]) == {"base_rate", "funnel_rule", "logistic", "gbm"}
    assert "metrics.test_log_loss_ci_high" in runs.columns
    assert (runs["tags.git_commit"] == "abc").all()


def test_trace_tags_identify_code_and_data(tmp_path):
    from mta_abtest_engine.pipelines.propensity import trace_tags

    data = tmp_path / "tp.parquet"
    data.write_bytes(b"same bytes")
    tags = trace_tags(data)
    assert tags["data_sha256"] == trace_tags(data)["data_sha256"]  # deterministic
    assert tags["git_commit"]


def test_run_writes_report_and_logs(tmp_path, monkeypatch, cfg, synthetic):
    from mta_abtest_engine.pipelines import propensity

    data = tmp_path / "tp.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(data)
    cfg.data.touchpoints_path = data
    cfg.propensity.report_path = tmp_path / "report.csv"
    cfg.propensity.bootstrap_samples = 50
    monkeypatch.setattr(propensity, "build_dataset", lambda tp, c: synthetic)
    settings = Settings(
        _env_file=None, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'm.db'}"
    )
    summary = propensity.run(cfg, settings)
    assert cfg.propensity.report_path.exists()
    assert set(summary["model"]) == {"base_rate", "funnel_rule", "logistic", "gbm"}


def test_gate_tags_are_logged(tmp_path, cfg, synthetic):
    import mlflow

    from mta_abtest_engine.pipelines.propensity import gate_results

    points, features = synthetic
    models = candidate_models(cfg)
    summary, boot = evaluate_models(models, points, features, 50, 0)
    gates = gate_results(boot, list(models), "funnel_rule")
    assert set(gates) == {"logistic", "gbm"}

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(_env_file=None, mlflow_tracking_uri=uri)
    log_runs(
        summary, points, cfg, settings, {"git_commit": "a", "data_sha256": "b"}, gates
    )
    runs = mlflow.search_runs(experiment_names=[cfg.propensity.experiment_name])
    gated = runs[runs["tags.model"].isin(["logistic", "gbm"])]
    assert gated["tags.gate_passed"].isin(["True", "False"]).all()
    assert gated["tags.gate_reference"].eq("funnel_rule").all()
