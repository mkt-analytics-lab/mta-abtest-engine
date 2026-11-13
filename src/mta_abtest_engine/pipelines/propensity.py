"""Session propensity: build the dataset, fit models, evaluate, log to MLflow.

    uv run poe propensity

Pure steps (dataset, evaluation) are separate from I/O (MLflow, reports) so
they can be tested without files or a tracking server (ADR 0012).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ..config import ProjectConfig, Settings, get_settings, load_config
from ..evaluation.gate import GateResult, evaluate_gate
from ..evaluation.metrics import bootstrap_metrics, summarize
from ..features.journeys import build_scoring_points
from ..features.scoring import build_scoring_features
from ..training.baselines import BaseRateModel, FunnelRuleModel
from ..training.propensity_models import build_gbm, build_logistic


def build_dataset(
    touchpoints: pd.DataFrame, cfg: ProjectConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scoring points (labels, split, user) and their feature matrix, row-aligned."""
    j = cfg.journey
    points = build_scoring_points(touchpoints, j.horizon_days, j.test_start, j.data_end)
    features = build_scoring_features(touchpoints, points["session_key"])
    return points, features.reset_index(drop=True)


def candidate_models(cfg: ProjectConfig) -> dict[str, Any]:
    """The ladder: baselines first, then ML candidates (ADR 0012)."""
    seed = cfg.random_seed
    return {
        "base_rate": BaseRateModel(),
        "funnel_rule": FunnelRuleModel(),
        "logistic": build_logistic(cfg.propensity.logistic, seed),
        "gbm": build_gbm(cfg.propensity.gbm, seed),
    }


def gate_results(
    boot: pd.DataFrame, models: list[str], reference: str
) -> dict[str, GateResult]:
    """Gate every model except the baselines against the reference."""
    candidates = [m for m in models if m not in ("base_rate", reference)]
    return {m: evaluate_gate(boot, m, reference) for m in candidates}


def evaluate_models(
    models: dict[str, Any],
    points: pd.DataFrame,
    features: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on train, predict test; return (summary with CIs, bootstrap samples)."""
    train = (points["split"] == "train").to_numpy()
    test = (points["split"] == "test").to_numpy()
    y_train = points.loc[train, "label"].to_numpy()
    y_test = points.loc[test, "label"].to_numpy()
    predictions = {}
    for name, model in models.items():
        model.fit(features[train], y_train)
        predictions[name] = model.predict_proba(features[test])[:, 1]
    boot = bootstrap_metrics(
        y_test,
        predictions,
        points.loc[test, "user_pseudo_id"].to_numpy(),
        n_bootstrap,
        seed,
    )
    return summarize(y_test, predictions, boot), boot


def trace_tags(data_path: Path) -> dict[str, str]:
    """What code and data produced a run (reproducibility)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()[:16]
    return {"git_commit": commit, "data_sha256": data_hash}


def log_runs(
    summary: pd.DataFrame,
    points: pd.DataFrame,
    cfg: ProjectConfig,
    settings: Settings,
    tags: dict[str, str],
    gates: dict[str, GateResult] | None = None,
) -> None:
    """One MLflow run per model: params, metrics with CIs, trace and gate tags."""
    import mlflow  # training-only dependency

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.propensity.experiment_name)
    counts = points.groupby("split")["label"].agg(["size", "sum"])
    for model_name, rows in summary.groupby("model"):
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tags({**tags, "model": model_name})
            gate = (gates or {}).get(model_name)
            if gate is not None:
                mlflow.set_tags(
                    {"gate_reference": gate.reference, "gate_passed": gate.passed}
                )
                mlflow.log_metrics(
                    {
                        "gate_log_loss_diff_ci_high": gate.log_loss_diff[1],
                        "gate_pr_auc_diff_ci_high": gate.pr_auc_diff[1],
                        "gate_brier_diff_ci_low": gate.brier_diff[0],
                    }
                )
            mlflow.log_params(
                {
                    "horizon_days": cfg.journey.horizon_days,
                    "test_start": str(cfg.journey.test_start),
                    "n_train": int(counts.loc["train", "size"]),
                    "n_test": int(counts.loc["test", "size"]),
                    "positives_test": int(counts.loc["test", "sum"]),
                    "bootstrap_samples": cfg.propensity.bootstrap_samples,
                }
            )
            for row in rows.itertuples():
                mlflow.log_metrics(
                    {
                        f"test_{row.metric}": row.value,
                        f"test_{row.metric}_ci_low": row.ci_low,
                        f"test_{row.metric}_ci_high": row.ci_high,
                    }
                )


def run(cfg: ProjectConfig, settings: Settings) -> pd.DataFrame:
    touchpoints = pd.read_parquet(cfg.data.touchpoints_path)
    points, features = build_dataset(touchpoints, cfg)
    models = candidate_models(cfg)
    summary, boot = evaluate_models(
        models, points, features, cfg.propensity.bootstrap_samples, cfg.random_seed
    )
    gates = gate_results(boot, list(models), cfg.propensity.reference_model)
    report = cfg.propensity.report_path
    report.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report, index=False)
    log_runs(
        summary, points, cfg, settings, trace_tags(cfg.data.touchpoints_path), gates
    )
    logger.info("Propensity report -> {}\n{}", report, summary.round(4).to_string())
    for gate in gates.values():
        logger.info(
            "Gate {} vs {}: {} | log_loss diff CI {} | pr_auc diff CI {} | "
            "brier diff CI {}",
            gate.candidate,
            gate.reference,
            "PASS" if gate.passed else "FAIL",
            _fmt(gate.log_loss_diff),
            _fmt(gate.pr_auc_diff),
            _fmt(gate.brier_diff),
        )
    return summary


def _fmt(interval: tuple[float, float]) -> str:
    return f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"


def main() -> None:  # pragma: no cover - thin CLI wrapper
    settings = get_settings()
    run(load_config(settings.config_path), settings)


if __name__ == "__main__":  # pragma: no cover
    main()
