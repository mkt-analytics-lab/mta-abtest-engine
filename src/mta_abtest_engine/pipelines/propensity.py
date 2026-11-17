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
from ..evaluation.behavior import CheckResult, all_passed, run_behavioral_checks
from ..evaluation.gate import GateResult, evaluate_gate
from ..evaluation.metrics import bootstrap_metrics, summarize
from ..features.journeys import build_scoring_points
from ..features.scoring import build_scoring_features
from ..training.baselines import BaseRateModel, FunnelRuleModel
from ..training.propensity_models import build_gbm, build_logistic
from ..training.registry import (
    CHAMPION,
    PrefittedModel,
    load_alias,
    register_challenger,
    use_experiment,
)


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
    candidates = [m for m in models if m not in ("base_rate", "champion", reference)]
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
    behavior: dict[str, list[CheckResult]] | None = None,
) -> dict[str, str]:
    """One MLflow run per model: params, metrics with CIs, trace and gate tags.

    Returns the run id of each model (to attach the registered model later)."""
    import mlflow  # training-only dependency

    use_experiment(
        settings.mlflow_tracking_uri,
        cfg.propensity.experiment_name,
        settings.mlflow_artifact_location,
    )
    counts = points.groupby("split")["label"].agg(["size", "sum"])
    run_ids: dict[str, str] = {}
    for model_name, rows in summary.groupby("model"):
        with mlflow.start_run(run_name=model_name) as active:
            run_ids[str(model_name)] = active.info.run_id
            mlflow.set_tags({**tags, "model": model_name})
            checks = (behavior or {}).get(model_name)
            if checks is not None:
                mlflow.set_tag("behavior_passed", all_passed(checks))
                mlflow.log_dict(
                    {
                        c.name: {"kind": c.kind, "passed": c.passed, "detail": c.detail}
                        for c in checks
                    },
                    "behavior_checks.json",
                )
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
    return run_ids


def select_challenger(
    summary: pd.DataFrame,
    gates: dict[str, GateResult],
    behavior: dict[str, list[CheckResult]],
    champion_gates: dict[str, GateResult] | None,
) -> str | None:
    """Best promotable candidate by test log-loss (ADR 0015), or None.

    Promotable = passes the gate vs the reference rule, every behavioural check
    and, if a champion exists, the gate vs the champion.
    """
    promotable = [
        name
        for name, gate in gates.items()
        if gate.passed
        and all_passed(behavior[name])
        and (champion_gates is None or champion_gates[name].passed)
    ]
    if not promotable:
        return None
    loss = summary[summary["metric"] == "log_loss"].set_index("model")["value"]
    return min(promotable, key=lambda name: loss[name])


def run(cfg: ProjectConfig, settings: Settings) -> pd.DataFrame:
    touchpoints = pd.read_parquet(cfg.data.touchpoints_path)
    points, features = build_dataset(touchpoints, cfg)
    use_experiment(
        settings.mlflow_tracking_uri,
        cfg.propensity.experiment_name,
        settings.mlflow_artifact_location,
    )
    registry_name = cfg.propensity.registered_model_name
    models = candidate_models(cfg)
    champion = load_alias(registry_name, CHAMPION)
    if champion is not None:
        models["champion"] = PrefittedModel(champion)
    summary, boot = evaluate_models(
        models, points, features, cfg.propensity.bootstrap_samples, cfg.random_seed
    )
    gates = gate_results(boot, list(models), cfg.propensity.reference_model)
    champion_gates = (
        {m: evaluate_gate(boot, m, "champion") for m in gates}
        if champion is not None
        else None
    )
    base_rate = float(points.loc[points["split"] == "train", "label"].mean())
    behavior = {name: run_behavioral_checks(models[name], base_rate) for name in gates}
    report = cfg.propensity.report_path
    report.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report, index=False)
    run_ids = log_runs(
        summary,
        points,
        cfg,
        settings,
        trace_tags(cfg.data.touchpoints_path),
        gates,
        behavior,
    )
    logger.info("Propensity report -> {}\n{}", report, summary.round(4).to_string())
    for name, checks in behavior.items():
        for c in checks:
            logger.info(
                "Behaviour {} [{}] {}: {} ({})",
                name,
                c.kind,
                c.name,
                "PASS" if c.passed else "FAIL",
                c.detail,
            )
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
    challenger = select_challenger(summary, gates, behavior, champion_gates)
    if challenger is None:
        logger.info("No promotable candidate: registry unchanged")
        return summary
    test_rows = (points["split"] == "test").to_numpy()
    gate = gates[challenger]
    version_tags = {
        **trace_tags(cfg.data.touchpoints_path),
        "model": challenger,
        "gate_reference": gate.reference,
        "gate_log_loss_diff_ci": _fmt(gate.log_loss_diff),
        "beats_champion": str(champion_gates is not None),
    }
    register_challenger(
        run_ids[challenger],
        models[challenger],
        features[test_rows].head(100),
        registry_name,
        version_tags,
    )
    logger.info('Review it in MLflow, then: uv run poe promote --reason "<why>"')
    return summary


def _fmt(interval: tuple[float, float]) -> str:
    return f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"


def main() -> None:  # pragma: no cover - thin CLI wrapper
    settings = get_settings()
    run(load_config(settings.config_path), settings)


if __name__ == "__main__":  # pragma: no cover
    main()
