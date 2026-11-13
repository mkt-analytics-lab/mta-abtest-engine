"""Hyperparameter tuning of the GBM with Optuna (ADR 0013).

    uv run poe tune

Protocol: tune on a temporal validation INSIDE train (January is never seen),
refit the best configuration on the whole train, evaluate it ONCE on the test
set and gate it against the current GBM (same gate as ADR 0012). Each trial is
a nested MLflow run; the study is stored in SQLite so it can be resumed.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from ..config import ParamSpec, ProjectConfig, Settings, get_settings, load_config
from ..evaluation.gate import GateResult, evaluate_gate
from ..pipelines.propensity import build_dataset, evaluate_models, log_runs, trace_tags
from ..training.baselines import FunnelRuleModel
from ..training.propensity_models import build_gbm

EPS = 1e-6


def temporal_validation_masks(
    points: pd.DataFrame,
    horizon_days: int,
    validation_start: dt.date,
    test_start: dt.date,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-train and validation masks among train points (no label overlap)."""
    horizon = pd.Timedelta(days=horizon_days)
    val_ts = pd.Timestamp(validation_start, tz="UTC")
    test_ts = pd.Timestamp(test_start, tz="UTC")
    ts = points["scoring_ts"]
    is_train = (points["split"] == "train").to_numpy()
    sub_train = is_train & (ts + horizon <= val_ts).to_numpy()
    validation = is_train & ((ts >= val_ts) & (ts + horizon <= test_ts)).to_numpy()
    return sub_train, validation


def suggest_params(trial: optuna.Trial, space: dict[str, ParamSpec]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if spec.type == "int":
            params[name] = trial.suggest_int(
                name, int(spec.low), int(spec.high), log=spec.log
            )
        else:
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    return params


def staged_validation_loss(
    trial: optuna.Trial,
    params: dict[str, Any],
    fixed: dict[str, Any],
    seed: int,
    data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    step: int,
) -> float:
    """Grow the GBM in steps (warm start), report validation log-loss, prune early."""
    X_sub, y_sub, X_val, y_val = data
    max_iter = params["max_iter"]
    model = HistGradientBoostingClassifier(
        random_state=seed,
        warm_start=True,
        **fixed,
        **{k: v for k, v in params.items() if k != "max_iter"},
    )
    loss = np.inf
    for n_iter in [*range(step, max_iter, step), max_iter]:
        model.set_params(max_iter=n_iter)
        model.fit(X_sub, y_sub)
        p = np.clip(model.predict_proba(X_val)[:, 1], EPS, 1 - EPS)
        loss = float(log_loss(y_val, p, labels=[0, 1]))
        trial.report(loss, n_iter)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return loss


def run_study(
    cfg: ProjectConfig,
    points: pd.DataFrame,
    features: pd.DataFrame,
    storage: str,
    log_trial: Any = None,
) -> optuna.Study:
    """Optimize validation log-loss; ``log_trial(trial, params, loss)`` is optional."""
    t, j = cfg.tuning, cfg.journey
    sub, val = temporal_validation_masks(
        points, j.horizon_days, t.validation_start, j.test_start
    )
    data = (
        features[sub],
        points.loc[sub, "label"].to_numpy(),
        features[val],
        points.loc[val, "label"].to_numpy(),
    )
    logger.info("Tuning: {} sub-train / {} validation points", sub.sum(), val.sum())

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, t.search_space)
        loss = staged_validation_loss(
            trial, params, t.fixed, cfg.random_seed, data, t.pruning_step
        )
        if log_trial is not None:
            log_trial(trial, params, loss)
        return loss

    study = optuna.create_study(
        study_name=t.study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=t.n_trials, timeout=t.timeout_seconds)
    return study


def challenge_current_gbm(
    cfg: ProjectConfig,
    points: pd.DataFrame,
    features: pd.DataFrame,
    best_params: dict[str, Any],
) -> tuple[pd.DataFrame, GateResult]:
    """Refit the best params on the whole train, evaluate ONCE on test, gate it."""
    models = {
        "funnel_rule": FunnelRuleModel(),
        "gbm": build_gbm(cfg.propensity.gbm, cfg.random_seed),
        "gbm_tuned": build_gbm({**cfg.tuning.fixed, **best_params}, cfg.random_seed),
    }
    summary, boot = evaluate_models(
        models, points, features, cfg.propensity.bootstrap_samples, cfg.random_seed
    )
    return summary, evaluate_gate(boot, "gbm_tuned", "gbm")


def run(cfg: ProjectConfig, settings: Settings) -> GateResult:
    import mlflow

    points, features = build_dataset(pd.read_parquet(cfg.data.touchpoints_path), cfg)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.propensity.experiment_name)

    def log_trial(trial: optuna.Trial, params: dict[str, Any], loss: float) -> None:
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("validation_log_loss", loss)

    with mlflow.start_run(run_name=cfg.tuning.study_name):
        study = run_study(cfg, points, features, cfg.tuning.storage, log_trial)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_validation_log_loss", study.best_value)
        mlflow.set_tag("n_trials", len(study.trials))

    summary, gate = challenge_current_gbm(cfg, points, features, study.best_params)
    log_runs(
        summary[summary["model"] == "gbm_tuned"],
        points,
        cfg,
        settings,
        trace_tags(cfg.data.touchpoints_path),
        {"gbm_tuned": gate},
    )
    out = cfg.propensity.report_path.with_name("tuning_best_params.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(study.best_params, indent=2))
    logger.info(
        "Best validation log-loss {:.5f} with {} -> {}\n{}\nGate gbm_tuned vs gbm: {} "
        "| log_loss diff CI [{:+.4f}, {:+.4f}]",
        study.best_value,
        study.best_params,
        out,
        summary.round(4).to_string(),
        "PASS" if gate.passed else "FAIL",
        *gate.log_loss_diff,
    )
    return gate


def main() -> None:  # pragma: no cover - thin CLI wrapper
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    settings = get_settings()
    run(load_config(settings.config_path), settings)


if __name__ == "__main__":  # pragma: no cover
    main()
