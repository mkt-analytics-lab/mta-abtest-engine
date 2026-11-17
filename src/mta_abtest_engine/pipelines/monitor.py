"""Daily monitoring of the champion (ADR 0021).

    uv run poe monitor --start 2021-01-15 [--end 2021-01-31]

For a day D:
- feature drift (Evidently) and prediction drift (PSI) of the sessions scored
  on D against the training reference - available immediately;
- matured performance of the sessions scored on D - 14 days (their labels now
  exist): calibration ratio, log-loss, PR-AUC.
Thresholds in `configs/config.yaml` turn these into alerts and a retraining
recommendation. One row per day goes to `mta_results.monitoring_daily`.
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config import ProjectConfig, Settings, get_settings, load_config
from ..features.journeys import build_scoring_points
from ..features.scoring import build_scoring_features
from ..monitoring.checks import matured_performance, psi
from ..monitoring.drift import feature_drift
from ..serving.batch import sessions_to_score

MONITOR_COLUMNS = [
    "monitor_date",
    "model_version",
    "n_scored",
    "mean_propensity",
    "share_drifted_features",
    "drifted_features",
    "psi_propensity",
    "matured_date",
    "matured_n",
    "observed_rate",
    "calibration_ratio",
    "log_loss",
    "pr_auc",
    "brier",
    "top_decile_lift",
    "alert_feature_drift",
    "alert_prediction_drift",
    "alert_calibration",
    "retrain_recommended",
]


def reference_set(
    touchpoints: pd.DataFrame, points: pd.DataFrame, model: Any, n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and predictions of a sample of the training scoring points."""
    train = points[points["split"] == "train"]
    sample = train.sample(min(n, len(train)), random_state=seed)
    features = build_scoring_features(touchpoints, sample["session_key"])
    return features, model.predict_proba(features)[:, 1]


def day_predictions(
    touchpoints: pd.DataFrame, day: dt.date, model: Any
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Features, session keys and predictions of the sessions scored on ``day``."""
    end = pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)
    history = touchpoints[touchpoints["session_start_ts"] < end]
    sessions = sessions_to_score(history, day)
    if sessions.empty:
        return pd.DataFrame(), pd.Series(dtype=str), np.array([])
    features = build_scoring_features(history, sessions["session_key"])
    return features, sessions["session_key"], model.predict_proba(features)[:, 1]


def monitor_day(
    touchpoints: pd.DataFrame,
    points: pd.DataFrame,
    cfg: ProjectConfig,
    model: Any,
    model_version: str,
    day: dt.date,
    reference: tuple[pd.DataFrame, np.ndarray],
) -> dict[str, Any]:
    m = cfg.monitoring
    ref_features, ref_pred = reference
    row: dict[str, Any] = {"monitor_date": day, "model_version": model_version}

    features, _, pred = day_predictions(touchpoints, day, model)
    row["n_scored"] = len(pred)
    if len(pred):
        drift = feature_drift(
            ref_features, features, m.report_dir / f"drift_{day:%Y%m%d}.html"
        )
        row.update(
            mean_propensity=float(pred.mean()),
            share_drifted_features=drift.share_drifted,
            drifted_features=",".join(drift.drifted_columns),
            psi_propensity=psi(ref_pred, pred),
        )

    matured = day - dt.timedelta(days=cfg.journey.horizon_days)
    _, keys, matured_pred = day_predictions(touchpoints, matured, model)
    labels = points.set_index("session_key")["label"].reindex(keys).to_numpy()
    row.update(matured_date=matured, matured_n=len(matured_pred))
    if len(matured_pred) and not np.isnan(labels).any():
        row.update(matured_performance(labels.astype(int), matured_pred))

    row["alert_feature_drift"] = (
        row.get("share_drifted_features", 0) > m.share_drifted_max
    )
    row["alert_prediction_drift"] = row.get("psi_propensity", 0) > m.psi_max
    ratio = row.get("calibration_ratio")
    row["alert_calibration"] = ratio is not None and not (
        m.calibration_ratio_min <= ratio <= m.calibration_ratio_max
    )
    row["retrain_recommended"] = row["alert_calibration"] or (
        row["alert_feature_drift"] and row["alert_prediction_drift"]
    )
    return row


def run(
    cfg: ProjectConfig,
    settings: Settings,
    start: dt.date,
    end: dt.date,
    client: Any = None,
    champion: tuple[Any, str] | None = None,
) -> pd.DataFrame:
    import mlflow

    from ..serving.batch import bigquery_client, load_champion
    from ..serving.publish import ensure_dataset, write_partition

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model, version = champion or load_champion(cfg.propensity.registered_model_name)
    touchpoints = pd.read_parquet(cfg.data.touchpoints_path)
    j = cfg.journey
    points = build_scoring_points(touchpoints, j.horizon_days, j.test_start, j.data_end)
    reference = reference_set(
        touchpoints, points, model, cfg.monitoring.reference_sample, cfg.random_seed
    )
    rows = [
        monitor_day(touchpoints, points, cfg, model, version, day, reference)
        for day in pd.date_range(start, end, freq="D").date
    ]
    table = pd.DataFrame(rows).reindex(columns=MONITOR_COLUMNS)
    client = client or bigquery_client(cfg, settings)
    ensure_dataset(
        client,
        settings.gcp_project_id,
        cfg.serving.results_dataset,
        cfg.bigquery.location,
    )
    dest = (
        f"{settings.gcp_project_id}.{cfg.serving.results_dataset}."
        f"{cfg.monitoring.table}"
    )
    for day in table["monitor_date"]:
        day_rows = table[table["monitor_date"] == day]
        write_partition(client, day_rows, dest, day, "monitor_date")
    cols = [
        "monitor_date",
        "share_drifted_features",
        "psi_propensity",
        "calibration_ratio",
        "retrain_recommended",
    ]
    logger.info(
        "Monitoring\n{}", table.reindex(columns=cols).round(3).to_string(index=False)
    )
    return table


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Daily monitoring of the champion")
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat)
    args = parser.parse_args(argv)
    settings = get_settings()
    run(load_config(settings.config_path), settings, args.start, args.end or args.start)


if __name__ == "__main__":  # pragma: no cover
    main()
