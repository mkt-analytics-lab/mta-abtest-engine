"""Evaluation: compute metrics on held-out data and apply a quality gate.

The quality gate is what turns "a model was trained" into "a model is good
enough to ship". If the metric is below the threshold in
``configs/config.yaml`` the stage raises and the pipeline (and CI) fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from ..config import ProjectConfig
from ..data.ingest import TARGET_COLUMN
from ..training.train import metadata_path


class QualityGateError(RuntimeError):
    """Raised when a model does not meet the minimum quality bar."""


def compute_metrics(y_true: pd.Series, y_proba: pd.Series) -> dict[str, float]:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def run_evaluation(
    config: ProjectConfig, model_path: Path, tracking_uri: str | None = None
) -> dict[str, float]:
    """Evaluate the saved model on the test split, persist and log metrics."""
    model = joblib.load(model_path)
    test = pd.read_csv(config.data.test_path)
    X, y = test.drop(columns=[TARGET_COLUMN]), test[TARGET_COLUMN]
    metrics = compute_metrics(y, model.predict_proba(X)[:, 1])

    config.evaluation.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.evaluation.metrics_path.write_text(json.dumps(metrics, indent=2))

    meta_file = metadata_path(model_path)
    if tracking_uri and meta_file.exists():
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        run_id = json.loads(meta_file.read_text())["mlflow_run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    logger.info("Test metrics: {}", metrics)
    threshold = config.evaluation.min_roc_auc
    if metrics["roc_auc"] < threshold:
        raise QualityGateError(
            f"roc_auc={metrics['roc_auc']:.3f} is below the gate ({threshold})"
        )
    return metrics
