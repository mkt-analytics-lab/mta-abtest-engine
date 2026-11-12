"""Training: fit a model and record the run in MLflow.

``train_model`` is pure (data in -> fitted model out) so tests can call it
without MLflow. ``run_training`` adds the I/O: reading data, saving the model
artifact and logging params + model to MLflow for traceability.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ProjectConfig
from ..data.ingest import TARGET_COLUMN

MODEL_REGISTRY: dict[str, type[ClassifierMixin]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_estimator(name: str, params: dict[str, Any], seed: int) -> Pipeline:
    """Create an (unfitted) sklearn pipeline from the config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    estimator = MODEL_REGISTRY[name](random_state=seed, **params)
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def train_model(
    X: pd.DataFrame, y: pd.Series, name: str, params: dict[str, Any], seed: int
) -> Pipeline:
    """Fit and return a model (pure function: no I/O, no tracking)."""
    model = build_estimator(name, params, seed)
    model.fit(X, y)
    return model


def metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def run_training(config: ProjectConfig, model_path: Path, tracking_uri: str) -> Path:
    """Train on the train split, save the model and log the run to MLflow."""
    import mlflow  # imported here: serving never needs MLflow installed
    import mlflow.sklearn
    from mlflow.models import infer_signature

    train = pd.read_csv(config.data.train_path)
    X, y = train.drop(columns=[TARGET_COLUMN]), train[TARGET_COLUMN]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)
    with mlflow.start_run() as run:
        model = train_model(
            X, y, config.model.name, config.model.params, config.random_seed
        )
        mlflow.log_params(
            {"model_name": config.model.name, "seed": config.random_seed}
            | config.model.params
        )
        # Explicit signature = the model contract: float features in,
        # class probabilities out (what the API serves).
        signature = infer_signature(X.astype("float64"), model.predict_proba(X))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X.head(5).astype("float64"),
            pyfunc_predict_fn="predict_proba",
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        metadata = {
            "mlflow_run_id": run.info.run_id,
            "model_name": config.model.name,
            "params": config.model.params,
            "features": list(X.columns),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        metadata_path(model_path).write_text(json.dumps(metadata, indent=2))

    logger.info("Model saved -> {} (MLflow run {})", model_path, run.info.run_id)
    return model_path
