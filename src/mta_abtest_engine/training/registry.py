"""Model registry: challenger registration and human-approved promotion (ADR 0015).

Flow:
    poe propensity  -> a candidate that passes the metric gate (vs the reference
                       rule AND vs the current champion, if any) and every
                       behavioural check is registered and aliased `challenger`.
    poe promote     -> a person reviews it in MLflow and moves `champion` to it;
                       who, why and the previous champion are recorded as tags.
Serving always loads `models:/<name>@champion`. Rollback = promote an older version.
"""

from __future__ import annotations

import argparse
import getpass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION, CHALLENGER = "champion", "challenger"


def use_experiment(tracking_uri: str, experiment: str, artifact_location: str) -> None:
    """Point MLflow at the tracking store and the experiment, creating it with an
    explicit artifact location if needed (never a path relative to the cwd)."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        location = artifact_location
        if "://" not in location:
            location = Path(location).resolve().as_uri()
        mlflow.create_experiment(experiment, artifact_location=location)
    mlflow.set_experiment(experiment)


class PrefittedModel:
    """Lets an already trained model (the champion) join a fit/predict ladder."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def fit(self, X: pd.DataFrame, y: Any) -> PrefittedModel:
        return self  # never retrained: evaluated exactly as deployed

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


def load_alias(name: str, alias: str = CHAMPION) -> Any | None:
    """The sklearn model behind an alias, or None if the alias does not exist."""
    import mlflow.sklearn

    try:
        MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException:
        return None
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def register_challenger(
    run_id: str,
    model: Any,
    features_sample: pd.DataFrame,
    name: str,
    tags: dict[str, str],
) -> str:
    """Log the fitted model into its run, register it and alias it `challenger`."""
    import mlflow
    import mlflow.sklearn
    import skops.io as sio
    from mlflow.models import infer_signature

    # MLflow stores sklearn models with skops (safe loading, no arbitrary code).
    # Our own model's internal types are declared trusted explicitly.
    trusted = sio.get_untrusted_types(data=sio.dumps(model))
    sample = features_sample.astype(float)  # features are floats by contract
    signature = infer_signature(sample, model.predict_proba(sample)[:, 1])
    with mlflow.start_run(run_id=run_id):
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            pyfunc_predict_fn="predict_proba",
            registered_model_name=name,
            skops_trusted_types=trusted,
        )
    version = str(info.registered_model_version)
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    client.set_registered_model_alias(name, CHALLENGER, version)
    logger.info("Registered {} v{} as {}", name, version, CHALLENGER)
    return version


def promote(
    name: str,
    reason: str,
    version: str | None = None,
    user: str | None = None,
) -> str:
    """Move `champion` to the challenger (or to ``version``, e.g. a rollback)."""
    if not reason.strip():
        raise ValueError("a promotion needs a reason")
    client = MlflowClient()
    if version is None:
        try:
            version = str(client.get_model_version_by_alias(name, CHALLENGER).version)
        except MlflowException as exc:
            raise RuntimeError(
                f"No '{CHALLENGER}' for {name}: nothing to promote"
            ) from exc
    try:
        previous = str(client.get_model_version_by_alias(name, CHAMPION).version)
    except MlflowException:
        previous = "none"
    client.set_registered_model_alias(name, CHAMPION, str(version))
    audit = {
        "promoted_by": user or getpass.getuser(),
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "promotion_reason": reason,
        "previous_champion": str(previous),
    }
    for key, value in audit.items():
        client.set_model_version_tag(name, str(version), key, value)
    try:
        if str(client.get_model_version_by_alias(name, CHALLENGER).version) == str(
            version
        ):
            client.delete_registered_model_alias(name, CHALLENGER)
    except MlflowException:
        pass
    logger.info("{} v{} is now {} (previous: {})", name, version, CHAMPION, previous)
    return str(version)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    import mlflow

    from ..config import get_settings, load_config

    parser = argparse.ArgumentParser(description="Promote a model version to champion")
    parser.add_argument("--reason", required=True, help="Why (recorded in MLflow)")
    parser.add_argument(
        "--version", help="Explicit version (rollback); default: challenger"
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    cfg = load_config(settings.config_path)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    promote(cfg.propensity.registered_model_name, args.reason, args.version)


if __name__ == "__main__":  # pragma: no cover
    main()
