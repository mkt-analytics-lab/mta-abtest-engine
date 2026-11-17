"""Export a model for the serving image (ADR 0018).

    uv run poe export-champion      # registry champion -> models/
    uv run poe export-test-model    # tiny model on synthetic journeys (CI only)

The image is built FROM models/ (Dockerfile COPY): image tag = model version.
`model.meta.json` records name, version and feature columns; the API refuses
to start if the features differ from the code (training-serving skew).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from ..features.scoring import FEATURE_COLUMNS, build_scoring_features


def write_model(model: Any, out: Path, name: str, version: str, run_id: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    meta = {
        "registered_model": name,
        "model_version": version,
        "mlflow_run_id": run_id,
        "feature_columns": FEATURE_COLUMNS,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Exported {} v{} -> {}", name, version, out)
    return out


def export_champion(name: str, out: Path) -> Path:
    from mlflow import MlflowClient

    from ..training.registry import CHAMPION, load_alias

    model = load_alias(name, CHAMPION)
    if model is None:
        raise RuntimeError(f"No champion for {name}")
    mv = MlflowClient().get_model_version_by_alias(name, CHAMPION)
    return write_model(model, out, name, str(mv.version), mv.run_id or "")


def export_test_model(out: Path, seed: int = 0) -> Path:
    """A small GBM on synthetic journeys: lets CI build and smoke-test the image."""
    from ..training.propensity_models import build_gbm

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-11-01", tz="UTC")
    rows = []
    for u in range(500):
        for i in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "session_key": f"{u}-{i}",
                    "user_pseudo_id": str(u),
                    "touch_position": i + 1,
                    "session_start_ts": start
                    + pd.Timedelta(days=int(u % 60), hours=i * 20),
                    "channel_group": rng.choice(
                        ["Organic Search", "Referral", "Direct"]
                    ),
                    "funnel_step": int(rng.integers(0, 4)),
                }
            )
    tp = pd.DataFrame(rows)
    X = build_scoring_features(tp, tp["session_key"])
    y = rng.binomial(1, 0.02 + 0.05 * X["max_funnel_step"])
    model = build_gbm({"max_iter": 20}, seed).fit(X, y)
    return write_model(model, out, "ci-test-model", "ci-test", "")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    from ..config import get_settings, load_config

    parser = argparse.ArgumentParser(description="Export a model for serving")
    parser.add_argument("what", choices=["champion", "test-model"])
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.what == "test-model":
        export_test_model(settings.model_path)
        return
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    cfg = load_config(settings.config_path)
    export_champion(cfg.propensity.registered_model_name, settings.model_path)


if __name__ == "__main__":  # pragma: no cover
    main()
