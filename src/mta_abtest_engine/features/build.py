"""Feature engineering.

RULE: this module is the ONLY place where features are computed. Training
(``pipelines/run.py``) and serving (``serving/app.py``) both call
``build_features``. If serving re-implemented the logic, the model would
see different inputs in production than in training: training-serving skew.

Keep functions pure (DataFrame in -> DataFrame out, no I/O) so they are easy
to test and reuse.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from ..data.ingest import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_FEATURES: list[str] = [*FEATURE_COLUMNS, "x1_x2"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input matrix (only model features, in a fixed order)."""
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    features = df[FEATURE_COLUMNS].astype(float).copy()
    features["x1_x2"] = features["x1"] * features["x2"]  # example interaction
    return features[MODEL_FEATURES]


def make_feature_splits(
    input_path: Path,
    train_path: Path,
    test_path: Path,
    test_size: float,
    seed: int,
) -> tuple[Path, Path]:
    """Build features and persist a stratified train/test split."""
    df = pd.read_csv(input_path)
    features = build_features(df)
    features[TARGET_COLUMN] = df[TARGET_COLUMN].to_numpy()

    train, test = train_test_split(
        features,
        test_size=test_size,
        random_state=seed,
        stratify=features[TARGET_COLUMN],
    )
    for path, split in ((train_path, train), (test_path, test)):
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
    logger.info("Features: train={} test={} rows", len(train), len(test))
    return train_path, test_path
