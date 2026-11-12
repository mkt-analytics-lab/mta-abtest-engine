"""Ingestion: bring raw data into ``data/raw``.

The template ships with a SYNTHETIC dataset so the whole pipeline runs
end-to-end on day one (walking skeleton). Replace ``generate_dataset`` with
your real extraction (API, warehouse query, files...) and keep the same
contract: write a CSV that passes ``data.validation.RawDataSchema``.

The synthetic data has a known ground truth, which makes behavioural model
tests meaningful:  P(target=1) = sigmoid(2*x1 - 1.5*x2 + 0.5*x3)
``x4`` is pure noise and should NOT influence predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

FEATURE_COLUMNS: list[str] = ["x1", "x2", "x3", "x4"]
TARGET_COLUMN = "target"


def generate_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    """Create a reproducible synthetic binary-classification dataset."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, len(FEATURE_COLUMNS)))
    logits = 2.0 * features[:, 0] - 1.5 * features[:, 1] + 0.5 * features[:, 2]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    target = rng.binomial(1, probabilities)

    df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    df[TARGET_COLUMN] = target
    return df


def ingest(output_path: Path, n_samples: int, seed: int) -> Path:
    """Ingest raw data and persist it to ``output_path``."""
    df = generate_dataset(n_samples=n_samples, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Ingested {} rows -> {}", len(df), output_path)
    return output_path
