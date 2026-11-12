"""Data contracts: what "valid data" means, written as code.

Validating at the boundary (right after ingestion) stops bad data before it
reaches training. A failed contract should fail the pipeline loudly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from loguru import logger

RawDataSchema = pa.DataFrameSchema(
    columns={
        "x1": pa.Column(float, nullable=False),
        "x2": pa.Column(float, nullable=False),
        "x3": pa.Column(float, nullable=False),
        "x4": pa.Column(float, nullable=False),
        "target": pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
    },
    strict=True,  # unexpected columns are a contract violation
    coerce=True,
    checks=pa.Check(lambda df: len(df) > 0, error="dataset is empty"),
)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a dataframe against the raw data contract (raises on failure)."""
    return RawDataSchema.validate(df, lazy=True)


def validate(input_path: Path, output_path: Path) -> Path:
    """Validate the raw file and write the validated copy."""
    df = validate_dataframe(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Validated {} rows -> {}", len(df), output_path)
    return output_path
