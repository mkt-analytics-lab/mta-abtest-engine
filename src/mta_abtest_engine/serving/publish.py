"""Publish result tables to BigQuery (dataset `mta_results`, ADR 0017).

dbt owns `mta_marts`; Python owns `mta_results`. Writes are idempotent:
- daily scores go to their date partition (partition decorator + WRITE_TRUNCATE),
  so re-scoring a day replaces it instead of duplicating rows;
- report tables (attribution) are replaced as a whole.
Load jobs are free in BigQuery (no bytes billed).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
from google.cloud import bigquery
from loguru import logger


def ensure_dataset(client: Any, project: str, dataset: str, location: str) -> None:
    ds = bigquery.Dataset(f"{project}.{dataset}")
    ds.location = location
    client.create_dataset(ds, exists_ok=True)


def write_partition(
    client: Any, df: pd.DataFrame, table: str, day: dt.date, field: str
) -> str:
    """Replace one day partition of a column-partitioned table."""
    destination = f"{table}${day:%Y%m%d}"
    config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field=field
        ),
    )
    client.load_table_from_dataframe(df, destination, job_config=config).result()
    logger.info("Wrote {} rows -> {}", len(df), destination)
    return destination


def write_table(client: Any, df: pd.DataFrame, table: str) -> str:
    """Replace a whole table."""
    config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    client.load_table_from_dataframe(df, table, job_config=config).result()
    logger.info("Wrote {} rows -> {}", len(df), table)
    return table
