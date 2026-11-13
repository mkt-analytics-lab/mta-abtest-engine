"""Extraction: download the dbt journey marts from BigQuery to local Parquet.

    uv run poe extract

Each mart is read with an explicit column list (the columns of its contract),
under the per-query cost guard, validated, and only then written to disk
(ADR 0008). Authentication: Application Default Credentials (ADR 0004).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from google.cloud import bigquery
from loguru import logger

from ..config import ProjectConfig, Settings, get_settings, load_config
from .contracts import SCHEMAS, validate_mart


def build_query(project: str, dataset: str, table: str) -> str:
    """SELECT only the contract columns of a mart (no SELECT *)."""
    columns = ",\n    ".join(SCHEMAS[table].columns)
    return f"select\n    {columns}\nfrom `{project}.{dataset}.{table}`"


def extract_mart(
    client: Any, sql: str, name: str, max_bytes_billed: int, output_path: Path
) -> Path:
    """Run one query under the cost guard, validate the result, write Parquet."""
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes_billed)
    df = client.query(sql, job_config=job_config).to_dataframe()
    df = validate_mart(name, df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Extracted {} rows of {} -> {}", len(df), name, output_path)
    return output_path


def extract_marts(
    cfg: ProjectConfig, settings: Settings, client: Any | None = None
) -> dict[str, Path]:
    """Extract every journey mart. ``client`` is injectable for tests."""
    project = settings.gcp_project_id
    if not project:
        raise RuntimeError(
            "APP_GCP_PROJECT_ID is not set: copy .env.example to .env and set it"
        )
    client = client or bigquery.Client(project=project, location=cfg.bigquery.location)
    targets = {
        "fct_touchpoints": cfg.data.touchpoints_path,
        "fct_user_journeys": cfg.data.journeys_path,
    }
    return {
        name: extract_mart(
            client,
            build_query(project, cfg.bigquery.marts_dataset, name),
            name,
            cfg.bigquery.maximum_bytes_billed,
            path,
        )
        for name, path in targets.items()
    }


def main() -> None:  # pragma: no cover - thin CLI wrapper, needs BigQuery
    settings = get_settings()
    extract_marts(load_config(settings.config_path), settings)


if __name__ == "__main__":  # pragma: no cover
    main()
