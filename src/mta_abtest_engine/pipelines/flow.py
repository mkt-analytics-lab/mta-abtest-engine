"""Prefect flows: the project's steps, orchestrated (ADR 0020).

    uv run poe flow daily --date 2021-01-15 [--skip-dbt]   # run once
    uv run poe flow retrain                                 # run once
    uv run poe flow-serve        # schedules: daily 06:10, weekly Monday 07:10 (Berlin)
    uv run prefect server start  # optional persistent UI at http://127.0.0.1:4200

Prefect adds retries, run history, logs and schedules. The step logic is not
duplicated: every task calls the same function as the `poe` commands.
- daily_scoring: dbt build -> extract marts -> score the day -> publish scores.
- weekly_retraining: extract -> propensity (may register a `challenger`; promotion
  stays human, ADR 0015) -> attribution -> publish attribution.
Warehouse steps retry (transient BigQuery / network errors); training does not
(a failure there is a bug or a data problem, retrying hides it).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

from prefect import flow, task

from ..config import ProjectConfig, Settings, get_settings, load_config

WAREHOUSE_RETRIES = {"retries": 2, "retry_delay_seconds": 60}


def _context() -> tuple[ProjectConfig, Settings]:
    settings = get_settings()
    return load_config(settings.config_path), settings


@task(**WAREHOUSE_RETRIES)
def dbt_build() -> None:
    from dbt.cli.main import dbtRunner

    _, settings = _context()
    if settings.gcp_project_id:  # dbt reads the project from the environment
        os.environ.setdefault("APP_GCP_PROJECT_ID", settings.gcp_project_id)
    result = dbtRunner().invoke(
        ["build", "--project-dir", "dbt", "--profiles-dir", "dbt"]
    )
    if not result.success:
        raise RuntimeError(f"dbt build failed: {result.exception}")


@task(**WAREHOUSE_RETRIES)
def extract() -> None:
    from ..data.extract import extract_marts

    cfg, settings = _context()
    extract_marts(cfg, settings)


@task(**WAREHOUSE_RETRIES)
def score(day: dt.date) -> int:
    from ..serving.batch import run_scoring

    cfg, settings = _context()
    return run_scoring(cfg, settings, day, day)


@task
def train_and_gate() -> None:
    from . import propensity

    cfg, settings = _context()
    propensity.run(cfg, settings)


@task
def attribute() -> None:
    from . import attribution

    cfg, settings = _context()
    attribution.run(cfg, settings)


@task(**WAREHOUSE_RETRIES)
def publish() -> None:
    from ..serving.batch import publish_attribution

    cfg, settings = _context()
    publish_attribution(cfg, settings)


@flow(name="daily-scoring", log_prints=True)
def daily_scoring(score_date: dt.date | None = None, skip_dbt: bool = False) -> int:
    """Score yesterday's sessions (or ``score_date``) with the champion."""
    day = score_date or dt.date.today() - dt.timedelta(days=1)
    if not skip_dbt:
        dbt_build()
    extract()
    return score(day)


@flow(name="weekly-retraining", log_prints=True)
def weekly_retraining() -> None:
    extract()
    train_and_gate()
    attribute()
    publish()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Run or serve the Prefect flows")
    sub = parser.add_subparsers(dest="command", required=True)
    daily = sub.add_parser("daily")
    daily.add_argument("--date", type=dt.date.fromisoformat)
    daily.add_argument("--skip-dbt", action="store_true")
    sub.add_parser("retrain")
    sub.add_parser("serve")
    args = parser.parse_args(argv)
    if args.command == "daily":
        daily_scoring(args.date, args.skip_dbt)
    elif args.command == "retrain":
        weekly_retraining()
    else:
        from prefect import serve

        serve(
            daily_scoring.to_deployment(
                name="daily", cron="10 6 * * *", timezone="Europe/Berlin"
            ),
            weekly_retraining.to_deployment(
                name="weekly", cron="10 7 * * 1", timezone="Europe/Berlin"
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    main()
