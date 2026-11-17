"""Daily batch scoring with the champion model (ADR 0017).

    uv run poe score --start 2021-01-01 --end 2021-01-31
    uv run poe publish-attribution

For each day: sessions that started that day and ended WITHOUT a purchase are
scored with the production model (`models:/<name>@champion`), using only the
history up to that day and the same feature function as training. Scores and
attribution tables are published to BigQuery for Looker.
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config import ProjectConfig, Settings, get_settings, load_config
from ..features.scoring import build_scoring_features

SCORE_COLUMNS = [
    "score_date",
    "session_key",
    "user_pseudo_id",
    "scoring_ts",
    "channel_group",
    "funnel_stage",
    "propensity",
    "score_decile",
    "model_name",
    "model_version",
    "scored_at",
]


def sessions_to_score(touchpoints: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    """Sessions of ``day`` (UTC) without a purchase in that session."""
    start = pd.Timestamp(day, tz="UTC")
    in_day = (touchpoints["session_start_ts"] >= start) & (
        touchpoints["session_start_ts"] < start + pd.Timedelta(days=1)
    )
    return touchpoints[in_day & ~touchpoints["is_conversion_touch"]]


def score_day(
    touchpoints: pd.DataFrame,
    day: dt.date,
    model: Any,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    """Scores for one day; history after ``day`` is never used."""
    end = pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)
    history = touchpoints[touchpoints["session_start_ts"] < end]
    sessions = sessions_to_score(history, day)
    if sessions.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    features = build_scoring_features(history, sessions["session_key"])
    propensity = model.predict_proba(features)[:, 1]
    ranks = pd.Series(propensity).rank(method="first", ascending=False)
    deciles = np.ceil(ranks / len(ranks) * 10).astype(int).to_numpy()
    out = sessions.reset_index(drop=True).rename(
        columns={"session_start_ts": "scoring_ts"}
    )
    out["score_date"] = day
    out["propensity"] = propensity
    out["score_decile"] = deciles  # 1 = top 10% of the day
    out["model_name"] = model_name
    out["model_version"] = str(model_version)
    out["scored_at"] = pd.Timestamp.now(tz="UTC")
    return out[SCORE_COLUMNS]


def load_champion(name: str) -> tuple[Any, str]:
    from mlflow import MlflowClient

    from ..training.registry import CHAMPION, load_alias

    model = load_alias(name, CHAMPION)
    if model is None:
        raise RuntimeError(f"No champion for {name}: run poe propensity + poe promote")
    version = MlflowClient().get_model_version_by_alias(name, CHAMPION).version
    return model, str(version)


def bigquery_client(cfg: ProjectConfig, settings: Settings) -> Any:
    from google.cloud import bigquery

    if not settings.gcp_project_id:
        raise RuntimeError("APP_GCP_PROJECT_ID is not set (.env)")
    return bigquery.Client(
        project=settings.gcp_project_id, location=cfg.bigquery.location
    )


def run_scoring(
    cfg: ProjectConfig,
    settings: Settings,
    start: dt.date,
    end: dt.date,
    client: Any = None,
) -> int:
    import mlflow

    from .publish import ensure_dataset, write_partition

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    name = cfg.propensity.registered_model_name
    model, version = load_champion(name)
    touchpoints = pd.read_parquet(cfg.data.touchpoints_path)
    client = client or bigquery_client(cfg, settings)
    s = cfg.serving
    ensure_dataset(
        client, settings.gcp_project_id, s.results_dataset, cfg.bigquery.location
    )
    table = f"{settings.gcp_project_id}.{s.results_dataset}.{s.scores_table}"
    total = 0
    for day in pd.date_range(start, end, freq="D").date:
        scores = score_day(touchpoints, day, model, name, version)
        if not scores.empty:
            write_partition(client, scores, table, day, "score_date")
        total += len(scores)
    logger.info(
        "Scored {} sessions with {} v{} ({} to {})", total, name, version, start, end
    )
    return total


def publish_attribution(
    cfg: ProjectConfig, settings: Settings, client: Any = None
) -> list[str]:
    from .publish import ensure_dataset, write_table

    client = client or bigquery_client(cfg, settings)
    s = cfg.serving
    ensure_dataset(
        client, settings.gcp_project_id, s.results_dataset, cfg.bigquery.location
    )
    written = []
    for name in ("attribution_states", "attribution_channels"):
        df = pd.read_csv(cfg.attribution.report_dir / f"{name}.csv")
        table = f"{settings.gcp_project_id}.{s.results_dataset}.{name}"
        written.append(write_table(client, df, table))
    return written


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Batch scoring / publishing")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--start", type=dt.date.fromisoformat, required=True)
    score.add_argument("--end", type=dt.date.fromisoformat)
    sub.add_parser("publish-attribution")
    args = parser.parse_args(argv)
    settings = get_settings()
    cfg = load_config(settings.config_path)
    if args.command == "score":
        run_scoring(cfg, settings, args.start, args.end or args.start)
    else:
        publish_attribution(cfg, settings)


if __name__ == "__main__":  # pragma: no cover
    main()
