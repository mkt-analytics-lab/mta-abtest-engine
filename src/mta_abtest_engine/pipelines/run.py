"""Pipeline entry point.

    python -m <package>.pipelines.run --stage all
    python -m <package>.pipelines.run --stage train

Each stage reads its inputs from disk and writes its outputs to disk. That
makes stages independently re-runnable and lets tools such as DVC or Prefect
orchestrate them without changing this code.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from loguru import logger

from ..config import ProjectConfig, Settings, get_settings, load_config
from ..data.ingest import ingest
from ..data.validation import validate
from ..evaluation.evaluate import run_evaluation
from ..features.build import make_feature_splits
from ..training.train import run_training


def stage_ingest(cfg: ProjectConfig, _: Settings) -> None:
    ingest(cfg.data.raw_path, cfg.data.n_samples, cfg.random_seed)


def stage_validate(cfg: ProjectConfig, _: Settings) -> None:
    validate(cfg.data.raw_path, cfg.data.validated_path)


def stage_features(cfg: ProjectConfig, _: Settings) -> None:
    make_feature_splits(
        cfg.data.validated_path,
        cfg.data.train_path,
        cfg.data.test_path,
        cfg.data.test_size,
        cfg.random_seed,
    )


def stage_train(cfg: ProjectConfig, settings: Settings) -> None:
    run_training(cfg, settings.model_path, settings.mlflow_tracking_uri)


def stage_evaluate(cfg: ProjectConfig, settings: Settings) -> None:
    run_evaluation(cfg, settings.model_path, settings.mlflow_tracking_uri)


STAGES: dict[str, Callable[[ProjectConfig, Settings], None]] = {
    "ingest": stage_ingest,
    "validate": stage_validate,
    "features": stage_features,
    "train": stage_train,
    "evaluate": stage_evaluate,
}


def run(stage: str = "all") -> None:
    settings = get_settings()
    cfg = load_config(settings.config_path)
    selected = list(STAGES) if stage == "all" else [stage]
    for name in selected:
        logger.info("=== stage: {} ===", name)
        STAGES[name](cfg, settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pipeline stages")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=get_settings().log_level)
    run(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
