"""CLI entry point (`python -m mta_abtest_engine.pipelines.run`)."""

from pathlib import Path

import pytest

from mta_abtest_engine.config import get_settings
from mta_abtest_engine.pipelines.run import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cli_runs_a_single_stage(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/config.yaml").write_text(
        (PROJECT_ROOT / "configs/config.yaml").read_text()
    )
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        assert main(["--stage", "ingest"]) == 0
    finally:
        get_settings.cache_clear()
    assert (tmp_path / "data/raw/dataset.csv").exists()


def test_cli_rejects_unknown_stage():
    with pytest.raises(SystemExit):
        main(["--stage", "does-not-exist"])
