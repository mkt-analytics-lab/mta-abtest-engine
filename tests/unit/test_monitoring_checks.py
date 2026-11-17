"""PSI, matured performance and Evidently drift parsing."""

import numpy as np
import pandas as pd
import pytest

from mta_abtest_engine.monitoring.checks import matured_performance, psi
from mta_abtest_engine.monitoring.drift import feature_drift


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = rng.beta(1, 50, 5000)
    assert psi(ref, ref) == pytest.approx(0, abs=1e-9)
    assert psi(ref, ref * 3) > 0.2


def test_calibration_ratio_detects_over_prediction():
    labels = np.array([0] * 99 + [1])  # observed 1%
    perf = matured_performance(labels, np.full(100, 0.02))  # predicts 2%
    assert perf["calibration_ratio"] == pytest.approx(2.0)
    assert "log_loss" in perf


def test_no_positives_gives_calibration_but_no_ranking_metrics():
    perf = matured_performance(np.zeros(10, dtype=int), np.full(10, 0.01))
    assert perf["calibration_ratio"] == np.inf
    assert "pr_auc" not in perf


def test_feature_drift_flags_the_shifted_column(tmp_path):
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600)})
    ref.index = [f"s{i}" for i in range(600)]  # like real features: session keys
    cur = ref.assign(a=ref["a"] + 3)
    drift = feature_drift(ref, cur, tmp_path / "drift.html")
    assert drift.drifted_columns == ["a"]
    assert drift.share_drifted == pytest.approx(0.5)
    assert drift.report_path.exists()
