"""Metrics and user-level bootstrap."""

import numpy as np
import pytest

from mta_abtest_engine.evaluation.metrics import (
    bootstrap_metrics,
    compute_metrics,
    paired_difference,
    summarize,
    top_decile_lift,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 2000
    groups = rng.integers(0, 500, n)  # ~4 rows per user
    signal = rng.normal(size=n)
    p_true = 1 / (1 + np.exp(-(signal - 3)))
    y = rng.binomial(1, p_true)
    return y, {"good": p_true, "flat": np.full(n, y.mean())}, groups


def test_top_decile_lift_of_perfect_ranking():
    y = np.array([0] * 90 + [1] * 10)
    p = np.arange(100) / 100  # positives get the top 10 scores
    assert top_decile_lift(y, p) == pytest.approx(10.0)


def test_constant_prediction_has_lift_one_regardless_of_row_order():
    y = np.array([1] * 5 + [0] * 95)  # positives first: row order must not matter
    assert top_decile_lift(y, np.full(100, 0.3)) == pytest.approx(1.0)


def test_ties_at_the_cut_off_are_shared():
    # 20 rows share the top score (10 positives); the top 10% takes half of them.
    y = np.array([1] * 10 + [0] * 10 + [0] * 80)
    p = np.array([0.9] * 20 + [0.1] * 80)
    assert top_decile_lift(y, p) == pytest.approx((5 / 10) / 0.1)


def test_informative_model_beats_flat_prediction(data):
    y, preds, _ = data
    good, flat = compute_metrics(y, preds["good"]), compute_metrics(y, preds["flat"])
    assert good["log_loss"] < flat["log_loss"]
    assert good["pr_auc"] > flat["pr_auc"]
    assert flat["top_decile_lift"] == pytest.approx(1.0, abs=0.3)


def test_bootstrap_is_reproducible_and_intervals_contain_estimate(data):
    y, preds, groups = data
    boot_a = bootstrap_metrics(y, preds, groups, n_samples=60, seed=1)
    boot_b = bootstrap_metrics(y, preds, groups, n_samples=60, seed=1)
    assert boot_a.equals(boot_b)
    summary = summarize(y, preds, boot_a)
    inside = (summary["ci_low"] <= summary["value"]) & (
        summary["value"] <= summary["ci_high"]
    )
    assert inside.all()


def test_paired_difference_detects_a_real_improvement(data):
    y, preds, groups = data
    boot = bootstrap_metrics(y, preds, groups, n_samples=60, seed=2)
    low, high = paired_difference(boot, "good", "flat", "log_loss")
    assert high < 0  # good always has lower log-loss: interval excludes zero
