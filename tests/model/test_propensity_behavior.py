"""Behavioural checks: they pass on a model that learned the domain structure and
catch a model that learned it backwards (so the checks themselves are tested)."""

import numpy as np
import pandas as pd
import pytest

from mta_abtest_engine.evaluation.behavior import (
    START,
    STEPS,
    all_passed,
    make_journey,
    run_behavioral_checks,
    score,
)
from mta_abtest_engine.features.scoring import build_scoring_features
from mta_abtest_engine.training.propensity_models import build_gbm

CHANNELS = ["Organic Search", "Referral", "Direct", "Paid Search"]


@pytest.fixture(scope="module")
def synthetic_training():
    """Journeys where purchase probability grows with funnel depth and sessions."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(3000):
        n = rng.integers(1, 5)
        hours = np.cumsum(rng.uniform(1, 72, n))
        for i in range(n):
            rows.append(
                {
                    "session_key": f"{u}-{i}",
                    "user_pseudo_id": str(u),
                    "touch_position": i + 1,
                    "session_start_ts": START + pd.Timedelta(hours=float(hours[i])),
                    "channel_group": rng.choice(CHANNELS),
                    "funnel_step": int(rng.choice(4, p=[0.6, 0.25, 0.1, 0.05])),
                }
            )
    tp = pd.DataFrame(rows)
    X = build_scoring_features(tp, tp["session_key"])
    logit = (
        -5
        + 1.0 * X["max_funnel_step"]
        + 0.6 * X["cur_funnel_step"]
        + 0.2 * (X["n_sessions"])
    )
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return X, y


@pytest.fixture(scope="module")
def good_model(synthetic_training):
    X, y = synthetic_training
    return build_gbm({"max_iter": 100, "min_samples_leaf": 50}, seed=0).fit(X, y)


class ReversedModel:
    """Wraps a model and inverts its ranking: deeper funnel = lower score."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        p = 0.05 - self.model.predict_proba(X)[:, 1] * 0.04
        return np.column_stack([1 - p, p])


def test_make_journey_feeds_the_real_feature_function():
    journey = make_journey([("Referral", "browse", 0), ("Direct", "cart", 24)])
    assert list(journey["funnel_step"]) == [STEPS["browse"], STEPS["cart"]]
    assert journey["touch_position"].tolist() == [1, 2]


def test_well_behaved_model_passes_all_checks(good_model, synthetic_training):
    _, y = synthetic_training
    results = run_behavioral_checks(good_model, base_rate=float(np.mean(y)))
    failed = [r for r in results if not r.passed]
    assert all_passed(results), failed
    assert {r.kind for r in results} == {"MFT", "DIR", "INV"}


def test_backwards_model_is_caught(good_model, synthetic_training):
    _, y = synthetic_training
    results = run_behavioral_checks(ReversedModel(good_model), float(np.mean(y)))
    failed = {r.name for r in results if not r.passed}
    assert {"checkout_above_browse", "deeper_stage_not_lower"} <= failed
    assert "calendar_shift_invariant" not in failed  # still date-free


def test_score_uses_the_last_session(good_model):
    browse_then_cart = make_journey(
        [("Referral", "browse", 0), ("Referral", "cart", 5)]
    )
    cart_then_browse = make_journey(
        [("Referral", "cart", 0), ("Referral", "browse", 5)]
    )
    assert score(good_model, browse_then_cart) != score(good_model, cart_then_browse)
