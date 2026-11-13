"""Baseline models: base rate and calibrated funnel rule."""

import numpy as np
import pandas as pd
import pytest

from mta_abtest_engine.training.baselines import BaseRateModel, FunnelRuleModel


@pytest.fixture
def xy():
    X = pd.DataFrame({"max_funnel_step": [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 1, 1])
    return X, y


def test_base_rate_predicts_training_rate(xy):
    X, y = xy
    p = BaseRateModel().fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(p, 0.4)


def test_funnel_rule_is_monotonic_and_calibrated(xy):
    X, y = xy
    model = FunnelRuleModel(smoothing=0.0).fit(X, y)
    p = model.predict_proba(pd.DataFrame({"max_funnel_step": [0, 1, 2, 3]}))[:, 1]
    assert list(p) == [0.0, 0.5, 0.5, 1.0]


def test_funnel_rule_unseen_value_falls_back_to_prior(xy):
    X, y = xy
    model = FunnelRuleModel().fit(X, y)
    p = model.predict_proba(pd.DataFrame({"max_funnel_step": [9]}))[:, 1]
    assert p[0] == pytest.approx(0.4)
