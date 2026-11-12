"""Behavioural tests (Ribeiro et al., 2020 "Beyond Accuracy: CheckList").

Aggregate metrics can hide bad behaviour. These tests check that the model
behaves as domain knowledge says it should:

* MFT (Minimum Functionality Test): obvious cases must be predicted right.
* INV (Invariance): changes that should NOT matter do not move predictions.
* DIR (Directional expectation): changes that SHOULD matter move predictions
  in the expected direction.

For the synthetic data: x1 increases P(target), x2 decreases it, x4 is noise.
Replace these with expectations from YOUR domain.
"""

import pandas as pd

from mta_abtest_engine.features.build import build_features


def predict(model, rows: list[dict]) -> list[float]:
    return model.predict_proba(build_features(pd.DataFrame(rows)))[:, 1].tolist()


BASE = {"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0}


def test_mft_obvious_cases(trained_model):
    clearly_positive = BASE | {"x1": 3.0, "x2": -3.0}
    clearly_negative = BASE | {"x1": -3.0, "x2": 3.0}
    pos, neg = predict(trained_model, [clearly_positive, clearly_negative])
    assert pos > 0.9
    assert neg < 0.1


def test_inv_noise_feature_does_not_matter(trained_model):
    low, high = predict(trained_model, [BASE | {"x4": -2.0}, BASE | {"x4": 2.0}])
    assert abs(high - low) < 0.1


def test_dir_x1_increases_probability(trained_model):
    probs = predict(trained_model, [BASE | {"x1": v} for v in (-1.0, 0.0, 1.0)])
    assert probs == sorted(probs)


def test_dir_x2_decreases_probability(trained_model):
    probs = predict(trained_model, [BASE | {"x2": v} for v in (-1.0, 0.0, 1.0)])
    assert probs == sorted(probs, reverse=True)
