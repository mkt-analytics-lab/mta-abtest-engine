"""Training-serving parity: the API computes exactly the training features."""

import pandas as pd

from mta_abtest_engine.features.scoring import build_scoring_features
from mta_abtest_engine.serving.journey import FUNNEL_STEPS, last_session_features
from mta_abtest_engine.serving.schemas import Session


def test_api_session_fields_cover_the_feature_inputs():
    assert set(Session.model_fields) == {
        "session_start_ts",
        "channel_group",
        "funnel_stage",
    }


def test_api_features_equal_batch_features():
    sessions = [
        {
            "session_start_ts": "2021-01-08T09:00:00Z",
            "channel_group": "Organic Search",
            "funnel_stage": "product",
        },
        {
            "session_start_ts": "2021-01-10T18:30:00Z",
            "channel_group": "Referral",
            "funnel_stage": "cart",
        },
    ]
    touchpoints = pd.DataFrame(
        {
            "session_key": ["a", "b"],
            "user_pseudo_id": "u",
            "touch_position": [1, 2],
            "session_start_ts": pd.to_datetime(
                [s["session_start_ts"] for s in sessions], utc=True
            ),
            "channel_group": [s["channel_group"] for s in sessions],
            "funnel_step": [FUNNEL_STEPS[s["funnel_stage"]] for s in sessions],
        }
    )
    batch = build_scoring_features(touchpoints, pd.Series(["b"])).reset_index(drop=True)
    api = last_session_features(sessions).reset_index(drop=True)
    pd.testing.assert_frame_equal(api, batch)
