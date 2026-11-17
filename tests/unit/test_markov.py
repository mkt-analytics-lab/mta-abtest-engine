"""Markov attribution on tiny paths with answers computable by hand."""

import numpy as np
import pandas as pd
import pytest

from mta_abtest_engine.attribution.markov import (
    PathCounts,
    channel_groups,
    conversion_probability,
    credits,
    removal_effects,
    state_groups,
)


def _pc(rows):
    return PathCounts(pd.DataFrame(rows, columns=["path", "converted", "n"]))


def test_conversion_probability_matches_observed_rate():
    pc = _pc([(("A:browse",), True, 1), (("B:browse",), False, 1)])
    assert conversion_probability(pc.transition_counts(), pc.states) == pytest.approx(
        0.5
    )


def test_removal_effect_of_an_irrelevant_state_is_zero():
    # A always converts, B never: removing B changes nothing, removing A kills all.
    pc = _pc([(("A:browse",), True, 1), (("B:browse",), False, 1)])
    effects = removal_effects(
        pc.transition_counts(), pc.states, state_groups(pc.states)
    )
    assert effects["A:browse"] == pytest.approx(1.0)
    assert effects["B:browse"] == pytest.approx(0.0)
    assert credits(effects, 1.0)["A:browse"] == pytest.approx(1.0)


def test_sequential_states_share_credit():
    # A -> B -> conversion only: both are necessary, credit is split equally.
    pc = _pc([(("A:product", "B:cart"), True, 3)])
    effects = removal_effects(
        pc.transition_counts(), pc.states, state_groups(pc.states)
    )
    shares = credits(effects, 3.0)
    assert shares["A:product"] == pytest.approx(1.5)
    assert shares["B:cart"] == pytest.approx(1.5)


def test_channel_groups_remove_all_stages_of_a_channel():
    pc = _pc(
        [
            (("Referral:browse", "Referral:cart"), True, 1),
            (("Direct:browse",), False, 1),
        ]
    )
    groups = channel_groups(pc.states)
    assert sorted(groups["Referral"]) == ["Referral:browse", "Referral:cart"]
    effects = removal_effects(pc.transition_counts(), pc.states, groups)
    assert effects["Referral"] == pytest.approx(1.0)
    assert effects["Direct"] == pytest.approx(0.0)


def test_weights_change_transition_counts():
    pc = _pc([(("A:browse",), True, 1), (("B:browse",), False, 1)])
    heavy_b = pc.transition_counts(np.array([1.0, 9.0]))
    assert conversion_probability(heavy_b, pc.states) == pytest.approx(0.1)


def test_no_conversions_gives_zero_effects():
    pc = _pc([(("A:browse",), False, 2)])
    effects = removal_effects(
        pc.transition_counts(), pc.states, state_groups(pc.states)
    )
    assert (effects == 0).all()
