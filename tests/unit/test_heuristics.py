"""First / last touch and the GA4 last non-direct click proxy."""

import pandas as pd

from mta_abtest_engine.attribution.heuristics import heuristic_credits

JOURNEYS = pd.DataFrame(
    {
        "channels": [
            ["Paid Search", "Referral", "Direct"],
            ["Direct"],
            ["Email", "Organic Search"],
            ["Referral"],
        ],
        "converted": [True, True, True, False],
        "revenue_usd": [100.0, 20.0, 50.0, 0.0],
    }
)


def _credit(rule: str) -> dict[str, float]:
    table = heuristic_credits(JOURNEYS, rule)
    return dict(zip(table["channel"], table["conversions_credit"], strict=True))


def test_first_touch():
    assert _credit("first_touch") == {"Paid Search": 1, "Direct": 1, "Email": 1}


def test_last_touch():
    assert _credit("last_touch") == {"Direct": 2, "Organic Search": 1}


def test_last_non_direct_skips_direct_unless_direct_only():
    assert _credit("last_non_direct") == {
        "Referral": 1,
        "Direct": 1,
        "Organic Search": 1,
    }


def test_revenue_follows_the_credited_channel():
    table = heuristic_credits(JOURNEYS, "last_non_direct").set_index("channel")
    assert table.loc["Referral", "revenue_credit"] == 100.0
