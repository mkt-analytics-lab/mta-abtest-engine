"""Rule-based channel attribution: baselines and the GA4 benchmark proxy.

- first_touch: all credit to the first touchpoint's channel.
- last_touch: all credit to the last touchpoint's channel.
- last_non_direct: like GA4's cross-channel last click, Direct is ignored unless
  the journey is Direct only (proxy for GA4 rows in the benchmark, ADR 0002).
"""

from __future__ import annotations

import pandas as pd

DIRECT = "Direct"


def _pick(channels: list[str], rule: str) -> str:
    if rule == "first_touch":
        return channels[0]
    if rule == "last_touch":
        return channels[-1]
    non_direct = [c for c in channels if c != DIRECT]
    return non_direct[-1] if non_direct else DIRECT


def heuristic_credits(journeys: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Channel credit for converted journeys (columns: channels list, revenue_usd)."""
    converted = journeys[journeys["converted"]]
    winners = converted["channels"].map(lambda c: _pick(c, rule))
    return (
        pd.DataFrame({"channel": winners, "revenue_usd": converted["revenue_usd"]})
        .groupby("channel", as_index=False)
        .agg(
            conversions_credit=("revenue_usd", "size"),
            revenue_credit=("revenue_usd", "sum"),
        )
        .assign(model=rule)
    )
