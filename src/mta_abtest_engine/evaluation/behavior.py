"""Behavioural checks for the session propensity model (ADR 0014).

Metrics say HOW MUCH a model is right; behavioural checks say whether it is
right for the right reasons. Each check builds small hand-made journeys, turns
them into features with the SAME function used in training, and asserts an
expectation from domain knowledge:

- MFT (minimum functionality): obvious cases must be scored as obvious.
- DIR (directional): browse -> product -> cart must not lower the score, and a
  checkout abandoner must score above a product viewer. Checkout is NOT required
  to score above cart: train data shows first-session checkout abandoners buy
  less often than cart abandoners (4.44% vs 6.23%, n > 3,000 each; ADR 0014).
- INV (invariance): shifting a journey in calendar time must not change it
  (absolute dates are not features; this guards against leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..features.scoring import build_scoring_features

STEPS = {"browse": 0, "product": 1, "cart": 2, "checkout": 3}
START = pd.Timestamp("2020-11-10 10:00", tz="UTC")


@dataclass(frozen=True)
class CheckResult:
    name: str
    kind: str
    passed: bool
    detail: str


def make_journey(
    sessions: list[tuple[str, str, float]], start: pd.Timestamp = START
) -> pd.DataFrame:
    """Touchpoints of one user from (channel group, funnel stage, hours offset)."""
    return pd.DataFrame(
        {
            "session_key": [f"u-{i}" for i in range(len(sessions))],
            "user_pseudo_id": "u",
            "touch_position": range(1, len(sessions) + 1),
            "session_start_ts": [start + pd.Timedelta(hours=h) for _, _, h in sessions],
            "channel_group": [c for c, _, _ in sessions],
            "funnel_step": [STEPS[s] for _, s, _ in sessions],
        }
    )


def score(model: Any, journey: pd.DataFrame) -> float:
    """Propensity of the journey's last session."""
    features = build_scoring_features(journey, journey["session_key"].iloc[[-1]])
    return float(model.predict_proba(features)[:, 1][0])


def run_behavioral_checks(model: Any, base_rate: float) -> list[CheckResult]:
    results: list[CheckResult] = []

    def add(name: str, kind: str, passed: bool, detail: str) -> None:
        results.append(CheckResult(name, kind, bool(passed), detail))

    browse_once = score(model, make_journey([("Organic Search", "browse", 0)]))
    engaged = score(
        model,
        make_journey(
            [
                ("Organic Search", "product", 0),
                ("Referral", "cart", 30),
                ("Direct", "checkout", 50),
            ]
        ),
    )
    add(
        "single_browse_below_base_rate",
        "MFT",
        browse_once < base_rate,
        f"p={browse_once:.4f} vs base={base_rate:.4f}",
    )
    add(
        "engaged_checkout_above_base_rate",
        "MFT",
        engaged > base_rate,
        f"p={engaged:.4f} vs base={base_rate:.4f}",
    )

    by_stage = [
        score(model, make_journey([("Organic Search", stage, 0)])) for stage in STEPS
    ]
    early = by_stage[:3]  # browse, product, cart
    add(
        "early_funnel_not_lower",
        "DIR",
        all(b >= a for a, b in zip(early, early[1:], strict=False)),
        "browse/product/cart = " + ", ".join(f"{p:.4f}" for p in early),
    )
    add(
        "checkout_above_product",
        "DIR",
        by_stage[3] > by_stage[1],
        f"checkout {by_stage[3]:.4f} > product {by_stage[1]:.4f}",
    )

    one_cart = make_journey([("Referral", "browse", 0), ("Referral", "cart", 24)])
    two_carts = make_journey([("Referral", "cart", 0), ("Referral", "cart", 24)])
    p_one, p_two = score(model, one_cart), score(model, two_carts)
    add(
        "more_cart_sessions_not_lower",
        "DIR",
        p_two >= p_one,
        f"{p_two:.4f} >= {p_one:.4f}",
    )

    shifted = make_journey(
        [("Organic Search", "product", 0), ("Referral", "cart", 30)],
        start=START + pd.Timedelta(days=40),
    )
    original = make_journey(
        [("Organic Search", "product", 0), ("Referral", "cart", 30)]
    )
    p_orig, p_shift = score(model, original), score(model, shifted)
    add(
        "calendar_shift_invariant",
        "INV",
        abs(p_orig - p_shift) < 1e-9,
        f"{p_orig:.6f} vs {p_shift:.6f}",
    )
    return results


def all_passed(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results)
