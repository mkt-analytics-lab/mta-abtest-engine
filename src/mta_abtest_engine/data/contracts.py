"""Data contracts for the journey marts extracted from BigQuery (ADR 0008).

They repeat, at the Python boundary, the rules dbt tests in the warehouse
(defence in depth): if a mart changes shape or a rule breaks, training stops
here with a clear error instead of learning from bad data.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

CHANNEL_GROUPS = [
    "Organic Search",
    "Referral",
    "Direct",
    "Paid Search",
    "Affiliates",
    "Email",
    "Obfuscated",
    "Unknown",
]
FUNNEL_STAGES = ["browse", "product", "cart", "checkout", "purchase"]
UTC_TIMESTAMP = pd.DatetimeTZDtype(unit="ns", tz="UTC")


def _conversion_is_last_touch(df: pd.DataFrame) -> bool:
    """At most one conversion per user, and only on the last touchpoint."""
    conversions = df[df["is_conversion_touch"]]
    if conversions["user_pseudo_id"].duplicated().any():
        return False
    return bool((conversions["touch_position"] == conversions["n_touchpoints"]).all())


TouchpointsSchema = pa.DataFrameSchema(
    columns={
        "session_key": pa.Column(str, unique=True),
        "user_pseudo_id": pa.Column(str),
        "touch_position": pa.Column(int, pa.Check.ge(1)),
        "n_touchpoints": pa.Column(int, pa.Check.ge(1)),
        "session_start_ts": pa.Column(UTC_TIMESTAMP),
        "channel_group": pa.Column(str, pa.Check.isin(CHANNEL_GROUPS)),
        "session_medium": pa.Column(str),
        "session_source": pa.Column(str, nullable=True),
        "session_campaign": pa.Column(str, nullable=True),
        "funnel_step": pa.Column(int, pa.Check.in_range(0, 4)),
        "funnel_stage": pa.Column(str, pa.Check.isin(FUNNEL_STAGES)),
        "funnel_stage_pre_purchase": pa.Column(str, pa.Check.isin(FUNNEL_STAGES[:-1])),
        "is_conversion_touch": pa.Column(bool),
        "purchase_revenue_usd": pa.Column(float, pa.Check.ge(0)),
    },
    checks=[
        pa.Check(lambda df: len(df) > 0, error="mart is empty"),
        pa.Check(
            lambda df: (df["touch_position"] <= df["n_touchpoints"]).all(),
            error="touch_position beyond n_touchpoints",
        ),
        pa.Check(
            _conversion_is_last_touch,
            error="journeys must end at the first purchase (no touch after it)",
        ),
    ],
    strict=True,
    coerce=True,
)

JourneysSchema = pa.DataFrameSchema(
    columns={
        "user_pseudo_id": pa.Column(str, unique=True),
        "journey_start_ts": pa.Column(UTC_TIMESTAMP),
        "journey_end_ts": pa.Column(UTC_TIMESTAMP),
        "n_touchpoints": pa.Column(int, pa.Check.ge(1)),
        "n_channels": pa.Column(int, pa.Check.ge(1)),
        "converted": pa.Column(bool),
        "revenue_usd": pa.Column(float, pa.Check.ge(0)),
        "first_channel": pa.Column(str, pa.Check.isin(CHANNEL_GROUPS)),
        "last_channel": pa.Column(str, pa.Check.isin(CHANNEL_GROUPS)),
        "channel_path": pa.Column(str),
        "state_path": pa.Column(str),
    },
    checks=[
        pa.Check(lambda df: len(df) > 0, error="mart is empty"),
        pa.Check(
            lambda df: (df["journey_end_ts"] >= df["journey_start_ts"]).all(),
            error="journey ends before it starts",
        ),
        pa.Check(
            lambda df: (df["n_channels"] <= df["n_touchpoints"]).all(),
            error="more channels than touchpoints",
        ),
    ],
    strict=True,
    coerce=True,
)

SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "fct_touchpoints": TouchpointsSchema,
    "fct_user_journeys": JourneysSchema,
}


def validate_mart(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate a mart against its contract; raises SchemaErrors listing all issues."""
    return SCHEMAS[name].validate(df, lazy=True)
