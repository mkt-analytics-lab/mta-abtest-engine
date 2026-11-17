"""Journey attribution: Markov removal effects + heuristic baselines (ADR 0016).

    uv run poe attribution

Uses the same 14-day journey window as the propensity model (ADR 0009) and only
journeys whose label is fully observed. Outputs:
- reports/attribution_states.csv: removal effect and credit per channel:stage.
- reports/attribution_channels.csv: channel credit per model (benchmark contract).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from ..attribution.heuristics import heuristic_credits
from ..attribution.markov import (
    PathCounts,
    channel_groups,
    conversion_probability,
    credits,
    removal_effects,
    state_groups,
)
from ..config import ProjectConfig, Settings, get_settings, load_config
from ..data.contracts import ChannelCreditSchema
from ..features.journeys import build_journey_dataset


def build_paths(touchpoints: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """One row per fully observed journey: states, channels, label, revenue."""
    j = cfg.journey
    windowed, journeys = build_journey_dataset(
        touchpoints, j.horizon_days, j.test_start, j.data_end
    )
    windowed = windowed[windowed["split"] != "excluded"].sort_values(
        ["user_pseudo_id", "touch_position"]
    )
    windowed = windowed.assign(
        state=windowed["channel_group"] + ":" + windowed["funnel_stage_pre_purchase"]
    )
    paths = windowed.groupby("user_pseudo_id").agg(
        states=("state", tuple), channels=("channel_group", list)
    )
    labels = journeys.set_index("user_pseudo_id")[["converted", "revenue_usd"]]
    return paths.join(labels, how="inner").reset_index()


def markov_attribution(
    paths: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """State and channel credits (channel credits with 95% bootstrap intervals)."""
    unique = (
        paths.groupby(["states", "converted"], as_index=False)
        .agg(n=("user_pseudo_id", "size"), revenue=("revenue_usd", "sum"))
        .rename(columns={"states": "path"})
    )
    pc = PathCounts(unique)
    counts = pc.transition_counts()
    conversions = float(paths["converted"].sum())
    revenue = float(paths["revenue_usd"].sum())

    state_effects = removal_effects(counts, pc.states, state_groups(pc.states))
    visits = pd.Series(counts.sum(axis=0), index=pc.states)
    states = pd.DataFrame(
        {
            "state": state_effects.index,
            "visits": visits[state_effects.index].to_numpy(),
            "removal_effect": state_effects.to_numpy(),
            "conversions_credit": credits(state_effects, conversions).to_numpy(),
        }
    ).sort_values("removal_effect", ascending=False, ignore_index=True)

    groups = channel_groups(pc.states)
    effects = removal_effects(counts, pc.states, groups)
    share = credits(effects, 1.0)
    rng = np.random.default_rng(seed)
    weights = unique["n"].to_numpy(float)
    samples = []
    for _ in range(n_bootstrap):
        w = rng.multinomial(int(weights.sum()), weights / weights.sum()).astype(float)
        s = credits(removal_effects(pc.transition_counts(w), pc.states, groups), 1.0)
        conv = w[unique["converted"].to_numpy()].sum()
        samples.append(s * conv)
    boot = pd.DataFrame(samples)
    channels = pd.DataFrame(
        {
            "model": "markov",
            "channel": share.index,
            "removal_effect": effects.to_numpy(),
            "conversions_credit": (share * conversions).to_numpy(),
            "revenue_credit": (share * revenue).to_numpy(),
            "conversions_ci_low": boot.quantile(0.025).reindex(share.index).to_numpy(),
            "conversions_ci_high": boot.quantile(0.975).reindex(share.index).to_numpy(),
        }
    )
    return states, channels, conversion_probability(counts, pc.states)


def channel_benchmark(paths: pd.DataFrame, markov: pd.DataFrame) -> pd.DataFrame:
    """All models in the benchmark contract (one row per model and channel)."""
    rules = [
        heuristic_credits(paths, "first_touch"),
        heuristic_credits(paths, "last_touch"),
        heuristic_credits(paths, "last_non_direct").assign(
            model="ga4_last_click_proxy"
        ),
    ]
    table = pd.concat([markov, *rules], ignore_index=True)
    return ChannelCreditSchema.validate(table, lazy=True)


def run(cfg: ProjectConfig, settings: Settings) -> pd.DataFrame:
    from ..pipelines.propensity import trace_tags
    from ..training.registry import use_experiment

    paths = build_paths(pd.read_parquet(cfg.data.touchpoints_path), cfg)
    a = cfg.attribution
    states, markov, p_conv = markov_attribution(
        paths, a.bootstrap_samples, cfg.random_seed
    )
    benchmark = channel_benchmark(paths, markov)
    a.report_dir.mkdir(parents=True, exist_ok=True)
    states_path = a.report_dir / "attribution_states.csv"
    channels_path = a.report_dir / "attribution_channels.csv"
    states.to_csv(states_path, index=False)
    benchmark.to_csv(channels_path, index=False)

    import mlflow

    use_experiment(
        settings.mlflow_tracking_uri,
        a.experiment_name,
        settings.mlflow_artifact_location,
    )
    with mlflow.start_run(run_name="markov"):
        mlflow.set_tags(trace_tags(cfg.data.touchpoints_path))
        mlflow.log_params(
            {
                "horizon_days": cfg.journey.horizon_days,
                "journeys": len(paths),
                "states": len(states),
                "bootstrap_samples": a.bootstrap_samples,
            }
        )
        mlflow.log_metrics(
            {
                "conversions": float(paths["converted"].sum()),
                "markov_conversion_probability": p_conv,
            }
        )
        mlflow.log_artifact(str(states_path))
        mlflow.log_artifact(str(channels_path))

    pivot = (
        benchmark.pivot(index="channel", columns="model", values="conversions_credit")
        .fillna(0)
        .round(1)
    )
    logger.info(
        "Journeys: {} | conversions: {} | P(conv): {:.4f}\nTop states:\n{}"
        "\nChannel credit (conversions):\n{}",
        len(paths),
        int(paths["converted"].sum()),
        p_conv,
        states.head(10).round(4).to_string(),
        pivot.to_string(),
    )
    return benchmark


def main() -> None:  # pragma: no cover - thin CLI wrapper
    settings = get_settings()
    run(load_config(settings.config_path), settings)


if __name__ == "__main__":  # pragma: no cover
    main()
