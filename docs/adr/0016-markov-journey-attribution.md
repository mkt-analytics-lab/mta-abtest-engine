# 0016. Markov journey attribution and the channel benchmark

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
ADR 0010 chose a Markov chain over `channel:stage` states as the journey /
attribution model, plus heuristic baselines and a GA4 benchmark through a
pluggable contract (ADR 0002). States come from the 14-day journey window
(ADR 0009) with `funnel_stage_pre_purchase`, so the purchase is never a state.

## Options considered
- Order: **first order** (state depends on the previous one) vs higher order
  (more context, states explode with ~3k converting journeys).
- Credit: **removal effect normalised to conversions** vs Shapley over channels
  (exact Shapley over 8 channels is feasible but answers a different question
  and is not defined for channel:stage states).
- Uncertainty: point estimates vs **bootstrap over journeys**.

## Decision
- First-order Markov, absorbing states `(conversion)` / `(null)`; conversion
  probability from the fundamental matrix; removal = redirect the state to null.
- Outputs: removal effect and credit per **state** (journey insight: which steps
  matter) and per **channel** (all its states removed: benchmark), the channel
  credit with 95% intervals from 100 multinomial resamples of journeys.
- Baselines: first touch, last touch, and `ga4_last_click_proxy` (last non-direct
  click, as GA4's cross-channel last click).
- Only journeys with a fully observed label (train + test splits, not excluded).
- `ChannelCreditSchema` (`model, channel, conversions_credit, revenue_credit`)
  is the contract every source must follow: our models now, a GA4 export adapter
  and the MMM project later. Revenue credit = channel share x observed revenue.
- `uv run poe attribution` writes `reports/attribution_states.csv` and
  `reports/attribution_channels.csv` and logs a run in MLflow
  (`journey-attribution`).

## Rationale
First order is the standard, explainable baseline for journey attribution and is
estimable with this data. State-level removal effects answer the journey
question; channel-level ones feed the GA4 benchmark and the MTA-MMM comparison.

## Consequences
- Attribution remains correlational (ADR 0002): validate with experiments.
- Rare states (Email, Affiliates) have wide intervals: read with their visits.
- GA4 proxy is not GA4: real DDA / last click must come from the GA4 adapter.
