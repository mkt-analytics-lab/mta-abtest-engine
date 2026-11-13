# 0007. Journey marts, channel grouping and missing channels

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
`int_ga4__sessions` (ADR 0006) has one row per session with its medium. 26% of
sessions have no medium (`'(missing)'`) and two values are obfuscated by Google
(`<Other>`, `(data deleted)`). Purchasers keep browsing after buying: those
sessions would leak the label into features.

## Options considered
Missing channel:
1. **Own category "Unknown"** - honest, the model measures its weight; part of the
   credit is not actionable.
2. **Treat as Direct** - GA4 default, easy comparison; inflates Direct.
3. **Inherit the previous session's channel** - fewer unknowns; an assumption that
   may over-credit earlier channels.

## Decision
- Option 1. Channel groups follow GA4 naming, defined in the seed
  `dbt/seeds/channel_groups.csv`: organic -> Organic Search, referral -> Referral,
  (none) -> Direct, cpc -> Paid Search, affiliate -> Affiliates, email -> Email,
  `<Other>` / `(data deleted)` -> Obfuscated, `(missing)` -> Unknown. The GA4
  benchmark adapter may map Unknown to Direct without touching the model.
- **Journeys are cut at the first purchase** (the purchase session included).
- `funnel_stage_pre_purchase` shows the converting session as `checkout`, so
  features and Markov states never contain the outcome.
- Marts (dataset `mta_marts`): `fct_touchpoints` (one row per touchpoint) and
  `fct_user_journeys` (one row per user: label, revenue, channel and state paths).

## Rationale
Business rules live in the marts and in a seed that marketing can edit; raw facts
stay in intermediate. A singular dbt test enforces the cut at the first purchase.

## Consequences
- An unmapped new medium fails the `not_null` test on `channel_group`: the seed
  must be updated deliberately.
- Right-censoring: users near the end of the data may still convert later; the
  temporal split and labels must account for it (phase 3).
- A full build reads `event_params` about twice (sessions + staging tests),
  ~6-7 GB: fine for the quota; scope tests if it becomes a problem.
