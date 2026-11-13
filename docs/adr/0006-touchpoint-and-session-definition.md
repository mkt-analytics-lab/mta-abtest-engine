# 0006. Touchpoint and session definition

- **Status:** Accepted
- **Date:** 2026-11-13

## Context
GA4 exploration (phase 1, `dbt/analyses/exploration`) showed:
- `ga_session_id` on 100% of events; it is the session start in Unix seconds, so it
  is only unique together with `user_pseudo_id`.
- `medium` is empty on `session_start`; it appears on ~1/3 of events (the session
  entrance). 26% of sessions have no medium at all.
- Only ~6-8 mediums (organic, referral, (none), cpc, affiliate, email, plus the
  obfuscated `<Other>` and `(data deleted)`).
- 61% of purchasers have 2+ sessions through 2+ channels: multi-touch matters.

## Options considered
1. **Session + channel** - classic MTA, directly comparable with GA4; poor journeys
   with so few channels, says nothing about where users drop off.
2. **Session + channel + funnel depth** - each session carries its channel and the
   deepest funnel stage reached; matches the journey-optimization goal and still
   aggregates by channel for the GA4 benchmark. More states (~7 x 5).
3. **Every event** - maximum detail; noise, clickstream rather than attribution.

## Decision
Option 2. Table `int_ga4__sessions`, one row per session:
- **Session key:** `user_pseudo_id` + `ga_session_id`.
- **Session channel:** first non-null `medium` (and `source`, `campaign`) by event
  time; `'(missing)'` when none. Channel grouping and the treatment of missing /
  obfuscated values are decided in the marts.
- **Funnel stage** (deepest reached): browse (0) < product (1: view_item,
  select_item) < cart (2: add_to_cart) < checkout (3: begin_checkout,
  add_shipping_info, add_payment_info) < purchase (4).

## Rationale
Gives marketing both "which channel" and "how far in the funnel"; keeps raw facts
in the intermediate layer and pushes business rules (grouping, missing values) to
the marts, where they can change without rebuilding sessions.

## Consequences
- Journeys must be cut at the first purchase (sessions after it leak the label):
  rule applied in the marts.
- Materialized as a full table for now; switch to incremental if build cost grows.
- Sessions spanning midnight keep one row (grouped by session id, not by day).
