# 0017. Serving: daily batch scoring + API; results in BigQuery

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
The champion (ADR 0015) scores sessions that ended without a purchase for
re-engagement. Marketing reads results in Looker Studio from BigQuery
(ADR 0002). The template ships a FastAPI service and a Docker image verified by CI.

## Options considered
1. **Batch only** - matches daily re-engagement and Looker; the API and Docker of
   the template stay unused.
2. **Batch + API** - batch for marketing, an endpoint for real-time uses (e.g. a
   site deciding to show an incentive); reuses Docker and CI; more to maintain.
3. **API only** - does not fit Looker or list-based re-engagement.

## Decision
Option 2, in two increments; this ADR covers the batch part (the API follows).
- `uv run poe score --start D1 [--end D2]`: for each day, sessions that started
  that day without a purchase are scored with `models:/session-propensity@champion`,
  using only history up to that day and the training feature function. Output
  table `mta_results.propensity_scores`, partitioned by `score_date`, with
  propensity, daily decile (1 = top 10%), model name and version.
- Idempotent writes: each day replaces its partition (decorator + WRITE_TRUNCATE).
- `uv run poe publish-attribution`: `mta_results.attribution_states` and
  `attribution_channels` (replaced as a whole).
- Dataset ownership: dbt owns `mta_marts`, Python owns `mta_results`.

## Rationale
Batch is the natural serving mode for list-based activation; partitions make
re-runs and backfills safe; storing the model version with every score keeps each
score traceable to a registry version.

## Consequences
- The GA4 sample is static: "daily" scoring is a backfill over Jan 2021, the same
  mechanism a scheduler (phase 6) would run every morning on new data.
- Load jobs are free; storage of scores is small (well under the free tier).

## Update 2026-11-17: schema evolution
Adding monitoring columns (ADR 0021) made the partition load fail ("Cannot add
fields"). Partition loads now allow field addition and relaxation: new columns are
added, existing partitions get NULL. Removing or renaming a column still requires
a deliberate migration.
