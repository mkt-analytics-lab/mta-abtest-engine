# 0022. Reporting layer for Looker Studio

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
Marketing reads results in Looker Studio (ADR 0002). Looker Studio is configured
by clicks: logic written there (custom SQL, calculated fields) is not versioned or
tested. Results live in `mta_marts` (dbt) and `mta_results` (Python).

## Options considered
1. **Connect Looker to the raw tables** and build logic in Looker - fastest; logic
   untested, duplicated across charts.
2. **Custom SQL inside Looker** - flexible; invisible to git and CI.
3. **dbt reporting views, one per chart** - logic versioned and tested in dbt;
   Looker only draws.

## Decision
Option 3: `dbt/models/reporting/` -> dataset `mta_reporting` (views):
`rpt_funnel_daily`, `rpt_funnel_steps` (long format for the funnel chart), `rpt_channel_credit`, `rpt_journey_states`,
`rpt_scores_daily`, `rpt_monitoring`. Python tables are declared as a dbt
source (`results`), so lineage covers Python outputs too. A build guide lives in
`docs/looker_dashboard.md` (4 pages: funnel, attribution, re-engagement, model
health).

## Consequences
- `dbt build` now needs the `mta_results` tables to exist (run scoring, monitoring
  and publishing first on a fresh project).
- The dashboard itself is not code: the guide is its specification.
