# 0005. ELT architecture: dbt on BigQuery

- **Status:** Accepted
- **Date:** 2026-11-12

## Context
GA4 events are nested (`event_params` arrays) and sharded by day
(`events_YYYYMMDD`, ~4M events). The ETL must look like a real-work setup and the
results must live in BigQuery for Looker Studio. Measured size: a worst-case
`SELECT *` over all days scans 3.34 GiB.

## Options considered
ELT architecture:
1. **ELT in BigQuery with dbt** - layered SQL (staging -> intermediate -> marts),
   versioned and tested; the standard with GA4 + BigQuery.
2. **Python ETL** (queries + pandas/polars) - simplest; SQL hidden in strings, no
   lineage or SQL tests.
3. **Hybrid local** (extract to Parquet, dbt-duckdb) - cheap, testable offline;
   unrealistic for GA4 and results must be loaded back for Looker.

How dbt enters the project (the template's dbt component targets DuckDB):
a. Enable the template component and adapt it - recurring conflicts on `copier update`.
b. **Add dbt-bigquery manually in the project** - clean, domain-specific.
c. Add a `dbt_adapter` choice to the template (v0.4.0) - generalizes from one case.

## Decision
Option 1 with option b. Data stays in BigQuery; dbt builds the layers; Python only
downloads the training mart and uploads results.
- Staging as **views**; intermediate and marts as **tables**, **incremental** by
  event date where it pays off.
- Select only needed columns (no `SELECT *`).
- `maximum_bytes_billed` = **5 GiB per query** (cost barrier 3, ADR 0003), in the
  dbt profile and in the Python client.
- `use_dbt` stays `false` in the template answers.

## Rationale
Cost is not a differentiator: reading GA4 costs the same in all options. Estimated
full pipeline run 5-10 GiB -> roughly 100-200 runs per month inside the 1 TiB free
tier; the daily quota (0.1 TiB) caps the worst case. dbt is added manually because
generalizing into the template should wait for a second real use case.

## Consequences
- dbt data tests need BigQuery: run locally by the owner. CI runs Python tests on
  fixtures and checks that dbt models parse without a warehouse connection.
- If a second project needs dbt on BigQuery, move this setup into the template
  (option c).
