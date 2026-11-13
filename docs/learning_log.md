# Learning log

Concepts introduced in each increment, to review step by step once the project
works end to end. Each entry: where to look, the idea, how to verify it yourself.

## Phase 0 - Framing
- `docs/adr/0002-*`: why MTA first, why GA4 sample, why log-loss as primary metric.
- Attribution is correlational; causal claims need experiments (A/B module).

## Phase 1 - Data
| Concept | Where | Verify yourself |
|---|---|---|
| Cost guards (budget, quota, `maximum_bytes_billed`) | ADR 0003, `configs/config.yaml`, `dbt/profiles.yml` | BigQuery Job history: bytes per query |
| ADC authentication, no keys in the repo | ADR 0004, `.env.example` | `gcloud auth application-default print-access-token` works; no key file in git |
| ELT with dbt: staging -> intermediate -> marts | ADR 0005, `dbt/` | `uv run poe dbt-build` creates `mta_staging`, `mta_intermediate` |
| `source()` / `ref()` are resolved by dbt | `dbt/target/compiled/...` | Compare a model with its compiled SQL |
| `parse_date`, `timestamp_micros`: change types, not formats | `stg_ga4__events.sql` | `dbt show --inline` raw vs typed columns |
| `unnest(event_params)`: nested list -> rows -> one value | `stg_ga4__events.sql` | Inline query with `_TABLE_SUFFIX` on one day |
| `dbt parse` (no warehouse) vs `dbt build` (runs on BigQuery) | CI workflow, `pyproject.toml` | Break a column name: parse passes, build fails |
| CI checks code; orchestration runs the pipeline | `.github/workflows/ci.yml` | GitHub Actions tab |
| DAG: tests skip when their model fails | `dbt build` output | `SKIP=n` after a model error |
| Explore before designing (analyses + `dbt show`) | `dbt/analyses/exploration/` | Q4: medium was not on session_start |
| Session = touchpoint; grain and keys | ADR 0006, `int_ga4__sessions.sql` | `unique` test on `session_key` |
