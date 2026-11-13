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
| Seeds: small reference tables versioned as CSV | `dbt/seeds/channel_groups.csv` | `mta_reference.channel_groups` in BigQuery |
| Business rules in marts, raw facts in intermediate | ADR 0007 | Change the seed, rebuild: only marts change |
| Data leakage: cut journeys at the first purchase | `fct_touchpoints.sql` | `dbt/tests/assert_journeys_end_at_first_purchase.sql` |
| Singular tests (custom SQL returning bad rows) vs generic tests | `dbt/tests/` | Break the cut on purpose: the test fails |
| Grain of a table (one row per what?) | `fct_touchpoints` vs `fct_user_journeys` | `unique` tests on their keys |
| `dbt show`: the row limit goes in `--limit`, never inside the SQL | `pyproject.toml` (dbt-show task) | A `limit` in the inline SQL fails with "Expected end of input" |
| Extract -> validate -> write (an invalid mart never reaches disk) | `data/extract.py` | `tests/unit/test_extract.py::test_invalid_mart_is_not_written` |
| Data contracts with pandera; defence in depth with dbt tests | `data/contracts.py` | `tests/data/test_mart_contracts.py` |
| Dependency injection: pass a fake client to test without the cloud | `extract_marts(..., client=)` | `FakeClient` in `tests/unit/test_extract.py` |
| Hand-written fixtures covering edge cases | `tests/fixtures/` | Each user in the README is one case |
| Right censoring: recent users have not had time to convert | ADR 0009 | `test_split_boundaries` in `tests/unit/test_journeys.py` |
| Fixed label horizon + dropping touches after it (no future info) | `features/journeys.py` | `test_short_horizon_drops_late_touches_and_conversions` |
| Temporal split by journey start (not random) | `assign_split` | Split counts per period on the real Parquet |
| Cross-field config validation (fail fast on impossible settings) | `JourneyConfig` in `config.py` | `test_test_period_shorter_than_horizon_fails` |
| Superseding a decision with a new ADR (never delete ADRs) | ADR 0002 status, ADR 0010 | Read both: context changed, decision changed |
| MTA vs MMM vs journey analytics: different questions | ADR 0010 | The portfolio split (project 1 vs 2) |
| Scoring point: predict from the past only (session without purchase) | `build_scoring_points` | `test_scoring_label_is_purchase_within_horizon` |
