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
| Running (cumulative) features: no future information by construction | `features/scoring.py` | `test_no_leakage_from_later_sessions` |
| Fixed feature order shared by training and scoring | `FEATURE_COLUMNS` | `test_columns_and_order_are_fixed` |
| Baselines first: base rate and a calibrated heuristic rule | `training/baselines.py` | `poe propensity` report: rule vs base rate |
| User-level bootstrap (correlated rows) and paired differences | `evaluation/metrics.py` | `test_paired_difference_detects_a_real_improvement` |
| Relative quality gate under distribution shift | ADR 0012 | Compare models on the same January test |
| MLflow runs with params, metrics + CIs and trace tags | `pipelines/propensity.py` | `uv run poe mlflow-ui`, experiment `session-propensity` |
| Means can mislead with sentinel values (-1 for "no previous session") | `hours_since_prev_session` | Compare medians or exclude -1 |
| Ties in ranking metrics: a metric must not depend on row order | `top_decile_lift` | `test_constant_prediction_has_lift_one_regardless_of_row_order` |
| No class re-weighting when probabilities matter (calibration) | `training/propensity_models.py` | Compare Brier / log-loss with `class_weight="balanced"` |
| Gate as code: decisions from paired difference intervals | `evaluation/gate.py` | `tests/unit/test_gate.py` (noise vs real improvement) |
| Never tune on the test set | ADR 0012 update | Tuning needs a validation period inside train |
| Hyperparameter tuning (not "fine-tuning"): temporal validation inside train | `training/tuning.py` | `test_validation_masks_never_touch_test_or_overlap` |
| Bayesian search (TPE), pruning with warm start, resumable studies | ADR 0013 | `test_study_finds_params_and_is_resumable`; `optuna.db` |
| Nested MLflow runs: one parent per study, one child per trial | `tuning.run` | MLflow UI, run `gbm-tuning` |
| Test touched once: tuned challenger gated against current model | `challenge_current_gbm` | Gate line `gbm_tuned vs gbm` |
| Behavioural tests: MFT / DIR / INV from domain knowledge | `evaluation/behavior.py` | `poe propensity` prints each check |
| Test your tests: a deliberately backwards model must be caught | `tests/model/test_propensity_behavior.py` | `test_backwards_model_is_caught` |
| A tuning result that does not pass the gate is still a result | ADR 0013 update | MLflow run `gbm_tuned`, `gate_passed=False` |
| When a behavioural check fails: test the assumption on raw data before changing model or check | ADR 0014 update | Cart vs checkout rates with a two-proportion interval |
| Clean test output: silence known third-party warnings narrowly, keep real ones | `tests/conftest.py` | `uv run poe test` shows no ResourceWarning |
| Registry, aliases `champion` / `challenger`, signature | `training/registry.py`, ADR 0015 | MLflow UI > Models > `session-propensity` |
| Human-approved promotion with audit tags and rollback | `poe promote --reason ...` | Version tags `promoted_by`, `previous_champion` |
| Champion re-evaluated on the same data (no retraining) | `PrefittedModel` | `champion` row in the propensity report |
| Safe model serialization (skops) and explicit trusted types | `register_challenger` | `MLmodel` file of the registered version |
| Explicit artifact location (never relative to the cwd) | `use_experiment`, `.env.example` | Experiment details in MLflow UI |
| A commit hash is not enough: record uncommitted changes (`git_dirty`) | `trace_tags` | Train with a modified file: tag is True |
| Absorbing Markov chain and removal effect | `attribution/markov.py` | `tests/unit/test_markov.py` hand-computed cases |
| Heuristic baselines and a GA4 last-click proxy | `attribution/heuristics.py` | `test_last_non_direct_skips_direct_unless_direct_only` |
| A contract for pluggable sources (models, GA4, MMM) | `ChannelCreditSchema` | `test_contract_rejects_unknown_channel_and_duplicates` |
| Bootstrap by resampling path counts (multinomial) | `markov_attribution` | Width of `conversions_ci_*` per channel |
| A modelling choice can create a fake finding (checkout states = disguised last touch) | ADR 0016 update | `role` column in `attribution_states.csv` |
| Batch scoring as of a date (no future history) | `serving/batch.py` | `test_scoring_never_uses_later_history` |
| Idempotent writes: partition decorator + WRITE_TRUNCATE | `serving/publish.py` | Re-run `poe score` for a day: row count unchanged |
| Dataset ownership (dbt -> marts, Python -> results) | ADR 0017 | BigQuery: `mta_marts` vs `mta_results` |
| Every score carries the model version | `propensity_scores.model_version` | Compare with the registry champion |
