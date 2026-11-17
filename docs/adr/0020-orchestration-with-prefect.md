# 0020. Orchestration with Prefect (local), via template update

- **Status:** Accepted
- **Date:** 2026-11-17

## Context
Every step is run by hand with `poe`. In production the daily path (dbt -> extract
-> score -> publish) and a periodic retraining must run unattended, with retries,
history and logs. The MLflow registry and the Parquet extracts live on the
owner's machine.

## Options considered
1. **Prefect, local** - flows in Python with retries, schedules and a UI; the
   template ships the component; runs only while the machine is on; the same
   flows can later move to Prefect Cloud or a server.
2. **Scheduled GitHub Actions** - no infra of our own; needs the registry and
   data in the cloud plus Workload Identity Federation.
3. **Linux cron** - simplest; no retries, history or UI.
4. **Cloud Run Jobs + Cloud Scheduler** - managed production; remote MLflow,
   artifacts in GCS, image registry: a project of its own, with cost.

## Decision
Option 1, enabled with `uvx copier update --data use_prefect=true` (also brings
template v0.3.1). `pipelines/flow.py`:
- `daily-scoring(score_date=yesterday, skip_dbt=False)`: dbt build -> extract ->
  score the day (publishes its partition).
- `weekly-retraining`: extract -> propensity (may register a `challenger`;
  promotion stays human, ADR 0015) -> attribution -> publish attribution.
- Warehouse tasks retry twice (60 s); training tasks do not retry (a failure there
  is a bug or a data problem; retrying hides it).
- `poe flow daily --date ...` / `poe flow retrain` run once; `poe flow-serve`
  schedules daily 06:10 and Monday 07:10 (Europe/Berlin) while the process runs.

## Template update notes
- `copier update` did not recreate the retired skeleton (ADR 0019): v0.3.0 ->
  v0.3.1 did not touch those files.
- Conflicts in `README.md` and `pyproject.toml` resolved by hand.
- The template's `flow.py` / `test_flow.py` call the skeleton (`pipelines.run`);
  replaced with project flows. Template backlog: make the Prefect component
  independent of the skeleton.

## Consequences
- The GA4 sample is static: `daily-scoring` is exercised with `--date` in Jan 2021;
  default "yesterday" scores nothing until real daily data flows in.
- A daily dbt build reads ~6-7 GB: ~200 GB/month, inside the 1 TiB free tier and
  far below the 0.1 TiB daily quota.
- Moving to option 4 later changes where flows run, not what they do.
