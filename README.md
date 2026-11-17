# MTA AB Test Engine

User-level multi-touch attribution and journey optimization

> Scaffolded from [mlops-project-template](https://github.com/damfello/mlops-project-template).
> Engineering standards: [`docs/mlops_blueprint.md`](docs/mlops_blueprint.md) ·
> Architecture decisions: [`docs/adr/`](docs/adr/)

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/), git, a Google Cloud project with
BigQuery (see `docs/adr/0003` and `0004`) and Docker (optional).

```bash
uv sync                          # create .venv with all dependency groups
uv run pre-commit install        # quality checks on every commit
cp .env.example .env             # set APP_GCP_PROJECT_ID
gcloud auth application-default login
uv run poe check                 # lint + tests + dbt parse (same as CI)
```

Run `uv run poe` to list every task.

## Project structure

```
├── configs/              # versioned pipeline parameters (validated with Pydantic)
├── data/                 # raw / interim / processed (git-ignored)
├── dbt/                  # ELT on BigQuery: staging -> intermediate -> marts
├── docs/
│   ├── adr/              # Architecture Decision Records
│   ├── learning_log.md   # concepts per increment and how to verify them
│   └── mlops_blueprint.md
├── models/               # exported serving model (git-ignored, baked into the image)
├── notebooks/            # exploration only; reusable logic goes to src/
├── reports/              # metrics and reports (git-ignored)
├── src/mta_abtest_engine/
│   ├── config.py         # settings (env) + project config (YAML)
│   ├── data/             # mart extraction + data contracts
│   ├── features/         # journeys, labels, split, scoring features (train = serve)
│   ├── training/         # baselines, models, tuning, registry
│   ├── evaluation/       # metrics, bootstrap, gate, behavioural checks
│   ├── attribution/      # Markov journey model + heuristic baselines
│   ├── serving/          # batch scoring, BigQuery publishing, FastAPI app
│   ├── monitoring/       # Evidently drift, PSI, calibration checks
│   └── pipelines/        # propensity, attribution and Prefect flows
└── tests/
    ├── unit/             # pure functions, config, parity
    ├── data/             # data contracts of the journey marts
    ├── model/            # behavioural tests (MFT / INV / DIR)
    └── integration/      # pipelines, registry, API, flows
```

## Pipeline

| Step | Command | Output |
|---|---|---|
| ELT on BigQuery | `uv run poe dbt-build` | `mta_staging`, `mta_intermediate`, `mta_marts` |
| Extract marts | `uv run poe extract` | `data/raw/*.parquet` (validated) |
| Propensity models | `uv run poe propensity` | MLflow runs, gate, `challenger` in the registry |
| Tuning (optional) | `uv run poe tune` | Optuna study, `gbm_tuned` gated vs `gbm` |
| Promote | `uv run poe promote --reason "..."` | `champion` alias with audit tags |
| Attribution | `uv run poe attribution` | Markov + baselines, `reports/attribution_*.csv` |
| Batch scoring | `uv run poe score --start D1 --end D2` | `mta_results.propensity_scores` |
| Publish attribution | `uv run poe publish-attribution` | `mta_results.attribution_*` |
| Monitoring | `uv run poe monitor --start D1 --end D2` | `mta_results.monitoring_daily`, `reports/monitoring/*.html` |
| Orchestration | `uv run poe flow daily --date D` / `poe flow-serve` | Prefect runs (daily + weekly) |

Inspect experiments and the registry: `uv run poe mlflow-ui` -> http://localhost:5000.

## Serving

```bash
uv run poe export-champion   # registry champion -> models/
uv run poe docker-build      # image mta-abtest-engine:model-v<N>
uv run poe docker-run        # http://localhost:8000/docs
curl -X POST localhost:8000/score -H "Content-Type: application/json" \
  -d '{"sessions": [{"session_start_ts": "2021-01-10T18:30:00Z",
                     "channel_group": "Referral", "funnel_stage": "cart"}]}'
```

## Optional components enabled
- **Prefect** – orchestration with retries and run history: `uv run poe flow`.
- **Evidently** – data drift report: `uv run poe drift` -> `reports/drift_report.html`.

## Dependency updates

Dependabot (`.github/dependabot.yml`) opens monthly PRs for Python packages, GitHub
Actions and Docker base images; CI validates them before you merge. Update pre-commit
hooks with `uv run pre-commit autoupdate`.

## Updating from the template

```bash
uvx copier update --trust    # review with git diff, run `uv run poe check`, commit
```
