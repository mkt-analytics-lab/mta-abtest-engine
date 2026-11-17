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

## Dependency updates

Dependabot (`.github/dependabot.yml`) opens monthly PRs for Python packages, GitHub
Actions and Docker base images; CI validates them before you merge. Update pre-commit
hooks with `uv run pre-commit autoupdate`.

## Updating from the template

```bash
uvx copier update --trust    # review with git diff, run `uv run poe check`, commit
```
