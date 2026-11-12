# MTA AB Test Engine

User-level multi-touch attribution and journey optimization

> Scaffolded from [mlops-project-template](https://github.com/damfello/mlops-project-template).
> Engineering standards: [`docs/mlops_blueprint.md`](docs/mlops_blueprint.md) ·
> Architecture decisions: [`docs/adr/`](docs/adr/)

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and git (Docker optional).

```bash
uv sync                      # create .venv with all dependency groups
uv run pre-commit install    # run quality checks on every commit
uv run poe check             # lint + tests
uv run poe pipeline          # ingest -> validate -> features -> train -> evaluate
uv run poe serve             # API at http://localhost:8000/docs
```

Run `uv run poe` to list every task.

## Project structure

```
├── configs/              # versioned pipeline parameters (validated with Pydantic)
├── data/                 # raw / interim / processed (git-ignored)
├── docs/
│   ├── adr/              # Architecture Decision Records
│   └── mlops_blueprint.md
├── models/               # trained model artifacts (git-ignored)
├── notebooks/            # exploration only; reusable logic goes to src/
├── reports/              # metrics and reports (git-ignored)
├── src/mta_abtest_engine/
│   ├── config.py         # settings (env) + project config (YAML)
│   ├── data/             # ingestion + data contracts
│   ├── features/         # feature engineering shared by training AND serving
│   ├── training/         # model training + MLflow tracking
│   ├── evaluation/       # metrics + quality gate
│   ├── serving/          # FastAPI app
│   └── pipelines/        # stage orchestration (CLI)
└── tests/
    ├── unit/             # pure functions, config, training-serving parity
    ├── data/             # data contract tests
    ├── model/            # behavioural tests (MFT / INV / DIR)
    └── integration/      # API + end-to-end pipeline
```

## Pipeline

| Stage      | Input                        | Output                              |
|------------|------------------------------|-------------------------------------|
| `ingest`   | source (synthetic example)   | `data/raw/dataset.csv`              |
| `validate` | raw data                     | `data/interim/validated.csv`        |
| `features` | validated data               | `data/processed/{train,test}.csv`   |
| `train`    | train split                  | `models/model.joblib` + MLflow run  |
| `evaluate` | model + test split           | `reports/metrics.json` + quality gate |

Run one stage: `uv run python -m mta_abtest_engine.pipelines.run --stage train`.
Inspect experiments: `uv run poe mlflow-ui` -> http://localhost:5000.

## Serving

```bash
uv run poe serve
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"records": [{"x1": 1.0, "x2": -0.5, "x3": 0.2, "x4": 0.0}]}'
```

Docker (serving image only contains core + `serving` dependencies):

```bash
uv run poe pipeline          # make sure models/model.joblib exists
uv run poe docker-build
uv run poe docker-run        # then open http://localhost:8000/health
```

## Dependency updates

Dependabot (`.github/dependabot.yml`) opens monthly PRs for Python packages, GitHub
Actions and Docker base images; CI validates them before you merge. Update pre-commit
hooks with `uv run pre-commit autoupdate`.

## Updating from the template

```bash
uvx copier update --trust    # review with git diff, run `uv run poe check`, commit
```
