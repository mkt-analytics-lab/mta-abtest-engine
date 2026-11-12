# syntax=docker/dockerfile:1
# Serving image: only core + `serving` dependencies (no training/notebook libs).
# Multi-stage build: the builder installs dependencies with uv; the runtime
# stage copies only the virtual environment -> smaller, safer image.

# ---- uv binary (pinned version: reproducible builds) -----------------------
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

# ---- builder ---------------------------------------------------------------
FROM python:3.13-slim AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# 1) dependencies first (cached layer: only rebuilt when the lockfile changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-default-groups --group serving

# 2) then the project code
COPY src/ src/
RUN uv sync --locked --no-default-groups --group serving --no-editable

# ---- runtime ---------------------------------------------------------------
FROM python:3.13-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs/ configs/
COPY --chown=app:app models/ models/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    APP_MODEL_PATH=/app/models/model.joblib

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "mta_abtest_engine.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
