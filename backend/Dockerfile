FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

CMD ["sh", "-c", "uv run --no-sync python -m scripts.seed_db && uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000"]
