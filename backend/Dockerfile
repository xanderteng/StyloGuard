FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./

# 1. Force CPU-only PyTorch >=2.6 and NumPy < 2 to satisfy transformers (CVE-2025-32434)
# Wrapped packages in quotes to prevent shell redirection/comparison interpretation
RUN uv pip install --system --no-cache "torch>=2.6" "numpy<2" --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the dependencies from pyproject.toml
RUN uv pip install --system --no-cache -r pyproject.toml --extra-index-url https://pypi.org/simple

COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY model_artifacts ./model_artifacts

EXPOSE 8000

CMD ["sh", "-c", "python -m scripts.seed_db && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
