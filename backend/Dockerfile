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
RUN uv pip install --system --no-cache gdown && \
    mkdir -p model_artifacts/tokenizer && \
    gdown 1ojrNng6Nqam7KmS5L-1VXJ3jocohZZJH -O model_artifacts/feature_fusion_transformer.pt && \
    gdown 16HxrdZnQTmPgH_6I0JT_NH8JRt1TRCiA -O model_artifacts/label_map.json && \
    gdown 133vTULQJeb2dvMTl4nJh5jEySnI-n0rf -O model_artifacts/standard_scaler.joblib && \
    gdown 1SP05x_WvokWirJcImJf5P3rz6W9KDvZB -O model_artifacts/tokenizer/tokenizer.json && \
    gdown 1S5GC3xSdtZJuxiGAyrUPX5Hh-mwL1ki- -O model_artifacts/tokenizer/tokenizer_config.json

EXPOSE 7860

CMD ["sh", "-c", "python -m scripts.seed_db && uvicorn app.main:app --host 0.0.0.0 --port 7860"]
