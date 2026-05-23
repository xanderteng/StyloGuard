"""
Centralised configuration for model artefact paths and inference parameters.

All paths are relative to the backend root (the directory containing pyproject.toml).
At startup the ModelManager resolves them to absolute paths and validates that the
files exist.
"""

from __future__ import annotations

from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/
MODEL_DIR = BACKEND_ROOT / "model_artifacts"

# ── Model artefact files ─────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = MODEL_DIR / "feature_fusion_transformer.pt"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
SCALER_PATH = MODEL_DIR / "standard_scaler.joblib"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

# ── Inference settings ───────────────────────────────────────────────────────
MAX_TOKEN_LENGTH = 256
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p1"

# ── Label constants ──────────────────────────────────────────────────────────
LABEL_AI = "AI"
