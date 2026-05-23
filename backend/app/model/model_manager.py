"""
ModelManager — singleton that lazily loads all inference artefacts.

Loaded once during the FastAPI ``lifespan`` and shared across all requests.
The three artefacts are:

1. ``FeatureFusionTransformer`` weights  (``.pt`` state dict)
2. HuggingFace ``AutoTokenizer`` (saved directory)
3. Scikit-Learn ``StandardScaler`` (joblib-serialised)
4. ``label_map.json`` mapping class index → label string

If any file is missing the manager enters *graceful-degradation* mode and
``is_ready`` returns ``False``.  The ``/predict`` endpoint can check this
and return 503 until the model is available.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

from app.core.config import (
    INDOBERT_MODEL_NAME,
    LABEL_MAP_PATH,
    MAX_TOKEN_LENGTH,
    MODEL_WEIGHTS_PATH,
    SCALER_PATH,
    TOKENIZER_DIR,
)
from app.model.feature_fusion_transformer import FeatureFusionTransformer
from app.model.stylometry_extractor import FEATURE_NAMES, NUM_STYLOMETRIC_FEATURES

logger = logging.getLogger(__name__)


class ModelManager:
    """Holds the loaded model, tokenizer, scaler, and label map."""

    _instance: ModelManager | None = None

    def __init__(self) -> None:
        self.model: FeatureFusionTransformer | None = None
        self.tokenizer = None
        self.scaler = None
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ready = False

    # ── Singleton access ─────────────────────────────────────────────────
    @classmethod
    def get(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Loading ──────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load all artefacts from disk.  Call once at startup."""
        missing: list[str] = []
        for path, name in [
            (MODEL_WEIGHTS_PATH, "model weights"),
            (SCALER_PATH, "standard scaler"),
            (LABEL_MAP_PATH, "label map"),
        ]:
            if not Path(path).exists():
                missing.append(f"{name} ({path})")

        # Tokenizer may be a saved directory OR we can fall back to the
        # pretrained hub model.
        tokenizer_source = str(TOKENIZER_DIR) if Path(TOKENIZER_DIR).exists() else INDOBERT_MODEL_NAME

        if missing:
            logger.warning(
                "Model artefacts not found — running in degraded mode: %s",
                "; ".join(missing),
            )
            self._ready = False
            return

        # 1. Label map
        with open(LABEL_MAP_PATH, encoding="utf-8") as fh:
            raw_map: dict[str, str] = json.load(fh)
        self.id2label = {int(k): v for k, v in raw_map.items()}
        self.label2id = {v: int(k) for k, v in raw_map.items()}
        num_classes = len(self.id2label)

        # 2. Tokenizer
        logger.info("Loading tokenizer from %s", tokenizer_source)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        # 3. Scaler
        logger.info("Loading StandardScaler from %s", SCALER_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        # 4. Model
        logger.info("Loading FeatureFusionTransformer from %s", MODEL_WEIGHTS_PATH)
        self.model = FeatureFusionTransformer(
            model_name=INDOBERT_MODEL_NAME,
            num_classes=num_classes,
            num_sty_features=NUM_STYLOMETRIC_FEATURES,
        )
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=self.device, weights_only=True)

        # Handle DataParallel state dicts (keys prefixed with "module.")
        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key.removeprefix("module.")] = value
        self.model.load_state_dict(cleaned)

        self.model.to(self.device)
        self.model.eval()
        self._ready = True

        logger.info(
            "ModelManager ready — %d classes, device=%s",
            num_classes,
            self.device,
        )

    # ── Inference helpers ────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return self._ready

    def predict(
        self,
        text: str,
        stylometry_vector: list[float],
    ) -> tuple[str, dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
        """
        Run a single inference pass.

        Returns:
            (predicted_label, prob_map, xai_tokens, xai_stylometry)
        """
        assert self.model is not None and self.tokenizer is not None and self.scaler is not None

        # Scale stylometric features
        sty_array = np.array([stylometry_vector], dtype=np.float32)
        sty_scaled = self.scaler.transform(sty_array).astype(np.float32)
        
        # Enable gradient tracking for sty_tensor
        sty_tensor = torch.tensor(sty_scaled, device=self.device, requires_grad=True)

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 1. Run backbone without tracking gradients to save memory/speed
        with torch.no_grad():
            outputs = self.model.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            cls_vec = outputs.last_hidden_state[:, 0, :]
            attentions = outputs.attentions

        # 2. Run fusion dense branches WITH gradient tracking to compute feature importances
        sty_vec = self.model.sty_branch(sty_tensor)
        fused = torch.cat([cls_vec, sty_vec], dim=1)
        logits = self.model.classifier(fused)
        
        # Softmax probabilities (detach from graph)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        # Build probability map
        prob_map: dict[str, float] = {
            self.id2label[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }
        predicted_idx = int(probabilities.argmax())
        predicted_label = self.id2label[predicted_idx]

        # 3. Calculate gradients of the predicted class logit with respect to sty_tensor
        pred_logit = logits[0, predicted_idx]
        self.model.zero_grad()
        pred_logit.backward()

        # Extract gradient and calculate attribution (Gradient * Input)
        sty_grad = sty_tensor.grad.squeeze(0).cpu().numpy()
        attributions = sty_grad * sty_scaled[0]

        # Map back to feature names
        xai_sty_pairs = [
            {"feature": name, "importance": round(float(attr), 6)}
            for name, attr in zip(FEATURE_NAMES, attributions)
        ]
        # Sort descending by absolute attribution
        xai_sty_pairs.sort(key=lambda x: abs(x["importance"]), reverse=True)
        # Keep top 10
        xai_stylometry = xai_sty_pairs[:10]

        # ── xAI Extraction (Tokens) ──────────────────────────────────────────
        # attentions is a tuple of (batch_size, num_heads, seq_len, seq_len)
        # We take the last layer, first item in batch
        last_layer_attention = attentions[-1][0]
        # Average across all attention heads
        avg_attention = torch.mean(last_layer_attention, dim=0)
        # Attention from [CLS] token (index 0) to all other tokens
        cls_attention = avg_attention[0].cpu().numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        token_att_pairs = [
            (tok, float(att)) 
            for tok, att in zip(tokens, cls_attention) 
            if tok not in ["[CLS]", "[SEP]", "[PAD]"]
        ]
        # Sort descending by attention weight
        token_att_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top 10
        xai_tokens = [
            {"token": tok, "attention": round(att, 4)}
            for tok, att in token_att_pairs[:10]
        ]

        return predicted_label, prob_map, xai_tokens, xai_stylometry
