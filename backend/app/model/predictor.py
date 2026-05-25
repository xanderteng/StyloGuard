"""
Authorship prediction — PyTorch inference pipeline.

Replaces the legacy statistical-distance approach with a single forward pass
through the ``FeatureFusionTransformer``.

Business-logic label mapping:
  • model predicts "AI"                     →  ``ai_generated``
  • model predicts a human ≠ claimed_author →  ``human_imposter``
  • model predicts claimed_author           →  ``authentic``
"""

from __future__ import annotations

from app.core.config import LABEL_AI
from app.model.model_manager import ModelManager
from app.model.stylometry_extractor import (
    extract_features,
    extract_features_dict,
)


def predict_authorship(claimed_author: str, text: str) -> dict:
    """
    Run the full prediction pipeline.

    1. Extract 52 stylometric features.
    2. Scale features with the fitted StandardScaler.
    3. Tokenize text with the IndoBERT tokenizer.
    4. Forward-pass through the FeatureFusionTransformer.
    5. Map the softmax output to the API response schema.
    """
    manager = ModelManager.get()

    if not manager.is_ready:
        return {
            "label": "unavailable",
            "confidence": 0.0,
            "author_similarity": 0.0,
            "ai_likelihood": 0.0,
            "stylometry": extract_features_dict(text),
            "class_probabilities": {},
            "explanation": (
                "The deep-learning model has not been loaded yet. "
                "Please ensure the model artefacts are present in "
                "backend/model_artifacts/ and restart the server."
            ),
        }

    # ── Feature extraction ───────────────────────────────────────────────
    stylometry_vector = extract_features(text)
    stylometry_dict = extract_features_dict(text)

    # ── Model inference ──────────────────────────────────────────────────
    predicted_label, prob_map, xai_tokens, xai_stylometry = manager.predict(text, stylometry_vector)
    confidence = prob_map.get(predicted_label, 0.0)
    ai_probability = prob_map.get(LABEL_AI, 0.0)

    # ── Business-logic label mapping ─────────────────────────────────────
    claimed_lower = claimed_author.strip().lower()

    if confidence < 0.7:
        return {
            "label": "authentic",
            "confidence": round(confidence, 4),
            "author_similarity": round(prob_map.get(claimed_author, _fuzzy_lookup(prob_map, claimed_lower)), 4),
            "ai_likelihood": round(ai_probability, 4),
            "stylometry": stylometry_dict,
            "class_probabilities": prob_map,
            "xai_tokens": xai_tokens,
            "xai_stylometry": xai_stylometry,
            "explanation": (
                "The model's prediction is inconclusive (confidence below 50%), "
                "so no imposter or AI signal is flagged."
            ),
        }

    if predicted_label == LABEL_AI:
        return {
            "label": "ai_generated",
            "confidence": round(confidence, 4),
            "author_similarity": round(prob_map.get(claimed_author, _fuzzy_lookup(prob_map, claimed_lower)), 4),
            "ai_likelihood": round(ai_probability, 4),
            "stylometry": stylometry_dict,
            "class_probabilities": prob_map,
            "xai_tokens": xai_tokens,
            "xai_stylometry": xai_stylometry,
            "explanation": (
                "The text shows strong heuristic and stylistic alignment "
                "with AI-generated content."
            ),
        }

    # Check if the predicted human author matches the claimed one
    if predicted_label.strip().lower() == claimed_lower:
        return {
            "label": "authentic",
            "confidence": round(confidence, 4),
            "author_similarity": round(confidence, 4),
            "ai_likelihood": round(ai_probability, 4),
            "stylometry": stylometry_dict,
            "class_probabilities": prob_map,
            "xai_tokens": xai_tokens,
            "xai_stylometry": xai_stylometry,
            "explanation": (
                "The submitted text is stylistically consistent with the "
                "claimed author's writing profile."
            ),
        }

    # Predicted a different human author
    return {
        "label": "human_imposter",
        "confidence": round(confidence, 4),
        "author_similarity": round(_fuzzy_lookup(prob_map, claimed_lower), 4),
        "ai_likelihood": round(ai_probability, 4),
        "stylometry": stylometry_dict,
        "class_probabilities": prob_map,
        "xai_tokens": xai_tokens,
        "xai_stylometry": xai_stylometry,
        "explanation": (
            f"This text is stylistically aligned with {predicted_label}, "
            f"not the claimed author {claimed_author.strip()}."
        ),
    }


def _fuzzy_lookup(prob_map: dict[str, float], target_lower: str) -> float:
    """Case-insensitive probability lookup."""
    for label, prob in prob_map.items():
        if label.strip().lower() == target_lower:
            return prob
    return 0.0
