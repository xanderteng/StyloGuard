from __future__ import annotations

import threading

from sqlalchemy.orm import Session

from app.db.models import Article
from app.model.preprocess import normalize_text
from app.model.stylometry_extractor import (
    FEATURE_NAMES,
    extract_features,
    mean_features,
    std_features,
)

MIN_AUTHOR_ARTICLES = 3
MAX_COMPETING_AUTHORS = 250
_AUTHOR_INDEX: dict | None = None
_AUTHOR_INDEX_COUNT = -1
_AUTHOR_INDEX_LOCK = threading.Lock()

FEATURE_WEIGHTS = {
    "word_count": 0.0,
    "sentence_count": 0.0,
    "paragraph_count": 0.0,
    "avg_word_length": 1.4,
    "avg_sentence_length": 1.1,
    "sentence_length_variance": 0.7,
    "lexical_diversity": 1.2,
    "punctuation_density": 1.4,
    "comma_ratio": 1.1,
    "period_ratio": 0.9,
    "question_ratio": 0.5,
    "exclamation_ratio": 0.5,
    "quote_ratio": 0.9,
    "semicolon_ratio": 0.8,
    "uppercase_ratio": 0.7,
    "numeric_ratio": 0.5,
    "stopword_ratio": 1.0,
    "avg_paragraph_length": 0.6,
}

for feature_name in FEATURE_NAMES:
    if feature_name.startswith("fw_"):
        FEATURE_WEIGHTS[feature_name] = 1.3

FEATURE_FLOORS = {
    "avg_word_length": 0.35,
    "avg_sentence_length": 4.0,
    "sentence_length_variance": 35.0,
    "lexical_diversity": 0.08,
    "punctuation_density": 0.012,
    "comma_ratio": 0.006,
    "period_ratio": 0.006,
    "question_ratio": 0.002,
    "exclamation_ratio": 0.002,
    "quote_ratio": 0.004,
    "semicolon_ratio": 0.003,
    "uppercase_ratio": 0.015,
    "numeric_ratio": 0.02,
    "stopword_ratio": 0.045,
    "avg_paragraph_length": 60.0,
}

for feature_name in FEATURE_NAMES:
    if feature_name.startswith("fw_"):
        FEATURE_FLOORS[feature_name] = 0.01


def _build_profile_from_features(article_features: list[dict[str, float]]) -> dict:
    means = mean_features(article_features)
    stds = std_features(article_features, means)
    return {"means": means, "stds": stds, "article_count": len(article_features)}


def _get_author_index(db: Session) -> dict:
    global _AUTHOR_INDEX, _AUTHOR_INDEX_COUNT

    with _AUTHOR_INDEX_LOCK:
        article_count = db.query(Article).count()
        if _AUTHOR_INDEX is not None and _AUTHOR_INDEX_COUNT == article_count:
            return _AUTHOR_INDEX

        grouped: dict[str, list[dict[str, float]]] = {}
        exact_matches: dict[str, set[str]] = {}
        for article in db.query(Article).all():
            if not article.author or not article.text:
                continue
            grouped.setdefault(article.author, []).append(extract_features(article.text))
            exact_matches.setdefault(normalize_text(article.text), set()).add(article.author)

        next_index = {
            "authors": {
                author: {
                    "article_count": len(article_features),
                    "article_features": article_features,
                    "profile": _build_profile_from_features(article_features),
                }
                for author, article_features in grouped.items()
            },
            "exact_matches": exact_matches,
        }
        _AUTHOR_INDEX = next_index
        _AUTHOR_INDEX_COUNT = article_count
        return next_index


def _feature_scale(feature_name: str, mean: float, std: float) -> float:
    floor = FEATURE_FLOORS.get(feature_name, 0.01)
    relative_scale = abs(mean) * 0.35
    author_scale = std * 1.25
    return max(floor, relative_scale, author_scale)


def _profile_distance(features: dict[str, float], profile: dict) -> float:
    weighted_total = 0.0
    weight_sum = 0.0

    for feature_name in FEATURE_NAMES:
        weight = FEATURE_WEIGHTS.get(feature_name, 1.0)
        if weight <= 0:
            continue

        mean = profile["means"][feature_name]
        std = profile["stds"][feature_name]
        scale = _feature_scale(feature_name, mean, std)
        normalized_difference = min(abs(features[feature_name] - mean) / scale, 3.0)
        weighted_total += weight * normalized_difference
        weight_sum += weight

    return weighted_total / weight_sum if weight_sum else 3.0


def _single_text_distance(left: dict[str, float], right: dict[str, float]) -> float:
    profile = {
        "means": right,
        "stds": {feature_name: 0.0 for feature_name in FEATURE_NAMES},
        "article_count": 1,
    }
    return _profile_distance(left, profile)


def _author_distance(features: dict[str, float], author_entry: dict) -> float:
    profile_distance = _profile_distance(features, author_entry["profile"])
    nearest_article_distance = min(
        _single_text_distance(features, article_features)
        for article_features in author_entry["article_features"]
    )
    return profile_distance * 0.35 + nearest_article_distance * 0.65


def _distance_to_similarity(distance: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - distance / 2.2)), 4)


def estimate_ai_likelihood(features: dict[str, float]) -> float:
    lexical_signal = max(0.0, (0.45 - features["lexical_diversity"]) / 0.45)
    punctuation_signal = max(0.0, (0.035 - features["punctuation_density"]) / 0.035)
    sentence_signal = max(0.0, (features["avg_sentence_length"] - 28) / 28)
    stopword_signal = max(0.0, (features["stopword_ratio"] - 0.34) / 0.34)
    score = (
        lexical_signal * 0.35
        + punctuation_signal * 0.2
        + sentence_signal * 0.25
        + stopword_signal * 0.2
    )
    return round(max(0.0, min(score, 1.0)), 4)


def predict_authorship(db: Session, claimed_author: str, text: str) -> dict:
    submitted_features = extract_features(text)
    normalized_input = normalize_text(text)
    author_index = _get_author_index(db)
    author_entries = author_index["authors"]
    exact_matches = author_index["exact_matches"]
    claimed_author_key = next(
        (author for author in author_entries if author.lower() == claimed_author.strip().lower()),
        None,
    )
    claimed_entry = author_entries.get(claimed_author_key) if claimed_author_key else None
    known_article_count = claimed_entry["article_count"] if claimed_entry else 0

    ai_likelihood = estimate_ai_likelihood(submitted_features)

    matching_authors = sorted(exact_matches.get(normalized_input, set()))
    if matching_authors:
        claimed_matches = [
            author
            for author in matching_authors
            if author.lower() == claimed_author.strip().lower()
        ]

        if claimed_matches:
            return {
                "label": "authentic",
                "confidence": 0.99,
                "author_similarity": 1.0,
                "ai_likelihood": ai_likelihood,
                "known_articles": known_article_count,
                "stylometry": submitted_features,
                "explanation": "This exact article is present in the indexed dataset under the claimed author.",
            }

        return {
            "label": "human_imposter",
            "confidence": 0.99,
            "author_similarity": 0.0,
            "ai_likelihood": ai_likelihood,
            "known_articles": known_article_count,
            "stylometry": submitted_features,
            "explanation": f"This exact article is indexed under {', '.join(matching_authors[:3])}, not {claimed_author.strip()}.",
        }

    if known_article_count < MIN_AUTHOR_ARTICLES:
        label = "unknown"
        confidence = 0.35
        similarity = 0.0
        explanation = (
            f"Only {known_article_count} known article(s) were found for this author. "
            f"At least {MIN_AUTHOR_ARTICLES} are needed for a reliable style profile."
        )
    else:
        claimed_distance = _author_distance(submitted_features, claimed_entry)
        similarity = _distance_to_similarity(claimed_distance)

        competitor_scores = []
        for author, author_entry in author_entries.items():
            if author == claimed_author_key or author_entry["article_count"] < MIN_AUTHOR_ARTICLES:
                continue
            distance = _author_distance(submitted_features, author_entry)
            competitor_scores.append((distance, author, author_entry["article_count"]))

        competitor_scores.sort(key=lambda item: item[0])
        competitor_scores = competitor_scores[:MAX_COMPETING_AUTHORS]
        best_competitor = competitor_scores[0] if competitor_scores else None
        competitor_margin = (best_competitor[0] - claimed_distance) if best_competitor else 1.0

        if ai_likelihood >= 0.68 and similarity < 0.72:
            label = "ai_generated"
            confidence = max(ai_likelihood, 1 - similarity)
            explanation = "The text diverges from the claimed author's profile and shows heuristic AI-like regularity."
        elif best_competitor and (similarity < 0.78 or competitor_margin < -0.16):
            label = "human_imposter"
            confidence = min(0.95, max(0.55, 1 - similarity, abs(competitor_margin) / 0.55))
            explanation = (
                "The text falls outside the claimed author's calibrated style profile "
                "relative to other indexed authors."
            )
        elif similarity >= 0.78 and competitor_margin >= -0.2:
            label = "authentic"
            confidence = similarity
            explanation = "The submitted text is close to the claimed author's historical stylometric profile."
        elif similarity >= 0.64:
            label = "uncertain"
            confidence = similarity
            explanation = "The text partly matches the author profile, but the evidence is not strong enough."
        else:
            label = "human_imposter"
            confidence = 1 - similarity
            explanation = "The submitted text differs substantially from the claimed author's historical style profile."

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "author_similarity": round(float(similarity), 4),
        "ai_likelihood": ai_likelihood,
        "known_articles": known_article_count,
        "stylometry": submitted_features,
        "explanation": explanation,
    }
