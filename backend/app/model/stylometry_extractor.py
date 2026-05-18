from __future__ import annotations

import math
import string

from app.model.preprocess import split_sentences, tokenize_words

FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "sentence_length_variance",
    "lexical_diversity",
    "punctuation_density",
    "comma_ratio",
    "period_ratio",
    "question_ratio",
    "exclamation_ratio",
    "quote_ratio",
    "semicolon_ratio",
    "uppercase_ratio",
    "numeric_ratio",
    "stopword_ratio",
    "paragraph_count",
    "avg_paragraph_length",
]

FUNCTION_WORDS = [
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "dengan",
    "untuk",
    "pada",
    "ini",
    "itu",
    "tidak",
    "akan",
    "juga",
    "karena",
    "sebagai",
]

FEATURE_NAMES.extend(f"fw_{word}" for word in FUNCTION_WORDS)

INDONESIAN_STOPWORDS = {
    "ada", "adalah", "agar", "akan", "atau", "bahwa", "bagi", "dalam", "dan",
    "dari", "dengan", "di", "dia", "ini", "itu", "jadi", "juga", "karena",
    "ke", "kemudian", "kepada", "lebih", "maka", "mereka", "namun", "pada",
    "para", "sebagai", "secara", "sehingga", "sementara", "serta", "sudah",
    "telah", "tersebut", "tidak", "untuk", "yang",
}


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def extract_features(text: str) -> dict[str, float]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    paragraph_lengths = [len(tokenize_words(paragraph)) for paragraph in paragraphs]
    chars = [char for char in text if not char.isspace()]
    punctuation = [char for char in text if char in string.punctuation]
    uppercase_chars = [char for char in text if char.isupper()]
    numeric_tokens = [word for word in words if word.isdigit()]
    stopwords = [word for word in words if word in INDONESIAN_STOPWORDS]

    word_count = len(words)
    sentence_count = len(sentences)
    char_count = len(chars)
    avg_sentence_length = _safe_divide(word_count, sentence_count)
    sentence_length_variance = _safe_divide(
        sum((length - avg_sentence_length) ** 2 for length in sentence_lengths),
        len(sentence_lengths),
    )

    features = {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_word_length": _safe_divide(sum(len(word) for word in words), word_count),
        "avg_sentence_length": avg_sentence_length,
        "sentence_length_variance": sentence_length_variance,
        "lexical_diversity": _safe_divide(len(set(words)), word_count),
        "punctuation_density": _safe_divide(len(punctuation), char_count),
        "comma_ratio": _safe_divide(text.count(","), char_count),
        "period_ratio": _safe_divide(text.count("."), char_count),
        "question_ratio": _safe_divide(text.count("?"), char_count),
        "exclamation_ratio": _safe_divide(text.count("!"), char_count),
        "quote_ratio": _safe_divide(sum(text.count(quote) for quote in ['"', "'", "“", "”", "‘", "’"]), char_count),
        "semicolon_ratio": _safe_divide(text.count(";") + text.count(":"), char_count),
        "uppercase_ratio": _safe_divide(len(uppercase_chars), char_count),
        "numeric_ratio": _safe_divide(len(numeric_tokens), word_count),
        "stopword_ratio": _safe_divide(len(stopwords), word_count),
        "paragraph_count": float(len(paragraphs) or 1),
        "avg_paragraph_length": _safe_divide(sum(paragraph_lengths), len(paragraph_lengths) or 1),
    }

    for function_word in FUNCTION_WORDS:
        features[f"fw_{function_word}"] = _safe_divide(words.count(function_word), word_count)

    return {name: round(features[name], 6) for name in FEATURE_NAMES}


def feature_vector(features: dict[str, float]) -> list[float]:
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        provided = sorted(features.keys())
        raise ValueError(f"Missing features for feature_vector: {missing}. Provided keys: {provided}")
    return [features[name] for name in FEATURE_NAMES]


def mean_features(feature_sets: list[dict[str, float]]) -> dict[str, float]:
    if not feature_sets:
        return {name: 0.0 for name in FEATURE_NAMES}

    return {
        name: sum(features[name] for features in feature_sets) / len(feature_sets)
        for name in FEATURE_NAMES
    }


def std_features(feature_sets: list[dict[str, float]], means: dict[str, float]) -> dict[str, float]:
    if len(feature_sets) < 2:
        return {name: 0.0 for name in FEATURE_NAMES}

    return {
        name: math.sqrt(
            sum((features[name] - means[name]) ** 2 for features in feature_sets)
            / (len(feature_sets) - 1)
        )
        for name in FEATURE_NAMES
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"cosine_similarity vector length mismatch: {len(left)} vs {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))
