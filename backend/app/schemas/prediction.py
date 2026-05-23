from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    claimed_author: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=100)


class StylometryFeatures(BaseModel):
    """Accepts the full 52-feature dictionary from the extractor."""
    model_config = ConfigDict(extra="allow")

    word_count: float
    sentence_count: float
    avg_word_length: float
    avg_sentence_length: float
    sentence_length_variance: float
    lexical_diversity: float
    punctuation_density: float
    comma_ratio: float
    period_ratio: float
    question_ratio: float
    exclamation_ratio: float
    semicolon_colon_ratio: float
    dash_ratio: float
    digit_char_ratio: float
    uppercase_ratio: float
    numeric_ratio: float
    stopword_ratio: float
    paragraph_count: float
    avg_paragraph_length: float
    short_word_ratio: float
    long_word_ratio: float
    suffix_nya_ratio: float
    suffix_lah_ratio: float
    suffix_kah_ratio: float


class TokenAttention(BaseModel):
    token: str
    attention: float


class StylometryImportance(BaseModel):
    feature: str
    importance: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    author_similarity: float
    ai_likelihood: float
    stylometry: StylometryFeatures
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    xai_tokens: list[TokenAttention] = Field(default_factory=list)
    xai_stylometry: list[StylometryImportance] = Field(default_factory=list)
    explanation: str
