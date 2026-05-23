from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    claimed_author: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=100)


class StylometryFeatures(BaseModel):
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
    quote_ratio: float
    semicolon_ratio: float
    uppercase_ratio: float
    numeric_ratio: float
    stopword_ratio: float
    paragraph_count: float
    avg_paragraph_length: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    author_similarity: float
    ai_likelihood: float
    known_articles: int
    stylometry: StylometryFeatures
    explanation: str
