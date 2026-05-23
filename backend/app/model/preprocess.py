import re

WORD_RE = re.compile(r"[\w]+", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.UNICODE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(normalize_text(text))]


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.findall(normalize_text(text))]
    return [sentence for sentence in sentences if sentence]
