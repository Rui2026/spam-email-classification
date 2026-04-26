"""Text cleaning and optional tokenization for spam/ham classification."""

from __future__ import annotations

import re
import string
from typing import Iterable

import nltk
from nltk.corpus import stopwords

from . import config

_STOPWORDS_EN: frozenset[str] | None = None


def _ensure_stopwords() -> frozenset[str]:
    global _STOPWORDS_EN
    if _STOPWORDS_EN is not None:
        return _STOPWORDS_EN
    try:
        _STOPWORDS_EN = frozenset(stopwords.words("english"))
    except LookupError:
        nltk.download(config.NLTK_STOPWORDS_RESOURCE[1], quiet=True)
        _STOPWORDS_EN = frozenset(stopwords.words("english"))
    return _STOPWORDS_EN


def strip_punctuation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def strip_digits(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_simple(text: str) -> list[str]:
    return [t for t in text.split() if t]


def strip_stopwords(tokens: Iterable[str], stopwords_set: frozenset[str]) -> list[str]:
    return [t for t in tokens if t not in stopwords_set]


def preprocess_text(
    text: str,
    *,
    lowercase: bool = config.LOWERCASE,
    strip_punct: bool = config.REMOVE_PUNCTUATION,
    strip_nums: bool = config.REMOVE_DIGITS,
    normalize_whitespace_flag: bool = config.NORMALIZE_WHITESPACE,
    remove_stopwords: bool = config.REMOVE_STOPWORDS,
    simple_tokenize: bool = config.SIMPLE_TOKENIZE,
) -> str:
    """Return cleaned text (space-joined tokens if tokenization enabled)."""
    s = text if isinstance(text, str) else str(text)
    if lowercase:
        s = s.lower()
    if strip_punct:
        s = strip_punctuation(s)
    if strip_nums:
        s = strip_digits(s)
    if normalize_whitespace_flag:
        s = normalize_whitespace(s)

    if simple_tokenize:
        tokens = tokenize_simple(s)
        if remove_stopwords:
            sw = _ensure_stopwords()
            tokens = strip_stopwords(tokens, sw)
        return " ".join(tokens)
    return s


def preprocess_series(texts: Iterable[str]) -> list[str]:
    return [preprocess_text(t) for t in texts]

