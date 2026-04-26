"""Vectorization: BoW / TF-IDF for sklearn; vocab builder for LSTM."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config

FeatureKind = Literal["bow", "tfidf"]


def build_vectorizer(kind: FeatureKind) -> CountVectorizer | TfidfVectorizer:
    common = dict(
        max_features=min(config.MAX_FEATURES, config.MLP_MAX_DENSE_FEATURES),
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
    )
    if kind == "bow":
        return CountVectorizer(**common)
    return TfidfVectorizer(**common)


def fit_transform_vectorizer(
    vectorizer: CountVectorizer | TfidfVectorizer,
    train_texts: list[str],
    test_texts: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Fit on train; transform train and test. Returns sparse CSR matrices."""
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return x_train.tocsr(), x_test.tocsr()


def sparse_to_dense_float32(mat: sparse.csr_matrix) -> np.ndarray:
    """Dense array for PyTorch MLP (vocab width already capped in vectorizer)."""
    return mat.toarray().astype(np.float32)


class Vocabulary:
    """Simple word -> index mapping with PAD/UNK tokens for LSTM."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_size: int = config.LSTM_VOCAB_MAX_SIZE) -> None:
        self.max_size = max_size
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

    def fit(self, tokenized_lines: list[list[str]]) -> None:
        freq: dict[str, int] = {}
        for tokens in tokenized_lines:
            for w in tokens:
                freq[w] = freq.get(w, 0) + 1
        specials = [self.PAD_TOKEN, self.UNK_TOKEN]
        most_common = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        words = specials + [w for w, _ in most_common[: self.max_size - len(specials)]]
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.word2idx.get(self.UNK_TOKEN, 1)
        return [self.word2idx.get(t, unk) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)


def texts_to_token_lists(texts: list[str]) -> list[list[str]]:
    return [t.split() for t in texts]


def pad_sequences(seqs: list[list[int]], max_len: int, pad_value: int = 0) -> np.ndarray:
    out = np.full((len(seqs), max_len), pad_value, dtype=np.int64)
    for i, s in enumerate(seqs):
        trunc = s[:max_len]
        if trunc:
            out[i, : len(trunc)] = trunc
    return out


def build_padded_matrix(
    texts: list[str], vocab: Vocabulary, max_len: int = config.LSTM_MAX_LEN
) -> np.ndarray:
    token_lists = texts_to_token_lists(texts)
    encoded = [vocab.encode(toks) for toks in token_lists]
    pad_id = vocab.word2idx.get(vocab.PAD_TOKEN, 0)
    return pad_sequences(encoded, max_len, pad_value=pad_id)

