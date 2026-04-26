"""Project-wide paths and hyperparameters."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Paths (relative to project root = parent of `src/`)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"

# Default CSV path (user can override via CLI)
DEFAULT_DATA_PATH: Path = DATA_RAW_DIR / "emails.csv"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "label"

# ---------------------------------------------------------------------------
# Preprocessing toggles
# ---------------------------------------------------------------------------
LOWERCASE: bool = True
REMOVE_PUNCTUATION: bool = True
REMOVE_DIGITS: bool = True
NORMALIZE_WHITESPACE: bool = True
REMOVE_STOPWORDS: bool = False
SIMPLE_TOKENIZE: bool = True  # whitespace tokenization after cleaning

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
STRATIFY: bool = True

# ---------------------------------------------------------------------------
# Reproducibility / logging
# ---------------------------------------------------------------------------
SEED: int = 42
LOG_LEVEL: str = "INFO"
DEFAULT_RUNS: int = 1

# ---------------------------------------------------------------------------
# Feature extraction (traditional ML + MLP on vectors)
# ---------------------------------------------------------------------------
FeatureKind = Literal["bow", "tfidf"]
DEFAULT_FEATURE_KIND: FeatureKind = "tfidf"

MAX_FEATURES: int = 5000
NGRAM_RANGE: tuple[int, int] = (1, 2)
MIN_DF: int = 1
MAX_DF: float = 0.95

# ---------------------------------------------------------------------------
# Classical ML
# ---------------------------------------------------------------------------
LOGISTIC_C: float = 1.0
LOGISTIC_MAX_ITER: int = 2000

KNN_N_NEIGHBORS: int = 5
KNN_METRIC: str = "cosine"

NAIVE_BAYES_ALPHA: float = 1.0

# ---------------------------------------------------------------------------
# Deep learning (PyTorch)
# ---------------------------------------------------------------------------
DL_BATCH_SIZE: int = 64
DL_NUM_EPOCHS: int = 10
DL_LEARNING_RATE: float = 1e-3
DL_WEIGHT_DECAY: float = 1e-4
DL_HIDDEN_DIM: int = 128
DL_DROPOUT: float = 0.3

# Class imbalance: weight the cross-entropy loss by inverse class frequency.
# Critical for spam datasets where one class dominates (e.g. ham ≈ 88%).
USE_CLASS_WEIGHT: bool = True

# Gradient clipping max-norm (None = disabled). Helps stabilise LSTM training.
GRAD_CLIP_NORM: float | None = 5.0

# LSTM
LSTM_EMBED_DIM: int = 128
LSTM_HIDDEN_DIM: int = 128
LSTM_NUM_LAYERS: int = 1
LSTM_MAX_LEN: int = 200
LSTM_VOCAB_MAX_SIZE: int = 8000
LSTM_BIDIRECTIONAL: bool = True

# MLP on vectorized text — cap dense width for memory
MLP_MAX_DENSE_FEATURES: int = 5000

# ---------------------------------------------------------------------------
# NLTK (optional stopwords)
# ---------------------------------------------------------------------------
NLTK_STOPWORDS_RESOURCE: tuple[str, str] = ("corpora", "stopwords")

