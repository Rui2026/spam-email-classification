"""Filesystem helpers, label encoding, logging and seeding."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_CONFIGURED: bool = False
LOGGER_NAME: str = "spam"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure a single project-wide logger (idempotent)."""
    global _LOG_CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if not _LOG_CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
        _LOG_CONFIGURED = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # local import: keeps utils light when only ML is run

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_directories() -> None:
    for d in (
        config.DATA_RAW_DIR,
        config.DATA_PROCESSED_DIR,
        config.MODELS_DIR,
        config.FIGURES_DIR,
        config.METRICS_DIR,
        config.PREDICTIONS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Place a CSV with columns "
            f"'{config.TEXT_COLUMN}' and '{config.LABEL_COLUMN}' under data/raw/."
        )
    try:
        return pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def encode_labels(series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """Map labels to 0/1 (ham=0, spam=1). Supports ints and common string aliases."""
    if pd.api.types.is_numeric_dtype(series):
        arr = series.astype(np.int64).to_numpy()
        uniq = np.unique(arr)
        if len(uniq) != 2:
            raise ValueError(f"Expected exactly two label values; got {uniq}")
        if set(map(int, uniq.tolist())) == {0, 1}:
            return arr, {"label_0": 0, "label_1": 1}
        high, low = int(uniq.max()), int(uniq.min())
        arr01 = np.where(arr == high, 1, 0).astype(np.int64)
        return arr01, {str(low): 0, str(high): 1}

    s = series.astype(str).str.lower().str.strip()
    alias_positive = {"spam", "1", "true", "yes"}
    alias_negative = {"ham", "0", "false", "no"}
    if s.isin(alias_positive | alias_negative).all():
        mapped = s.map(lambda v: 1 if v in alias_positive else 0)
        return mapped.to_numpy(dtype=np.int64), {"ham": 0, "spam": 1}

    le = LabelEncoder()
    y = le.fit_transform(s)
    if len(le.classes_) != 2:
        raise ValueError("Only binary labels are supported (e.g. spam/ham).")
    return y.astype(np.int64), {str(le.classes_[i]): i for i in range(2)}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_processed_df(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_directories()
    out = path or (config.DATA_PROCESSED_DIR / "processed.csv")
    df.to_csv(out, index=False)
    return out

