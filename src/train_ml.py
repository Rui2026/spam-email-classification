"""Train classical ML models on sparse BoW / TF-IDF features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from . import config
from .metrics import compute_metrics
from .features import FeatureKind, build_vectorizer, fit_transform_vectorizer
from .ml_models import build_ml_models
from .utils import get_logger


def train_ml_models(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_kind: FeatureKind,
    models_dir: Path | None = None,
) -> tuple[dict, dict[str, np.ndarray], object, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Fit vectorizer + each sklearn model.
    Returns (metrics_per_model, predictions_per_model, vectorizer, x_train_sp, x_test_sp).
    """
    models_dir = models_dir or config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = build_vectorizer(feature_kind)
    x_train, x_test = fit_transform_vectorizer(vectorizer, train_texts, test_texts)

    vec_path = models_dir / f"vectorizer_{feature_kind}.pkl"
    joblib.dump(vectorizer, vec_path)

    log = get_logger()
    estimators = build_ml_models()
    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, est in estimators.items():
        log.info("[ML] fitting %s on %s features (n=%d, dim=%d)", name, feature_kind, x_train.shape[0], x_train.shape[1])
        est.fit(x_train, y_train)
        model_path = models_dir / f"{name}_{feature_kind}.pkl"
        joblib.dump(est, model_path)

        preds = est.predict(x_test)
        predictions[name] = preds
        m = compute_metrics(y_test, preds, prefix="test_")
        metrics[name] = m
        log.info(
            "[ML] %s acc=%.3f f1=%.3f",
            name,
            m["test_accuracy"],
            m["test_f1"],
        )

    return metrics, predictions, vectorizer, x_train, x_test

