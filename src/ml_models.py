"""Classical ML model constructors for spam classification."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from . import config


def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        max_iter=config.LOGISTIC_MAX_ITER,
        C=config.LOGISTIC_C,
        class_weight="balanced",
        solver="saga",
    )


def make_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=config.KNN_N_NEIGHBORS,
        metric=config.KNN_METRIC,
        algorithm="brute",
        n_jobs=-1,
    )


def make_naive_bayes() -> MultinomialNB:
    return MultinomialNB(alpha=config.NAIVE_BAYES_ALPHA)


def build_ml_models() -> dict[str, object]:
    """Return name -> unfitted estimator (sparse-friendly)."""
    return {
        "logistic_regression": make_logistic_regression(),
        "knn": make_knn(),
        "naive_bayes": make_naive_bayes(),
    }

