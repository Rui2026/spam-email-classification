"""Metrics, confusion matrix plots, and model comparison figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .metrics import compute_metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
    labels: tuple[str, str] = ("ham", "spam"),
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison_bar(
    rows: list[dict[str, float | str]],
    out_path: Path,
    metric_keys: tuple[str, ...] = ("accuracy", "precision", "recall", "f1"),
) -> None:
    df = pd.DataFrame(rows)
    model_col = "model"
    melted = df.melt(
        id_vars=[model_col],
        value_vars=list(metric_keys),
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Model comparison")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def predictions_to_csv(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": texts, "y_true": y_true, "y_pred": y_pred}).to_csv(out_path, index=False)


def aggregate_metric_row(model_name: str, m: dict[str, float]) -> dict[str, float | str]:
    """Normalize keys like test_accuracy -> accuracy for plotting."""
    row: dict[str, float | str] = {"model": model_name}
    for k, v in m.items():
        short = k.split("_", 1)[-1] if "_" in k else k
        if short in {"accuracy", "precision", "recall", "f1"}:
            row[short] = float(v)
    return row


def plot_train_curves(history: dict[str, list[float]], title: str, out_path: Path) -> None:
    """Plot training/validation loss curves from a history dict."""
    plt.figure(figsize=(6, 4))
    for label, values in history.items():
        if not values:
            continue
        plt.plot(range(1, len(values) + 1), values, marker="o", label=label)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_summary_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    """Save a tidy comparison table (one row per model) for reports."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def aggregate_runs(
    per_run_rows: list[list[dict[str, float | str]]],
    metric_keys: tuple[str, ...] = ("accuracy", "precision", "recall", "f1"),
) -> list[dict[str, float | str]]:
    """Take a list of per-run row lists; return mean ± std per model."""
    if not per_run_rows:
        return []
    flat = pd.concat([pd.DataFrame(r) for r in per_run_rows], ignore_index=True)
    out: list[dict[str, float | str]] = []
    for model, group in flat.groupby("model", sort=False):
        row: dict[str, float | str] = {"model": str(model), "n_runs": int(len(group))}
        for k in metric_keys:
            if k in group.columns:
                row[f"{k}_mean"] = float(group[k].mean())
                row[f"{k}_std"] = float(group[k].std(ddof=0))
        out.append(row)
    return out

