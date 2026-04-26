#!/usr/bin/env python3
"""Unified entry point: load data, preprocess, train ML/DL models, evaluate, save artifacts."""

from __future__ import annotations

import os
from pathlib import Path

# Matplotlib reads MPLCONFIGDIR at import time — set before importing plotting code.
_ROOT = Path(__file__).resolve().parent
_MPL = _ROOT / "outputs" / ".mplconfig"
_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
# Non-interactive backend — avoids GUI/Apple frameworks crashes on some macOS setups.
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split

from src import config
from src.evaluate import (
    aggregate_metric_row,
    aggregate_runs,
    plot_confusion_matrix,
    plot_model_comparison_bar,
    plot_train_curves,
    predictions_to_csv,
    save_summary_csv,
)
from src.preprocess import preprocess_series
from src.train_ml import train_ml_models
from src.utils import (
    encode_labels,
    ensure_directories,
    get_logger,
    load_csv,
    save_json,
    save_processed_df,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spam email classification — ML & DL baselines.")
    p.add_argument(
        "--mode",
        choices=["ml", "dl", "all"],
        default="all",
        help="Which pipeline to run: classical ML, deep learning, or both.",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_DATA_PATH,
        help="Path to CSV with text + label columns.",
    )
    p.add_argument(
        "--feature",
        choices=["bow", "tfidf"],
        default=config.DEFAULT_FEATURE_KIND,
        help="Bag-of-words or TF-IDF for sklearn models and MLP.",
    )
    p.add_argument(
        "--text-col",
        default=config.TEXT_COLUMN,
        help="Name of the text column in the CSV (default: 'text').",
    )
    p.add_argument(
        "--label-col",
        default=config.LABEL_COLUMN,
        help="Name of the label column in the CSV (default: 'label').",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding. Use 'latin-1' for the Kaggle SMS Spam Collection.",
    )
    p.add_argument("--seed", type=int, default=config.SEED, help="Base random seed.")
    p.add_argument(
        "--runs",
        type=int,
        default=config.DEFAULT_RUNS,
        help="Repeat training N times with seeds [seed, seed+1, ...] and report mean ± std.",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override DL epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override DL batch size.")
    p.add_argument("--lr", type=float, default=None, help="Override DL learning rate.")
    p.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable class-balanced loss for DL models (default: enabled).",
    )
    p.add_argument(
        "--tag",
        default=None,
        help=(
            "Optional experiment tag. Artifacts go to outputs/<tag>/... and "
            "models/<tag>/... so different runs don't overwrite each other. "
            "Use --tag auto for an automatic timestamp."
        ),
    )
    p.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _apply_tag(tag: str | None) -> str | None:
    """Re-route output / model directories to a tagged subfolder.

    Mutates the four module-level paths in `src.config` so that every downstream
    module (which reads them lazily inside functions) picks up the new layout.
    """
    if not tag:
        return None
    if tag == "auto":
        tag = time.strftime("%Y%m%d-%H%M%S")
    config.FIGURES_DIR = config.OUTPUTS_DIR / tag / "figures"
    config.METRICS_DIR = config.OUTPUTS_DIR / tag / "metrics"
    config.PREDICTIONS_DIR = config.OUTPUTS_DIR / tag / "predictions"
    config.MODELS_DIR = config.PROJECT_ROOT / "models" / tag
    return tag


def _split(cleaned: list[str], y: np.ndarray, seed: int) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    stratify = y if config.STRATIFY and len(np.unique(y)) > 1 else None
    return train_test_split(
        cleaned,
        y,
        test_size=config.TEST_SIZE,
        random_state=seed,
        stratify=stratify,
    )


def _run_once(
    cleaned: list[str],
    y: np.ndarray,
    args: argparse.Namespace,
    run_idx: int,
    seed: int,
) -> tuple[dict, list[dict], dict[str, list[float]] | None, dict[str, list[float]] | None]:
    """One full train/eval cycle. Returns (per_model_metrics, comparison_rows, mlp_history, lstm_history)."""
    log = get_logger()
    set_seed(seed)
    feature_kind = args.feature
    suffix = f"run{run_idx}_seed{seed}"

    log.info("=" * 60)
    log.info("Run %d/%d  seed=%d  feature=%s  mode=%s", run_idx + 1, args.runs, seed, feature_kind, args.mode)
    log.info("=" * 60)

    x_train, x_test, y_train, y_test = _split(cleaned, y, seed)
    log.info("Split: train=%d, test=%d", len(x_train), len(x_test))

    all_metrics: dict[str, dict] = {}
    rows: list[dict] = []
    mlp_history: dict[str, list[float]] | None = None
    lstm_history: dict[str, list[float]] | None = None

    if args.mode in ("ml", "all"):
        ml_metrics, ml_preds, _, _, _ = train_ml_models(
            x_train, x_test, y_train, y_test, feature_kind=feature_kind
        )
        for name, m in ml_metrics.items():
            key = f"ml_{name}_{feature_kind}"
            all_metrics[key] = m
            rows.append(aggregate_metric_row(key, m))
            plot_confusion_matrix(
                y_test,
                ml_preds[name],
                title=f"{name} ({feature_kind})  {suffix}",
                out_path=config.FIGURES_DIR / f"cm_ml_{name}_{feature_kind}_{suffix}.png",
            )
            predictions_to_csv(
                x_test,
                y_test,
                ml_preds[name],
                config.PREDICTIONS_DIR / f"pred_ml_{name}_{feature_kind}_{suffix}.csv",
            )

    if args.mode in ("dl", "all"):
        from src.train_dl import train_lstm, train_mlp

        mlp_metrics, y_true_mlp, y_pred_mlp, mlp_history = train_mlp(
            x_train,
            x_test,
            y_train,
            y_test,
            feature_kind=feature_kind,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=seed,
            use_class_weight=not args.no_class_weight,
        )
        key_mlp = f"mlp_{feature_kind}"
        all_metrics[key_mlp] = mlp_metrics
        rows.append(aggregate_metric_row(key_mlp, mlp_metrics))
        plot_confusion_matrix(
            y_true_mlp,
            y_pred_mlp,
            title=f"MLP ({feature_kind})  {suffix}",
            out_path=config.FIGURES_DIR / f"cm_mlp_{feature_kind}_{suffix}.png",
        )
        predictions_to_csv(
            x_test,
            y_true_mlp,
            y_pred_mlp,
            config.PREDICTIONS_DIR / f"pred_mlp_{feature_kind}_{suffix}.csv",
        )
        plot_train_curves(
            mlp_history,
            title=f"MLP training curves ({suffix})",
            out_path=config.FIGURES_DIR / f"curves_mlp_{feature_kind}_{suffix}.png",
        )

        lstm_metrics, y_true_lstm, y_pred_lstm, lstm_history = train_lstm(
            x_train,
            x_test,
            y_train,
            y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=seed,
            use_class_weight=not args.no_class_weight,
        )
        all_metrics["lstm"] = lstm_metrics
        rows.append(aggregate_metric_row("lstm", lstm_metrics))
        plot_confusion_matrix(
            y_true_lstm,
            y_pred_lstm,
            title=f"LSTM  {suffix}",
            out_path=config.FIGURES_DIR / f"cm_lstm_{suffix}.png",
        )
        predictions_to_csv(
            x_test,
            y_true_lstm,
            y_pred_lstm,
            config.PREDICTIONS_DIR / f"pred_lstm_{suffix}.csv",
        )
        plot_train_curves(
            lstm_history,
            title=f"LSTM training curves ({suffix})",
            out_path=config.FIGURES_DIR / f"curves_lstm_{suffix}.png",
        )

    save_json(config.METRICS_DIR / f"metrics_{args.mode}_{feature_kind}_{suffix}.json", all_metrics)
    return all_metrics, rows, mlp_history, lstm_history


def main() -> None:
    args = parse_args()
    log = setup_logging(args.log_level)
    resolved_tag = _apply_tag(args.tag)
    ensure_directories()
    if resolved_tag:
        log.info(
            "Tag='%s' → outputs/%s/  models/%s/",
            resolved_tag,
            resolved_tag,
            resolved_tag,
        )

    log.info("Loading dataset: %s", args.data)
    df = load_csv(args.data, encoding=args.encoding)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{args.text_col}' and '{args.label_col}'. "
            f"Got columns: {list(df.columns)}. Use --text-col / --label-col to override."
        )

    texts = df[args.text_col].astype(str).tolist()
    y, label_map = encode_labels(df[args.label_col])
    log.info("Loaded %d rows | label distribution: %s", len(y), dict(zip(*np.unique(y, return_counts=True))))

    cleaned = preprocess_series(texts)
    df_out = df.copy()
    df_out["clean_text"] = cleaned
    save_processed_df(df_out)

    save_json(
        config.METRICS_DIR / "label_mapping.json",
        {k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in label_map.items()},
    )

    per_run_rows: list[list[dict]] = []
    last_metrics: dict[str, dict] = {}
    started = time.time()
    for i in range(max(1, args.runs)):
        seed = args.seed + i
        all_metrics, rows, _mlp_h, _lstm_h = _run_once(cleaned, y, args, i, seed)
        per_run_rows.append(rows)
        last_metrics = all_metrics

    elapsed = time.time() - started
    log.info("Total wall time: %.1fs across %d run(s)", elapsed, args.runs)

    # Per-model bar comparison from the last run (visual, quick look)
    if per_run_rows and per_run_rows[-1]:
        plot_model_comparison_bar(
            per_run_rows[-1],
            config.FIGURES_DIR / f"model_comparison_{args.mode}_{args.feature}.png",
        )

    # Aggregated mean ± std table when multiple runs requested
    summary_rows = (
        aggregate_runs(per_run_rows) if args.runs > 1 else per_run_rows[-1] if per_run_rows else []
    )
    if summary_rows:
        save_summary_csv(
            summary_rows,
            config.METRICS_DIR / f"summary_{args.mode}_{args.feature}.csv",
        )
        save_json(
            config.METRICS_DIR / f"summary_{args.mode}_{args.feature}.json",
            {"runs": args.runs, "rows": summary_rows},
        )

    print(json.dumps(last_metrics, indent=2))
    if resolved_tag:
        log.info(
            "Done. Artifacts under outputs/%s/ and models/%s/.",
            resolved_tag,
            resolved_tag,
        )
    else:
        log.info("Done. Artifacts under outputs/ and models/.")


if __name__ == "__main__":
    main()
