"""PyTorch training for MLP (vectorized text) and LSTM (token sequences)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import ArrayDataset, SequenceDataset
from .dl_models import LSTMClassifier, MLPClassifier
from .metrics import compute_metrics
from .features import (
    FeatureKind,
    Vocabulary,
    build_padded_matrix,
    build_vectorizer,
    fit_transform_vectorizer,
    sparse_to_dense_float32,
    texts_to_token_lists,
)
from .utils import get_logger

History = dict[str, list[float]]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_train_val(
    x: np.ndarray, y: np.ndarray, val_fraction: float = 0.1, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = seed if seed is not None else config.RANDOM_STATE
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(yb)
    return total_loss / max(1, len(loader.dataset))


def _class_weights(y_train: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights, normalised so weighted-mean ≈ 1.

    For y in {0,1} with counts n0, n1: w_c = N / (K * n_c)  (sklearn 'balanced').
    """
    counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div-by-zero
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * len(yb)
            pred = torch.argmax(logits, dim=1)
            ys.append(yb.cpu().numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    denom = max(1, len(loader.dataset))
    return total_loss / denom, y_true, y_pred


def train_mlp(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_kind: FeatureKind,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    seed: int | None = None,
    models_dir: Path | None = None,
    use_class_weight: bool | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, History]:
    log = get_logger()
    epochs = epochs or config.DL_NUM_EPOCHS
    batch_size = batch_size or config.DL_BATCH_SIZE
    lr = lr if lr is not None else config.DL_LEARNING_RATE
    use_class_weight = config.USE_CLASS_WEIGHT if use_class_weight is None else use_class_weight
    models_dir = models_dir or config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    device = _device()

    vectorizer = build_vectorizer(feature_kind)
    x_tr_sp, x_te_sp = fit_transform_vectorizer(vectorizer, train_texts, test_texts)
    x_train = sparse_to_dense_float32(x_tr_sp)
    x_test = sparse_to_dense_float32(x_te_sp)

    x_tr, y_tr, x_val, y_val = _split_train_val(x_train, y_train, val_fraction=0.1, seed=seed)

    train_ds = ArrayDataset(x_tr, y_tr)
    val_ds = ArrayDataset(x_val, y_val)
    test_ds = ArrayDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = MLPClassifier(input_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config.DL_WEIGHT_DECAY,
    )
    if use_class_weight:
        cw = _class_weights(y_tr).to(device)
        log.info("[MLP] class weights: %s", cw.cpu().numpy().round(3).tolist())
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    history: History = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss = _train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=config.GRAD_CLIP_NORM
        )
        val_loss, _, _ = _evaluate_model(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        log.info("[MLP][epoch %d/%d] train_loss=%.4f val_loss=%.4f", epoch, epochs, tr_loss, val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "model_state": model.state_dict(),
        "input_dim": x_train.shape[1],
        "feature_kind": feature_kind,
    }
    torch.save(ckpt, models_dir / f"mlp_{feature_kind}.pt")

    _, y_true, y_pred = _evaluate_model(model, test_loader, criterion, device)
    return compute_metrics(y_true, y_pred, prefix="test_"), y_true, y_pred, history


def train_lstm(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    seed: int | None = None,
    models_dir: Path | None = None,
    use_class_weight: bool | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, History]:
    log = get_logger()
    epochs = epochs or config.DL_NUM_EPOCHS
    batch_size = batch_size or config.DL_BATCH_SIZE
    lr = lr if lr is not None else config.DL_LEARNING_RATE
    use_class_weight = config.USE_CLASS_WEIGHT if use_class_weight is None else use_class_weight
    models_dir = models_dir or config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    device = _device()

    vocab = Vocabulary()
    token_lists = texts_to_token_lists(train_texts)
    vocab.fit(token_lists)

    x_train = build_padded_matrix(train_texts, vocab, config.LSTM_MAX_LEN)
    x_test = build_padded_matrix(test_texts, vocab, config.LSTM_MAX_LEN)

    x_tr, y_tr, x_val, y_val = _split_train_val(x_train, y_train, val_fraction=0.1, seed=seed)

    train_ds = SequenceDataset(x_tr, y_tr)
    val_ds = SequenceDataset(x_val, y_val)
    test_ds = SequenceDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = LSTMClassifier(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config.DL_WEIGHT_DECAY,
    )
    if use_class_weight:
        cw = _class_weights(y_tr).to(device)
        log.info("[LSTM] class weights: %s", cw.cpu().numpy().round(3).tolist())
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    history: History = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss = _train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=config.GRAD_CLIP_NORM
        )
        val_loss, _, _ = _evaluate_model(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        log.info("[LSTM][epoch %d/%d] train_loss=%.4f val_loss=%.4f", epoch, epochs, tr_loss, val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "model_state": model.state_dict(),
        "vocab_word2idx": vocab.word2idx,
        "max_len": config.LSTM_MAX_LEN,
    }
    torch.save(ckpt, models_dir / "lstm.pt")

    _, y_true, y_pred = _evaluate_model(model, test_loader, criterion, device)
    return compute_metrics(y_true, y_pred, prefix="test_"), y_true, y_pred, history

