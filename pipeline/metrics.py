"""
Evaluation: accuracy, macro F1, raw confusion matrix, threshold sweep.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def _collect_probs_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    last_step_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model over loader, returning flattened (probs, labels) arrays.

    last_step_only=True  → use only the final timestep of each SEQ_LEN sequence
                           (for threshold-sweep style eval)
    last_step_only=False → use every (B * SEQ_LEN) timestep
                           (for the main F1/confusion-matrix eval)
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)  # (B, SEQ_LEN)
            probs = torch.sigmoid(logits)

            if last_step_only:
                p = probs[:, -1].cpu().numpy()
                y = y_batch[:, -1].cpu().numpy().astype(int)
            else:
                p = probs.cpu().numpy().flatten()
                y = y_batch.cpu().numpy().flatten()

            all_probs.append(p)
            all_labels.append(y)

    probs = np.concatenate(all_probs) if all_probs else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return probs, labels.astype(int)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, np.ndarray]:
    """
    Sweep thresholds from 0.10 to 0.85 and report the best macro-F1 result.
    Uses only the final timestep of each sequence (matches evaluate_threshold_sweep).

    Returns
    -------
    best_threshold, best_macro_f1, accuracy, apnea_recall, confusion_matrix
    """
    probs, labels = _collect_probs_and_labels(model, loader, device, last_step_only=True)

    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.10, 0.90, 0.05):
        preds = (probs >= thresh).astype(int)
        score = f1_score(labels, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    final_preds = (probs >= best_thresh).astype(int)
    acc = accuracy_score(labels, final_preds)
    cm = confusion_matrix(labels, final_preds)

    apnea_recall = float("nan")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        apnea_recall = tp / (tp + fn + 1e-8)

    print(f"Optimal Threshold : {best_thresh:.2f}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Macro F1          : {best_f1:.4f}")
    print("Confusion Matrix  :")
    print(cm)

    return best_thresh, best_f1, acc, apnea_recall, cm


def evaluate_threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Print macro F1 / sensitivity / precision across thresholds 0.10–0.85,
    using only the last timestep of each sequence (matches orchestrator usage).
    """
    probs, labels = _collect_probs_and_labels(model, loader, device, last_step_only=True)

    print(f"\n{'=' * 60}")
    print(f"  THRESHOLD SWEEP EVALUATION")
    print(f"{'=' * 60}")
    for t in np.arange(0.10, 0.90, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        sens = recall_score(labels, preds, pos_label=1, zero_division=0)
        prec = precision_score(labels, preds, pos_label=1, zero_division=0)
        print(f"t={t:.2f}  macro_F1={f1:.3f}  sens={sens:.3f}  prec={prec:.3f}")
    print(f"{'=' * 60}\n")
