"""
Sanity check: run the full pipeline on a SINGLE patient for both train and test.

The goal is NOT a useful model — it's to confirm the entire pipeline
(loader → windows → dataset → model → train → eval) can overfit a single
patient. If loss doesn't drop to near-zero here, something is wired wrong
upstream and training on the full set will be wasted.

Usage examples:
    # Default — train + test on ucddb002 for 15 epochs
    python -m pipeline.sanity

    # Pick a different patient
    python -m pipeline.sanity --patient ucddb003

    # Different epoch / batch-size combination
    python -m pipeline.sanity --epochs 5 --batch-size 32

    # All flags at once
    python -m pipeline.sanity --patient ucddb005 --epochs 20 --batch-size 8

Programmatic use:
    from pipeline.sanity import run_sanity
    summary = run_sanity(patient_id="ucddb002", num_epochs=10)
    print(summary["passed"])
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config, dataset, loader, metrics, models, train as train_mod
from .dataset import ApneaSequenceDataset


def run_sanity(
    patient_id: str = "ucddb002",
    num_epochs: int = 15,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict:
    """
    Train + test on a single patient. Expect loss to collapse near 0 and
    accuracy to approach 1.0 within a handful of epochs.

    Returns a summary dict with key training metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.RANDOM_SEED)
    print(f"\n{'=' * 60}")
    print(f"  SANITY CHECK — Patient: {patient_id}  |  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    # ── Gate check first ────────────────────────────────────────────────────
    print(f"[1/5] Gate check on {patient_id} ...")
    loader.run_gate_check(patient_id)

    # ── Build a single-patient dataset ──────────────────────────────────────
    print(f"[2/5] Building single-patient dataset ...")
    wins, labs = loader.process_patient(patient_id)
    n_total = len(labs)
    n_apnea = int(labs.sum())
    print(f"  Windows: {n_total} total, {n_apnea} apnea "
          f"({100 * n_apnea / n_total:.1f}%)")
    print(f"  Window shape: {wins.shape}")

    if n_apnea < 2:
        print("  WARNING: very few apnea windows — sanity may be unreliable.")

    train_windows = [wins]
    train_labels = [labs]
    test_windows = [wins]
    test_labels = [labs]

    train_ds = ApneaSequenceDataset(train_windows, train_labels)
    test_ds = ApneaSequenceDataset(test_windows, test_labels)

    print(f"  Train sequences: {len(train_ds)}  (SEQ_LEN={config.SEQ_LEN})")
    print(f"  Test  sequences: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"\n[3/5] Building model ...")
    in_channels = wins.shape[1]
    model = models.ApneaCNNLSTM(in_channels=in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ApneaCNNLSTM — in_channels={in_channels}, params={n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── Train ───────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training {num_epochs} epochs on single patient ...")
    train_losses, val_losses = train_mod.fit(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        num_epochs=num_epochs,
    )

    # ── Evaluate ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating (same patient) ...")
    best_thresh, best_f1, acc, apnea_recall, cm = metrics.evaluate(
        model, test_loader, device
    )
    metrics.evaluate_threshold_sweep(model, test_loader, device)

    # ── Verdict ─────────────────────────────────────────────────────────────
    final_train_loss = train_losses[-1]
    print(f"\n{'=' * 60}")
    print(f"  SANITY VERDICT")
    print(f"{'=' * 60}")
    print(f"  Final train loss  : {final_train_loss:.4f}")
    print(f"  Final val   loss  : {val_losses[-1]:.4f}")
    print(f"  Best macro F1     : {best_f1:.4f}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Apnea recall      : {apnea_recall:.3f}")

    passed = (final_train_loss < 0.10) and (acc >= 0.90)
    print(
        f"  Result            : "
        f"{'✓ PASS — pipeline can overfit single patient' if passed else '✗ FAIL — something is broken upstream'}"
    )
    print(f"{'=' * 60}\n")

    return {
        "patient_id": patient_id,
        "n_windows": n_total,
        "n_apnea": n_apnea,
        "final_train_loss": final_train_loss,
        "final_val_loss": val_losses[-1],
        "best_threshold": best_thresh,
        "best_macro_f1": best_f1,
        "accuracy": acc,
        "apnea_recall": apnea_recall,
        "passed": bool(passed),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UCDDB pipeline sanity check — single-patient smoke test."
    )
    parser.add_argument(
        "--patient",
        default="ucddb002",
        help="Patient ID to use for sanity check (default: ucddb002).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    args = parser.parse_args()

    run_sanity(
        patient_id=args.patient,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
