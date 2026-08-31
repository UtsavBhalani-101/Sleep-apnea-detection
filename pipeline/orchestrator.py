"""
End-to-end orchestrator: builds datasets, instantiates the model, trains,
validates, and evaluates on the held-out test set.

Run with:
    python -m pipeline.orchestrator
or:
    from pipeline.orchestrator import run

When `WANDB_DISABLED` is not set to a truthy value, every run is uploaded to
Weights & Biases (project name from `WANDB_PROJECT` env var, default
`sleep-apnea-experiments`). The run is auto-named `EXP_<next>` based on how
many `EXP_*.md` files are in the experiments/ directory.
"""

from __future__ import annotations

import os
import socket
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import config, dataset, loader, metrics, models, train as train_mod
from .wandb_log import WandbLogger


def _build_sampler(train_ds: dataset.ApneaSequenceDataset) -> WeightedRandomSampler:
    """
    A sequence is 'apnea' if any of its SEQ_LEN labels is 1.
    Apnea sequences get weight (n_normal / n_apnea) to balance the batches.
    """
    seq_is_apnea = (train_ds.y.float().mean(dim=1) > 0).float()
    n_apnea = int(seq_is_apnea.sum().item())
    n_normal = len(seq_is_apnea) - n_apnea

    if n_apnea == 0:
        # Degenerate fallback — uniform weights
        weights = torch.ones(len(seq_is_apnea), dtype=torch.float32)
    else:
        weights = torch.where(
            seq_is_apnea == 1,
            torch.tensor(float(n_normal) / n_apnea),
            torch.tensor(1.0),
        )

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def _next_exp_id() -> str:
    """Return EXP_NNN where NNN is one higher than the largest existing exp file."""
    exp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
    if not os.path.isdir(exp_dir):
        return "EXP_001"
    existing = []
    for name in os.listdir(exp_dir):
        if name.startswith("EXP_") and name.endswith(".md"):
            digits = "".join(ch for ch in name if ch.isdigit())
            if digits:
                existing.append(int(digits))
    nxt = (max(existing) + 1) if existing else 1
    return f"EXP_{nxt:03d}"


def run(sanity: bool = False, exp_id: str | None = None, notes: str | None = None) -> None:
    """
    Run the full pipeline. Set sanity=True to use the single sanity-check
    patient for both train and test (useful for overfitting smoke tests).

    Parameters
    ----------
    sanity : bool
        If True, train + test on the sanity-check patient.
    exp_id : str | None
        Override the auto-generated EXP_NNN id (e.g. ``"EXP_011"``).
    notes : str | None
        Optional text attached to the wandb run.
    """
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_id = exp_id or _next_exp_id()

    # ── W&B logger (auto-disabled if WANDB_DISABLED=true or wandb missing) ──
    logger = WandbLogger(
        exp_id=exp_id,
        run_name=f"{exp_id} — {datetime.now().strftime('%Y-%m-%d %H:%M')} @ {socket.gethostname()}",
        notes=notes,
        tags=["sanity" if sanity else "full", "pipeline-live"],
        config={
            "sanity": sanity,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "random_seed": config.RANDOM_SEED,
            "channels": list(config.EDF_CHANNELS),
            "in_channels": len(config.EDF_CHANNELS),
            "target_sfreq": config.TARGET_SFREQ,
            "window_seconds": config.WINDOW_SECONDS,
            "samples_per_window": config.SAMPLES_PER_WIN,
            "overlap_threshold_secs": config.OVERLAP_THRESHOLD_SECS,
            "clip_sigma": config.CLIP_SIGMA,
            "seq_len": config.SEQ_LEN,
            "batch_size": config.BATCH_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "learn_rate": config.LEARN_RATE,
            "dropout": config.DROPOUT,
            "train_patients": config.SANITY_CHECK_PATIENTS if sanity else config.TRAIN_PATIENTS,
            "test_patients": config.SANITY_CHECK_PATIENTS if sanity else config.TEST_PATIENTS,
        },
    )

    # ── Step 6: Gate check ──────────────────────────────────────────────────
    print("\n[STEP 6] Running diagnostic gate check on ucddb002 ...")
    loader.run_gate_check("ucddb002")

    # ── Step 8: Build datasets ──────────────────────────────────────────────
    train_patients = config.SANITY_CHECK_PATIENTS if sanity else config.TRAIN_PATIENTS
    test_patients = config.SANITY_CHECK_PATIENTS if sanity else config.TEST_PATIENTS

    print("\n[STEP 8] Building TRAIN dataset ...")
    train_windows, train_labels = dataset.build_dataset_per_patient(train_patients)

    print("\n[STEP 8] Building TEST dataset ...")
    test_windows, test_labels = dataset.build_dataset_per_patient(test_patients)

    train_ds = dataset.ApneaSequenceDataset(train_windows, train_labels)
    test_ds = dataset.ApneaSequenceDataset(test_windows, test_labels)

    sampler = _build_sampler(train_ds)

    loader_workers = 2
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(loader_workers),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(loader_workers),
    )

    # ── Step 7: Model ───────────────────────────────────────────────────────
    in_channels = train_windows[0].shape[1] if train_windows else len(config.EDF_CHANNELS)
    model = models.ApneaCNNLSTM(in_channels=in_channels).to(device)
    logger.watch_model(model)

    # Sampler balances the batches — no pos_weight needed in the loss.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARN_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    # ── Step 8: Train ───────────────────────────────────────────────────────
    print(f"\n[STEP 8] Training for up to {config.NUM_EPOCHS} epochs ...")
    train_losses, val_losses = train_mod.fit(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        num_epochs=config.NUM_EPOCHS,
        scheduler=scheduler,
        early_stop_patience=config.EARLY_STOP_PATIENCE,
    )
    for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
        logger.log_epoch(epoch, train_loss=tl, val_loss=vl)
    logger.log_epoch_curve(train_losses, val_losses)

    # ── Step 9: Evaluate ────────────────────────────────────────────────────
    print("\n[STEP 9] Evaluating on test patients ...")
    best_thresh, best_f1, acc, recall_v, precision_v, cm = metrics.evaluate(model, test_loader, device)
    sweep = metrics.evaluate_threshold_sweep(model, test_loader, device)

    tn = fp = fn = tp = None
    if cm is not None and cm.size == 4:
        tn, fp, fn, tp = (int(v) for v in cm.ravel())

    logger.log_final(
        optimal_threshold=float(best_thresh),
        accuracy=float(acc),
        macro_f1=float(best_f1),
        sensitivity=float(recall_v) if recall_v == recall_v else None,  # NaN -> None
        precision=float(precision_v) if precision_v == precision_v else None,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        threshold_sweep=sweep,
        confusion_matrix=cm,
    )

    logger.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UCDDB sleep apnea pipeline.")
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Train + test on a single sanity-check patient (smoke test).",
    )
    args = parser.parse_args()
    run(sanity=args.sanity)
