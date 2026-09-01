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

from . import config, dataset, datasets_spec as specs, loader, metrics, models, train as train_mod
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


def run(
    sanity: bool = False,
    exp_id: str | None = None,
    notes: str | None = None,
    *,
    train_dataset: str = "ucddb",
    test_dataset: str = "ucddb",
    train_patients: list[str] | None = None,
    test_patients: list[str] | None = None,
) -> None:
    """
    Run the full pipeline. Set sanity=True to use the single sanity-check
    patient for both train and test (useful for overfitting smoke tests).

    Cross-dataset usage
    -------------------
    Pass ``train_dataset="shhs1"`` and ``test_dataset="ucddb"`` to pretrain
    on SHHS and evaluate on UCDDB (the headline cross-dataset experiment).
    ``train_patients`` / ``test_patients`` default to the UCDDB splits in
    ``config``. Override either list to subset further.

    Parameters
    ----------
    sanity : bool
        If True, train + test on the sanity-check patient.
    exp_id : str | None
        Override the auto-generated EXP_NNN id (e.g. ``"EXP_011"``).
    notes : str | None
        Optional text attached to the wandb run.
    train_dataset : str
        Logical dataset name for the training set (e.g. "shhs1", "ucddb").
    test_dataset : str
        Logical dataset name for the held-out test set.
    train_patients : list[str] | None
    test_patients  : list[str] | None
        Explicit patient ID lists. If None, defaults are pulled from
        ``config.TRAIN_PATIENTS`` / ``config.TEST_PATIENTS`` when the
        corresponding ``<dataset>`` matches "ucddb", else a sensible
        auto-discovery is used.
    """
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_id = exp_id or _next_exp_id()

    train_spec = specs.get_spec(train_dataset)
    test_spec = specs.get_spec(test_dataset)

    if train_patients is None:
        if sanity:
            train_patients = list(config.SANITY_CHECK_PATIENTS)
        elif train_dataset.lower() == "ucddb":
            train_patients = list(config.TRAIN_PATIENTS)
        else:
            # Cross-dataset: caller didn't pass an explicit list. We refuse
            # to silently train on everything in DATA_DIR — force a list.
            raise ValueError(
                f"train_patients is required when train_dataset={train_dataset!r} "
                "(no default UCDDB split applies)."
            )

    if test_patients is None:
        if sanity:
            test_patients = list(config.SANITY_CHECK_PATIENTS)
        elif test_dataset.lower() == "ucddb":
            test_patients = list(config.TEST_PATIENTS)
        else:
            raise ValueError(
                f"test_patients is required when test_dataset={test_dataset!r}."
            )

    print(
        f"Train on {train_spec.name}  ({len(train_patients)} patients)  →  "
        f"Test on {test_spec.name}  ({len(test_patients)} patients)"
    )

    # ── W&B logger (auto-disabled if WANDB_DISABLED=true or wandb missing) ──
    logger = WandbLogger(
        exp_id=exp_id,
        run_name=f"{exp_id} — {datetime.now().strftime('%Y-%m-%d %H:%M')} @ {socket.gethostname()}",
        notes=notes,
        tags=[
            "sanity" if sanity else "full",
            "pipeline-live",
            f"train={train_spec.name}",
            f"test={test_spec.name}",
        ],
        config={
            "sanity": sanity,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "random_seed": config.RANDOM_SEED,
            "train_dataset": train_spec.name,
            "test_dataset": test_spec.name,
            "train_channels": list(train_spec.channels),
            "test_channels": list(test_spec.channels),
            "in_channels": len(train_spec.channels),
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
            "weight_decay": config.WEIGHT_DECAY,
            "lstm_dropout": config.LSTM_DROPOUT,
            "early_stop_patience": config.EARLY_STOP_PATIENCE,
            "lr_scheduler_patience": config.LR_SCHEDULER_PATIENCE,
            "lr_scheduler_factor": config.LR_SCHEDULER_FACTOR,
            "train_patients": train_patients,
            "test_patients": test_patients,
        },
    )

    # ── Step 6: Gate check (always on the test dataset, first patient) ──────
    if test_patients:
        gate_pid = test_patients[0]
        print(f"\n[STEP 6] Running diagnostic gate check on {gate_pid} ...")
        loader.run_gate_check(gate_pid)

    # ── Step 8: Build datasets ──────────────────────────────────────────────
    print("\n[STEP 8] Building TRAIN dataset ...")
    train_windows, train_labels = dataset.build_dataset_per_patient(
        train_patients, dataset_name=train_spec.name, spec=train_spec
    )

    print("\n[STEP 8] Building TEST dataset ...")
    test_windows, test_labels = dataset.build_dataset_per_patient(
        test_patients, dataset_name=test_spec.name, spec=test_spec
    )

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
    in_channels = train_windows[0].shape[1] if train_windows else len(train_spec.channels)
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

    parser = argparse.ArgumentParser(description="Sleep apnea pipeline.")
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Train + test on a single sanity-check patient (smoke test).",
    )
    parser.add_argument(
        "--train-dataset",
        default="ucddb",
        help="Logical dataset name for the training set (e.g. ucddb, shhs1).",
    )
    parser.add_argument(
        "--test-dataset",
        default="ucddb",
        help="Logical dataset name for the held-out test set.",
    )
    parser.add_argument(
        "--train-patients",
        nargs="*",
        default=None,
        help="Explicit patient IDs to train on. Defaults to config.TRAIN_PATIENTS for UCDDB.",
    )
    parser.add_argument(
        "--test-patients",
        nargs="*",
        default=None,
        help="Explicit patient IDs to test on. Defaults to config.TEST_PATIENTS for UCDDB.",
    )
    args = parser.parse_args()
    run(
        sanity=args.sanity,
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        train_patients=args.train_patients,
        test_patients=args.test_patients,
    )
