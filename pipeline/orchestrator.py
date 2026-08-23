"""
End-to-end orchestrator: builds datasets, instantiates the model, trains,
validates, and evaluates on the held-out test set.

Run with:
    python -m pipeline.orchestrator
or:
    from pipeline.orchestrator import run
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import config, dataset, loader, metrics, models, train as train_mod


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


def run(sanity: bool = False) -> None:
    """
    Run the full pipeline. Set sanity=True to use the single sanity-check
    patient for both train and test (useful for overfitting smoke tests).
    """
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── Step 7: Model ───────────────────────────────────────────────────────
    in_channels = train_windows[0].shape[1] if train_windows else len(config.EDF_CHANNELS)
    model = models.ApneaCNNLSTM(in_channels=in_channels).to(device)

    # Sampler balances the batches — no pos_weight needed in the loss.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARN_RATE)

    # ── Step 8: Train ───────────────────────────────────────────────────────
    print(f"\n[STEP 8] Training for {config.NUM_EPOCHS} epochs ...")
    train_mod.fit(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        num_epochs=config.NUM_EPOCHS,
    )

    # ── Step 9: Evaluate ────────────────────────────────────────────────────
    print("\n[STEP 9] Evaluating on test patients ...")
    metrics.evaluate(model, test_loader, device)
    metrics.evaluate_threshold_sweep(model, test_loader, device)


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
