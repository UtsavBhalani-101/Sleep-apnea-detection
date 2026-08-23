"""
PyTorch Dataset + per-patient dataset assembly.

Builds (SEQ_LEN, C, SAMPLES_PER_WIN) sequences from each patient's windows
WITHOUT bridging the boundary between two patients (no patient A tail
contaminating patient B head).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from .loader import process_patient


class ApneaSequenceDataset(Dataset):
    """
    Each item is a (SEQ_LEN, C, SAMPLES_PER_WIN) window sequence and the
    matching (SEQ_LEN,) label sequence. Sequences are built independently
    for each patient so we never cross patient boundaries.
    """

    def __init__(
        self,
        all_windows: list[np.ndarray],
        all_labels: list[np.ndarray],
        seq_len: int = config.SEQ_LEN,
    ):
        self.seq_len = seq_len

        sequences = []
        seq_labels = []
        for wins, labs in zip(all_windows, all_labels):
            n = len(labs)
            if n < seq_len:
                continue
            for i in range(n - seq_len + 1):
                sequences.append(wins[i : i + seq_len])
                seq_labels.append(labs[i : i + seq_len])

        self.X = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.y = torch.tensor(np.array(seq_labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_dataset_per_patient(
    patient_ids: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run the full per-patient pipeline and return parallel lists of
    per-patient windows and per-patient labels. Used downstream by
    ApneaSequenceDataset.
    """
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for pid in patient_ids:
        print(f"  Processing {pid} ...", end=" ", flush=True)
        try:
            wins, labs = process_patient(pid)
            all_windows.append(wins)
            all_labels.append(labs)
            print(
                f"{len(labs)} windows | {int(labs.sum())} apnea "
                f"({100 * labs.mean():.1f}%)"
            )
        except Exception as e:  # noqa: BLE001 — surface but keep going
            print(f"FAILED — {e}")
    return all_windows, all_labels
