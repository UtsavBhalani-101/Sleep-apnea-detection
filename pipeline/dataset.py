"""
PyTorch Dataset + per-patient dataset assembly.

Builds (SEQ_LEN, C, SAMPLES_PER_WIN) sequences from each patient's windows
WITHOUT bridging the boundary between two patients (no patient A tail
contaminating patient B head).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config, datasets_spec as specs
from .loader import process_patient


CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "windows_cache"
CACHE_VERSION = "v3"


def _config_fingerprint() -> str:
    """Stable hash of all config knobs that change window content."""
    raw = (
        f"{CACHE_VERSION}|{config.TARGET_SFREQ}|{config.WINDOW_SECONDS}"
        f"|{config.OVERLAP_THRESHOLD_SECS}|{config.CLIP_SIGMA}"
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def dataset_root(dataset_name: str) -> Path:
    """Return the bulk-preprocessed root for a named dataset (e.g. 'ucddb')."""
    root = CACHE_DIR / f"{dataset_name}_{_config_fingerprint()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def preprocessed_paths(dataset_name: str, pid: str) -> tuple[Path, Path]:
    root = dataset_root(dataset_name)
    return root / f"{pid}_w.npy", root / f"{pid}_l.npy"


def is_preprocessed(dataset_name: str, patient_ids: list[str]) -> bool:
    """True iff every patient in the list has both cache files present."""
    return all(
        preprocessed_paths(dataset_name, p)[0].exists()
        and preprocessed_paths(dataset_name, p)[1].exists()
        for p in patient_ids
    )


def build_dataset_per_patient(
    patient_ids: list[str],
    *,
    dataset_name: str = "ucddb",
    spec: specs.DatasetSpec | None = None,
    use_cache: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run the full per-patient pipeline and return parallel lists of
    per-patient windows and per-patient labels. Used downstream by
    ApneaSequenceDataset.

    Lookup order per patient:
      1. Bulk-preprocessed ``artifacts/windows_cache/<dataset>_<hash>/<pid>_*.npy``
         (produced by ``pipeline.preprocess_dataset``).
      2. Fall back to per-patient processing + per-patient cache.

    If ``use_cache`` is True, both the bulk and per-patient fallbacks are
    memoized to disk so subsequent runs skip the EDF/resample/window pass.
    """
    if spec is None:
        spec = specs.get_spec(dataset_name)

    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for pid in patient_ids:
        print(f"  Processing {pid} ...", end=" ", flush=True)
        wins_path, labs_path = preprocessed_paths(dataset_name, pid)

        wins: np.ndarray | None = None
        labs: np.ndarray | None = None

        if use_cache and wins_path.exists() and labs_path.exists():
            wins = np.load(wins_path)
            labs = np.load(labs_path)
        else:
            try:
                wins, labs, _ch_names = process_patient(pid, spec=spec)
            except Exception as e:  # noqa: BLE001 — surface but keep going
                print(f"FAILED — {e}")
                continue
            if use_cache:
                np.save(wins_path, wins)
                np.save(labs_path, labs)

        all_windows.append(wins)
        all_labels.append(labs)
        print(
            f"{len(labs)} windows | {int(labs.sum())} apnea "
            f"({100 * labs.mean():.1f}%)"
        )
    return all_windows, all_labels


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
