"""
Bulk preprocessing entry point.

Run once per dataset to materialize every patient's processed windows to
``artifacts/windows_cache/<dataset_name>_<config_hash>/<patient_id>_w.npy``
(and matching ``_l.npy``). The resulting folder can be zipped and downloaded
from a Kaggle notebook, then re-attached (or extracted into the repo) so
that subsequent training runs skip the entire EDF/resample/window pipeline.

Usage (CLI)
-----------
    # Process every patient in config.ALL_PATIENTS (default dataset name 'ucddb')
    python -m pipeline.preprocess_dataset

    # Process an explicit list of patient IDs
    python -m pipeline.preprocess_dataset --patients ucddb002 ucddb003 ...

    # Custom dataset name (so UCDDB / SHHS / MESA caches don't collide)
    python -m pipeline.preprocess_dataset --dataset-name shhs \\
        --patients shhs-200001 shhs-200002 ...

    # Force rebuild even if cache files already exist
    python -m pipeline.preprocess_dataset --force

Programmatic
------------
    from pipeline.preprocess_dataset import preprocess_all
    preprocess_all(patient_ids=["ucddb002", ...], dataset_name="ucddb", force=False)

The orchestrator (``pipeline.orchestrator``) will detect the bulk-preprocessed
files via :func:`pipeline.dataset.is_preprocessed` and use them directly,
never re-running MNE/EDF loading or resampling.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from . import config, dataset, loader
from .dataset import preprocessed_paths


def _discover_patients(dataset_name: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)

    candidates: list[str] = []
    try:
        for entry in sorted(os.listdir(config.DATA_DIR)):
            full = Path(config.DATA_DIR) / entry
            if entry.lower().endswith(config.EDF_SUFFIX.lower()) and full.is_file():
                candidates.append(entry[: -len(config.EDF_SUFFIX)])
    except FileNotFoundError as exc:
        raise SystemExit(
            f"DATA_DIR not found: {config.DATA_DIR!r}. "
            "Set the DATA_DIR environment variable to the dataset root."
        ) from exc

    if not candidates:
        raise SystemExit(
            f"No EDF files found under DATA_DIR={config.DATA_DIR!r}. "
            "Pass --patients explicitly if your filenames use a different convention."
        )
    return candidates


def preprocess_all(
    patient_ids: Iterable[str],
    *,
    dataset_name: str = "ucddb",
    force: bool = False,
) -> dict[str, tuple[Path, Path]]:
    """
    Materialize the per-patient preprocessed cache for every patient.

    Returns a ``{patient_id: (wins_path, labels_path)}`` mapping. Patients
    whose files already exist (and ``force`` is False) are skipped.
    """
    out_dir = dataset.dataset_root(dataset_name)
    print(f"[preprocess] dataset_name : {dataset_name}")
    print(f"[preprocess] output dir   : {out_dir}")
    print(f"[preprocess] DATA_DIR     : {config.DATA_DIR}")
    print(f"[preprocess] config hash  : {out_dir.name.split('_')[-1]}")
    print(
        f"[preprocess] channels     : {config.EDF_CHANNELS} | "
        f"sfreq={config.TARGET_SFREQ} Hz | win={config.WINDOW_SECONDS}s"
    )
    print()

    written: dict[str, tuple[Path, Path]] = {}
    skipped = 0
    failed = 0
    t0 = time.perf_counter()

    for pid in patient_ids:
        wins_path, labs_path = preprocessed_paths(dataset_name, pid)
        if not force and wins_path.exists() and labs_path.exists():
            skipped += 1
            print(f"  [skip] {pid}  (cached)")
            continue

        print(f"  [....] {pid} ...", end=" ", flush=True)
        try:
            wins, labs = loader.process_patient(pid)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAILED — {exc}")
            continue
        np_save = __import__("numpy").save
        np_save(wins_path, wins)
        np_save(labs_path, labs)
        written[pid] = (wins_path, labs_path)
        print(
            f"{len(labs)} windows | {int(labs.sum())} apnea "
            f"({100 * labs.mean():.1f}%)"
        )

    elapsed = time.perf_counter() - t0
    print()
    print(
        f"[preprocess] done in {elapsed:.1f}s — "
        f"written={len(written)}  skipped={skipped}  failed={failed}"
    )
    print(f"[preprocess] output : {out_dir}")
    print(
        "[preprocess] tip    : zip the folder above and download it from Kaggle, "
        "then drop it back into artifacts/windows_cache/ on a future run."
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-preprocess a sleep-apnea dataset into per-patient .npy caches.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ucddb",
        help="Logical dataset name used as the cache subfolder (default: ucddb).",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Explicit patient IDs to process. Defaults to every EDF in DATA_DIR.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache files even if they already exist.",
    )
    args = parser.parse_args(argv)

    pids = _discover_patients(args.dataset_name, args.patients)
    print(f"[preprocess] {len(pids)} patient(s) discovered\n")
    preprocess_all(pids, dataset_name=args.dataset_name, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())