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
import warnings
from pathlib import Path
from typing import Iterable

# Silence the runpy "found in sys.modules... prior to execution" warning that
# Python emits when you run `python -m pipeline.preprocess_dataset` after the
# orchestrator has already imported the `pipeline` package. The actual code is
# safe; the warning is purely about import order.
warnings.filterwarnings(
    "ignore",
    message=r".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
)

from . import config, dataset, datasets_spec as specs, loader
from .dataset import preprocessed_paths


def _discover_patients(spec: specs.DatasetSpec, explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)

    root = spec.data_dir or config.DATA_DIR
    candidates: list[str] = []
    try:
        for entry in sorted(os.listdir(root)):
            full = Path(root) / entry
            if not full.is_file():
                continue
            pid = spec.parse_patient_id(entry)
            if pid is not None:
                candidates.append(pid)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Data dir not found: {root!r} (spec={spec.name}). "
            "Set DATA_DIR or pass --patients explicitly."
        ) from exc

    if not candidates:
        raise SystemExit(
            f"No EDF files matching spec {spec.name!r} found under {root!r}. "
            "Pass --patients explicitly if your filenames use a different convention."
        )
    return candidates


def preprocess_all(
    patient_ids: Iterable[str],
    *,
    dataset_name: str = "ucddb",
    spec: specs.DatasetSpec | None = None,
    force: bool = False,
) -> dict[str, tuple[Path, Path]]:
    """
    Materialize the per-patient preprocessed cache for every patient.

    Returns a ``{patient_id: (wins_path, labels_path)}`` mapping. Patients
    whose files already exist (and ``force`` is False) are skipped.
    """
    if spec is None:
        spec = specs.get_spec(dataset_name)

    out_dir = dataset.dataset_root(dataset_name)
    print(f"[preprocess] dataset_name : {spec.name}")
    print(f"[preprocess] output dir   : {out_dir}")
    print(f"[preprocess] data dir     : {spec.data_dir or config.DATA_DIR}")
    print(f"[preprocess] channels     : {spec.channels}")
    print(f"[preprocess] annotation   : {spec.annotation_format} ({spec.annotation_suffix})")
    print(
        f"[preprocess] window       : {config.WINDOW_SECONDS}s @ "
        f"{config.TARGET_SFREQ} Hz (samples/win={config.SAMPLES_PER_WIN})"
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
            wins, labs, _ch = loader.process_patient(pid, spec=spec)
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
        help="Logical dataset name (e.g. ucddb, shhs1, shhs2, mesa).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the spec's data directory (use when running outside Kaggle).",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Explicit patient IDs to process. Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache files even if they already exist.",
    )
    args = parser.parse_args(argv)

    spec = specs.get_spec(args.dataset_name)
    if args.data_dir is not None:
        spec = specs.DatasetSpec(
            **{**spec.__dict__, "data_dir": args.data_dir}
        )

    pids = _discover_patients(spec, args.patients)
    print(f"[preprocess] {len(pids)} patient(s) discovered\n")
    preprocess_all(pids, dataset_name=spec.name, spec=spec, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())