"""
EDF loading, resampling, respevt / NSRR-XML parsing, and the diagnostic gate check.

Functions:
    load_edf          — read EDF + select channels (spec-aware)
    resample_to_10hz  — anti-aliased decimation to TARGET_SFREQ
    parse_respevt     — read respevt.txt → list of (onset_sec, duration_sec)
    parse_events      — respevt vs nsrr-xml dispatch based on DatasetSpec
    run_gate_check    — end-to-end diagnostic print for one patient
    process_patient   — full Steps 1–5 for a single patient
"""

from __future__ import annotations

import contextlib
import datetime
import os
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from . import config
from . import datasets_spec as specs
from . import windows as windows_mod


@contextlib.contextmanager
def _silence_mne_warnings():
    """Suppress MNE's per-file filter warnings (highpass/lowpass disagreement).

    MNE emits a ``RuntimeWarning`` per EDF when channels disagree on the
    highpass or lowpass filter cutoffs. We do not recompute filters, so
    these messages add noise without value.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Channels contain different highpass filters. Highest filter setting will be stored.",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Channels contain different lowpass filters. Lowest filter setting will be stored.",
            category=RuntimeWarning,
        )
        yield


mne.set_log_level("ERROR")


def load_edf(
    patient_id: str,
    spec: specs.DatasetSpec | None = None,
) -> tuple[np.ndarray, float, list[str], datetime.datetime | None, specs.DatasetSpec]:
    """
    Load the EDF for a given patient and select the configured channels.

    Parameters
    ----------
    patient_id : e.g. "ucddb002" or "shhs1-200001"
    spec       : DatasetSpec describing the dataset. Defaults to UCDDB.

    Returns
    -------
    data         : np.ndarray, shape (n_channels, total_samples)
    native_sfreq : float — sampling frequency stored in the EDF header
    ch_names     : list[str] — channel names actually present after pick
    meas_date    : datetime | None — recording start time from the EDF header
    spec         : the DatasetSpec that was used (echoed for callers)
    """
    if spec is None:
        spec = specs.UCDDB
    edf_path = spec.edf_path(patient_id)

    with _silence_mne_warnings():
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    filtered_raw = raw.copy().pick(picks=list(spec.channels))

    native_sfreq = filtered_raw.info["sfreq"]
    ch_names = filtered_raw.info["ch_names"]
    data = filtered_raw.get_data()
    meas_date = filtered_raw.info["meas_date"]

    return data, native_sfreq, ch_names, meas_date, spec


def resample_to_10hz(data: np.ndarray, native_sfreq: float) -> np.ndarray:
    """
    Downsample signal data from native_sfreq to TARGET_SFREQ.

    Uses scipy resample_poly which applies an anti-aliasing FIR filter
    before decimation — no Nyquist aliasing artifacts.
    """
    up = int(config.TARGET_SFREQ)
    down = int(native_sfreq)
    return resample_poly(data, up, down, axis=1)


def parse_respevt(
    patient_id: str,
    meas_date: datetime.datetime | None = None,
    spec: specs.DatasetSpec | None = None,
) -> list[tuple[float, float]]:
    """
    Parse the respevt.txt file for a patient into (onset_sec, duration_sec).

    The file uses a 3-line header followed by whitespace-separated columns:
      col 0: HH:MM:SS time
      col 1: event type (APNEA, HYP, etc.)
      col 2+: PB/CS flag, duration, SpO2, etc. — Duration is the first
              whole-number column from col 2 onward.

    Parameters
    ----------
    meas_date : datetime | None
        Recording start time from the EDF header. If None, falls back to
        re-opening the EDF (slower; only needed when calling parse_respevt
        without first calling load_edf).
    spec : DatasetSpec | None
        Defaults to UCDDB. Only the EDF/annotation paths are used here.
    """
    if spec is None:
        spec = specs.UCDDB

    if meas_date is None:
        edf_path = spec.edf_path(patient_id)
        with _silence_mne_warnings():
            meas_date = mne.io.read_raw_edf(edf_path, preload=False, verbose=False).info["meas_date"]

    edf_start_sec = meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second

    evt_path = spec.annotation_path(patient_id)
    if not os.path.exists(evt_path):
        return []

    df = pd.read_csv(
        evt_path,
        sep=r"\s+",
        header=None,
        skiprows=3,
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = range(df.shape[1])
    df = df.rename(columns={0: "time_str", 1: "event_type"})

    is_apnea_or_hyp = (
        df["event_type"].str.contains("APNEA", na=False)
        | df["event_type"].str.contains("HYP", na=False)
    )
    df = df[is_apnea_or_hyp].copy()

    # Find the duration column (first whole-number column from col 2 onward)
    duration_col = None
    candidate_cols = [c for c in df.columns if isinstance(c, int) and c >= 2]
    for col in candidate_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        is_whole = (numeric == numeric.round()) & numeric.notna()
        if is_whole.any():
            duration_col = col
            break

    if duration_col is None:
        return []

    df["duration_sec"] = pd.to_numeric(df[duration_col], errors="coerce")
    df = df.dropna(subset=["duration_sec"])
    df["duration_sec"] = df["duration_sec"].astype(int)

    def hhmmss_to_edf_seconds(t_str: str) -> float:
        parts = [int(p) for p in t_str.split(":")]
        event_sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
        if event_sec < edf_start_sec - 1800:
            event_sec += 86400
        return event_sec - edf_start_sec

    df["onset_sec"] = df["time_str"].apply(hhmmss_to_edf_seconds)
    return list(zip(df["onset_sec"], df["duration_sec"]))


def parse_events(
    patient_id: str,
    *,
    meas_date: datetime.datetime | None = None,
    spec: specs.DatasetSpec | None = None,
) -> list[tuple[float, float]]:
    """
    Dispatch annotation parsing by dataset. Returns apnea/hypopnea events
    as ``(onset_sec, duration_sec)``.
    """
    if spec is None:
        spec = specs.UCDDB

    if spec.annotation_format == "respevt":
        return parse_respevt(patient_id, meas_date=meas_date, spec=spec)
    if spec.annotation_format == "nsrr-xml":
        raw = specs.parse_nsrr_xml_events(spec.annotation_path(patient_id))
        return specs.filter_apnea_events(raw, spec)
    raise ValueError(
        f"Unknown annotation_format {spec.annotation_format!r} for spec {spec.name!r}"
    )


def process_patient(
    patient_id: str,
    spec: specs.DatasetSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Run Steps 1–5 for one patient.

    Returns
    -------
    windows     : (n_windows, C, SAMPLES_PER_WIN), float32, z-score normalized
    labels      : (n_windows,), int64, binary
    ch_names    : list[str] — channel order actually used (== spec.channels)
    """
    data, native_sfreq, ch_names, meas_date, spec = load_edf(patient_id, spec=spec)
    data_10hz = resample_to_10hz(data, native_sfreq)
    events = parse_events(patient_id, meas_date=meas_date, spec=spec)
    windows, labels = windows_mod.make_windows_and_labels(data_10hz, events)
    windows_norm = windows_mod.normalize(windows, labels, channel_names=list(spec.channels))
    return windows_norm, labels, list(spec.channels)


def run_gate_check(patient_id: str = "ucddb002") -> None:
    """
    Run the full pipeline for one patient and print a diagnostic block.

    Read the output carefully before proceeding to model training —
    a model trained on broken data will still produce a plausible loss curve.
    """
    print(f"\n{'=' * 60}")
    print(f"  GATE CHECK — Patient: {patient_id}")
    print(f"{'=' * 60}")

    # Step 1: Load
    data, native_sfreq, ch_names, meas_date, _spec = load_edf(patient_id)
    print(f"  Channels      : {ch_names}")
    print(f"  Native sfreq  : {native_sfreq} Hz")
    print(f"  Raw shape     : {data.shape}")

    # Step 2: Resample
    data_10hz = resample_to_10hz(data, native_sfreq)
    print(f"  After resample: {data_10hz.shape}  (10 Hz)")

    # Step 3: Parse events
    events = parse_events(patient_id, meas_date=meas_date)
    print(f"  Events parsed : {len(events)}")
    if events:
        print(
            f"  First event   : onset={events[0][0]:.0f}s, "
            f"duration={events[0][1]:.0f}s"
        )

    # Step 4: Window + label
    windows, labels = windows_mod.make_windows_and_labels(data_10hz, events)
    n_total = len(labels)
    n_apnea = int(labels.sum())
    n_normal = n_total - n_apnea
    print(f"  Total windows : {n_total}")
    print(f"  Apnea windows : {n_apnea}  ({100 * n_apnea / n_total:.1f}%)")
    print(f"  Normal windows: {n_normal}  ({100 * n_normal / n_total:.1f}%)")
    print(f"  Window shape  : {windows.shape[1:]}")
    print(f"  Label dtype   : {labels.dtype}")

    # Step 5: Normalize
    print(f"  Per-channel stats (from normal windows):")
    windows_norm = windows_mod.normalize(windows, labels)
    sig_min = float(windows_norm.min())
    sig_max = float(windows_norm.max())
    print(
        f"  Signal range  : [{sig_min:.2f}, {sig_max:.2f}]  "
        f"(after clip to ±{config.CLIP_SIGMA})"
    )

    # Checks
    print(f"\n  Checks:")
    apnea_rate = n_apnea / n_total
    print(
        f"  {'✓' if 0.10 <= apnea_rate <= 0.40 else '✗'} "
        f"Apnea rate in [10%, 40%]  → {100 * apnea_rate:.1f}%"
    )
    expected_shape = (len(_spec.channels), config.SAMPLES_PER_WIN)
    print(
        f"  {'✓' if windows.shape[1:] == expected_shape else '✗'} "
        f"Window shape is {expected_shape}  → {windows.shape[1:]}"
    )
    print(
        f"  {'✓' if -6 <= sig_min and sig_max <= 6 else '✗'} "
        f"Signal range in [-6, +6]  → [{sig_min:.2f}, {sig_max:.2f}]"
    )
    print(
        f"  {'✓' if labels.dtype == np.int64 else '✗'} "
        f"Labels are int64           → {labels.dtype}"
    )
    print(
        f"  {'✓' if len(events) > 0 else '✗'} "
        f"Events parsed > 0          → {len(events)}"
    )
    print(f"{'=' * 60}\n")
