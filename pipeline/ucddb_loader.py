"""
UCDDB Sleep Apnea Detection — V1 Pipeline
==========================================
Everything in one file, in execution order:
  1. Load EDF
  2. Resample to 10 Hz
  3. Parse respevt.txt events
  4. Create 30-second windows + binary labels
  5. Z-score normalization (normal-window stats only)
  6. Diagnostic gate check (print before any model code)
  7. 1D CNN model definition
  8. Dataset + DataLoader + training loop
  9. Evaluation: accuracy, macro F1, raw confusion matrix

Dataset: UCDDB (_lifecard.edf has 3 channels: Chan 1 / Chan 2 / Chan 3)
Target: macro F1 > 0.5 on held-out test patients
"""

import os
import pandas as pd
import numpy as np
import mne
from scipy.signal import resample_poly
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths/hyperparameters as needed
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = r"d:\Sleep irregularity\datasets\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0\files"

# All 25 patient IDs present in the dataset (004 and 016 are missing from UCDDB)
ALL_PATIENTS = [
    "ucddb002", "ucddb003", "ucddb005", "ucddb006", "ucddb007",
    "ucddb008", "ucddb009", "ucddb010", "ucddb011", "ucddb012",
    "ucddb013", "ucddb014", "ucddb015", "ucddb017", "ucddb018",
    "ucddb019", "ucddb020", "ucddb021", "ucddb022", "ucddb023",
    "ucddb024", "ucddb025", "ucddb026", "ucddb027", "ucddb028",
]

# Patient-level 80/20 split — all windows from a patient stay on one side
TRAIN_PATIENTS = ALL_PATIENTS[:20]  # ucddb002 → ucddb023
TEST_PATIENTS = ALL_PATIENTS[20:]  # ucddb024 → ucddb028

# Signal constants
TARGET_SFREQ = 10.0  # Hz — all channels resampled to this
WINDOW_SECONDS = 30  # seconds per window
SAMPLES_PER_WIN = int(WINDOW_SECONDS * TARGET_SFREQ)  # 300 samples

# Labeling rule: event must overlap window by at least this many seconds
OVERLAP_THRESHOLD_SECS = 10

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARN_RATE = 1e-3
DROPOUT = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load EDF and extract all 3 channels as a raw numpy array
# ─────────────────────────────────────────────────────────────────────────────


def load_edf(patient_id: str) -> tuple[np.ndarray, float, list[str]]:
    """
    Load the _lifecard.edf for a given patient.

    Returns
    -------
    data        : np.ndarray, shape (3, total_samples) — raw signal in native sfreq
    native_sfreq: float — the sampling frequency stored in the EDF header
    ch_names    : list[str] — the actual channel names found in the file
    """
    edf_path = os.path.join(DATA_DIR, f"{patient_id}_lifecard.edf")

    # preload=True loads all signal data into RAM immediately
    # verbose=False silences MNE's chatty info prints
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    native_sfreq = raw.info["sfreq"]
    ch_names = raw.info["ch_names"]

    # get_data() returns shape (n_channels, total_samples)
    data = raw.get_data()

    return data, native_sfreq, ch_names


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Resample all channels from native sfreq → 10 Hz
# ─────────────────────────────────────────────────────────────────────────────


def resample_to_10hz(data: np.ndarray, native_sfreq: float) -> np.ndarray:
    """
    Downsample signal data from native_sfreq to TARGET_SFREQ (10 Hz).

    Uses scipy resample_poly which applies an anti-aliasing FIR filter
    before decimation — no Nyquist aliasing artifacts.

    Parameters
    ----------
    data         : (n_channels, total_samples) at native_sfreq
    native_sfreq : original sampling rate (e.g. 128.0 Hz)

    Returns
    -------
    resampled : (n_channels, new_total_samples) at 10 Hz
    """
    # resample_poly needs integer up/down ratios
    # e.g. 128 Hz → 10 Hz: up=10, down=128 → gcd reduces to up=5, down=64
    up = int(TARGET_SFREQ)
    down = int(native_sfreq)

    # axis=1 means resample along the time axis (columns), not the channel axis
    resampled = resample_poly(data, up, down, axis=1)

    return resampled


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Parse respevt.txt into a DataFrame, then extract event tuples
# ─────────────────────────────────────────────────────────────────────────────


def parse_respevt(patient_id: str) -> list[tuple[float, float]]:
    """
    Parse the _respevt.txt file for a patient using pandas.

    The file looks like this (fixed-width, 3-line header, then data rows):

        Time       Type   PB/CS  Duration  Low    %Drop  ...
        00:29:13  HYP-C             16       89.9    4.1  ...
        01:41:32  APNEA-O           15       90.8    5.2  ...

    Strategy:
      - Skip the 3-line header with skiprows=3
      - Let pandas parse all whitespace-separated tokens into numbered columns
      - Column 0 = time string (HH:MM:SS)
      - Column 1 = event type string (APNEA-O, HYP-C, etc.)
      - Column 2 = PB/CS flag (optional — can be blank, shifting later cols)
      - Duration lives in column 2 OR column 3 depending on whether PB/CS
        is present. We find it by scanning cols 2 onward for the first
        value that is a whole number (no decimal).
      - Filter: keep rows where Type contains 'APNEA' or 'HYP'
      - Drop rows where Duration could not be found (NaN after coercion)

    Returns
    -------
    events : list of (onset_seconds, duration_seconds) tuples
             Only apnea and hypopnea events. Rows with missing duration skipped.
    """
    evt_path = os.path.join(DATA_DIR, f"{patient_id}_respevt.txt")

    # --- Read the file into a DataFrame -----------------------------------
    # sep=r'\s+' : split on any whitespace (handles variable spacing)
    # header=None : no column names in file — we name them ourselves
    # skiprows=3  : skip the 3-line title/header block
    # engine='python': needed for regex separator
    # on_bad_lines='skip': silently drop rows that can't be parsed
    df = pd.read_csv(
        evt_path,
        sep=r"\s+",
        header=None,
        skiprows=3,
        engine="python",
        on_bad_lines="skip",
    )

    # Assign readable names to the columns we actually use
    # (remaining columns are SpO2, %Drop, snore, arousal, etc. — ignored)
    df.columns = range(df.shape[1])  # ensure integer column indices
    df = df.rename(columns={0: "time_str", 1: "event_type"})

    # --- Filter to apnea and hypopnea rows only ---------------------------
    # .str.contains is case-sensitive by default; na=False drops NaN rows
    is_apnea_or_hyp = df["event_type"].str.contains("APNEA", na=False) | df[
        "event_type"
    ].str.contains("HYP", na=False)
    df = df[is_apnea_or_hyp].copy()

    # --- Find the Duration column -----------------------------------------
    # Duration is a whole-number integer (16, 22, etc.).
    # PB/CS (col 2) is sometimes a string flag like 'CS' and sometimes blank,
    # which pushes Duration from col 2 to col 3.
    # Solution: scan cols 2 onward; coerce each to numeric; take the first
    # column whose values are whole numbers (no fractional part) as Duration.
    duration_col = None
    candidate_cols = [c for c in df.columns if isinstance(c, int) and c >= 2]

    for col in candidate_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")  # non-numeric → NaN
        # A whole number has zero fractional part after coercion
        is_whole = (numeric == numeric.round()) & numeric.notna()
        if is_whole.any():
            duration_col = col
            break

    if duration_col is None:
        # No duration column found at all — return empty (will surface in gate check)
        return []

    # Coerce to numeric so rows with blank/non-numeric duration become NaN
    df["duration_sec"] = pd.to_numeric(df[duration_col], errors="coerce")

    # Drop rows where duration is missing
    df = df.dropna(subset=["duration_sec"])
    df["duration_sec"] = df["duration_sec"].astype(int)

    # --- Convert HH:MM:SS time string → seconds ---------------------------
    def hhmmss_to_seconds(t: str) -> float:
        """'01:29:42' → 5382.0"""
        parts = t.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    df["onset_sec"] = df["time_str"].apply(hhmmss_to_seconds)

    # --- Return as list of (onset, duration) tuples -----------------------
    events = list(zip(df["onset_sec"].astype(float), df["duration_sec"].astype(float)))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Create 30-second non-overlapping windows and assign binary labels
# ─────────────────────────────────────────────────────────────────────────────


def make_windows_and_labels(
    data: np.ndarray,
    events: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice resampled data into 30-second non-overlapping windows and label each.

    Label = 1 if ANY apnea/hypopnea event overlaps the window by ≥ 10 seconds.
    Label = 0 otherwise.

    Parameters
    ----------
    data   : (3, total_samples) at 10 Hz
    events : list of (onset_sec, duration_sec) from parse_respevt()

    Returns
    -------
    windows : np.ndarray, shape (n_windows, 3, 300)
    labels  : np.ndarray, shape (n_windows,), dtype int64
    """
    n_channels, total_samples = data.shape

    # How many complete 30-second windows fit in the recording?
    n_windows = total_samples // SAMPLES_PER_WIN

    # Trim trailing samples that don't fill a complete window
    trimmed = data[:, : n_windows * SAMPLES_PER_WIN]

    # Reshape into (n_windows, n_channels, samples_per_window)
    # Step-by-step:
    #   trimmed              → (3, n_windows * 300)
    #   reshape(3, n, 300)   → (3, n_windows, 300)
    #   swapaxes(0, 1)       → (n_windows, 3, 300)
    windows = trimmed.reshape(n_channels, n_windows, SAMPLES_PER_WIN).swapaxes(0, 1)

    labels = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        # Time boundaries of this window in seconds
        w_start = i * WINDOW_SECONDS  # e.g. window 0 → 0s
        w_end = w_start + WINDOW_SECONDS  # e.g. window 0 → 30s

        for onset_sec, duration_sec in events:
            event_end = onset_sec + duration_sec

            # Overlap = intersection of [w_start, w_end] and [onset, event_end]
            overlap = min(w_end, event_end) - max(w_start, onset_sec)
            overlap = max(0.0, overlap)  # clamp negative values to zero

            if overlap >= OVERLAP_THRESHOLD_SECS:
                labels[i] = 1
                break  # one overlapping event is enough to label this window

    return windows, labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Per-subject z-score normalization using ONLY normal-window stats
# ─────────────────────────────────────────────────────────────────────────────

# Artifact spikes (patient movement, sensor detachment) survive z-score
# normalization because z-score rescales but does not remove outliers.
# Anything beyond ±CLIP_SIGMA standard deviations is almost certainly
# an artifact — not real respiratory signal. We clip after normalization.
CLIP_SIGMA = 5.0


def normalize(
    windows: np.ndarray,
    labels: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply per-channel z-score normalization to all windows, then clip.

    Statistics (mean, std) are computed ONLY from label=0 (normal) windows
    to avoid letting apnea signal distort the baseline.
    The same stats are then applied to ALL windows (including label=1).

    After z-scoring, values are clipped to [-CLIP_SIGMA, +CLIP_SIGMA].
    This suppresses motion artifact spikes that are many standard deviations
    from normal breathing — they are real in the raw signal but carry no
    useful apnea classification information and destabilize training.

    Parameters
    ----------
    windows : (n_windows, 3, 300)
    labels  : (n_windows,) — binary int64
    verbose : if True, print per-channel stats (used by gate check)

    Returns
    -------
    normalized : (n_windows, 3, 300) — float32, clipped to [-5, +5]
    """
    normal_mask = labels == 0
    normal_wins = windows[normal_mask]  # only normal windows for stats

    normalized = windows.astype(np.float32).copy()

    for ch in range(windows.shape[1]):
        # Flatten all samples from normal windows for this channel
        ch_samples = normal_wins[:, ch, :].flatten()

        ch_mean = ch_samples.mean()
        ch_std = ch_samples.std()

        # Apply: (x - mean) / (std + epsilon)
        # epsilon = 1e-8 prevents division by zero on flat channels
        normalized[:, ch, :] = (normalized[:, ch, :] - ch_mean) / (ch_std + 1e-8)

        if verbose:
            raw_min = float(windows[:, ch, :].min())
            raw_max = float(windows[:, ch, :].max())
            print(
                f"    Chan {ch}: raw=[{raw_min:.2f}, {raw_max:.2f}]  "
                f"mean={ch_mean:.4f}  std={ch_std:.4f}"
            )

    # Clip z-scores to suppress artifact spikes.
    # Values beyond ±CLIP_SIGMA are real signal extremes (43 sigma = motion artifact),
    # not apnea patterns. Clamping them protects gradient stability during training.
    normalized = np.clip(normalized, -CLIP_SIGMA, CLIP_SIGMA)

    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Diagnostic gate check: run pipeline for ONE patient, print stats
# ─────────────────────────────────────────────────────────────────────────────


def run_gate_check(patient_id: str = "ucddb002") -> None:
    """
    Run the full pipeline for one patient and print a diagnostic block.
    Read the output carefully before proceeding to model training.
    A model trained on broken data will still produce a plausible loss curve.
    """
    print(f"\n{'=' * 60}")
    print(f"  GATE CHECK — Patient: {patient_id}")
    print(f"{'=' * 60}")

    # --- Step 1: Load ---
    data, native_sfreq, ch_names = load_edf(patient_id)
    print(f"  Channels      : {ch_names}")
    print(f"  Native sfreq  : {native_sfreq} Hz")
    print(f"  Raw shape     : {data.shape}")

    # --- Step 2: Resample ---
    data_10hz = resample_to_10hz(data, native_sfreq)
    print(f"  After resample: {data_10hz.shape}  (10 Hz)")

    # --- Step 3: Parse events ---
    events = parse_respevt(patient_id)
    print(f"  Events parsed : {len(events)}")
    if events:
        print(
            f"  First event   : onset={events[0][0]:.0f}s, duration={events[0][1]:.0f}s"
        )

    # --- Step 4: Window + label ---
    windows, labels = make_windows_and_labels(data_10hz, events)
    n_total = len(labels)
    n_apnea = int(labels.sum())
    n_normal = n_total - n_apnea
    print(f"  Total windows : {n_total}")
    print(f"  Apnea windows : {n_apnea}  ({100 * n_apnea / n_total:.1f}%)")
    print(f"  Normal windows: {n_normal}  ({100 * n_normal / n_total:.1f}%)")
    print(f"  Window shape  : {windows.shape[1:]}")
    print(f"  Label dtype   : {labels.dtype}")

    # --- Step 5: Normalize ---
    # verbose=True prints per-channel raw range, mean, and std before clipping
    print(f"  Per-channel stats (from normal windows):")
    windows_norm = normalize(windows, labels, verbose=True)
    sig_min = float(windows_norm.min())
    sig_max = float(windows_norm.max())
    print(
        f"  Signal range  : [{sig_min:.2f}, {sig_max:.2f}]  (after clip to ±{CLIP_SIGMA})"
    )

    # --- Checks ---
    print(f"\n  Checks:")
    apnea_rate = n_apnea / n_total
    print(
        f"  {'✓' if 0.10 <= apnea_rate <= 0.40 else '✗'} Apnea rate in [10%, 40%]  → {100 * apnea_rate:.1f}%"
    )
    print(
        f"  {'✓' if windows.shape[1:] == (3, 300) else '✗'} Window shape is (3, 300)  → {windows.shape[1:]}"
    )
    print(
        f"  {'✓' if -6 <= sig_min and sig_max <= 6 else '✗'} Signal range in [-6, +6]  → [{sig_min:.2f}, {sig_max:.2f}]"
    )
    print(
        f"  {'✓' if labels.dtype == np.int64 else '✗'} Labels are int64           → {labels.dtype}"
    )
    print(
        f"  {'✓' if len(events) > 0 else '✗'} Events parsed > 0          → {len(events)}"
    )
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline for a single patient → returns normalized windows + labels
# ─────────────────────────────────────────────────────────────────────────────


def process_patient(patient_id: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Steps 1–5 for one patient.

    Returns
    -------
    windows : (n_windows, 3, 300), float32, z-score normalized
    labels  : (n_windows,), int64, binary
    """
    data, native_sfreq, _ = load_edf(patient_id)
    data_10hz = resample_to_10hz(data, native_sfreq)
    events = parse_respevt(patient_id)
    windows, labels = make_windows_and_labels(data_10hz, events)
    windows_norm = normalize(windows, labels)
    return windows_norm, labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — 1D CNN Model Definition
# ─────────────────────────────────────────────────────────────────────────────


class ApneaCNN(nn.Module):
    """
    1D Convolutional Neural Network for binary apnea classification.

    Input  : (batch, C, 300*N)  — C channels × (300 × CONTEXT_WINDOWS) timesteps
             With CONTEXT_WINDOWS=1  → (batch, C, 300)   — 30-second window
             With CONTEXT_WINDOWS=5  → (batch, C, 1500)  — 2.5-minute window
    Output : (batch, 1)             — P(apnea) via sigmoid

    AdaptiveAvgPool1d(1) collapses ANY time length to a fixed 256-dim vector,
    so this architecture requires NO changes for different CONTEXT_WINDOWS values.

    Architecture rationale:
    - Conv1D sees 300 timesteps as the sequence; 3 channels as input depth
    - Filters increase (64→128→256): each layer learns richer abstractions
    - Kernel size decreases (7→5→3): early layers need wide receptive fields
      to capture full breathing cycles (~3-5s); later layers detect fine features
    - GlobalAveragePooling collapses the time dimension — no fixed-size assumption
    - Dropout(0.3) before Dense to prevent overfitting on small dataset
    """

    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1: broad temporal patterns (breathing rhythm ~3-5 seconds)
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Block 2: intermediate features (onset/offset of events)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Block 3: fine-grained abstract features
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # GlobalAveragePooling over the time dimension → collapses (B, 256, 300) to (B, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # raw logit — sigmoid applied in loss and at inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 14, 300)
        x = self.conv_block(x)  # → (B, 256, 300)
        x = self.gap(x).squeeze(-1)  # → (B, 256)
        x = self.classifier(x)  # → (B, 1)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — PyTorch Dataset, DataLoader, and training loop
# ─────────────────────────────────────────────────────────────────────────────

class ApneaDataset(Dataset):
    """Wraps numpy windows + labels arrays into a PyTorch Dataset."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        # Convert to tensors once at init time (not per __getitem__ call)
        self.X = torch.from_numpy(windows).float()   # (N, 3, 300)
        self.y = torch.from_numpy(labels).float()    # (N,) — float for BCEWithLogitsLoss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataset(patient_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full pipeline for a list of patients and concatenate results.
    Prints a one-line progress update per patient.

    Channel-count guard: the first successfully processed patient sets the
    expected channel count. Any subsequent patient whose EDF contains a
    different number of channels is skipped rather than crashing np.concatenate
    with a dimension-1 mismatch (UCDDB files vary between 14 and 16 channels).
    """
    all_windows: list[np.ndarray] = []
    all_labels:  list[np.ndarray] = []
    expected_channels: int | None = None   # locked from first good patient

    for pid in patient_ids:
        print(f"  Processing {pid} ...", end=" ", flush=True)
        try:
            wins, labs = process_patient(pid)
            n_ch = wins.shape[1]

            # Lock channel count on first success
            if expected_channels is None:
                expected_channels = n_ch

            # Skip patients whose EDF has a different channel layout
            if n_ch != expected_channels:
                print(f"SKIPPED — channel mismatch ({n_ch} ch, expected {expected_channels} ch)")
                continue

            n_apnea = int(labs.sum())
            apnea_percentage = 100 * (n_apnea / len(labs))

            # Check condition (5.0 = 5%)
            if apnea_percentage < 5.0:
                print(f"SKIPPED — near normal patient ({apnea_percentage:.1f}%)")
                continue

            # ── Temporal stacking — MUST happen per-patient ────────────────────
            # stack_windows() groups consecutive windows WITHIN a single patient.
            # If we concatenated all patients first and stacked afterwards, groups
            # would silently bridge the boundary between two different patients,
            # leaking signal from patient A's last windows into patient B's first
            # windows. Stacking here prevents that contamination entirely.
            wins_stacked, labs_stacked = stack_windows(wins, labs, n=CONTEXT_WINDOWS)

            all_windows.append(wins_stacked)
            all_labels.append(labs_stacked)

            n_apnea_stacked = int(labs_stacked.sum())
            print(
                f"{len(labs)} raw → {len(labs_stacked)} stacked windows  "
                f"| {n_apnea_stacked} apnea ({100*n_apnea_stacked/len(labs_stacked):.1f}%)"
            )

        except Exception as e:
            print(f"FAILED — {e}")

    windows = np.concatenate(all_windows, axis=0)
    labels  = np.concatenate(all_labels,  axis=0)
    return windows, labels


def train(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    epoch:      int,
) -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)  # (B,1) → (B,)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    mean_loss = total_loss / len(loader.dataset)
    print(f"  Epoch {epoch:02d} | train loss: {mean_loss:.4f}")
    return mean_loss

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Evaluation: accuracy, macro F1, raw confusion matrix
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Evaluate the model on a DataLoader and print:
      - Overall accuracy
      - Macro F1 score  (the number that matters for v1)
      - Raw-count confusion matrix

    Threshold: sigmoid(logit) >= 0.5 → predicted apnea (label=1)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.long().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {acc:.4f}  ({100 * acc:.1f}%)")
    print(
        f"  Macro F1  : {macro_f1:.4f}  {'✓ PASS (>0.5)' if macro_f1 > 0.5 else '✗ FAIL — check data pipeline'}"
    )
    print(f"\n  Confusion Matrix (raw counts):")
    print(f"                     Predicted Normal  Predicted Apnea")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  Actually Normal  |     TN = {tn:<8d}  |  FP = {fp:<8d}|")
        print(f"  Actually Apnea   |     FN = {fn:<8d}  |  TP = {tp:<8d}|")
        print(f"\n  Apnea recall (sensitivity): {tp / (tp + fn + 1e-8):.3f}")
        print(f"  Apnea precision           : {tp / (tp + fp + 1e-8):.3f}")
    else:
        print(cm)
    print(f"{'=' * 60}\n")


def evaluate_threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Sweeps classification decision thresholds from 0.10 to 0.85 to calculate
    macro F1, sensitivity (recall), and precision across candidate thresholds.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device)).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Sweep thresholds
    print(f"\n{'=' * 60}")
    print(f"  THRESHOLD SWEEP EVALUATION")
    print(f"{'=' * 60}")
    thresholds = np.arange(0.1, 0.9, 0.05)
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        sens = recall_score(all_labels, preds, pos_label=1, zero_division=0)
        prec = precision_score(all_labels, preds, pos_label=1, zero_division=0)
        print(f"t={t:.2f}  macro_F1={f1:.3f}  sens={sens:.3f}  prec={prec:.3f}")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── STEP 6: Gate check ──────────────────────────────────────────────────
    # Run this first. Do not skip. Read the output carefully.
    print("\n[STEP 6] Running diagnostic gate check on ucddb002 ...")
    run_gate_check("ucddb002")

    # ── STEP 8a: Build train/test datasets ──────────────────────────────────
    print("[STEP 8] Building TRAIN dataset ...")
    X_train, y_train = build_dataset(TRAIN_PATIENTS)

    print("\n[STEP 8] Building TEST dataset ...")
    X_test, y_test = build_dataset(TEST_PATIENTS)

    print(
        f"\n  Train: {len(y_train)} windows  ({int(y_train.sum())} apnea, {int((y_train == 0).sum())} normal)"
    )
    print(
        f"  Test : {len(y_test)}  windows  ({int(y_test.sum())}  apnea, {int((y_test == 0).sum())}  normal)"
    )

    # ── Compute class weight for imbalanced labels ──────────────────────────
    # Weight apnea windows more heavily so the model is penalized for missing them.
    # Raw ratio can exceed 15x when pooling many near-normal patients together
    # (e.g. ucddb018 at 0.9% apnea). At that level the model overcorrects and
    # floods the output with false positives. Cap at MAX_POS_WEIGHT.
    MAX_POS_WEIGHT = 10.0
    n_normal_train = int((y_train == 0).sum())
    n_apnea_train = int(y_train.sum())
    raw_weight = n_normal_train / (n_apnea_train + 1e-8)
    weight_apnea = min(raw_weight, MAX_POS_WEIGHT)
    print(f"\n  Raw class ratio       : {raw_weight:.2f}x")
    print(
        f"  Class weight applied  : {weight_apnea:.2f}x  (capped at {MAX_POS_WEIGHT}x)"
    )

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_dataset = ApneaDataset(X_train, y_train)
    test_dataset = ApneaDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── STEP 7: Instantiate model ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[STEP 7] Model device: {device}")
    model = ApneaCNN().to(device)

    # BCEWithLogitsLoss = sigmoid + BCE in one numerically stable operation
    # pos_weight tells it to penalize false negatives on the apnea class
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight_apnea], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    # ── STEP 8b: Training loop ───────────────────────────────────────────────
    print(f"\n[STEP 8] Training for {NUM_EPOCHS} epochs ...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)

    # ── STEP 9: Evaluate on held-out test patients ───────────────────────────
    print("\n[STEP 9] Evaluating on test patients ...")
    evaluate(model, test_loader, device)
    evaluate_threshold_sweep(model, test_loader, device)
