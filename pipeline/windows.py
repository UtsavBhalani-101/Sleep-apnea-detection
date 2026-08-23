"""
Windowing, binary labeling, and per-subject z-score normalization.
"""

from __future__ import annotations

import numpy as np

from . import config

# SpO2 fixed min-max scaling parameters (channel index 0).
SPO2_MIN = 70.0
SPO2_RANGE = 30.0


def make_windows_and_labels(
    data: np.ndarray,
    events: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice resampled data into non-overlapping WINDOW_SECONDS windows and label each.

    Label = 1 if ANY apnea/hypopnea event overlaps the window by ≥
    OVERLAP_THRESHOLD_SECS. Label = 0 otherwise.

    Parameters
    ----------
    data   : (n_channels, total_samples) at TARGET_SFREQ
    events : list of (onset_sec, duration_sec)

    Returns
    -------
    windows : (n_windows, n_channels, SAMPLES_PER_WIN)
    labels  : (n_windows,), int64
    """
    n_channels, total_samples = data.shape
    n_windows = total_samples // config.SAMPLES_PER_WIN

    trimmed = data[:, : n_windows * config.SAMPLES_PER_WIN]
    windows = trimmed.reshape(
        n_channels, n_windows, config.SAMPLES_PER_WIN
    ).swapaxes(0, 1)

    labels = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        w_start = i * config.WINDOW_SECONDS
        w_end = w_start + config.WINDOW_SECONDS

        for onset_sec, duration_sec in events:
            event_end = onset_sec + duration_sec
            overlap = min(w_end, event_end) - max(w_start, onset_sec)
            overlap = max(0.0, overlap)
            if overlap >= config.OVERLAP_THRESHOLD_SECS:
                labels[i] = 1
                break

    return windows, labels


def normalize(windows: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Per-subject normalization. Statistics come ONLY from label=0 windows so apnea
    signal does not contaminate the baseline.

      Channel 0 (SpO2): linear scale into roughly [0, 1] (no clipping).
      Other channels  : z-score using normal-window stats, then clip to
                        ±CLIP_SIGMA to suppress artifact spikes.
    """
    normal_mask = labels == 0
    normal_wins = windows[normal_mask]

    normalized = windows.astype(np.float32).copy()

    for ch in range(windows.shape[1]):
        if ch == 0:
            normalized[:, ch, :] = (windows[:, ch, :] - SPO2_MIN) / SPO2_RANGE
        else:
            ch_samples = normal_wins[:, ch, :].flatten()
            ch_mean = ch_samples.mean()
            ch_std = ch_samples.std()
            normalized[:, ch, :] = np.clip(
                (windows[:, ch, :] - ch_mean) / (ch_std + 1e-8),
                -config.CLIP_SIGMA,
                config.CLIP_SIGMA,
            )
    return normalized
