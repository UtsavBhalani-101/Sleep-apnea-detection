"""
Windowing, binary labeling, and per-subject z-score normalization.
"""

from __future__ import annotations

import numpy as np

from . import config

# SpO2 fixed min-max scaling parameters (channel identified by name).
SPO2_MIN = 70.0
SPO2_RANGE = 30.0
SPO2_CHANNEL_NAME = "SpO2"


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


def normalize(
    windows: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str] | None = None,
) -> np.ndarray:
    """
    Per-subject normalization. Statistics come ONLY from label=0 windows so apnea
    signal does not contaminate the baseline.

      SpO2 channel (looked up by name): linear scale into roughly [0, 1],
                                        then clip to [0, 1] to suppress
                                        sensor-detach artifacts.
      Other channels: z-score using normal-window stats, then clip to
                      ±CLIP_SIGMA to suppress artifact spikes.

    Parameters
    ----------
    windows       : (n_windows, n_channels, SAMPLES_PER_WIN)
    labels        : (n_windows,)
    channel_names : list of channel names in the same order as windows' axis=1.
                    Defaults to config.EDF_CHANNELS. Required if you want
                    SpO2 to be identified by name rather than index 0.
    """
    if channel_names is None:
        channel_names = config.EDF_CHANNELS

    normal_mask = labels == 0
    normal_wins = windows[normal_mask]

    normalized = windows.astype(np.float32).copy()

    for ch in range(windows.shape[1]):
        if channel_names[ch] == SPO2_CHANNEL_NAME:
            normalized[:, ch, :] = np.clip(
                (windows[:, ch, :] - SPO2_MIN) / SPO2_RANGE,
                0.0,
                1.0,
            )
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
