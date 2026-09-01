"""
Sleep Apnea Detection Pipeline (refactored).

Modules:
    config          : paths, patient splits, signal + training hyperparameters
    datasets_spec   : per-dataset specs (UCDDB, SHHS, MESA) + NSRR XML parser
    loader          : EDF loading, resampling, annotation parsing, gate check
    windows         : windowing, labeling, z-score normalization
    dataset         : PyTorch Sequence Dataset + per-patient dataset builder
    models          : ApneaCNNLSTM model definition
    train           : train / validate loops
    metrics         : evaluate, evaluate_threshold_sweep
    orchestrator    : end-to-end entry point
    preprocess_dataset : bulk-preprocess a dataset to .npy caches
    sanity          : smoke checks
"""

from . import (
    config,
    datasets_spec,
    loader,
    windows,
    dataset,
    models,
    train,
    metrics,
    orchestrator,
    preprocess_dataset,
    sanity,
)

__all__ = [
    "config",
    "datasets_spec",
    "loader",
    "windows",
    "dataset",
    "models",
    "train",
    "metrics",
    "orchestrator",
    "preprocess_dataset",
    "sanity",
]
