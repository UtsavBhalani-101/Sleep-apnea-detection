"""
W&B integration for the sleep-apnea pipeline.

Provides a thin wrapper around the wandb SDK that:
  * Initializes a single run per orchestrator invocation.
  * Streams per-epoch train/val losses.
  * Logs final evaluation metrics, threshold-sweep table, and confusion matrix.
  * Captures config from `pipeline.config`.

Design goals
------------
* Zero overhead when wandb isn't installed / disabled. Every public function
  short-circuits when the module is in disabled mode.
* Disabled by default if `WANDB_DISABLED=true` (the official wandb env var) so
  CI runs don't try to upload anything.
* Mirror the metric names parsed by `upload_to_wandb.py` so the dashboard
  grouping/comparison works across old (markdown-uploaded) and new (live)
  runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.is_file():
        load_dotenv(_ENV_PATH, override=False)
except ImportError:  # pragma: no cover
    pass

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]


WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "false").lower() in {"1", "true", "yes"}
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sleep-apnea-experiments")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None


class WandbLogger:
    """Context-manager-style wrapper around a single wandb run.

    Usage
    -----
    >>> logger = WandbLogger(exp_id="EXP_011", config={"lr": 1e-3})
    >>> for epoch in range(30):
    ...     logger.log_epoch(epoch + 1, train_loss=..., val_loss=...)
    >>> logger.log_final(metrics={...}, sweep=[...], cm=np.array([[...]]))
    >>> logger.finish()
    """

    def __init__(
        self,
        exp_id: str,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and not WANDB_DISABLED and wandb is not None
        self._run = None

        if not self.enabled:
            print("[wandb] logging disabled (set WANDB_DISABLED=false to enable)")
            return

        init_kwargs: dict[str, Any] = {
            "project": WANDB_PROJECT,
            "entity": WANDB_ENTITY,
            "name": run_name or exp_id,
            "config": {"experiment_id": exp_id, **(config or {})},
        }
        if notes:
            init_kwargs["notes"] = notes
        if tags:
            init_kwargs["tags"] = tags

        self._run = wandb.init(**init_kwargs)
        print(f"[wandb] run initialized: {wandb.run.get_url() if wandb.run else '<offline>'}")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float | None = None) -> None:
        """Log one epoch's loss. No-op when disabled."""
        if not self.enabled:
            return
        payload: dict[str, Any] = {"train_loss": train_loss}
        if val_loss is not None:
            payload["val_loss"] = val_loss
        wandb.log(payload, step=epoch)

    def log_epoch_curve(
        self,
        train_losses: list[float],
        val_losses: list[float] | None = None,
        title: str = "Train / Val Loss vs Epoch",
    ) -> None:
        """Log the full per-epoch loss curve as a named line chart.

        Replaces the per-step scalars with one consolidated chart in the
        Charts panel so it's easy to eyeball overfitting (e.g. EXP_010's
        train loss falling while val loss climbs).
        """
        if not self.enabled or not train_losses:
            return
        epochs = list(range(1, len(train_losses) + 1))
        series: list[tuple[str, list[float]]] = [("train_loss", list(train_losses))]
        if val_losses and len(val_losses) == len(train_losses):
            series.append(("val_loss", list(val_losses)))
        wandb.log(
            {
                "loss_curve": wandb.plot.line_series(
                    xs=epochs,
                    ys=[vals for _, vals in series],
                    keys=[name for name, _ in series],
                    title=title,
                )
            }
        )

    def log_threshold_curve(
        self,
        sweep: list[dict[str, float]],
        title: str = "Precision / Recall / Macro-F1 vs Decision Threshold",
    ) -> None:
        """Log a 3-line chart: precision, recall, macro-F1 vs threshold.

        Expects each row in ``sweep`` to contain keys ``threshold``,
        ``precision``, ``sensitivity``, ``macro_f1``.
        """
        if not self.enabled or not sweep:
            return
        thresholds = [float(r["threshold"]) for r in sweep]
        precision = [float(r["precision"]) for r in sweep]
        recall = [float(r["sensitivity"]) for r in sweep]
        macro_f1 = [float(r["macro_f1"]) for r in sweep]
        wandb.log(
            {
                "threshold_curve": wandb.plot.line_series(
                    xs=thresholds,
                    ys=[precision, recall, macro_f1],
                    keys=["precision", "recall", "macro_f1"],
                    title=title,
                )
            }
        )

    # ------------------------------------------------------------------ #
    # Final evaluation
    # ------------------------------------------------------------------ #
    def log_final(
        self,
        optimal_threshold: float | None = None,
        accuracy: float | None = None,
        macro_f1: float | None = None,
        sensitivity: float | None = None,
        precision: float | None = None,
        tn: int | None = None,
        fp: int | None = None,
        fn: int | None = None,
        tp: int | None = None,
        threshold_sweep: list[dict[str, float]] | None = None,
        confusion_matrix: np.ndarray | None = None,
        class_names: tuple[str, str] = ("Normal", "Apnea"),
    ) -> None:
        """Log final test-set metrics. Safe to call with any subset of fields."""
        if not self.enabled:
            return

        # 1) Scalar summary
        summary: dict[str, Any] = {
            k: v
            for k, v in {
                "optimal_threshold": optimal_threshold,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "sensitivity": sensitivity,
                "precision": precision,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }.items()
            if v is not None
        }
        if summary:
            wandb.log(summary)

        if macro_f1 is not None and macro_f1 > 0.5:
            wandb.log({"passes_macro_f1_0.5": 1})

        # 2) Threshold sweep as a Table + per-step series + named line chart
        if threshold_sweep:
            table = wandb.Table(
                columns=["threshold", "macro_f1", "sensitivity", "precision"],
                data=[
                    [row["threshold"], row["macro_f1"], row["sensitivity"], row["precision"]]
                    for row in threshold_sweep
                ],
            )
            wandb.log({"threshold_sweep": table})

            # Named line chart with all three metrics on one plot so it's easy
            # to see how precision / recall / macro-F1 trade off vs threshold.
            self.log_threshold_curve(threshold_sweep)

            for row in threshold_sweep:
                wandb.log(
                    {
                        "threshold_sweep/macro_f1": row["macro_f1"],
                        "threshold_sweep/sensitivity": row["sensitivity"],
                        "threshold_sweep/precision": row["precision"],
                    },
                    step=int(row["threshold"] * 100),
                )

        # 3) Confusion matrix
        if confusion_matrix is not None and confusion_matrix.size == 4:
            tn_v, fp_v, fn_v, tp_v = confusion_matrix.ravel().tolist()
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=[0] * int(tn_v + fn_v) + [1] * int(fp_v + tp_v),
                        preds=[0] * int(tn_v) + [1] * int(fn_v) + [0] * int(fp_v) + [1] * int(tp_v),
                        class_names=list(class_names),
                    )
                }
            )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def watch_model(self, model) -> None:
        """Optional: track gradients/parameters. Skip if disabled."""
        if not self.enabled:
            return
        wandb.watch(model, log="gradients", log_freq=100)

    def finish(self) -> None:
        if not self.enabled:
            return
        wandb.finish()
        print("[wandb] run finished")
