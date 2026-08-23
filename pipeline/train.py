"""
Training + validation loops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """One training epoch. Returns mean loss over the loader."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)  # (B, SEQ_LEN)

        optimizer.zero_grad()
        logits = model(X_batch)        # (B, SEQ_LEN)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    mean_loss = total_loss / len(loader.dataset)
    print(f"  Epoch {epoch:02d} | train loss: {mean_loss:.4f}")
    return mean_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute mean validation loss for one pass over the loader."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.numel()
            n_samples += y_batch.numel()

    mean_loss = total_loss / max(n_samples, 1)
    model.train()
    return mean_loss


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
) -> tuple[list[float], list[float]]:
    """
    Run a full training schedule.

    Returns
    -------
    train_losses, val_losses : per-epoch loss lists
    """
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss = validate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses
