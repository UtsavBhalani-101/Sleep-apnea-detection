"""
Model definitions.

ApneaCNNLSTM
------------
    Input : (batch, SEQ_LEN, C, SAMPLES_PER_WIN)
    Output: (batch, SEQ_LEN)  — one logit per timestep

A CNN encodes each 30-second window independently to a fixed-size
embedding. A bidirectional LSTM then runs across the SEQ_LEN embeddings
to inject temporal context across consecutive windows.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config


class ApneaCNNLSTM(nn.Module):
    """CNN encoder per-window + BiLSTM across windows."""

    def __init__(
        self,
        in_channels: int = len(config.EDF_CHANNELS),
        cnn_out_dim: int = 128,
        lstm_hidden: int = 64,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        conv_out_channels = 128
        pooled_size = 8

        # CNN encoder: one window (C, 300) → embedding (conv_out_channels, pooled_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # → (32, 150)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # → (64, 75)

            nn.Conv1d(64, conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pooled_size),  # → (conv_out_channels, pooled_size)
        )

        # Flatten (conv_out_channels * pooled_size) features and project to embedding dim
        self.fc_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_channels * pooled_size, cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM over SEQ_LEN window embeddings
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C, T)
        B, S, C, T = x.shape
        x = x.view(B * S, C, T)

        feat = self.cnn(x)          # (B*S, cnn_out_dim, 1)
        feat = self.fc_proj(feat)   # (B*S, cnn_out_dim)
        feat = feat.view(B, S, -1)  # (B, S, cnn_out_dim)

        out, _ = self.lstm(feat)        # (B, S, lstm_hidden * 2)
        logits = self.classifier(out)   # (B, S, 1)
        return logits.squeeze(-1)       # (B, S)
