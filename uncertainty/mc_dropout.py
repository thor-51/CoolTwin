"""
mc_dropout.py

Monte Carlo Dropout uncertainty quantification (Gal & Ghahramani, 2016):
keep dropout active at inference time and run N stochastic forward passes.
The spread across those passes approximates the model's predictive
uncertainty -- cheap (no extra models to train or store) and a reasonable
first uncertainty signal for the hybrid twin's residual correction.

This is one of two independent UQ methods used in CoolTwin (see
ensembles.py for the second, Deep Ensembles) -- Phase 4 of the roadmap
explicitly compares both rather than committing to just one.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MCDropoutResidualLSTM(nn.Module):
    """Same architecture role as twin.residual_lstm.ResidualLSTM, but with
    dropout applied both between LSTM layers and in the head, so it can be
    used for MC Dropout at inference time. A separate class (rather than
    just adding dropout to ResidualLSTM) keeps the Phase 2 model's exact
    behavior untouched for reproducibility of the Phase 2 result."""

    def __init__(self, n_features: int = 6, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout_p = dropout
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def predict_with_uncertainty(model: MCDropoutResidualLSTM, x: torch.Tensor, n_samples: int = 30):
    """Runs n_samples stochastic forward passes with dropout left ACTIVE
    (model.train() mode, but gradients disabled) and returns (mean, std)
    of the predictive distribution for each input in the batch.

    x: (batch, window, n_features)
    returns: mean (batch,), std (batch,)
    """
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x))
    preds = torch.stack(preds, dim=0)  # (n_samples, batch)
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    model.eval()
    return mean.numpy(), std.numpy()
