"""
ensembles.py

Deep Ensembles (Lakshminarayanan et al., 2017): train N independently
initialized ResidualLSTM models on the same data, and use the spread of
their predictions as the uncertainty estimate. Generally gives better
calibrated uncertainty than MC Dropout at the cost of N x training/storage
-- Phase 4 trains both so the report can honestly compare them (see
notebooks/03_uncertainty_quantification.py).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from twin.residual_lstm import ResidualLSTM, train_model


class DeepEnsemble:
    def __init__(self, n_models: int = 5, n_features: int = 6, hidden_size: int = 32):
        self.n_models = n_models
        self.models = [ResidualLSTM(n_features=n_features, hidden_size=hidden_size) for _ in range(n_models)]

    def fit(self, train_ds: Dataset, val_ds: Dataset, epochs: int = 15, verbose: bool = False):
        for i, model in enumerate(self.models):
            torch.manual_seed(1000 + i)  # different init per ensemble member
            if verbose:
                print(f"Training ensemble member {i+1}/{self.n_models}...")
            train_model(model, train_ds, val_ds, epochs=epochs, verbose=False)

    def predict_with_uncertainty(self, x: torch.Tensor):
        """x: (batch, window, n_features). Returns (mean, std) across
        ensemble members, each (batch,)."""
        preds = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                preds.append(model(x))
        preds = torch.stack(preds, dim=0)  # (n_models, batch)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean.numpy(), std.numpy()
