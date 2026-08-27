"""
residual_lstm.py

The learned correction component of the hybrid digital twin. Given a window
of recent exogenous inputs and the nominal RC-model's own prediction, this
LSTM predicts the *residual* (ground_truth - physics_prediction) at the next
step. The hybrid twin's final prediction is:

    T_in_hybrid = T_in_physics + residual_lstm(window)

This is the standard grey-box "physics + residual ML" pattern: the physics
model carries the interpretable, extrapolation-safe backbone; the ML layer
mops up whatever systematic and stochastic effects (solar gain, sensor
noise, model-parameter mismatch) the physics model can't represent.

Also included: a pure-ML baseline (`DirectLSTM`) that predicts T_in directly
from the same features with no physics input at all, used in evaluation to
show the hybrid approach isn't just "physics does all the work" or "ML does
all the work" -- both baselines are needed for an honest comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


FEATURE_NAMES = ["T_out", "hour_sin", "hour_cos", "Q_hvac_norm", "Q_gain_norm", "T_in_physics"]


def _build_features(ep: dict) -> np.ndarray:
    hour = ep["hour"]
    feats = np.stack(
        [
            ep["T_out"],
            np.sin(hour / 24 * 2 * np.pi),
            np.cos(hour / 24 * 2 * np.pi),
            ep["Q_hvac"] / 1500.0,
            ep["Q_gain_base"] / 400.0,
            ep["T_in_physics"],
        ],
        axis=1,
    )
    return feats.astype(np.float32)


class ResidualWindowDataset(Dataset):
    """Sliding-window dataset: given `window` past steps of features, predict
    the residual at the *next* step."""

    def __init__(self, episodes: list[dict], window: int = 8, target: str = "residual"):
        self.window = window
        self.samples_X = []
        self.samples_y = []
        for ep in episodes:
            feats = _build_features(ep)
            if target == "residual":
                y = ep["residual"]
            elif target == "T_in_true":
                y = ep["T_in_true"]
            else:
                raise ValueError(target)

            n = len(y)
            for t in range(window, n):
                self.samples_X.append(feats[t - window : t])
                self.samples_y.append(y[t])

        self.samples_X = np.stack(self.samples_X)
        self.samples_y = np.array(self.samples_y, dtype=np.float32)

    def __len__(self):
        return len(self.samples_y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples_X[idx]), torch.tensor(self.samples_y[idx])


class ResidualLSTM(nn.Module):
    """Predicts the physics-model residual from a window of features."""

    def __init__(self, n_features: int = 6, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):  # x: (batch, window, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class DirectLSTM(nn.Module):
    """Pure-ML baseline: predicts T_in directly from the same features, with
    no physics prior at all. Same architecture as ResidualLSTM so the
    comparison isolates the effect of the physics prior, not model capacity."""

    def __init__(self, n_features: int = 6, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = True,
):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X, y in train_loader:
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                val_losses.append(loss_fn(pred, y).item())

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        if verbose:
            print(
                f"epoch {epoch+1:2d}/{epochs}  train_mse={history['train_loss'][-1]:.4f}  "
                f"val_mse={history['val_loss'][-1]:.4f}"
            )

    return history
