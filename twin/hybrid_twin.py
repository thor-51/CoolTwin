"""
hybrid_twin.py

Combines the RC physics model with a trained ResidualLSTM to produce the
final hybrid prediction:

    T_in_hybrid[t] = T_in_physics[t] + ResidualLSTM(window of features up to t)

This is what Phase 3 (RL environment) will eventually use in place of the
physics-only prediction inside CoolTwinEnv, once the model is trained and
validated (see notebooks/01_train_residual_lstm.py for training +
evaluation).
"""

from __future__ import annotations

import numpy as np
import torch

from twin.residual_lstm import ResidualLSTM, _build_features


class HybridTwin:
    """Wraps a trained ResidualLSTM to correct physics-model predictions
    online, given a rolling window of recent features."""

    def __init__(self, model: ResidualLSTM, window: int = 8):
        self.model = model
        self.model.eval()
        self.window = window

    def correct_episode(self, ep: dict) -> np.ndarray:
        """Given a full episode dict (as produced by twin/data_gen.py),
        returns the hybrid-corrected T_in prediction for every step where a
        full window is available (first `window` steps fall back to the raw
        physics prediction, since there isn't enough history yet)."""
        feats = _build_features(ep)
        n = len(ep["T_in_physics"])
        hybrid = ep["T_in_physics"].copy()

        with torch.no_grad():
            for t in range(self.window, n):
                x = torch.from_numpy(feats[t - self.window : t]).unsqueeze(0)
                residual_pred = self.model(x).item()
                hybrid[t] = ep["T_in_physics"][t] + residual_pred

        return hybrid
