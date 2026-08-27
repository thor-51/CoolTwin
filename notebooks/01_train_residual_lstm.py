"""
01_train_residual_lstm.py

Phase 2 deliverable: train the residual LSTM, and produce the core Phase-2
result -- an RMSE comparison table showing hybrid (physics + residual ML)
beats both physics-only and pure-ML baselines on held-out episodes.

Usage:
    PYTHONPATH=. python notebooks/01_train_residual_lstm.py
"""

from __future__ import annotations

import numpy as np
import torch

from twin.data_gen import generate_dataset
from twin.residual_lstm import (
    ResidualLSTM,
    DirectLSTM,
    ResidualWindowDataset,
    train_model,
)
from twin.hybrid_twin import HybridTwin


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    torch.manual_seed(0)

    print("Generating synthetic dataset (train/val/test episodes)...")
    train_episodes = generate_dataset(n_episodes=24, n_steps=672, seed=1)
    val_episodes = generate_dataset(n_episodes=6, n_steps=672, seed=2)
    test_episodes = generate_dataset(n_episodes=6, n_steps=672, seed=3)

    window = 8

    # --- train the residual model ---
    print("\nTraining ResidualLSTM (predicts physics-model error)...")
    train_ds_res = ResidualWindowDataset(train_episodes, window=window, target="residual")
    val_ds_res = ResidualWindowDataset(val_episodes, window=window, target="residual")
    residual_model = ResidualLSTM(n_features=6, hidden_size=32)
    train_model(residual_model, train_ds_res, val_ds_res, epochs=15)

    # --- train the pure-ML baseline (same features, predicts T_in directly) ---
    print("\nTraining DirectLSTM (pure-ML baseline, no physics prior)...")
    train_ds_direct = ResidualWindowDataset(train_episodes, window=window, target="T_in_true")
    val_ds_direct = ResidualWindowDataset(val_episodes, window=window, target="T_in_true")
    direct_model = DirectLSTM(n_features=6, hidden_size=32)
    train_model(direct_model, train_ds_direct, val_ds_direct, epochs=15)

    # --- evaluate all three approaches on held-out test episodes ---
    print("\nEvaluating on held-out test episodes...")
    hybrid_twin = HybridTwin(residual_model, window=window)

    physics_rmses, direct_rmses, hybrid_rmses = [], [], []

    direct_model.eval()
    for ep in test_episodes:
        T_true = ep["T_in_true"]

        # physics-only
        physics_rmses.append(rmse(ep["T_in_physics"][window:], T_true[window:]))

        # hybrid
        hybrid_pred = hybrid_twin.correct_episode(ep)
        hybrid_rmses.append(rmse(hybrid_pred[window:], T_true[window:]))

        # pure-ML (DirectLSTM)
        from twin.residual_lstm import _build_features

        feats = _build_features(ep)
        direct_preds = []
        with torch.no_grad():
            for t in range(window, len(T_true)):
                x = torch.from_numpy(feats[t - window : t]).unsqueeze(0)
                direct_preds.append(direct_model(x).item())
        direct_rmses.append(rmse(np.array(direct_preds), T_true[window:]))

    print("\n" + "=" * 55)
    print("Phase 2 result: RMSE vs ground truth (held-out test episodes)")
    print("=" * 55)
    print(f"{'Approach':<25}{'Mean RMSE (C)':>15}{'Std':>12}")
    print(f"{'Physics-only (RC net)':<25}{np.mean(physics_rmses):>15.3f}{np.std(physics_rmses):>12.3f}")
    print(f"{'Pure-ML (DirectLSTM)':<25}{np.mean(direct_rmses):>15.3f}{np.std(direct_rmses):>12.3f}")
    print(f"{'Hybrid (RC + Residual)':<25}{np.mean(hybrid_rmses):>15.3f}{np.std(hybrid_rmses):>12.3f}")
    print("=" * 55)

    improvement_vs_physics = (1 - np.mean(hybrid_rmses) / np.mean(physics_rmses)) * 100
    print(f"\nHybrid improves on physics-only by {improvement_vs_physics:.1f}%")

    return {
        "physics_rmse": np.mean(physics_rmses),
        "direct_rmse": np.mean(direct_rmses),
        "hybrid_rmse": np.mean(hybrid_rmses),
    }


if __name__ == "__main__":
    main()
