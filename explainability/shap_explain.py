"""
shap_explain.py

SHAP-based feature attribution for two different models in the pipeline:

  1. The residual model (twin/residual_lstm.py) -- "why did the twin correct
     the physics prediction the way it did?"
  2. The RL policy -- "why did the agent take this action?"

Both models take structured, low-dimensional inputs (a short feature window
for the residual model; a 7-dim observation for the policy), so a
model-agnostic SHAP KernelExplainer is used rather than a model-specific
explainer (DeepExplainer support for LSTMs is inconsistent across SHAP
versions, and KernelExplainer keeps this code robust to swapping algorithms
later, at the cost of being slower -- fine given how small these inputs are).
"""

from __future__ import annotations

import numpy as np
import torch
import shap


def _flatten_window(window: np.ndarray) -> np.ndarray:
    """(window, n_features) -> (window * n_features,) for SHAP, which
    expects flat feature vectors."""
    return window.reshape(-1)


def _unflatten_window(flat: np.ndarray, window_len: int, n_features: int) -> np.ndarray:
    return flat.reshape(window_len, n_features)


def explain_residual_model(
    model,
    background_windows: np.ndarray,   # (n_background, window, n_features)
    instance_window: np.ndarray,      # (window, n_features)
    feature_names: list[str],
    n_samples: int = 100,
):
    """Returns a dict {feature_name: shap_value} attributing the residual
    model's prediction for `instance_window` to each (timestep, feature)
    input. Feature names are repeated per-timestep since the model sees a
    window of history, not a single step -- e.g. 'T_out (t-3)'.
    """
    window_len, n_features = instance_window.shape
    background_flat = np.stack([_flatten_window(w) for w in background_windows])

    def predict_fn(flat_batch: np.ndarray) -> np.ndarray:
        windows = np.stack([_unflatten_window(f, window_len, n_features) for f in flat_batch])
        with torch.no_grad():
            x = torch.from_numpy(windows.astype(np.float32))
            model.eval()
            preds = model(x)
        return preds.numpy()

    explainer = shap.KernelExplainer(predict_fn, background_flat, silent=True)
    instance_flat = _flatten_window(instance_window).reshape(1, -1)
    shap_values = explainer.shap_values(instance_flat, nsamples=n_samples, silent=True)
    shap_values = np.array(shap_values).reshape(window_len, n_features)

    result = {}
    for t in range(window_len):
        lag = window_len - 1 - t  # t=window_len-1 is the most recent step (lag 0)
        for f, fname in enumerate(feature_names):
            result[f"{fname} (t-{lag})"] = float(shap_values[t, f])

    return result


def explain_policy(
    predict_fn,               # callable(obs_batch: np.ndarray) -> action array (n, 1)
    background_obs: np.ndarray,  # (n_background, obs_dim)
    instance_obs: np.ndarray,    # (obs_dim,)
    feature_names: list[str],
    n_samples: int = 100,
):
    """Returns a dict {feature_name: shap_value} attributing the RL policy's
    action for `instance_obs` to each observation dimension."""

    def wrapped(obs_batch: np.ndarray) -> np.ndarray:
        return predict_fn(obs_batch).reshape(-1)

    explainer = shap.KernelExplainer(wrapped, background_obs, silent=True)
    shap_values = explainer.shap_values(instance_obs.reshape(1, -1), nsamples=n_samples, silent=True)
    shap_values = np.array(shap_values).reshape(-1)

    return {name: float(val) for name, val in zip(feature_names, shap_values)}


def top_k_features(shap_dict: dict, k: int = 3) -> list[tuple[str, float]]:
    """Returns the k features with the largest absolute SHAP value, sorted
    by magnitude -- used to build the LLM explanation context without
    dumping every feature into the prompt."""
    return sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
