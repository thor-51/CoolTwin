"""
safety_layer.py

Wraps a trained RL policy with an uncertainty-gated fallback: if the hybrid
twin's residual model reports high predictive uncertainty for the current
state (i.e. the twin doesn't trust its own temperature correction), control
falls back to the conservative rule-based thermostat instead of trusting
the RL policy's action.

This directly answers "how do you know your RL agent is safe to deploy?" --
the honest answer is: it isn't, unconditionally, but it's coupled to a
cheap uncertainty signal that catches out-of-distribution states and
defers to a known-safe controller.

Note: uncertainty here is computed on the residual model's prediction over
the recent feature window, not on the RL policy's action distribution --
these are two different (also legitimate) notions of "uncertainty" in this
system; we use the twin's predictive uncertainty since it's what's
available cheaply at every control step without extra RL-specific machinery.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.baselines import RuleBasedThermostat


class UncertaintyGatedController:
    def __init__(
        self,
        rl_model,
        uncertainty_fn,
        std_threshold: float = 0.5,
        fallback=None,
    ):
        """
        rl_model: trained stable-baselines3 model exposing .predict(obs)
        uncertainty_fn: callable(feature_window: np.ndarray) -> std (float),
            e.g. a closure over an MCDropoutResidualLSTM or DeepEnsemble
        std_threshold: predictive std (deg C) above which control defers to
            the fallback controller
        fallback: a controller exposing .act(obs); defaults to a rule-based
            thermostat
        """
        self.rl_model = rl_model
        self.uncertainty_fn = uncertainty_fn
        self.std_threshold = std_threshold
        self.fallback = fallback or RuleBasedThermostat()
        self.fallback_triggered_count = 0
        self.total_steps = 0

    def act(self, obs: np.ndarray, feature_window: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Returns (action, used_fallback). If feature_window is None (not
        enough history yet), defaults to the RL policy -- consistent with
        how HybridTwin/CoolTwinEnv treat the warm-up period."""
        self.total_steps += 1

        if feature_window is not None:
            std = self.uncertainty_fn(feature_window)
            if std > self.std_threshold:
                self.fallback_triggered_count += 1
                return self.fallback.act(obs), True

        action, _ = self.rl_model.predict(obs, deterministic=True)
        return action, False

    @property
    def fallback_rate(self) -> float:
        return self.fallback_triggered_count / max(self.total_steps, 1)
