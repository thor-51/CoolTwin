"""
env.py

CoolTwinEnv: a Gymnasium environment wrapping the RC thermal digital twin
(Phase 1: physics-only; Phase 2 will swap in the hybrid physics+residual
model here without changing this interface).

State (observation):
    [T_in, T_out, T_setpoint, occupancy (0/1), price ($/kWh), hour_of_day, day_progress]

Action:
    continuous scalar in [-1, 1]: HVAC power fraction
    -1 = full cooling, 0 = off, +1 = full heating

Reward:
    weighted sum of (negative) energy cost, comfort violation, carbon
    emissions and peak-demand penalty. See rl/reward.py once Phase 3 starts
    for the full multi-objective version with configurable weights; this
    env exposes a simple default so Phase 1 baselines are runnable standalone.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from twin.rc_model import RCThermalZone, RCParams
from rl.reward import RewardWeights, compute_reward


class CoolTwinEnv(gym.Env):
    """
    residual_model: optional trained ResidualLSTM (twin/residual_lstm.py).
    If provided, the env corrects each step's physics prediction with the
    learned residual (i.e. steps through the hybrid twin, per Phase 2)
    instead of the raw RC-network-only prediction. If None (default), the
    env behaves exactly as in Phase 1 -- physics-only -- so existing
    baselines and tests remain unaffected.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_hours: int = 24 * 7,   # one week per episode
        dt_seconds: float = 900.0,      # 15-minute control steps
        comfort_band: tuple[float, float] = (21.0, 25.0),
        hvac_max_watts: float = 2000.0,
        reward_weights: RewardWeights | None = None,
        residual_model=None,
        residual_window: int = 8,
        seed: int | None = None,
    ):
        super().__init__()
        self.zone = RCThermalZone(RCParams())
        self.dt_seconds = dt_seconds
        self.n_steps = int(episode_hours * 3600 / dt_seconds)
        self.comfort_low, self.comfort_high = comfort_band
        self.hvac_max_watts = hvac_max_watts
        self.weights = reward_weights or RewardWeights()

        self.residual_model = residual_model
        self.residual_window = residual_window
        self._feature_history = []  # rolling window for the residual model

        self.observation_space = spaces.Box(
            low=np.array([-20, -30, 15, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([50, 50, 30, 1, 1, 24, 1], dtype=np.float32),
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # -- synthetic exogenous signals (Phase 1 stand-in for Sinergym/weather API) --
    def _outdoor_temp(self, step: int) -> float:
        hour = (step * self.dt_seconds / 3600.0) % 24
        day = step * self.dt_seconds / 3600.0 / 24.0
        seasonal = 3 * np.sin(day / 30 * 2 * np.pi)
        diurnal = 6 * np.sin((hour - 9) / 24 * 2 * np.pi)
        return 24 + seasonal + diurnal + self._rng.normal(0, 0.3)

    def _occupancy(self, step: int) -> float:
        hour = (step * self.dt_seconds / 3600.0) % 24
        return 1.0 if 9 <= hour < 18 else 0.0

    def _price(self, step: int) -> float:
        hour = (step * self.dt_seconds / 3600.0) % 24
        # simple time-of-use pricing: peak 14:00-19:00
        return 0.30 if 14 <= hour < 19 else 0.12

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self.T_wall = 22.0
        self.T_in = 22.0
        self.cumulative_energy_kwh = 0.0
        self.cumulative_carbon_kg = 0.0
        self.peak_power_w = 0.0
        self._feature_history = []

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        T_out = self._outdoor_temp(self.t)
        occ = self._occupancy(self.t)
        price = self._price(self.t)
        hour = (self.t * self.dt_seconds / 3600.0) % 24
        progress = self.t / self.n_steps
        return np.array(
            [self.T_in, T_out, 23.0, occ, price, hour, progress], dtype=np.float32
        )

    def _apply_residual_correction(self, T_out, Q_hvac, Q_gain, T_in_physics) -> float:
        """Uses the trained ResidualLSTM to correct the physics prediction,
        via the shared feature-construction helper (twin/residual_lstm.py)
        so online (here) and offline (dataset-based) feature vectors never
        drift apart. Falls back to the raw physics prediction until enough
        history has accumulated to fill a full window."""
        import torch
        from twin.residual_lstm import build_feature_vector

        hour = (self.t * self.dt_seconds / 3600.0) % 24
        feat = build_feature_vector(T_out, hour, Q_hvac, Q_gain, T_in_physics)
        self._feature_history.append(feat)
        if len(self._feature_history) < self.residual_window:
            return T_in_physics

        window = np.stack(self._feature_history[-self.residual_window :])
        with torch.no_grad():
            x = torch.from_numpy(window).unsqueeze(0)
            residual = self.residual_model(x).item()
        return T_in_physics + residual

    def step(self, action: np.ndarray):
        action = float(np.clip(action[0], -1.0, 1.0))
        Q_hvac = action * self.hvac_max_watts

        T_out = self._outdoor_temp(self.t)
        occ = self._occupancy(self.t)
        Q_gain = 300.0 if occ else 50.0
        price = self._price(self.t)

        T_wall_new, T_in_physics = self.zone.step(
            self.T_wall, self.T_in, T_out, Q_hvac, Q_gain, self.dt_seconds
        )
        self.T_wall = T_wall_new

        if self.residual_model is not None:
            self.T_in = self._apply_residual_correction(T_out, Q_hvac, Q_gain, T_in_physics)
        else:
            self.T_in = T_in_physics

        # --- energy / cost ---
        power_w = abs(Q_hvac)
        energy_kwh = power_w * self.dt_seconds / 3600.0 / 1000.0
        cost = energy_kwh * price
        self.cumulative_energy_kwh += energy_kwh
        self.peak_power_w = max(self.peak_power_w, power_w)

        # --- carbon (simple static grid intensity, kg CO2 / kWh) ---
        carbon_intensity = 0.45
        carbon_kg = energy_kwh * carbon_intensity
        self.cumulative_carbon_kg += carbon_kg

        # --- comfort violation (only penalized while occupied) ---
        if occ:
            discomfort = max(0.0, self.comfort_low - self.T_in) + max(
                0.0, self.T_in - self.comfort_high
            )
        else:
            discomfort = 0.0

        # --- multi-term reward (configurable weights, see rl/reward.py --
        #     Phase 3 trains multiple policies across weightings to trace a
        #     Pareto front rather than using one fixed weighted sum) ---
        peak_frac = power_w / self.hvac_max_watts
        reward = compute_reward(cost, discomfort, carbon_kg, peak_frac, self.weights)

        self.t += 1
        terminated = False
        truncated = self.t >= self.n_steps

        info = {
            "energy_kwh": energy_kwh,
            "cost": cost,
            "carbon_kg": carbon_kg,
            "discomfort": discomfort,
            "T_in": self.T_in,
            "T_in_physics": T_in_physics,
            "T_out": T_out,
            "Q_hvac": Q_hvac,
            "Q_gain": Q_gain,
        }

        return self._get_obs(), reward, terminated, truncated, info
