"""
baselines.py

Non-RL controllers used as comparison points against the trained agent
(Phase 6 evaluation). Both operate purely on the current observation, with
no training required.
"""

from __future__ import annotations

import numpy as np


class RuleBasedThermostat:
    """A simple on/off thermostat around a fixed setpoint band -- the kind of
    controller most real buildings actually run today. This is the primary
    baseline the abstract compares against."""

    def __init__(self, setpoint: float = 23.0, deadband: float = 1.0, full_power: float = 1.0):
        self.setpoint = setpoint
        self.deadband = deadband
        self.full_power = full_power

    def act(self, obs: np.ndarray) -> np.ndarray:
        T_in = obs[0]
        if T_in > self.setpoint + self.deadband:
            action = -self.full_power  # cool
        elif T_in < self.setpoint - self.deadband:
            action = self.full_power  # heat
        else:
            action = 0.0
        return np.array([action], dtype=np.float32)


class PIDController:
    """A simple PID controller regulating T_in to a fixed setpoint. Output is
    clipped to the environment's [-1, 1] action range."""

    def __init__(self, setpoint: float = 23.0, kp: float = 0.3, ki: float = 0.01, kd: float = 0.05):
        self.setpoint = setpoint
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        T_in = obs[0]
        error = self.setpoint - T_in  # positive error -> too cold -> heat
        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        action = np.clip(output, -1.0, 1.0)
        return np.array([action], dtype=np.float32)


class RandomController:
    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-1, 1, size=(1,)).astype(np.float32)


def run_episode(env, controller) -> dict:
    """Runs one full episode with any controller exposing `.act(obs)`, and
    returns summary metrics matching what evaluation/metrics.py expects."""
    if hasattr(controller, "reset"):
        controller.reset()

    obs, _ = env.reset()
    total_reward = 0.0
    total_energy_kwh = 0.0
    total_carbon_kg = 0.0
    total_discomfort = 0.0
    max_power_w = 0.0
    steps = 0

    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_energy_kwh += info["energy_kwh"]
        total_carbon_kg += info["carbon_kg"]
        total_discomfort += info["discomfort"]
        max_power_w = max(max_power_w, abs(action[0]) * env.hvac_max_watts)
        steps += 1

    return {
        "total_reward": total_reward,
        "total_energy_kwh": total_energy_kwh,
        "total_carbon_kg": total_carbon_kg,
        "total_discomfort": total_discomfort,
        "peak_power_w": max_power_w,
        "steps": steps,
    }
