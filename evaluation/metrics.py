"""
metrics.py

Runs a trained stable-baselines3 policy for one full episode on CoolTwinEnv
and returns the same summary metrics dict as evaluation/baselines.py's
run_episode(), so RL agents and rule-based/PID baselines can be compared
apples-to-apples in one table (Phase 6).
"""

from __future__ import annotations

import numpy as np


def run_policy_episode(env, model, deterministic: bool = True) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    total_energy_kwh = 0.0
    total_carbon_kg = 0.0
    total_discomfort = 0.0
    max_power_w = 0.0
    steps = 0

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_energy_kwh += info["energy_kwh"]
        total_carbon_kg += info["carbon_kg"]
        total_discomfort += info["discomfort"]
        max_power_w = max(max_power_w, abs(float(action[0])) * env.hvac_max_watts)
        steps += 1

    return {
        "total_reward": total_reward,
        "total_energy_kwh": total_energy_kwh,
        "total_carbon_kg": total_carbon_kg,
        "total_discomfort": total_discomfort,
        "peak_power_w": max_power_w,
        "steps": steps,
    }
