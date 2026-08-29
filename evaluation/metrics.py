"""
metrics.py

Runs a trained stable-baselines3 policy for one full episode on CoolTwinEnv
and returns the same summary metrics dict as evaluation/baselines.py's
run_episode(), so RL agents and rule-based/PID baselines can be compared
apples-to-apples in one table (Phase 6).
"""

from __future__ import annotations

import numpy as np


def compute_comfort_hours(total_discomfort_k_step: float, dt_seconds: float) -> float:
    """Converts the internal discomfort accumulator (sum of °C-over-band per
    control step) into actual °C-hours -- the unit the abstract and report
    should use, since raw step-summed units depend on the control interval
    and aren't meaningful to compare across configurations."""
    return total_discomfort_k_step * dt_seconds / 3600.0


def benchmark_inference_latency(act_fn, obs: np.ndarray, n_calls: int = 200) -> dict:
    """Measures wall-clock time per single-step action decision. act_fn must
    be a zero-argument callable (already bound to a fixed obs) or accept obs
    as its only argument -- see call sites for the exact signature used.
    Returns mean/p50/p95 latency in milliseconds."""
    import time

    latencies = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        act_fn(obs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


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
