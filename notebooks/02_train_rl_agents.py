"""
02_train_rl_agents.py

Phase 3 deliverable: trains PPO and SAC on the (physics-only, for training
speed) CoolTwinEnv with the default "balanced" reward weighting, then
evaluates both against the rule-based thermostat, PID, and random baselines
on a held-out episode. Produces the core Phase 3 comparison table.

Usage:
    PYTHONPATH=. python notebooks/02_train_rl_agents.py
"""

from __future__ import annotations

import time

from twin.env import CoolTwinEnv
from rl.reward import RewardWeights
from rl.train_ppo import train_ppo
from rl.train_sac import train_sac
from evaluation.baselines import RuleBasedThermostat, PIDController, RandomController, run_episode
from evaluation.metrics import run_policy_episode


def main(total_timesteps: int = 50_000, eval_seed: int = 123):
    weights = RewardWeights()  # balanced default

    print(f"Training PPO for {total_timesteps} timesteps...")
    t0 = time.time()
    ppo_model = train_ppo(weights=weights, total_timesteps=total_timesteps, seed=0,
                           save_path="results/ppo_balanced")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Training SAC for {total_timesteps} timesteps...")
    t0 = time.time()
    sac_model = train_sac(weights=weights, total_timesteps=total_timesteps, seed=0,
                           save_path="results/sac_balanced")
    print(f"  done in {time.time()-t0:.1f}s")

    print("\nEvaluating all controllers on a held-out week-long episode...\n")

    results = {}

    for name, ctrl in [
        ("Random", RandomController(seed=eval_seed)),
        ("Rule-based thermostat", RuleBasedThermostat()),
        ("PID", PIDController()),
    ]:
        env = CoolTwinEnv(episode_hours=24 * 7, reward_weights=weights, seed=eval_seed)
        results[name] = run_episode(env, ctrl)

    for name, model in [("PPO", ppo_model), ("SAC", sac_model)]:
        env = CoolTwinEnv(episode_hours=24 * 7, reward_weights=weights, seed=eval_seed)
        env.reset(seed=eval_seed)
        results[name] = run_policy_episode(env, model)

    print("=" * 80)
    print(f"{'Controller':<24}{'Reward':>12}{'Energy(kWh)':>14}{'Discomfort':>14}{'Carbon(kg)':>13}")
    print("=" * 80)
    for name, m in results.items():
        print(f"{name:<24}{m['total_reward']:>12.1f}{m['total_energy_kwh']:>14.1f}"
              f"{m['total_discomfort']:>14.1f}{m['total_carbon_kg']:>13.1f}")
    print("=" * 80)

    best = max(results.items(), key=lambda kv: kv[1]["total_reward"])
    print(f"\nBest by total reward: {best[0]}")

    return results


if __name__ == "__main__":
    main()
