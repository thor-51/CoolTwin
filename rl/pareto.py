"""
pareto.py

Trains a PPO policy for each named reward weighting in rl.reward.PARETO_WEIGHT_SET,
evaluates each on a held-out episode, and plots the resulting cost-vs-comfort
trade-off curve. This is the actual "multi-objective optimization" deliverable
from the abstract: a Pareto front of policies, not one fixed-weight policy.

Usage:
    PYTHONPATH=. python rl/pareto.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twin.env import CoolTwinEnv
from rl.reward import PARETO_WEIGHT_SET
from rl.train_ppo import train_ppo
from evaluation.metrics import run_policy_episode


def run_pareto_sweep(total_timesteps: int = 200_000, eval_seed: int = 42):
    results = {}
    for name, weights in PARETO_WEIGHT_SET.items():
        print(f"Training PPO for weighting: {name} ({weights})...")
        model = train_ppo(weights=weights, total_timesteps=total_timesteps, seed=0)

        eval_env = CoolTwinEnv(episode_hours=24 * 7, reward_weights=weights, seed=eval_seed)
        eval_env.reset(seed=eval_seed)
        metrics = run_policy_episode(eval_env, model)
        metrics["weights"] = weights
        results[name] = metrics
        print(
            f"  -> cost proxy(energy kWh)={metrics['total_energy_kwh']:.1f}  "
            f"discomfort={metrics['total_discomfort']:.1f}  "
            f"carbon_kg={metrics['total_carbon_kg']:.1f}"
        )

    return results


def plot_pareto_front(results: dict, save_path: str = "results/pareto_front.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, m in results.items():
        ax.scatter(m["total_energy_kwh"], m["total_discomfort"], s=90, label=name)
        ax.annotate(name, (m["total_energy_kwh"], m["total_discomfort"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Total energy consumption (kWh) — proxy for cost")
    ax.set_ylabel("Total discomfort (K·step)")
    ax.set_title("CoolTwin: Pareto front across reward weightings (1-week eval episode)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved Pareto front plot to {save_path}")


if __name__ == "__main__":
    results = run_pareto_sweep(total_timesteps=200_000)
    plot_pareto_front(results)

    print("\nSummary:")
    print(f"{'Weighting':<18}{'Energy(kWh)':>14}{'Discomfort':>14}{'Carbon(kg)':>13}")
    for name, m in results.items():
        print(f"{name:<18}{m['total_energy_kwh']:>14.1f}{m['total_discomfort']:>14.1f}{m['total_carbon_kg']:>13.1f}")
