"""
reward_decomposition.py

Breaks the scalar reward at each control step down into its four weighted
components (cost, comfort, carbon, peak), so the agent's decisions can be
explained in terms a building operator actually cares about instead of a
single opaque number.

This is the cheapest, most concrete form of explainability in the system:
the components are already computed by CoolTwinEnv.step() (see the `info`
dict) -- this module just organizes them into something that can be shown
on a dashboard, fed to the LLM explanation layer, or plotted as a bar chart.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl.reward import RewardWeights


@dataclass
class RewardDecomposition:
    cost_term: float
    comfort_term: float
    carbon_term: float
    peak_term: float

    @property
    def total(self) -> float:
        return -(self.cost_term + self.comfort_term + self.carbon_term + self.peak_term)

    def as_dict(self) -> dict:
        return {
            "cost": self.cost_term,
            "comfort": self.comfort_term,
            "carbon": self.carbon_term,
            "peak": self.peak_term,
        }

    def dominant_factor(self) -> str:
        """Which weighted penalty term contributed most to this step's
        (negative) reward -- i.e. what the agent was "most reacting to"."""
        d = self.as_dict()
        return max(d, key=d.get)

    def percentages(self) -> dict:
        d = self.as_dict()
        total = sum(d.values())
        if total <= 1e-9:
            return {k: 0.0 for k in d}
        return {k: 100.0 * v / total for k, v in d.items()}


def decompose_reward(cost: float, discomfort: float, carbon_kg: float, peak_frac: float, weights: RewardWeights) -> RewardDecomposition:
    """Mirrors the exact math in rl/reward.py's compute_reward(), broken out
    per-term instead of summed, so decomposition and the actual reward used
    for training can never silently drift apart."""
    return RewardDecomposition(
        cost_term=weights.cost * cost,
        comfort_term=weights.comfort * discomfort,
        carbon_term=weights.carbon * carbon_kg,
        peak_term=weights.peak * peak_frac,
    )


def plot_decomposition(decomps: list[RewardDecomposition], save_path: str, title: str = "Reward decomposition over episode"):
    """Stacked bar chart of the four (weighted) penalty terms over a
    sequence of steps -- e.g. one week of an evaluation episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cost = np.array([d.cost_term for d in decomps])
    comfort = np.array([d.comfort_term for d in decomps])
    carbon = np.array([d.carbon_term for d in decomps])
    peak = np.array([d.peak_term for d in decomps])
    steps = np.arange(len(decomps))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(steps, cost, label="cost", width=1.0)
    ax.bar(steps, comfort, bottom=cost, label="comfort", width=1.0)
    ax.bar(steps, carbon, bottom=cost + comfort, label="carbon", width=1.0)
    ax.bar(steps, peak, bottom=cost + comfort + carbon, label="peak", width=1.0)
    ax.set_xlabel("Control step")
    ax.set_ylabel("Weighted penalty contribution")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return save_path
