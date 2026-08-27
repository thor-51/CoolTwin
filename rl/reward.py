"""
reward.py

The configurable multi-objective reward used by CoolTwinEnv from Phase 3
onward. This replaces the fixed-weight reward that was hardcoded directly
into twin/env.py for the Phase 1 baseline.

R = -( w_cost * cost + w_comfort * discomfort + w_carbon * carbon_kg + w_peak * peak_penalty )

Exposing the weights as a dataclass lets us:
  1. Train multiple policies at different weightings (rl/pareto.py) to trace
     out a Pareto front of cost/comfort/carbon/peak trade-offs -- this is
     the actual "multi-objective optimization" claim from the abstract,
     not just a fixed weighted sum.
  2. Keep the reward math in one place, documented, and unit-testable
     independent of the Gym env's step() bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    cost: float = 1.0
    comfort: float = 2.0
    carbon: float = 0.5
    peak: float = 0.1

    def as_tuple(self):
        return (self.cost, self.comfort, self.carbon, self.peak)


# A small set of named weightings spanning the trade-off space, used to trace
# the Pareto front in rl/pareto.py. Ranges are chosen so each policy clearly
# emphasizes a different objective.
PARETO_WEIGHT_SET = {
    "cost_focused":    RewardWeights(cost=3.0, comfort=1.0, carbon=0.3, peak=0.1),
    "balanced":        RewardWeights(cost=1.0, comfort=2.0, carbon=0.5, peak=0.1),
    "comfort_focused": RewardWeights(cost=0.5, comfort=4.0, carbon=0.3, peak=0.1),
    "carbon_focused":  RewardWeights(cost=0.5, comfort=1.5, carbon=2.5, peak=0.1),
    "peak_focused":    RewardWeights(cost=0.5, comfort=1.5, carbon=0.3, peak=1.5),
}


def compute_reward(cost: float, discomfort: float, carbon_kg: float, peak_frac: float, weights: RewardWeights) -> float:
    """peak_frac is the current step's HVAC power as a fraction of max (0-1),
    matching how twin/env.py computes it."""
    return -(
        weights.cost * cost
        + weights.comfort * discomfort
        + weights.carbon * carbon_kg
        + weights.peak * peak_frac
    )
