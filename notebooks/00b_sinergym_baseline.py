"""
00b_sinergym_baseline.py

Phase 1 deliverable (the part that was originally deferred): run a
random-action agent on a real EnergyPlus-backed Sinergym scenario for one
episode, confirming the higher-fidelity simulator actually works end-to-end
in this repo -- not just documented as a future option.

This does NOT replace twin/env.py (the custom RC-network environment) as the
default for training and evaluation elsewhere in the repo. That decision --
made explicitly in docs/sinergym_setup.md -- still holds: EnergyPlus is a
heavy, platform-specific system dependency, a poor fit for guaranteeing
"clone and run" works in CI or for every judge/teammate's machine. What
changes here is that the Sinergym path is now a *validated, running* option
for anyone who does have EnergyPlus installed, not just a documented
instruction set.

Requires:
  - `pip install sinergym` (already in requirements.txt, sinergym extra)
  - A working EnergyPlus install (see docs/sinergym_setup.md), with two
    environment variables set BEFORE importing sinergym:
        EPLUS_PATH   -- the EnergyPlus install directory
                        (e.g. /usr/local/EnergyPlus-25-1-0)
        PYTHONPATH   -- must include that same directory, so
                        `from pyenergyplus.api import EnergyPlusAPI` resolves

If either is missing, this script (and its test) skip cleanly rather than
failing -- see the `require_sinergym()` guard below and
tests/test_sinergym_baseline.py.
"""

from __future__ import annotations

import os
import sys


def require_sinergym():
    """Returns the sinergym module if it and EnergyPlus are both usable,
    otherwise raises RuntimeError with a clear, actionable message pointing
    at docs/sinergym_setup.md. Centralized here so the notebook script and
    the test use the exact same check."""
    eplus_path = os.environ.get("EPLUS_PATH")
    if not eplus_path:
        raise RuntimeError(
            "EPLUS_PATH is not set. See docs/sinergym_setup.md for how to install "
            "EnergyPlus and set EPLUS_PATH / PYTHONPATH before running this."
        )
    if eplus_path not in sys.path:
        sys.path.insert(0, eplus_path)

    try:
        import gymnasium as gym
        import sinergym  # noqa: F401 -- registers the Eplus-* env IDs
    except ImportError as e:
        raise RuntimeError(
            f"sinergym or its EnergyPlus bindings aren't importable ({e}). "
            "See docs/sinergym_setup.md."
        ) from e

    return gym


def run_sinergym_baseline(
    env_id: str = "Eplus-5zone-hot-continuous-v1",
    seed: int = 0,
):
    """Runs one random-action episode on a real EnergyPlus-backed Sinergym
    scenario. `5zone` is Sinergym's standard small-building scenario --
    the closest match to this project's "single building zone" scope among
    Sinergym's shipped scenarios. Continuous action space, matching the
    same scope decision made for twin/env.py."""
    gym = require_sinergym()

    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)

    print(f"Sinergym baseline: {env_id}")
    print(f"  Action space:      {env.action_space}")
    print(f"  Observation shape: {env.observation_space.shape}")

    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    # A full Sinergym episode is a full simulated year (35,040 steps at the
    # default 15-minute timestep) -- far more than needed to confirm the
    # pipeline works. Cap at a representative slice so this runs in seconds,
    # not minutes, matching the "fast sanity check" spirit of Phase 1's
    # existing random-agent baseline.
    max_steps = 500

    while not (terminated or truncated) and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()

    mean_reward = total_reward / steps if steps else float("nan")
    print(f"  Steps run:         {steps}")
    print(f"  Total reward:      {total_reward:.3f}")
    print(f"  Mean reward/step:  {mean_reward:.4f}")

    return {
        "env_id": env_id,
        "steps": steps,
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "action_space": str(env.action_space),
        "observation_shape": list(env.observation_space.shape),
    }


if __name__ == "__main__":
    run_sinergym_baseline()
