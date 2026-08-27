"""
00_baseline_random_agent.py

Phase 1 deliverable: run a random-action agent on CoolTwinEnv for one
episode, confirm the environment loop works end-to-end, and print summary
metrics. This is the sanity-check baseline that later gets compared against
rule-based control, PID, and the trained RL agent (see evaluation/baselines.py,
Phase 6).
"""

import numpy as np

from twin.env import CoolTwinEnv


def run_random_baseline(episode_hours: int = 24 * 7, seed: int = 0):
    env = CoolTwinEnv(episode_hours=episode_hours, seed=seed)
    obs, _ = env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    total_reward = 0.0
    total_energy_kwh = 0.0
    total_carbon_kg = 0.0
    total_discomfort = 0.0
    steps = 0

    terminated = truncated = False
    while not (terminated or truncated):
        action = rng.uniform(-1, 1, size=(1,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_energy_kwh += info["energy_kwh"]
        total_carbon_kg += info["carbon_kg"]
        total_discomfort += info["discomfort"]
        steps += 1

    print(f"Random baseline over {episode_hours}h ({steps} steps):")
    print(f"  Total reward:          {total_reward:10.2f}")
    print(f"  Total energy (kWh):    {total_energy_kwh:10.2f}")
    print(f"  Total carbon (kg CO2): {total_carbon_kg:10.2f}")
    print(f"  Total discomfort (K*step): {total_discomfort:10.2f}")

    return {
        "total_reward": total_reward,
        "total_energy_kwh": total_energy_kwh,
        "total_carbon_kg": total_carbon_kg,
        "total_discomfort": total_discomfort,
    }


if __name__ == "__main__":
    run_random_baseline()
