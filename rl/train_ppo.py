"""
train_ppo.py

Trains a PPO agent (stable-baselines3) on CoolTwinEnv with a given reward
weighting. Used both for the primary PPO agent and, with different
`weights`, as one leg of the Pareto sweep (see pareto.py).
"""

from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from twin.env import CoolTwinEnv
from rl.reward import RewardWeights


def make_env(weights: RewardWeights | None = None, episode_hours: int = 24 * 7, seed: int = 0):
    def _init():
        env = CoolTwinEnv(episode_hours=episode_hours, reward_weights=weights, seed=seed)
        return Monitor(env)

    return _init


def train_ppo(
    weights: RewardWeights | None = None,
    total_timesteps: int = 50_000,
    seed: int = 0,
    save_path: str | None = None,
) -> PPO:
    env = DummyVecEnv([make_env(weights, seed=seed)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        n_steps=672,          # one full episode per rollout
        batch_size=96,         # 672 / 96 = 7, divides n_steps evenly
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    if save_path:
        model.save(save_path)
    return model


if __name__ == "__main__":
    print("Training PPO on default (balanced) reward weights...")
    model = train_ppo(total_timesteps=50_000, save_path="results/ppo_balanced")
    print("Done. Saved to results/ppo_balanced.zip")
