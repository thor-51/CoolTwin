"""
train_sac.py

Trains a SAC agent (stable-baselines3) on CoolTwinEnv, as the second
algorithm in the PPO-vs-SAC comparison referenced in the abstract's
methodology. SAC is off-policy and typically more sample-efficient than PPO
on continuous-control problems like this one, at the cost of more
hyperparameter sensitivity -- worth reporting both training curves in the
final evaluation.
"""

from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from twin.env import CoolTwinEnv
from rl.reward import RewardWeights


def make_env(weights: RewardWeights | None = None, episode_hours: int = 24 * 7, seed: int = 0):
    def _init():
        env = CoolTwinEnv(episode_hours=episode_hours, reward_weights=weights, seed=seed)
        return Monitor(env)

    return _init


def train_sac(
    weights: RewardWeights | None = None,
    total_timesteps: int = 50_000,
    seed: int = 0,
    save_path: str | None = None,
) -> SAC:
    env = DummyVecEnv([make_env(weights, seed=seed)])
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        learning_starts=1000,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    if save_path:
        model.save(save_path)
    return model


if __name__ == "__main__":
    print("Training SAC on default (balanced) reward weights...")
    model = train_sac(total_timesteps=50_000, save_path="results/sac_balanced")
    print("Done. Saved to results/sac_balanced.zip")
