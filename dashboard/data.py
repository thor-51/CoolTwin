"""
data.py

Data-preparation layer for the CoolTwin Streamlit dashboard, kept
deliberately separate from dashboard/app.py so these functions can be unit
tested with plain pytest -- Streamlit apps themselves aren't easily unit
testable, but the logic that feeds them should be.

All "trained on the fly" functions here use small, fast training budgets
(seconds, not minutes) so the dashboard loads quickly. This mirrors the
same short-training-budget caveat already documented for Phase 3/6: these
are DEMO-quality numbers for interactivity, not the final report numbers
(which should come from the longer offline runs in notebooks/02 and
notebooks/05).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from twin.data_gen import generate_episode, generate_dataset
from twin.residual_lstm import ResidualWindowDataset, train_model, build_feature_vector
from twin.hybrid_twin import HybridTwin
from twin.env import CoolTwinEnv
from uncertainty.mc_dropout import MCDropoutResidualLSTM, predict_with_uncertainty
from rl.reward import RewardWeights, PARETO_WEIGHT_SET
from rl.train_ppo import train_ppo
from evaluation.baselines import RuleBasedThermostat
from explainability.reward_decomposition import decompose_reward


DEMO_WINDOW = 8


def train_demo_residual_model(seed: int = 0, epochs: int = 6) -> MCDropoutResidualLSTM:
    """Fast residual-model training for dashboard interactivity. Call this
    once per session and cache the result (see app.py's st.cache_resource)."""
    torch.manual_seed(seed)
    train_eps = generate_dataset(n_episodes=8, n_steps=150, seed=1)
    val_eps = generate_dataset(n_episodes=2, n_steps=150, seed=2)
    train_ds = ResidualWindowDataset(train_eps, window=DEMO_WINDOW, target="residual")
    val_ds = ResidualWindowDataset(val_eps, window=DEMO_WINDOW, target="residual")

    model = MCDropoutResidualLSTM(n_features=6, hidden_size=16, num_layers=1, dropout=0.2)
    train_model(model, train_ds, val_ds, epochs=epochs, verbose=False)
    return model


def train_demo_ppo(weights: RewardWeights | None = None, seed: int = 0, total_timesteps: int = 8_000):
    """Fast PPO training for dashboard interactivity -- see module docstring
    re: this being a demo-quality budget, not the final report numbers."""
    return train_ppo(weights=weights or RewardWeights(), total_timesteps=total_timesteps, seed=seed)


@dataclass
class TwinFidelityResult:
    hour: np.ndarray
    T_out: np.ndarray
    T_in_true: np.ndarray
    T_in_physics: np.ndarray
    T_in_hybrid: np.ndarray
    rmse_physics: float
    rmse_hybrid: float


def get_twin_fidelity_data(residual_model, n_steps: int = 96, seed: int = 42) -> TwinFidelityResult:
    """Replays one episode of the synthetic ground-truth building (same
    generator used in Phase 2) and compares physics-only vs hybrid-corrected
    predictions against it -- this is the 'does the twin actually work'
    view of the dashboard."""
    ep = generate_episode(n_steps=n_steps, seed=seed)
    twin = HybridTwin(residual_model, window=DEMO_WINDOW)
    hybrid_pred = twin.correct_episode(ep)

    w = DEMO_WINDOW
    rmse_physics = float(np.sqrt(np.mean((ep["T_in_physics"][w:] - ep["T_in_true"][w:]) ** 2)))
    rmse_hybrid = float(np.sqrt(np.mean((hybrid_pred[w:] - ep["T_in_true"][w:]) ** 2)))

    return TwinFidelityResult(
        hour=ep["hour"],
        T_out=ep["T_out"],
        T_in_true=ep["T_in_true"],
        T_in_physics=ep["T_in_physics"],
        T_in_hybrid=hybrid_pred,
        rmse_physics=rmse_physics,
        rmse_hybrid=rmse_hybrid,
    )


@dataclass
class ControlStepLog:
    t: int
    T_in: float
    T_out: float
    occupancy: float
    price: float
    action: float
    reward: float
    cost: dict = field(default_factory=dict)   # RewardDecomposition.as_dict()
    dominant_factor: str = ""
    uncertainty_std: float = 0.0
    used_fallback: bool = False


def run_control_episode(
    ppo_model,
    residual_model,
    weights: RewardWeights,
    episode_hours: int = 48,
    seed: int = 7,
    uncertainty_threshold: float = 0.15,
) -> list[ControlStepLog]:
    """Runs the trained PPO policy through the (physics-only) env, with an
    uncertainty-gated fallback computed from the separately-trained residual
    model -- mirrors notebooks/04_explainability.py's approach, factored out
    here for dashboard reuse."""
    env = CoolTwinEnv(episode_hours=episode_hours, reward_weights=weights, seed=seed)
    obs, _ = env.reset(seed=seed)
    fallback_ctrl = RuleBasedThermostat()

    feature_history = []
    logs = []
    t = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = ppo_model.predict(obs, deterministic=True)

        T_in, T_out, T_set, occ, price, hour, progress = obs
        Q_hvac = float(action[0]) * env.hvac_max_watts
        Q_gain = 300.0 if occ else 50.0
        feat = build_feature_vector(T_out, hour, Q_hvac, Q_gain, T_in)
        feature_history.append(feat)

        used_fallback = False
        std = 0.0
        if len(feature_history) >= DEMO_WINDOW:
            x = torch.from_numpy(np.stack(feature_history[-DEMO_WINDOW:])).unsqueeze(0)
            _, std_arr = predict_with_uncertainty(residual_model, x, n_samples=15)
            std = float(std_arr[0])
            if std > uncertainty_threshold:
                used_fallback = True
                action = fallback_ctrl.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        decomp = decompose_reward(info["cost"], info["discomfort"], info["carbon_kg"], abs(float(action[0])), weights)

        logs.append(ControlStepLog(
            t=t, T_in=T_in, T_out=T_out, occupancy=occ, price=price,
            action=float(action[0]), reward=reward, cost=decomp.as_dict(),
            dominant_factor=decomp.dominant_factor(), uncertainty_std=std,
            used_fallback=used_fallback,
        ))
        t += 1

    return logs


def get_pareto_preview(total_timesteps: int = 4_000, eval_seed: int = 42) -> dict:
    """A FAST preview Pareto sweep for dashboard interactivity -- explicitly
    not the final Pareto front (see notebooks/02 and rl/pareto.py for that,
    with a much larger timestep budget). Trains a small PPO per weighting
    and evaluates on a short episode, quickly enough to run in an
    interactive app."""
    from evaluation.metrics import run_policy_episode

    results = {}
    for name, weights in PARETO_WEIGHT_SET.items():
        model = train_ppo(weights=weights, total_timesteps=total_timesteps, seed=0)
        env = CoolTwinEnv(episode_hours=24 * 2, reward_weights=weights, seed=eval_seed)
        env.reset(seed=eval_seed)
        results[name] = run_policy_episode(env, model)
    return results
