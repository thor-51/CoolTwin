"""
04_explainability.py

Phase 5 deliverable: runs a trained PPO policy through the hybrid twin for
one evaluation episode, logging a full DecisionContext (reward
decomposition + uncertainty + top SHAP features) at every step. Then
demonstrates:
  1. The reward decomposition plot over the episode.
  2. SHAP attribution for a single interesting (high-discomfort) step.
  3. Natural-language explanation of that step (template mode by default;
     set ANTHROPIC_API_KEY to use the real LLM).
  4. The canned Q&A layer answering a few example questions.

Usage:
    PYTHONPATH=. python notebooks/04_explainability.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

from twin.env import CoolTwinEnv
from twin.data_gen import generate_dataset
from twin.residual_lstm import ResidualLSTM, ResidualWindowDataset, train_model, FEATURE_NAMES
from rl.reward import RewardWeights
from rl.train_ppo import train_ppo
from uncertainty.mc_dropout import MCDropoutResidualLSTM, predict_with_uncertainty
from evaluation.baselines import RuleBasedThermostat

from explainability.reward_decomposition import decompose_reward, plot_decomposition
from explainability.shap_explain import explain_residual_model, explain_policy, top_k_features
from explainability.llm_explainer import DecisionContext, generate_explanation, answer_question


OBS_NAMES = ["T_in", "T_out", "T_setpoint", "occupancy", "price", "hour", "day_progress"]


def main():
    weights = RewardWeights()
    uncertainty_threshold = 0.15  # deg C; illustrative fixed threshold for this demo

    print("Training a small residual model (for SHAP + uncertainty) and PPO policy...")
    train_eps = generate_dataset(n_episodes=10, n_steps=200, seed=1)
    val_eps = generate_dataset(n_episodes=3, n_steps=200, seed=2)
    train_ds = ResidualWindowDataset(train_eps, window=8, target="residual")
    val_ds = ResidualWindowDataset(val_eps, window=8, target="residual")

    torch.manual_seed(0)
    residual_model = MCDropoutResidualLSTM(n_features=6, hidden_size=16, num_layers=1, dropout=0.2)
    train_model(residual_model, train_ds, val_ds, epochs=10, verbose=False)

    ppo_model = train_ppo(weights=weights, total_timesteps=10_000, seed=0)

    print("Running one evaluation episode, logging decision context at each step...\n")
    env = CoolTwinEnv(episode_hours=48, reward_weights=weights, seed=7)  # 2-day episode for a fast demo
    obs, _ = env.reset(seed=7)

    window = 8
    feature_history = []
    decomps = []
    contexts: list[DecisionContext] = []
    fallback_ctrl = RuleBasedThermostat()

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = ppo_model.predict(obs, deterministic=True)

        T_in, T_out, T_set, occ, price, hour, progress = obs
        Q_hvac = float(action[0]) * env.hvac_max_watts
        Q_gain = 300.0 if occ else 50.0

        from twin.residual_lstm import build_feature_vector
        feat = build_feature_vector(T_out, hour, Q_hvac, Q_gain, T_in)
        feature_history.append(feat)

        used_fallback = False
        std = 0.0
        if len(feature_history) >= window:
            x = torch.from_numpy(np.stack(feature_history[-window:])).unsqueeze(0)
            _, std_arr = predict_with_uncertainty(residual_model, x, n_samples=20)
            std = float(std_arr[0])
            if std > uncertainty_threshold:
                used_fallback = True
                action = fallback_ctrl.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        decomp = decompose_reward(info["cost"], info["discomfort"], info["carbon_kg"], abs(float(action[0])), weights)
        decomps.append(decomp)

        contexts.append(
            DecisionContext(
                T_in=T_in, T_out=T_out, occupancy=bool(occ), price=price, action=float(action[0]),
                reward_components=decomp.as_dict(), dominant_factor=decomp.dominant_factor(),
                uncertainty_std=std, uncertainty_threshold=uncertainty_threshold,
                top_shap_features=[], used_fallback_control=used_fallback,
            )
        )

    print(f"Episode complete: {len(contexts)} steps logged.")
    print(f"Fallback control used on {sum(c.used_fallback_control for c in contexts)}/{len(contexts)} steps.\n")

    plot_decomposition(decomps, save_path="results/reward_decomposition.png")
    print("Saved results/reward_decomposition.png")

    # --- SHAP on the residual model, for the step with the highest discomfort ---
    worst_idx = max(range(len(contexts)), key=lambda i: contexts[i].reward_components["comfort"])
    print(f"\nRunning SHAP on the residual model for step {worst_idx} (highest discomfort penalty)...")

    # Background must be a set of full (window, n_features) windows sampled
    # from earlier in the episode, not individual per-step feature vectors.
    candidate_starts = list(range(window - 1, worst_idx, 2))
    background_starts = candidate_starts[-15:] if len(candidate_starts) > 15 else candidate_starts
    background = np.stack([
        np.stack(feature_history[s - window + 1: s + 1]) for s in background_starts
    ]) if background_starts else np.empty((0, window, len(FEATURE_NAMES)))
    instance = np.stack(feature_history[worst_idx - window + 1: worst_idx + 1])

    if len(background) >= 5:
        shap_result = explain_residual_model(residual_model, background, instance, FEATURE_NAMES, n_samples=50)
        top_feats = top_k_features(shap_result, k=3)
        contexts[worst_idx].top_shap_features = top_feats
        print("Top contributing features:", top_feats)
    else:
        print("Not enough history for a SHAP background set at this step -- skipping.")

    print("\n--- Natural-language explanation of that step ---")
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not use_llm:
        print("(ANTHROPIC_API_KEY not set -- using template fallback, not the real LLM)")
    print(generate_explanation(contexts[worst_idx], use_llm=use_llm))

    print("\n--- Canned Q&A demo ---")
    for q in ["Why is the room hot?", "How confident are you right now?", "Give me a weekly summary"]:
        print(f"\nQ: {q}")
        print(f"A: {answer_question(q, contexts, use_llm=use_llm)}")

    return contexts


if __name__ == "__main__":
    main()
