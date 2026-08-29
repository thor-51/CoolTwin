import numpy as np
import torch

from rl.reward import RewardWeights
from explainability.reward_decomposition import decompose_reward, RewardDecomposition
from explainability.shap_explain import explain_residual_model, explain_policy, top_k_features
from explainability.llm_explainer import DecisionContext, generate_explanation, answer_question
from twin.residual_lstm import ResidualLSTM, FEATURE_NAMES


def test_decompose_reward_matches_compute_reward_sign_and_magnitude():
    """The decomposition's total must exactly match what rl/reward.py's
    compute_reward() would return for the same inputs -- otherwise the
    explanation could show numbers that don't match what the agent was
    actually trained on."""
    from rl.reward import compute_reward

    weights = RewardWeights(cost=1.0, comfort=2.0, carbon=0.5, peak=0.1)
    cost, discomfort, carbon_kg, peak_frac = 2.0, 1.0, 0.5, 0.3

    decomp = decompose_reward(cost, discomfort, carbon_kg, peak_frac, weights)
    direct_reward = compute_reward(cost, discomfort, carbon_kg, peak_frac, weights)

    assert np.isclose(decomp.total, direct_reward)


def test_dominant_factor_identifies_largest_term():
    decomp = RewardDecomposition(cost_term=0.1, comfort_term=5.0, carbon_term=0.2, peak_term=0.05)
    assert decomp.dominant_factor() == "comfort"


def test_percentages_sum_to_100():
    decomp = RewardDecomposition(cost_term=1.0, comfort_term=2.0, carbon_term=0.5, peak_term=0.5)
    pct = decomp.percentages()
    assert np.isclose(sum(pct.values()), 100.0)


def test_percentages_handles_all_zero():
    decomp = RewardDecomposition(cost_term=0.0, comfort_term=0.0, carbon_term=0.0, peak_term=0.0)
    pct = decomp.percentages()
    assert all(v == 0.0 for v in pct.values())


def test_explain_residual_model_returns_all_windowed_features():
    torch.manual_seed(0)
    model = ResidualLSTM(n_features=6, hidden_size=8)
    window, n_features = 4, 6

    background = np.random.randn(10, window, n_features).astype(np.float32)
    instance = np.random.randn(window, n_features).astype(np.float32)

    result = explain_residual_model(model, background, instance, FEATURE_NAMES, n_samples=20)
    # one SHAP value per (timestep, feature) combination
    assert len(result) == window * n_features
    for name in FEATURE_NAMES:
        assert any(name in k for k in result)


def test_explain_policy_returns_one_value_per_obs_dim():
    def fake_predict(obs_batch):
        # simple linear function so SHAP has something structured to attribute
        return (obs_batch[:, 0] - obs_batch[:, 1]).reshape(-1, 1)

    obs_names = ["T_in", "T_out", "T_setpoint", "occupancy", "price", "hour", "day_progress"]
    background = np.random.uniform(-1, 1, size=(15, 7)).astype(np.float32)
    instance = np.array([2.0, -1.0, 0, 0, 0, 0, 0], dtype=np.float32)

    result = explain_policy(fake_predict, background, instance, obs_names, n_samples=30)
    assert set(result.keys()) == set(obs_names)


def test_top_k_features_sorted_by_magnitude():
    shap_dict = {"a": 0.1, "b": -0.9, "c": 0.5, "d": 0.05}
    top2 = top_k_features(shap_dict, k=2)
    assert [name for name, _ in top2] == ["b", "c"]


def _make_context(dominant="comfort", used_fallback=False):
    return DecisionContext(
        T_in=27.0, T_out=30.0, occupancy=True, price=0.25, action=-0.6,
        reward_components={"cost": 0.2, "comfort": 1.8, "carbon": 0.1, "peak": 0.05},
        dominant_factor=dominant, uncertainty_std=0.08, uncertainty_threshold=0.15,
        top_shap_features=[("T_out (t-1)", 0.03)], used_fallback_control=used_fallback,
    )


def test_template_explanation_mentions_key_numbers():
    ctx = _make_context()
    text = generate_explanation(ctx, use_llm=False)
    assert "27.0" in text
    assert "comfort" in text.lower()


def test_template_explanation_flags_fallback_control():
    ctx = _make_context(used_fallback=True)
    text = generate_explanation(ctx, use_llm=False)
    assert "fallback" in text.lower() or "rule-based" in text.lower()


def test_answer_question_why_routes_to_explanation():
    history = [_make_context()]
    answer = answer_question("Why is the room warm?", history, use_llm=False)
    assert "27.0" in answer


def test_answer_question_confidence_routes_correctly():
    history = [_make_context()]
    answer = answer_question("How confident are you right now?", history, use_llm=False)
    assert "uncertainty" in answer.lower() or "confidence" in answer.lower()


def test_answer_question_summary_counts_dominant_factor():
    history = [_make_context(dominant="comfort"), _make_context(dominant="comfort"), _make_context(dominant="cost")]
    answer = answer_question("weekly summary please", history, use_llm=False)
    assert "comfort" in answer.lower()
    assert "2/3" in answer


def test_answer_question_empty_history():
    answer = answer_question("why is it hot", [], use_llm=False)
    assert "no decision history" in answer.lower()


def test_generate_explanation_uses_template_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = _make_context()
    text = generate_explanation(ctx, use_llm=None)  # auto-detect
    # should not raise, and should be the template's deterministic style
    assert "trained policy" in text.lower() or "fallback" in text.lower()
