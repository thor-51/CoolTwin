import numpy as np

from rl.reward import RewardWeights, compute_reward, PARETO_WEIGHT_SET
from twin.env import CoolTwinEnv
from evaluation.baselines import RuleBasedThermostat, PIDController, RandomController, run_episode


def test_compute_reward_matches_manual_calc():
    w = RewardWeights(cost=1.0, comfort=2.0, carbon=0.5, peak=0.1)
    r = compute_reward(cost=2.0, discomfort=1.0, carbon_kg=0.5, peak_frac=0.5, weights=w)
    expected = -(1.0 * 2.0 + 2.0 * 1.0 + 0.5 * 0.5 + 0.1 * 0.5)
    assert np.isclose(r, expected)


def test_higher_weight_increases_penalty_magnitude():
    low = RewardWeights(comfort=1.0)
    high = RewardWeights(comfort=5.0)
    r_low = compute_reward(cost=0, discomfort=1.0, carbon_kg=0, peak_frac=0, weights=low)
    r_high = compute_reward(cost=0, discomfort=1.0, carbon_kg=0, peak_frac=0, weights=high)
    assert r_high < r_low  # more negative reward = bigger penalty


def test_pareto_weight_set_has_expected_keys():
    expected = {"cost_focused", "balanced", "comfort_focused", "carbon_focused", "peak_focused"}
    assert set(PARETO_WEIGHT_SET.keys()) == expected
    for w in PARETO_WEIGHT_SET.values():
        assert isinstance(w, RewardWeights)


def test_env_accepts_custom_weights_and_uses_them():
    custom = RewardWeights(cost=10.0, comfort=0.0, carbon=0.0, peak=0.0)
    env = CoolTwinEnv(episode_hours=1, reward_weights=custom, seed=0)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    # with comfort/carbon/peak weights at zero, reward should equal -cost*10
    assert np.isclose(reward, -10.0 * info["cost"])


def test_rule_based_thermostat_direction():
    ctrl = RuleBasedThermostat(setpoint=23.0, deadband=1.0)
    hot_obs = np.array([26.0, 30, 23, 1, 0.2, 12, 0.5], dtype=np.float32)
    cold_obs = np.array([19.0, 10, 23, 1, 0.2, 12, 0.5], dtype=np.float32)
    neutral_obs = np.array([23.0, 20, 23, 1, 0.2, 12, 0.5], dtype=np.float32)

    assert ctrl.act(hot_obs)[0] < 0     # should cool
    assert ctrl.act(cold_obs)[0] > 0    # should heat
    assert ctrl.act(neutral_obs)[0] == 0.0  # within deadband -> off


def test_pid_controller_reduces_error_over_steps():
    """Sanity check: running PID in a simple env for a while should not
    diverge and should keep the action bounded."""
    env = CoolTwinEnv(episode_hours=6, seed=0)
    ctrl = PIDController(setpoint=23.0)
    metrics = run_episode(env, ctrl)
    assert np.isfinite(metrics["total_reward"])
    assert metrics["peak_power_w"] <= env.hvac_max_watts + 1e-3


def test_random_controller_is_bounded():
    ctrl = RandomController(seed=0)
    for _ in range(20):
        a = ctrl.act(np.zeros(7, dtype=np.float32))
        assert -1.0 <= a[0] <= 1.0


def test_env_with_residual_model_runs_end_to_end():
    """Confirms CoolTwinEnv correctly steps through the hybrid twin when a
    trained residual model is supplied (Phase 3 wiring of Phase 2's model)."""
    import torch
    from twin.residual_lstm import ResidualLSTM

    torch.manual_seed(0)
    residual_model = ResidualLSTM(n_features=6, hidden_size=8)
    residual_model.eval()

    env = CoolTwinEnv(episode_hours=6, residual_model=residual_model, residual_window=8, seed=0)
    obs, _ = env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        steps += 1
    assert steps == env.n_steps


def test_run_episode_returns_expected_keys():
    env = CoolTwinEnv(episode_hours=2, seed=0)
    ctrl = RandomController(seed=0)
    metrics = run_episode(env, ctrl)
    for key in ["total_reward", "total_energy_kwh", "total_carbon_kg", "total_discomfort", "peak_power_w", "steps"]:
        assert key in metrics
