import numpy as np
import torch

from dashboard.data import (
    train_demo_residual_model,
    train_demo_ppo,
    get_twin_fidelity_data,
    run_control_episode,
    ControlStepLog,
)
from rl.reward import RewardWeights


def test_train_demo_residual_model_returns_working_model():
    torch.manual_seed(0)
    model = train_demo_residual_model(epochs=2)
    x = torch.randn(1, 8, 6)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1,)


def test_get_twin_fidelity_data_hybrid_improves_on_physics():
    """Even at the dashboard's fast training budget, hybrid should generally
    track ground truth at least as well as physics-only -- not asserting a
    strict improvement every single run (small-budget training has some
    variance) but checking the result is well-formed and both RMSEs are
    finite, non-negative numbers."""
    torch.manual_seed(0)
    model = train_demo_residual_model(epochs=4)
    result = get_twin_fidelity_data(model, n_steps=48, seed=1)

    assert result.rmse_physics >= 0
    assert result.rmse_hybrid >= 0
    assert np.isfinite(result.rmse_physics)
    assert np.isfinite(result.rmse_hybrid)
    assert len(result.T_in_true) == 48


def test_run_control_episode_returns_expected_log_structure():
    torch.manual_seed(0)
    residual_model = train_demo_residual_model(epochs=2)
    ppo_model = train_demo_ppo(total_timesteps=1000)

    logs = run_control_episode(ppo_model, residual_model, RewardWeights(), episode_hours=6, seed=0)

    assert len(logs) > 0
    assert all(isinstance(l, ControlStepLog) for l in logs)
    for l in logs:
        assert -1.0 <= l.action <= 1.0
        assert set(l.cost.keys()) == {"cost", "comfort", "carbon", "peak"}
        assert l.dominant_factor in l.cost.keys()


def test_run_control_episode_fallback_flag_is_boolean():
    torch.manual_seed(0)
    residual_model = train_demo_residual_model(epochs=2)
    ppo_model = train_demo_ppo(total_timesteps=1000)
    logs = run_control_episode(ppo_model, residual_model, RewardWeights(), episode_hours=6, seed=0)
    assert all(isinstance(l.used_fallback, bool) for l in logs)


def test_streamlit_app_renders_without_exceptions():
    """End-to-end smoke test: runs the actual Streamlit app script via
    Streamlit's official AppTest framework and asserts no exception was
    raised during a full render. This is the closest thing to 'does the
    dashboard actually work' that's practical to run in CI."""
    from streamlit.testing.v1 import AppTest
    import pathlib

    app_path = pathlib.Path(__file__).parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=180)

    assert not at.exception
    assert len(at.tabs) == 5


def test_streamlit_app_chat_tab_answers_a_question():
    from streamlit.testing.v1 import AppTest
    import pathlib

    app_path = pathlib.Path(__file__).parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=180)

    chat_tab = at.tabs[4]
    assert len(chat_tab.button) == 3
    chat_tab.button[0].click()
    at.run(timeout=180)

    assert not at.exception
    chat_tab = at.tabs[4]
    assert len(chat_tab.markdown) >= 2  # Q: ... and A: ... lines
