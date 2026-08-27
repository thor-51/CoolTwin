import numpy as np
import torch

from twin.data_gen import generate_episode, generate_dataset
from twin.residual_lstm import ResidualLSTM, DirectLSTM, ResidualWindowDataset, train_model
from twin.hybrid_twin import HybridTwin


def test_generate_episode_shapes():
    ep = generate_episode(n_steps=96, seed=0)
    for key in ["T_out", "hour", "day_frac", "Q_hvac", "Q_gain_base", "T_in_true", "T_in_physics", "residual"]:
        assert key in ep
        assert len(ep[key]) == 96


def test_residual_is_nontrivial():
    """The synthetic ground truth must actually differ from the physics-only
    prediction -- otherwise there's nothing for the residual model to learn,
    and the whole Phase 2 comparison would be meaningless."""
    ep = generate_episode(n_steps=96, seed=0)
    residual_rmse = np.sqrt(np.mean(ep["residual"] ** 2))
    assert residual_rmse > 0.1  # meaningful gap, not just float noise


def test_generate_dataset_multiple_episodes():
    episodes = generate_dataset(n_episodes=3, n_steps=48, seed=0)
    assert len(episodes) == 3
    # different seeds -> different trajectories
    assert not np.allclose(episodes[0]["T_out"], episodes[1]["T_out"])


def test_residual_lstm_forward_shape():
    model = ResidualLSTM(n_features=6, hidden_size=8)
    x = torch.randn(4, 8, 6)  # batch=4, window=8, features=6
    out = model(x)
    assert out.shape == (4,)


def test_dataset_windowing():
    episodes = generate_dataset(n_episodes=2, n_steps=50, seed=0)
    ds = ResidualWindowDataset(episodes, window=8, target="residual")
    # each episode of length 50 with window 8 yields 42 samples
    assert len(ds) == 2 * (50 - 8)
    X, y = ds[0]
    assert X.shape == (8, 6)
    assert y.shape == ()


def test_training_reduces_loss():
    """Smoke test: a couple epochs of training should reduce loss, confirming
    gradients flow and the model is actually learning something (not a silent
    no-op)."""
    train_eps = generate_dataset(n_episodes=4, n_steps=100, seed=1)
    val_eps = generate_dataset(n_episodes=2, n_steps=100, seed=2)
    train_ds = ResidualWindowDataset(train_eps, window=8, target="residual")
    val_ds = ResidualWindowDataset(val_eps, window=8, target="residual")

    torch.manual_seed(0)
    model = ResidualLSTM(n_features=6, hidden_size=8)
    history = train_model(model, train_ds, val_ds, epochs=5, verbose=False)

    assert history["train_loss"][-1] < history["train_loss"][0]


def test_hybrid_twin_beats_physics_only():
    """The core Phase 2 claim, as a regression test: after a few epochs of
    training, hybrid-corrected predictions should be closer to ground truth
    than the raw physics-only predictions on the same episode."""
    train_eps = generate_dataset(n_episodes=8, n_steps=200, seed=1)
    val_eps = generate_dataset(n_episodes=2, n_steps=200, seed=2)
    test_ep = generate_episode(n_steps=200, seed=3)

    train_ds = ResidualWindowDataset(train_eps, window=8, target="residual")
    val_ds = ResidualWindowDataset(val_eps, window=8, target="residual")

    torch.manual_seed(0)
    model = ResidualLSTM(n_features=6, hidden_size=16)
    train_model(model, train_ds, val_ds, epochs=10, verbose=False)

    twin = HybridTwin(model, window=8)
    hybrid_pred = twin.correct_episode(test_ep)

    window = 8
    physics_rmse = np.sqrt(np.mean((test_ep["T_in_physics"][window:] - test_ep["T_in_true"][window:]) ** 2))
    hybrid_rmse = np.sqrt(np.mean((hybrid_pred[window:] - test_ep["T_in_true"][window:]) ** 2))

    assert hybrid_rmse < physics_rmse
