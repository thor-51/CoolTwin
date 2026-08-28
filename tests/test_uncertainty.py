import numpy as np
import torch

from twin.data_gen import generate_dataset
from twin.residual_lstm import ResidualWindowDataset, build_feature_vector, train_model
from twin.env import CoolTwinEnv
from uncertainty.mc_dropout import MCDropoutResidualLSTM, predict_with_uncertainty
from uncertainty.ensembles import DeepEnsemble
from uncertainty.calibration import compute_reliability_diagram, expected_calibration_error
from uncertainty.safety_layer import UncertaintyGatedController
from evaluation.baselines import RuleBasedThermostat


def test_build_feature_vector_matches_offline_construction():
    """The shared single-step feature builder (used online in env.py and in
    the uncertainty modules) must produce exactly the same encoding as the
    offline dataset builder in residual_lstm.py, or online/offline models
    would silently see different feature semantics."""
    from twin.data_gen import generate_episode
    from twin.residual_lstm import _build_features

    ep = generate_episode(n_steps=20, seed=0)
    offline_feats = _build_features(ep)

    t = 5
    online_feat = build_feature_vector(
        ep["T_out"][t], ep["hour"][t], ep["Q_hvac"][t], ep["Q_gain_base"][t], ep["T_in_physics"][t]
    )
    assert np.allclose(offline_feats[t], online_feat)


def test_mc_dropout_gives_nonzero_spread():
    """With dropout active across stochastic passes, predictive std should
    be nonzero (i.e. dropout is actually being applied at inference, not
    silently disabled)."""
    torch.manual_seed(0)
    model = MCDropoutResidualLSTM(n_features=6, hidden_size=16, num_layers=2, dropout=0.3)
    x = torch.randn(4, 8, 6)
    mean, std = predict_with_uncertainty(model, x, n_samples=20)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert np.all(std > 0)


def test_mc_dropout_model_is_deterministic_in_eval_mode():
    """After predict_with_uncertainty, the model should be back in eval mode
    so downstream code (e.g. HybridTwin) gets deterministic single-pass
    predictions, not accidentally-stochastic ones."""
    torch.manual_seed(0)
    model = MCDropoutResidualLSTM(n_features=6, hidden_size=8)
    x = torch.randn(1, 8, 6)
    predict_with_uncertainty(model, x, n_samples=5)
    assert not model.training

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


def test_deep_ensemble_predicts_and_has_spread():
    train_eps = generate_dataset(n_episodes=3, n_steps=60, seed=1)
    val_eps = generate_dataset(n_episodes=2, n_steps=60, seed=2)
    train_ds = ResidualWindowDataset(train_eps, window=8, target="residual")
    val_ds = ResidualWindowDataset(val_eps, window=8, target="residual")

    ensemble = DeepEnsemble(n_models=3, n_features=6, hidden_size=8)
    ensemble.fit(train_ds, val_ds, epochs=2)

    x = torch.randn(5, 8, 6)
    mean, std = ensemble.predict_with_uncertainty(x)
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(std >= 0)


def test_reliability_diagram_perfect_calibration():
    """If predictive intervals are constructed with the exact true std,
    observed coverage should track nominal confidence closely (allowing for
    finite-sample noise)."""
    rng = np.random.default_rng(0)
    n = 5000
    true_std = 1.0
    y_mean = rng.normal(0, 1, n)
    y_true = y_mean + rng.normal(0, true_std, n)
    y_std = np.full(n, true_std)

    nominal, observed = compute_reliability_diagram(y_true, y_mean, y_std)
    ece = expected_calibration_error(nominal, observed)
    assert ece < 0.03  # well-calibrated by construction


def test_reliability_diagram_overconfident_has_high_ece():
    """If predicted std is much smaller than the true noise, observed
    coverage should fall well below nominal (overconfident model), giving a
    high ECE -- sanity check that the metric actually detects miscalibration."""
    rng = np.random.default_rng(0)
    n = 5000
    y_mean = rng.normal(0, 1, n)
    y_true = y_mean + rng.normal(0, 2.0, n)  # true noise is 2.0
    y_std = np.full(n, 0.1)  # model claims much tighter uncertainty

    nominal, observed = compute_reliability_diagram(y_true, y_mean, y_std)
    ece = expected_calibration_error(nominal, observed)
    assert ece > 0.2


def test_fit_variance_scale_recovers_true_underestimated_factor():
    """If predicted std systematically underestimates true noise by a known
    factor, fit_variance_scale should recover approximately that factor."""
    from uncertainty.calibration import fit_variance_scale

    rng = np.random.default_rng(0)
    n = 4000
    true_std = 2.0
    y_mean = rng.normal(0, 1, n)
    y_true = y_mean + rng.normal(0, true_std, n)
    y_std = np.full(n, 0.5)  # underestimates true_std by a factor of 4

    scale = fit_variance_scale(y_true, y_mean, y_std, scale_range=(0.1, 10.0), n_grid=200)
    assert 3.0 < scale < 5.0  # should land close to the true factor of 4


def test_safety_layer_falls_back_when_uncertain():
    class DummyRLModel:
        def predict(self, obs, deterministic=True):
            return np.array([1.0], dtype=np.float32), None  # always "heat full"

    high_uncertainty_fn = lambda window: 999.0  # always triggers fallback
    low_uncertainty_fn = lambda window: 0.0     # never triggers fallback

    fallback = RuleBasedThermostat()
    obs = np.array([23.0, 20, 23, 1, 0.2, 12, 0.5], dtype=np.float32)
    dummy_window = np.zeros((8, 6), dtype=np.float32)

    ctrl_high = UncertaintyGatedController(DummyRLModel(), high_uncertainty_fn, std_threshold=0.5, fallback=fallback)
    action, used_fallback = ctrl_high.act(obs, dummy_window)
    assert used_fallback is True
    assert np.allclose(action, fallback.act(obs))

    ctrl_low = UncertaintyGatedController(DummyRLModel(), low_uncertainty_fn, std_threshold=0.5, fallback=fallback)
    action, used_fallback = ctrl_low.act(obs, dummy_window)
    assert used_fallback is False
    assert np.allclose(action, np.array([1.0], dtype=np.float32))


def test_safety_layer_defaults_to_rl_during_warmup():
    """With feature_window=None (not enough history yet), the controller
    should defer to the RL policy, matching how HybridTwin/CoolTwinEnv treat
    the warm-up period before a full window is available."""
    class DummyRLModel:
        def predict(self, obs, deterministic=True):
            return np.array([0.5], dtype=np.float32), None

    ctrl = UncertaintyGatedController(DummyRLModel(), lambda w: 999.0, std_threshold=0.1)
    obs = np.zeros(7, dtype=np.float32)
    action, used_fallback = ctrl.act(obs, feature_window=None)
    assert used_fallback is False
    assert np.allclose(action, np.array([0.5], dtype=np.float32))


def test_safety_layer_fallback_rate_property():
    class DummyRLModel:
        def predict(self, obs, deterministic=True):
            return np.array([0.0], dtype=np.float32), None

    ctrl = UncertaintyGatedController(DummyRLModel(), lambda w: 999.0, std_threshold=0.5)
    obs = np.zeros(7, dtype=np.float32)
    window = np.zeros((8, 6), dtype=np.float32)
    for _ in range(10):
        ctrl.act(obs, window)
    assert ctrl.fallback_rate == 1.0
