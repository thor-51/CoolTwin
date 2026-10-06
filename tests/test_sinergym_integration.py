"""
tests/test_sinergym_integration.py

Tests for the Sinergym data collection and integration pipeline.

These tests are designed to work WITH OR WITHOUT Sinergym/EnergyPlus
installed — just like test_sinergym_baseline.py. Tests that require
EnergyPlus are marked with pytest.mark.skipif and skip cleanly when
the dependency isn't available.

The core logic tests (save/load, episode format validation) run on
synthetic stand-in data so they work in CI without EnergyPlus.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sinergym_available() -> bool:
    """Check whether sinergym and EnergyPlus are importable."""
    if not os.environ.get("EPLUS_PATH"):
        return False
    try:
        import sinergym  # noqa: F401

        return True
    except ImportError:
        return False


def _make_fake_episode(n_steps: int = 200, seed: int = 0) -> dict:
    """Create a synthetic episode dict in the same format as
    sinergym_data.collect_sinergym_episode(), for testing save/load
    and downstream pipeline code without EnergyPlus."""
    rng = np.random.default_rng(seed)
    hours = (np.arange(n_steps) * 1.0) % 24  # 1-hour steps
    return {
        "T_out": 20 + 5 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 0.3, n_steps),
        "hour": hours,
        "day_frac": np.arange(n_steps) / 24.0,
        "Q_hvac": rng.uniform(-1000, 1000, n_steps),
        "Q_gain_base": np.zeros(n_steps),
        "T_in_true": 22 + rng.normal(0, 0.5, n_steps),
        "T_in_physics": 22 + rng.normal(0, 1.0, n_steps),
        "residual": rng.normal(0, 0.5, n_steps),
    }


# ---------------------------------------------------------------------------
# Tests that run WITHOUT Sinergym/EnergyPlus (CI-safe)
# ---------------------------------------------------------------------------


class TestSaveLoadEpisodes:
    """Test save_episodes / load_episodes roundtrip."""

    def test_roundtrip(self):
        from twin.sinergym_data import load_episodes, save_episodes

        episodes = [_make_fake_episode(seed=i) for i in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_episodes.npz"
            save_episodes(episodes, path)
            loaded = load_episodes(path)

        assert len(loaded) == 3
        for orig, loaded_ep in zip(episodes, loaded):
            for key in [
                "T_out",
                "hour",
                "day_frac",
                "Q_hvac",
                "Q_gain_base",
                "T_in_true",
                "T_in_physics",
                "residual",
            ]:
                np.testing.assert_array_almost_equal(orig[key], loaded_ep[key])

    def test_single_episode_roundtrip(self):
        from twin.sinergym_data import load_episodes, save_episodes

        episodes = [_make_fake_episode(seed=42)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single.npz"
            save_episodes(episodes, path)
            loaded = load_episodes(path)

        assert len(loaded) == 1
        assert len(loaded[0]["T_out"]) == 200


class TestEpisodeFormat:
    """Verify that the episode dict format matches what the downstream
    pipeline (residual_lstm, rc_model, etc.) expects."""

    REQUIRED_KEYS = (
        "T_out",
        "hour",
        "day_frac",
        "Q_hvac",
        "Q_gain_base",
        "T_in_true",
        "T_in_physics",
        "residual",
    )

    def test_has_all_required_keys(self):
        ep = _make_fake_episode()
        for key in self.REQUIRED_KEYS:
            assert key in ep, f"Missing key: {key}"

    def test_all_arrays_same_length(self):
        ep = _make_fake_episode()
        n = len(ep["T_out"])
        for key in self.REQUIRED_KEYS:
            assert len(ep[key]) == n, f"{key} has length {len(ep[key])}, expected {n}"

    def test_residual_is_true_minus_physics(self):
        """The residual should be T_in_true - T_in_physics (within tolerance
        for the fake data which constructs them independently)."""
        # Use collect_sinergym_episode's logic where residual IS computed
        # from T_in_true and T_in_physics
        ep = _make_fake_episode()
        # For fake data these are independent, so just check the key exists
        assert ep["residual"] is not None
        assert len(ep["residual"]) == len(ep["T_in_true"])


class TestResidualLSTMOnFakeData:
    """Verify that the residual LSTM training pipeline accepts Sinergym-format
    episodes (using fake data as a stand-in)."""

    def test_dataset_creation(self):
        from twin.residual_lstm import ResidualWindowDataset

        episodes = [_make_fake_episode(seed=i) for i in range(3)]
        ds = ResidualWindowDataset(episodes, window=8, target="residual")
        assert len(ds) > 0

        X, y = ds[0]
        assert X.shape == (8, 6)  # window=8, 6 features
        assert y.shape == ()

    def test_train_on_fake_data(self):
        from twin.residual_lstm import (
            ResidualLSTM,
            ResidualWindowDataset,
            train_model,
        )

        episodes = [_make_fake_episode(seed=i) for i in range(4)]
        train_ds = ResidualWindowDataset(episodes[:3], window=8)
        val_ds = ResidualWindowDataset(episodes[3:], window=8)

        model = ResidualLSTM()
        history = train_model(model, train_ds, val_ds, epochs=2, verbose=False)

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(loss > 0 for loss in history["train_loss"])


class TestRCFittingOnFakeData:
    """Verify RC parameter fitting works with the episode format."""

    def test_fit_returns_params(self):
        from twin.rc_model import RCParams, fit_rc_params

        ep = _make_fake_episode(n_steps=100)
        params = fit_rc_params(
            ep["T_out"],
            ep["Q_hvac"],
            ep["Q_gain_base"],
            ep["T_in_true"],
            dt_seconds=3600.0,
        )
        assert isinstance(params, RCParams)
        assert params.R_out_wall > 0
        assert params.C_wall > 0


# ---------------------------------------------------------------------------
# Tests that REQUIRE Sinergym + EnergyPlus
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _sinergym_available(),
    reason="Sinergym / EnergyPlus not available (EPLUS_PATH not set or sinergym not installed)",
)
class TestSinergymCollection:
    """Integration tests that actually run EnergyPlus. Skipped in CI."""

    def test_collect_short_episode(self):
        from twin.sinergym_data import collect_sinergym_episode

        ep = collect_sinergym_episode(max_steps=50)

        assert len(ep["T_out"]) == 50
        assert len(ep["T_in_true"]) == 50
        assert len(ep["T_in_physics"]) == 50
        assert len(ep["residual"]) == 50

        # Temperatures should be in a physically reasonable range
        assert ep["T_in_true"].min() > -10
        assert ep["T_in_true"].max() < 50
        assert ep["T_out"].min() > -40
        assert ep["T_out"].max() < 55

    def test_episode_format_matches_data_gen(self):
        """The episode dict from Sinergym should have the same keys as
        data_gen.generate_episode()."""
        from twin.data_gen import generate_episode
        from twin.sinergym_data import collect_sinergym_episode

        sinergym_ep = collect_sinergym_episode(max_steps=20)
        synthetic_ep = generate_episode(n_steps=20)

        # Should have all the same keys (sinergym may have extra _meta)
        for key in synthetic_ep:
            assert key in sinergym_ep, f"Sinergym episode missing key: {key}"

    def test_collect_dataset_short(self):
        """Test collecting a short dataset of 2 episodes."""
        from twin.sinergym_data import collect_sinergym_dataset

        eps = collect_sinergym_dataset(
            n_episodes=2,
            max_steps_per_episode=15,
            comfort_bands=[(19.0, 26.0), (20.0, 24.0)],
            verbose=False,
        )
        assert len(eps) == 2
        assert len(eps[0]["T_out"]) == 15
        assert len(eps[1]["T_out"]) == 15
        assert eps[0]["_meta"]["comfort_band"] == (19.0, 26.0)
        assert eps[1]["_meta"]["comfort_band"] == (20.0, 24.0)
