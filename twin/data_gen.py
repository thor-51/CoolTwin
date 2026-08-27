"""
data_gen.py

Generates synthetic "ground-truth" building data for training and evaluating
the residual correction model.

Why synthetic and not Sinergym here: Phase 2's job is to prove the *hybrid
modeling methodology* works (physics + residual ML beats either alone) before
paying the cost of a full EnergyPlus integration (see docs/sinergym_setup.md).
So we construct a "true" building that:

  1. Uses the same 3R2C structure as our nominal RC model, but with
     different (unknown to the controller) parameter values -- simulating
     the fact that a real building's true thermal parameters are never
     exactly what you assumed.
  2. Adds a nonlinear solar-gain effect (proportional to a daytime bump,
     not captured by the linear RC network) directly heating the indoor node.
  3. Adds sensor noise to the observed indoor temperature.

The "nominal" RC model (twin/rc_model.py, default RCParams) is what the
controller/agent actually has access to. The gap between its predictions and
this synthetic ground truth is exactly the kind of systematic + stochastic
error the residual LSTM is trained to correct.
"""

from __future__ import annotations

import numpy as np

from twin.rc_model import RCThermalZone, RCParams


TRUE_PARAMS = RCParams(
    R_out_wall=1.6e-3,
    R_wall_in=0.7e-3,
    R_in_out=4.5e-3,
    C_wall=6.0e6,
    C_in=4.2e5,
)


def _solar_gain(hour: float, day_frac: float, rng: np.random.Generator) -> float:
    """Nonlinear, weather-dependent solar heat gain on the indoor node (Watts).
    Peaks around midday, modulated by a slowly varying 'cloud cover' factor.
    This is the kind of effect a linear RC network structurally cannot
    represent without an explicit solar input channel -- exactly what we
    want the residual model to pick up on instead.
    """
    cloud_factor = 0.5 + 0.5 * np.sin(day_frac * 2 * np.pi / 5.0 + 1.0)  # slow ~5 day cycle
    daylight = max(0.0, np.sin((hour - 6) / 12 * np.pi)) if 6 <= hour <= 18 else 0.0
    return 400.0 * daylight**1.5 * cloud_factor + rng.normal(0, 15)


def generate_episode(
    n_steps: int = 672,  # one week at 15-min steps
    dt_seconds: float = 900.0,
    seed: int = 0,
):
    """Returns a dict of arrays: T_out, hour, day_frac, Q_hvac, Q_gain,
    T_in_true (ground truth, noisy), T_in_physics (nominal RC-only prediction).
    """
    rng = np.random.default_rng(seed)
    hours = (np.arange(n_steps) * dt_seconds / 3600.0) % 24
    days = np.arange(n_steps) * dt_seconds / 3600.0 / 24.0

    T_out = 24 + 3 * np.sin(days / 30 * 2 * np.pi) + 6 * np.sin((hours - 9) / 24 * 2 * np.pi)
    T_out += rng.normal(0, 0.3, n_steps)

    occ = ((hours >= 9) & (hours < 18)).astype(float)
    Q_gain_base = np.where(occ > 0, 300.0, 50.0)

    # A simple thermostat-ish HVAC signal so the data covers a realistic
    # operating range (this is NOT the RL policy -- just data-generation).
    Q_hvac = np.where(T_out > 26, -1200.0, np.where(T_out < 18, 800.0, 0.0))
    Q_hvac += rng.normal(0, 50, n_steps)

    solar = np.array([_solar_gain(hours[i], days[i], rng) for i in range(n_steps)])
    Q_gain_true = Q_gain_base + solar

    # --- ground truth: "true" params + solar gain + sensor noise ---
    true_zone = RCThermalZone(TRUE_PARAMS)
    traj_true = true_zone.simulate(T_out, Q_hvac, Q_gain_true, T_wall0=22, T_in0=22, dt_seconds=dt_seconds)
    T_in_true = traj_true[1:, 1] + rng.normal(0, 0.15, n_steps)  # sensor noise

    # --- what the controller's nominal physics model predicts (no solar term,
    #     nominal/incorrect params, uses only the base occupancy gain it knows about) ---
    nominal_zone = RCThermalZone(RCParams())
    traj_phys = nominal_zone.simulate(T_out, Q_hvac, Q_gain_base, T_wall0=22, T_in0=22, dt_seconds=dt_seconds)
    T_in_physics = traj_phys[1:, 1]

    return {
        "T_out": T_out,
        "hour": hours,
        "day_frac": days,
        "Q_hvac": Q_hvac,
        "Q_gain_base": Q_gain_base,
        "T_in_true": T_in_true,
        "T_in_physics": T_in_physics,
        "residual": T_in_true - T_in_physics,
    }


def generate_dataset(n_episodes: int = 20, n_steps: int = 672, seed: int = 0):
    """Generates multiple independent episodes (different random seeds) for
    train/val/test splitting."""
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n_episodes):
        ep_seed = int(rng.integers(0, 1_000_000))
        episodes.append(generate_episode(n_steps=n_steps, seed=ep_seed))
    return episodes


if __name__ == "__main__":
    ep = generate_episode(n_steps=96, seed=0)
    rmse_physics_only = np.sqrt(np.mean(ep["residual"] ** 2))
    print(f"Physics-only RMSE vs ground truth (1 day, no correction): {rmse_physics_only:.3f} C")
    print(f"Residual stats -- mean: {ep['residual'].mean():.3f}, std: {ep['residual'].std():.3f}")
