"""
sinergym_data.py

Collects episodes from Sinergym (EnergyPlus-backed environments) and converts
them into the same dict format produced by twin/data_gen.py:

    {T_out, hour, day_frac, Q_hvac, Q_gain_base, T_in_true, T_in_physics, residual}

This means the entire downstream pipeline -- RC parameter fitting, residual
LSTM training, RL training, uncertainty, explainability -- can switch from
synthetic to real EnergyPlus data by changing ONE import.

Design decisions
----------------
1. **Thermostat policy, not random actions.** Random HVAC actions produce
   physically nonsensical trajectories (wild temperature swings) that a thermal
   model can't meaningfully be fit to. A fixed comfort-band thermostat gives a
   trajectory that exercises the building's actual thermal dynamics over a
   realistic operating range.

2. **Electrical-to-thermal via COP.** Sinergym's stock 5-zone environments
   expose `HVAC_electricity_demand_rate` but not the underlying thermal load
   directly. We approximate Q_hvac = electricity * COP. The existing
   06_sinergym_validation.py documented that this is a known approximation --
   a custom Sinergym environment exposing the raw thermal rate would be better,
   but this gets the pipeline working end-to-end with stock environments.

3. **Q_gain = 0.** The stock environment doesn't expose a separate
   occupancy-driven internal gains signal. Setting Q_gain=0 means the RC model
   and LSTM both learn the building's behavior without a separate gain input --
   the LSTM residual naturally picks up whatever internal gains pattern
   EnergyPlus is simulating.

4. **Multiple episodes via different seeds and weather files.** EnergyPlus is
   deterministic for a given building + weather file, so seed variation alone
   doesn't give truly different episodes. When available, different weather
   files (via Sinergym env variants like hot/mixed/cool) provide real variety.

Requires: sinergym >= 3.0, a local EnergyPlus install. See docs/sinergym_setup.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Guard: fail fast with a helpful message if sinergym isn't available
# ---------------------------------------------------------------------------


def _check_sinergym():
    """Returns the sinergym module if it and EnergyPlus are both available.
    Raises RuntimeError with actionable diagnostics if not."""
    eplus_path = os.environ.get("EPLUS_PATH")
    if not eplus_path:
        raise RuntimeError(
            "EPLUS_PATH is not set. See docs/sinergym_setup.md for how to "
            "install EnergyPlus and configure the two required env vars."
        )
    try:
        import sinergym  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"sinergym is not importable ({e}). "
            "Install with: pip install -r requirements-sinergym.txt"
        ) from e
    return True


# ---------------------------------------------------------------------------
# COP assumption (documented, not hidden -- same as 06_sinergym_validation.py)
# ---------------------------------------------------------------------------

ASSUMED_COP = 3.0


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------


def collect_sinergym_episode(
    env_id: str = "Eplus-demo-v1",
    max_steps: int | None = None,
    seed: int = 0,
    comfort_band: tuple[float, float] = (19.0, 26.0),
    verbose: bool = False,
) -> dict:
    """Run one full Sinergym episode under a fixed thermostat policy and return
    the trajectory in the standard CoolTwin episode dict format.

    Parameters
    ----------
    env_id : str
        Sinergym environment ID. Stock options include 'Eplus-demo-v1',
        'Eplus-5zone-hot-continuous-v1', etc.
    max_steps : int or None
        Cap on episode length. None = run the full simulated year.
    seed : int
        Random seed passed to env.reset().
    comfort_band : tuple
        (htg_setpoint, clg_setpoint) sent as the action each step.
    verbose : bool
        Print progress every 1000 steps.

    Returns
    -------
    dict with keys matching twin/data_gen.py's output:
        T_out, hour, day_frac, Q_hvac, Q_gain_base,
        T_in_true, T_in_physics, residual
    """
    import gymnasium as gym
    import sinergym  # noqa: F401 -- registers the Eplus-* environments

    env = gym.make(env_id)
    obs, _info = env.reset(seed=seed)

    # Sinergym's Eplus-demo-v1 obs layout:
    # [month, day_of_month, hour, outdoor_temperature, htg_setpoint,
    #  clg_setpoint, air_temperature, air_humidity, HVAC_electricity_demand_rate]

    T_out_list = []
    T_in_list = []
    elec_list = []
    hour_list = []
    step_count = 0

    terminated = truncated = False
    action = np.array(list(comfort_band), dtype=np.float32)

    while not (terminated or truncated):
        obs, _reward, terminated, truncated, _info = env.step(action)

        T_out_list.append(float(obs[3]))
        T_in_list.append(float(obs[6]))
        elec_list.append(float(obs[8]))
        hour_list.append(float(obs[2]))

        step_count += 1
        if verbose and step_count % 1000 == 0:
            print(f"  step {step_count}: T_out={obs[3]:.1f}, T_in={obs[6]:.1f}")
        if max_steps is not None and step_count >= max_steps:
            break

    env.close()

    T_out = np.array(T_out_list)
    T_in = np.array(T_in_list)
    Q_hvac = np.array(elec_list) * ASSUMED_COP
    hours = np.array(hour_list)
    n = len(T_out)

    # Compute day_frac (cumulative days elapsed) from hours
    # Sinergym reports the hour-of-day (0-23), so we reconstruct cumulative time
    day_frac = np.arange(n) / 24.0  # 1-hour timestep in stock Eplus-demo-v1

    # No separate gain signal is exposed; set to zero (documented assumption)
    Q_gain_base = np.zeros(n)

    # Run our RC model with default params to get T_in_physics
    # (This is what the RC model predicts WITHOUT being fitted to this data --
    # the residual between this and T_in_true is what the LSTM will learn)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from twin.rc_model import RCParams, RCThermalZone

    nominal_zone = RCThermalZone(RCParams())
    traj = nominal_zone.simulate(
        T_out,
        Q_hvac,
        Q_gain_base,
        T_wall0=T_in[0],
        T_in0=T_in[0],
        dt_seconds=3600.0,  # Sinergym Eplus-demo-v1 has 1-hour timesteps
    )
    T_in_physics = traj[1:, 1]

    return {
        "T_out": T_out,
        "hour": hours,
        "day_frac": day_frac,
        "Q_hvac": Q_hvac,
        "Q_gain_base": Q_gain_base,
        "T_in_true": T_in,
        "T_in_physics": T_in_physics,
        "residual": T_in - T_in_physics,
        # Extra metadata (not used by downstream pipeline, but useful for diagnostics)
        "_meta": {
            "env_id": env_id,
            "n_steps": n,
            "dt_seconds": 3600.0,
            "assumed_cop": ASSUMED_COP,
            "comfort_band": comfort_band,
            "seed": seed,
        },
    }


DEFAULT_COMFORT_BANDS = [
    (19.0, 26.0),
    (20.0, 24.0),
    (18.0, 27.0),
    (21.0, 25.0),
    (19.5, 23.5),
]


def collect_sinergym_dataset(
    n_episodes: int = 5,
    env_id: str = "Eplus-demo-v1",
    max_steps_per_episode: int | None = 2160,  # 90 days = manageable size
    base_seed: int = 0,
    comfort_bands: list[tuple[float, float]] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Collect multiple episodes from Sinergym.

    Since EnergyPlus has deterministic weather for a given scenario, we vary
    the seed and comfort setpoint bands across episodes to explore diverse
    realistic operating conditions (e.g., tighter comfort bands, wider setback
    bands, differing occupant preferences).

    Parameters
    ----------
    n_episodes : int
        Number of episodes to collect.
    env_id : str
        Sinergym environment ID.
    max_steps_per_episode : int or None
        Cap per episode. 2160 steps = 90 days at hourly resolution.
    base_seed : int
        Seeds are generated as base_seed + i.
    comfort_bands : list of tuple or None
        List of (htg_setpoint, clg_setpoint) comfort bands to cycle through.
        If None, uses DEFAULT_COMFORT_BANDS.
    verbose : bool
        Print progress.

    Returns
    -------
    list[dict] — each element has the same keys as data_gen.generate_episode().
    """
    _check_sinergym()

    bands = comfort_bands or DEFAULT_COMFORT_BANDS

    episodes = []
    for i in range(n_episodes):
        seed = base_seed + i
        band = bands[i % len(bands)]
        if verbose:
            print(
                f"\n--- Episode {i+1}/{n_episodes} (seed={seed}, env={env_id}, band={band}) ---"
            )

        ep = collect_sinergym_episode(
            env_id=env_id,
            max_steps=max_steps_per_episode,
            seed=seed,
            comfort_band=band,
            verbose=verbose,
        )
        episodes.append(ep)

        if verbose:
            n = ep["_meta"]["n_steps"]
            rmse = float(np.sqrt(np.mean(ep["residual"] ** 2)))
            print(
                f"  Collected {n} steps. "
                f"T_in range: [{ep['T_in_true'].min():.1f}, {ep['T_in_true'].max():.1f}]°C, "
                f"Physics RMSE: {rmse:.2f}°C"
            )

    return episodes


# ---------------------------------------------------------------------------
# Save / load (so you don't re-run EnergyPlus every time)
# ---------------------------------------------------------------------------


def save_episodes(episodes: list[dict], path: str | Path) -> None:
    """Save collected episodes to a .npz file for fast reloading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for i, ep in enumerate(episodes):
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
            arrays[f"ep{i}_{key}"] = ep[key]
    arrays["n_episodes"] = np.array([len(episodes)])

    np.savez_compressed(str(path), **arrays)
    print(f"Saved {len(episodes)} episodes to {path}")


def load_episodes(path: str | Path) -> list[dict]:
    """Load episodes from a .npz file saved by save_episodes()."""
    data = np.load(str(path))
    n_episodes = int(data["n_episodes"][0])
    episodes = []
    for i in range(n_episodes):
        ep = {}
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
            ep[key] = data[f"ep{i}_{key}"]
        episodes.append(ep)
    return episodes


# ---------------------------------------------------------------------------
# CLI entry point: collect and save a dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect Sinergym episodes and save as .npz"
    )
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--env-id", type=str, default="Eplus-demo-v1")
    parser.add_argument("--max-steps", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/sinergym_episodes.npz")
    args = parser.parse_args()

    episodes = collect_sinergym_dataset(
        n_episodes=args.n_episodes,
        env_id=args.env_id,
        max_steps_per_episode=args.max_steps,
        base_seed=args.seed,
        verbose=True,
    )
    save_episodes(episodes, args.output)
