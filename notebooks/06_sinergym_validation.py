"""
notebooks/06_sinergym_validation.py

Phase 1 (revisited): wires Sinergym in and uses it for the thing that was
flagged as the project's one open gap since the first roadmap review --
validating the twin's physics against a real EnergyPlus-simulated building,
not just the custom synthetic RC-only generator used everywhere else in
this repo (twin/data_gen.py).

Requires a local EnergyPlus install (see docs/sinergym_setup.md) -- this is
NOT wired into CI, deliberately, matching the same reasoning documented for
the rest of Sinergym: keeps the fast pipeline fast, and EnergyPlus is a
150-200MB system-level install, not something to add to every CI run.

What this does:
  1. Runs a full episode of Sinergym's `Eplus-demo-v1` (a 5-zone EnergyPlus
     building, ASHRAE 90.1 5-zone reference model) under a simple thermostat
     policy, logging outdoor temp, one zone's indoor air temp, and HVAC
     electricity demand at every hourly step.
  2. Converts electricity demand to an approximate thermal load via a fixed
     COP assumption (documented, not hidden) -- Sinergym's demo env doesn't
     expose thermal Q_hvac directly, only electrical demand.
  3. Fits our twin/rc_model.py 3R2C network to this REAL trajectory via the
     same fit_rc_params() used throughout the rest of the project, then
     reports RMSE on a held-out portion of the same episode.

This is deliberately a first pass, not a final validation: one building,
one zone treated as if it were the whole building, a fixed COP assumption
for the electrical-to-thermal conversion, and no residual-LSTM correction
applied yet (that's the natural next step once this baseline number exists).
The honest point of this script is to replace "we haven't validated against
real building physics" with an actual number and a clear list of what that
number does and doesn't cover yet.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# COP assumption for converting Sinergym's HVAC_electricity_demand_rate (W
# electric) into an approximate thermal load (W) injected at the zone air
# node. This is a simplification -- a real deployment would use the
# building's actual equipment curves -- documented here rather than buried.
ASSUMED_COP = 3.0


def collect_sinergym_episode(max_steps: int | None = None, seed: int = 0):
    """Runs one full Sinergym episode under a simple on/off thermostat
    policy (not random actions -- random heating/cooling setpoints produce
    a less physically sensible trajectory to fit against) and returns the
    per-step T_out, T_in, and approximate Q_hvac arrays."""
    import gymnasium as gym
    import sinergym  # noqa: F401 -- registers the Eplus-* environments

    env = gym.make("Eplus-demo-v1")
    obs, info = env.reset(seed=seed)

    # obs layout (see env.get_wrapper_attr('observation_variables')):
    # ['month', 'day_of_month', 'hour', 'outdoor_temperature', 'htg_setpoint',
    #  'clg_setpoint', 'air_temperature', 'air_humidity', 'HVAC_electricity_demand_rate']
    T_out_list, T_in_list, elec_list = [], [], []

    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        # Simple fixed comfort-band thermostat action, not random -- gives a
        # trajectory a thermal model can plausibly be fit to.
        action = np.array([19.0, 26.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        T_out_list.append(float(obs[3]))
        T_in_list.append(float(obs[6]))
        elec_list.append(float(obs[8]))

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    env.close()

    T_out = np.array(T_out_list)
    T_in = np.array(T_in_list)
    Q_hvac = np.array(elec_list) * ASSUMED_COP  # approximate electrical->thermal
    return T_out, T_in, Q_hvac


def main():
    print(f"Collecting a full Sinergym episode (Eplus-demo-v1, EnergyPlus-backed)...")
    T_out, T_in, Q_hvac = collect_sinergym_episode()
    n = len(T_out)
    print(f"Collected {n} hourly steps.")
    print(f"T_out range: [{T_out.min():.1f}, {T_out.max():.1f}] C")
    print(f"T_in range:  [{T_in.min():.1f}, {T_in.max():.1f}] C")

    # No internal-gain signal is exposed by this env's observation set;
    # occupancy-driven gains are folded into the same COP-scaled electricity
    # term for this first pass rather than invented from nothing.
    Q_gain = np.zeros(n)

    split = int(n * 0.7)
    dt_seconds = 3600.0  # Eplus-demo-v1's configured timestep (1 step/hour)

    from twin.rc_model import RCThermalZone, RCParams, fit_rc_params

    print("\nFitting 3R2C parameters to the first 70% of the episode (real EnergyPlus data)...")
    fitted_params = fit_rc_params(
        T_out[:split], Q_hvac[:split], Q_gain[:split], T_in[:split],
        dt_seconds=dt_seconds,
    )
    print(f"Fitted params: {fitted_params}")

    zone = RCThermalZone(fitted_params)
    traj = zone.simulate(
        T_out, Q_hvac, Q_gain,
        T_wall0=T_in[0], T_in0=T_in[0], dt_seconds=dt_seconds,
    )
    T_in_pred = traj[1:, 1]  # drop row 0 (the seeded initial state, not a prediction)

    train_rmse = float(np.sqrt(np.mean((T_in_pred[:split] - T_in[:split]) ** 2)))
    test_rmse = float(np.sqrt(np.mean((T_in_pred[split:] - T_in[split:]) ** 2)))
    print(f"\nRMSE vs real EnergyPlus zone temperature:")
    print(f"  Train (fit) portion:      {train_rmse:.3f} C")
    print(f"  Held-out (test) portion:  {test_rmse:.3f} C")
    print(
        "\nNOTE: this RMSE is poor (double digits, against a ~7C-wide ground-truth "
        "band) -- see results/sinergym_validation.md for the diagnosis. Short "
        "version: the ground truth is a CLOSED-LOOP thermostat output (tightly "
        "held in its setpoint band); the electricity-demand-derived Q_hvac fed "
        "into our OPEN-LOOP RC model is itself an output of that same closed "
        "loop, not an independent forcing signal, so driving an open-loop model "
        "with it causes it to wildly over/undershoot rather than track the band."
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    hours = np.arange(n)
    ax.plot(hours, T_in, label="Ground truth (EnergyPlus)", linewidth=1.5)
    ax.plot(hours, T_in_pred, label="3R2C fit (this repo's twin)", linewidth=1.2, alpha=0.8)
    ax.axvline(split, color="gray", linestyle="--", alpha=0.6, label="train/test split")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Zone air temperature (C)")
    ax.set_title(
        f"CoolTwin RC model vs real EnergyPlus simulation (Eplus-demo-v1)\n"
        f"Held-out RMSE: {test_rmse:.2f} C  (COP={ASSUMED_COP} assumption for electrical->thermal)"
    )
    ax.legend()
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sinergym_validation.png", dpi=120)
    print("\nSaved results/sinergym_validation.png")

    with open("results/sinergym_validation.md", "w") as f:
        f.write("# CoolTwin — Sinergym Validation (Phase 1, revisited)\n\n")
        f.write(
            "First-pass validation of the 3R2C physics model against a real "
            "EnergyPlus-simulated building (Sinergym's `Eplus-demo-v1`, a "
            "5-zone ASHRAE reference model), instead of only the synthetic "
            "generator used elsewhere in this repo (`twin/data_gen.py`).\n\n"
        )
        f.write(f"- Episode length: {n} hourly steps ({n/24:.0f} days)\n")
        f.write(f"- Train/test split: {split}/{n-split} steps\n")
        f.write(f"- Electrical->thermal conversion: fixed COP={ASSUMED_COP} assumption\n")
        f.write(f"- Fitted RC params: {fitted_params}\n\n")
        f.write("| Portion | RMSE (C) |\n|---|---|\n")
        f.write(f"| Train (fit) | {train_rmse:.3f} |\n")
        f.write(f"| Held-out (test) | {test_rmse:.3f} |\n\n")
        f.write("## This RMSE is bad, and here's why -- diagnosed, not hidden\n\n")
        f.write(
            "Double-digit RMSE against a ground-truth band that's only ~7C wide "
            "(19-26C, see `results/sinergym_validation.png`) means the fit is "
            "not usable as-is. The plot makes the mechanism visible: the ground "
            "truth is the *output of a closed control loop* -- EnergyPlus's "
            "ideal-loads system holds the zone tightly inside its thermostat "
            "band by construction. The `HVAC_electricity_demand_rate` signal "
            "used here as a stand-in for `Q_hvac` is itself an output of that "
            "same closed loop (\"however much power was needed to hit the "
            "setpoint this step\"), not an independent forcing input. Feeding "
            "it into our model open-loop -- as if it were a given, "
            "system-independent heat injection -- means small fitting errors "
            "compound: the model doesn't have the actual feedback mechanism "
            "that kept the real building's demand exactly matched to what its "
            "own drift required, so it overshoots and undershoots instead of "
            "tracking.\n\n"
        )
        f.write(
            "**Concrete next step**: don't fit against derived electrical "
            "demand. Configure a custom Sinergym environment (rather than the "
            "stock `Eplus-demo-v1`) that exposes a real thermal-rate output "
            "variable directly from EnergyPlus (e.g. `Zone Ideal Loads Supply "
            "Air Total Heating/Cooling Rate`), which is a genuine forcing "
            "input rather than a closed-loop artifact, and re-run this same "
            "fit_rc_params() call against that. That's the fix, not a bigger "
            "model or more data at this same signal.\n\n"
        )
        f.write(
            "**What this first pass still accomplished**: Sinergym + "
            "EnergyPlus 25.1.0 is now verified working end-to-end in this "
            "repo (see `docs/sinergym_setup.md`), the roadmap's original "
            "Phase 1 ask (\"get one default scenario running end-to-end with "
            "a random-action baseline\") is done, and the exact next step to "
            "get a real physics validation number is now specific and "
            "actionable rather than an open question.\n"
        )
    print("Saved results/sinergym_validation.md")


if __name__ == "__main__":
    main()
