"""
rc_model.py

A 3R2C (three-resistor, two-capacitor) thermal network model of a single
building zone. This is the physics-based backbone of the CoolTwin digital
twin: it's cheap to simulate, physically interpretable, and gives a strong
grey-box prior that the residual ML model (see residual_lstm.py) corrects.

Nodes:
    T_out  -- outdoor air temperature (input, not a state)
    T_wall -- lumped wall / thermal-mass temperature (state)
    T_in   -- indoor (zone) air temperature (state)

Resistances:
    R_out_wall -- resistance between outdoor air and wall mass
    R_wall_in  -- resistance between wall mass and indoor air
    R_in_out   -- direct envelope leakage resistance (infiltration/windows)

Capacitances:
    C_wall -- thermal mass of the wall/structure
    C_in   -- thermal mass of the indoor air (+ furnishings)

HVAC input Q_hvac (Watts, +heating / -cooling) and internal gains Q_gain
(occupancy, equipment) are injected directly into the indoor air node.

State-space form:
    dT_wall/dt = [ (T_out - T_wall)/R_out_wall - (T_wall - T_in)/R_wall_in ] / C_wall
    dT_in/dt   = [ (T_wall - T_in)/R_wall_in + (T_out - T_in)/R_in_out
                   + Q_hvac + Q_gain ] / C_in
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class RCParams:
    """Physical parameters of the 3R2C network.

    Defaults are rough order-of-magnitude values for a small office zone
    (~50 m^2). These should be replaced with values fit to real/simulated
    data via `fit_rc_params` (Phase 2, parameter estimation step).
    """

    R_out_wall: float = 2.0e-3   # K/W
    R_wall_in: float = 1.0e-3    # K/W
    R_in_out: float = 5.0e-3     # K/W  (direct envelope leakage)
    C_wall: float = 5.0e6        # J/K
    C_in: float = 5.0e5          # J/K


class RCThermalZone:
    """Simulates a single zone's temperature dynamics under the 3R2C model."""

    def __init__(self, params: RCParams | None = None):
        self.params = params or RCParams()

    def _dynamics(self, t: float, y: np.ndarray, T_out: float, Q_hvac: float, Q_gain: float) -> np.ndarray:
        T_wall, T_in = y
        p = self.params

        dT_wall = (
            (T_out - T_wall) / p.R_out_wall - (T_wall - T_in) / p.R_wall_in
        ) / p.C_wall

        dT_in = (
            (T_wall - T_in) / p.R_wall_in
            + (T_out - T_in) / p.R_in_out
            + Q_hvac
            + Q_gain
        ) / p.C_in

        return np.array([dT_wall, dT_in])

    def step(
        self,
        T_wall: float,
        T_in: float,
        T_out: float,
        Q_hvac: float,
        Q_gain: float,
        dt_seconds: float = 900.0,
    ) -> tuple[float, float]:
        """Advance the zone state by one control timestep (default 15 min).

        Returns the new (T_wall, T_in).
        """
        sol = solve_ivp(
            self._dynamics,
            t_span=(0, dt_seconds),
            y0=[T_wall, T_in],
            args=(T_out, Q_hvac, Q_gain),
            method="RK45",
        )
        T_wall_new, T_in_new = sol.y[:, -1]
        return float(T_wall_new), float(T_in_new)

    def simulate(
        self,
        T_out_series: np.ndarray,
        Q_hvac_series: np.ndarray,
        Q_gain_series: np.ndarray,
        T_wall0: float = 22.0,
        T_in0: float = 22.0,
        dt_seconds: float = 900.0,
    ) -> np.ndarray:
        """Simulate a full episode given input series. Returns array of shape
        (N, 2) with columns [T_wall, T_in], including the initial state as row 0.
        """
        n = len(T_out_series)
        assert len(Q_hvac_series) == n and len(Q_gain_series) == n

        traj = np.zeros((n + 1, 2))
        traj[0] = [T_wall0, T_in0]

        T_wall, T_in = T_wall0, T_in0
        for i in range(n):
            T_wall, T_in = self.step(
                T_wall, T_in, T_out_series[i], Q_hvac_series[i], Q_gain_series[i], dt_seconds
            )
            traj[i + 1] = [T_wall, T_in]

        return traj


def fit_rc_params(
    T_out_series: np.ndarray,
    Q_hvac_series: np.ndarray,
    Q_gain_series: np.ndarray,
    T_in_ground_truth: np.ndarray,
    dt_seconds: float = 900.0,
    initial_guess: RCParams | None = None,
) -> RCParams:
    """Fit RC parameters to ground-truth indoor temperature via least-squares.

    This is the "parameter estimation" step referenced in the CoolTwin
    methodology: instead of hand-picking R/C values, fit them against real
    (or simulator-generated, e.g. Sinergym/EnergyPlus) zone temperature data.
    """
    from scipy.optimize import least_squares

    x0 = initial_guess or RCParams()
    x0_vec = np.array(
        [x0.R_out_wall, x0.R_wall_in, x0.R_in_out, x0.C_wall, x0.C_in]
    )

    def residuals(x):
        params = RCParams(*x)
        zone = RCThermalZone(params)
        traj = zone.simulate(
            T_out_series,
            Q_hvac_series,
            Q_gain_series,
            T_wall0=T_in_ground_truth[0],
            T_in0=T_in_ground_truth[0],
            dt_seconds=dt_seconds,
        )
        T_in_pred = traj[1:, 1]  # drop initial state to align with series
        return T_in_pred - T_in_ground_truth

    result = least_squares(
        residuals,
        x0_vec,
        bounds=(1e-6, np.inf),  # all physical params must be positive
        verbose=0,
    )
    return RCParams(*result.x)


if __name__ == "__main__":
    # Quick smoke test: simulate a day with a simple diurnal outdoor temp
    # and a step HVAC cooling input, print resulting indoor temp trajectory.
    n_steps = 96  # 15-min steps over 24h
    hours = np.arange(n_steps) * 0.25
    T_out = 24 + 6 * np.sin((hours - 9) / 24 * 2 * np.pi)  # peaks midday
    Q_hvac = np.where((hours > 9) & (hours < 18), -800.0, 0.0)  # cooling during day
    Q_gain = np.where((hours > 9) & (hours < 18), 300.0, 50.0)  # occupancy gains

    zone = RCThermalZone()
    traj = zone.simulate(T_out, Q_hvac, Q_gain)

    print("hour  T_out   T_wall   T_in")
    for i in range(0, n_steps, 8):
        print(f"{hours[i]:5.1f} {T_out[i]:7.2f} {traj[i+1,0]:8.2f} {traj[i+1,1]:7.2f}")
