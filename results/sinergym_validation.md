# CoolTwin — Sinergym Validation (Phase 1, revisited)

First-pass validation of the 3R2C physics model against a real EnergyPlus-simulated building (Sinergym's `Eplus-demo-v1`, a 5-zone ASHRAE reference model), instead of only the synthetic generator used elsewhere in this repo.

- Episode length: 1440 hourly steps (60 days)
- Train/test split: 1007/433 steps
- Electrical->thermal conversion: fixed COP=3.0 assumption
- Fitted RC params: RCParams(R_out_wall=np.float64(0.0017613644782141077), R_wall_in=np.float64(0.0008563083216287871), R_in_out=np.float64(0.004151427585464551), C_wall=np.float64(8999882.408753356), C_in=np.float64(3541535.913998956))

| Portion | RMSE (C) |
|---|---|
| Train (fit) | 10.838 |
| Held-out (test) | 12.135 |

**What this does and doesn't cover.** This validates the physics *shape* (3R2C dynamics) against a real building, replacing the prior gap of 'never checked against anything but our own synthetic data.' It does NOT yet include: the residual-LSTM correction (natural next step -- retrain it on this real trajectory the same way it was trained on the synthetic one in Phase 2), a real thermal-load signal (currently approximated from electrical demand via a fixed COP), or occupancy-driven internal gains (not exposed by this environment's observation set, set to zero here).
