# CoolTwin — Phase 6 Final Evaluation

One held-out week-long episode (seed=123), same balanced cost/comfort/carbon/peak
accounting used to evaluate every controller regardless of what it was trained on.

| Controller | Energy (kWh) | Comfort (°C-hr) | Peak (W) | Peak Δ% vs rule-based | Carbon (kg) | Latency (ms) |
|---|---|---|---|---|---|---|
| Random | 168.0 | 262.33 | 1999 | -0.0% | 75.6 | 0.002 |
| Rule-based thermostat | 210.5 | 95.46 | 2000 | +0.0% | 94.7 | 0.001 |
| PID | 261.6 | 91.76 | 2000 | +0.0% | 117.7 | 0.003 |
| Single-objective PPO (cost-only) | 17.1 | 255.74 | 312 | -84.4% | 7.7 | 0.170 |
| Multi-objective PPO (balanced) | 299.6 | 86.14 | 2000 | +0.0% | 134.8 | 0.184 |
| Multi-objective SAC (balanced) | 205.2 | 91.99 | 1979 | -1.0% | 92.3 | 0.259 |

## Calibration (Phase 4)

| Method | ECE (calibrated) |
|---|---|
| MC Dropout (calibrated) | 0.009 |
| Deep Ensemble (calibrated) | 0.029 |

## Key finding

The single-objective (cost-only) PPO policy uses dramatically less energy but at a large comfort cost, compared to the multi-objective policies trained on the same environment -- this is the empirical demonstration of the trade-off problem described in the project abstract's problem statement.
