# CoolTwin

**An Explainable, Uncertainty-Aware Digital Twin Framework for Autonomous HVAC Scheduling**
using Hybrid AI, Reinforcement Learning, and Multi-Objective Optimization.

Built for the Schneider Electric Co-Creation Challenge 2026.

Team: Aryan Vatsal, Anmol Tibrewal, Josithaa Joseph — Vellore Institute of Technology
Guide: Dr. Athira K

---

## What this is

A single-zone HVAC controller that combines:

- **A hybrid digital twin**: a physics-based RC (resistor-capacitor) thermal network, corrected
  by a learned residual model (LSTM), so the twin is both interpretable and accurate.
- **A reinforcement learning agent** (PPO / SAC) trained inside the twin, optimizing a
  multi-term reward over energy cost, comfort, carbon emissions, and peak demand.
- **Uncertainty quantification** (MC Dropout, Deep Ensembles) so the agent knows when it
  doesn't know — and can fall back to conservative rule-based control.
- **Explainability** (reward decomposition, SHAP, an LLM explanation layer) so a
  non-technical building operator can ask "why did you do that?" and get a real answer.

## Why this scope

This project intentionally does **not** try to build a full industrial platform
(Kubernetes, Kafka, knowledge graphs, multi-agent orchestration, etc.) as a working
prototype. Those are documented as deliberate future work in
[`docs/future_work.md`](docs/future_work.md). The goal here is a small number of things
built *properly*, evaluated honestly, and explainable end-to-end — not a long list of
half-built integrations.

## Status

- **Phase 1** (foundations) — repo scaffolding, baseline environment, RC thermal model. ✅
- **Phase 2** (hybrid twin) — residual LSTM correction model, trained and evaluated
  against physics-only and pure-ML baselines. ✅
  Result on held-out synthetic test episodes: **hybrid RMSE 0.18°C** vs. physics-only
  0.42°C and pure-ML 0.21°C — see `notebooks/01_train_residual_lstm.py`.
- **Phase 3** (RL agent) — PPO and SAC trained on the twin with a configurable
  multi-objective reward (cost/comfort/carbon/peak), evaluated against rule-based,
  PID, and random baselines; Pareto front traced across 5 reward weightings. ✅
  Result (50k-timestep training budget): **SAC beat all three baselines on reward,
  energy use, discomfort, and carbon** (202 kWh / 347 K·step discomfort / 91 kg CO2
  vs. the rule-based thermostat's 211 kWh / 382 K·step / 95 kg). PPO also beat both
  baselines on reward, trading more energy for the lowest discomfort among the RL
  agents — see `notebooks/02_train_rl_agents.py`.
  The Pareto sweep (200k timesteps per weighting) now shows a genuine trade-off
  curve: energy ranges 129–248 kWh across weightings, with `comfort_focused`
  trading roughly 2x the energy of the leanest weighting for the lowest discomfort
  — see `rl/pareto.py` and `results/pareto_front.png`.

  *Note on training budget*: these numbers use 50k timesteps (PPO/SAC comparison)
  and 200k timesteps per weighting (Pareto sweep) — enough for a clear, defensible
  signal, run on CPU in well under an hour total. Longer training (e.g. 500k+)
  would likely sharpen the separation further and is worth doing once with more
  compute before the final pitch numbers are locked in.

See [`docs/roadmap.md`](docs/roadmap.md) for the full build plan.

## Quickstart

```bash
git clone <your-repo-url>
cd CoolTwin
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the baseline: random-action agent on the RC-network zone
python notebooks/00_baseline_random_agent.py

# Train the hybrid twin's residual model and see the RMSE comparison table
python notebooks/01_train_residual_lstm.py

# Train PPO + SAC, compare against rule-based/PID/random baselines
python notebooks/02_train_rl_agents.py

# Trace the Pareto front across reward weightings (takes a few minutes)
python rl/pareto.py

# Run tests
pytest tests/
```

Sinergym / EnergyPlus is used later (Phase 2+) for higher-fidelity simulation and is
**not required** to run the Phase 1 baseline above — see
[`docs/sinergym_setup.md`](docs/sinergym_setup.md) for installing it when you're ready.

## Repo structure

```
CoolTwin/
├── twin/              # physics model, residual ML model, hybrid twin, gym env
├── rl/                # RL training (PPO/SAC), reward function, Pareto front
├── uncertainty/        # MC Dropout, ensembles, calibration
├── explainability/     # reward decomposition, SHAP, LLM explanation layer
├── dashboard/           # Streamlit app
├── evaluation/          # baselines, metrics
├── notebooks/           # exploratory scripts/notebooks
├── results/             # generated plots, tables (gitignored except .gitkeep)
├── docs/                 # architecture, roadmap, methodology, future work
└── tests/                 # unit tests
```

## License

MIT — see `LICENSE`.
