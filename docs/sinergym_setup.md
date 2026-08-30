# Setting up Sinergym + EnergyPlus

**Status: wired in and verified working (Phase 1, revisited).** The default
training/evaluation path everywhere else in this repo still runs on the
lightweight RC-network environment in `twin/env.py` — no EnergyPlus install
required for that. This doc covers the *optional*, higher-fidelity Sinergym
path: `notebooks/00b_sinergym_baseline.py` (a random-action sanity check) and
`notebooks/06_sinergym_validation.py` (fitting the 3R2C physics model against
a real EnergyPlus trajectory — see its results in
`results/sinergym_validation.md` and the honest discussion of what that first
validation pass did and didn't get right).

## Option A — Docker (recommended, avoids system-level installs)

```bash
docker pull sailab/sinergym:latest
docker run -it --rm -v $(pwd):/workspace sailab/sinergym:latest bash
```

This ships EnergyPlus preinstalled and avoids fighting with system dependencies on your
own machine or in CI.

## Option B — Native install (Ubuntu)

1. Install EnergyPlus (system package, ~150-200MB):
   ```bash
   wget https://github.com/NREL/EnergyPlus/releases/download/v25.1.0/EnergyPlus-25.1.0-...-Linux-Ubuntu24.04-x86_64.sh
   chmod +x EnergyPlus-*.sh
   sudo ./EnergyPlus-*.sh
   ```
   (check https://github.com/NREL/EnergyPlus/releases for the exact filename/version
   for your OS — this repo was verified against 25.1.0 on Ubuntu 24.04)

2. Install Sinergym:
   ```bash
   pip install sinergym
   ```

3. **Set two environment variables before importing sinergym** — this is the
   part that isn't obvious from Sinergym's own docs and will silently fail
   without it:
   ```bash
   export EPLUS_PATH=/usr/local/EnergyPlus-25-1-0   # wherever step 1 installed it
   export PYTHONPATH=$EPLUS_PATH:$PYTHONPATH        # so `pyenergyplus` is importable
   ```
   `EPLUS_PATH` unset raises `KeyError: 'EPLUS_PATH'` deep inside
   `sinergym/config/modeling.py`; `PYTHONPATH` missing the EnergyPlus dir raises
   `ModuleNotFoundError: No module named 'pyenergyplus'`. Both are checked
   explicitly (with a pointer back to this doc) by
   `notebooks/00b_sinergym_baseline.py`'s `require_sinergym()` guard.

4. Smoke test:
   ```bash
   python notebooks/00b_sinergym_baseline.py
   ```
   Should print the action/observation space and a mean reward over 500 steps
   of `Eplus-5zone-hot-continuous-v1` — a real EnergyPlus simulation, not a
   mock.

## Why this still isn't wired into CI or the default training path

EnergyPlus is a heavy, platform-specific system dependency (not a pure Python
package), which makes it a poor fit for guaranteeing "clone and run" works
for every team member and for judges evaluating the repo without a local
EnergyPlus install. `tests/test_sinergym_baseline.py` reflects this exactly:
it runs and asserts real results when EnergyPlus is available, and skips
cleanly (not fails) when it isn't — so this repo's default `pytest tests/`
run works identically with or without EnergyPlus installed.

The RC-network environment (`twin/env.py`) gives an identical Gym interface,
so nothing about the RL training code changes based on which environment
backs it — see `docs/architecture.md`.

## What the first real validation pass found

`notebooks/06_sinergym_validation.py` fits the project's 3R2C model against a
real EnergyPlus trajectory (Sinergym's `Eplus-demo-v1`) rather than only the
synthetic generator used elsewhere in this repo. The result was a genuinely
bad RMSE (~10-12°C against a ~7°C-wide ground-truth band) — and the
diagnosis, not just the number, is the useful part: the ground-truth
`HVAC_electricity_demand_rate` signal is the *output* of EnergyPlus's own
closed-loop ideal-loads controller, not an independent forcing input, so
feeding it into our open-loop RC model compounds error instead of tracking.
Full writeup, including the concrete fix (use a real thermal-rate output
variable instead of a closed-loop-derived one), is in
`results/sinergym_validation.md`.

