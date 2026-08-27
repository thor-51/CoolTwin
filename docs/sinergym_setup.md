# Setting up Sinergym + EnergyPlus (Phase 2+)

The Phase 1 baseline (`notebooks/00_baseline_random_agent.py`) runs entirely on the
lightweight RC-network environment in `twin/env.py` — no EnergyPlus install required.

Once you're ready to validate against a higher-fidelity simulator (recommended before
finalizing results for the pitch), set up Sinergym:

## Option A — Docker (recommended, avoids system-level installs)

```bash
docker pull sailab/sinergym:latest
docker run -it --rm -v $(pwd):/workspace sailab/sinergym:latest bash
```

This ships EnergyPlus preinstalled and avoids fighting with system dependencies on your
own machine or in CI.

## Option B — Native install (Ubuntu)

1. Install EnergyPlus (system package, ~300MB):
   ```bash
   wget https://github.com/NREL/EnergyPlus/releases/download/v24.1.0/EnergyPlus-24.1.0-...-Linux-Ubuntu22.04-x86_64.sh
   chmod +x EnergyPlus-*.sh
   sudo ./EnergyPlus-*.sh
   ```
   (check https://github.com/NREL/EnergyPlus/releases for the exact filename/version)

2. Install Sinergym:
   ```bash
   pip install sinergym[extras]
   ```

3. Smoke test:
   ```python
   import gymnasium as gym
   import sinergym
   env = gym.make("Eplus-5zone-hot-continuous-v1")
   obs, info = env.reset()
   print(obs)
   ```

## Why we didn't wire this into Phase 1

EnergyPlus is a heavy, platform-specific system dependency (not a pure Python package),
which makes it a poor fit for CI and for guaranteeing "clone and run" works for every
team member and for judges evaluating the repo. The RC-network environment gives an
identical Gym interface (see `twin/env.py`), so swapping to Sinergym later is a matter of
pointing training scripts at a different `gym.make(...)` — no RL code changes needed.
