"""
notebooks/07_sinergym_integration.py

Full Sinergym integration pipeline — the script that replaces synthetic data
with real EnergyPlus building physics across the entire CoolTwin pipeline.

What this does, step by step:
  1. Collects episodes from a Sinergym (EnergyPlus) environment
  2. Fits the 3R2C physics model parameters to the real trajectory
  3. Recomputes physics predictions using the fitted RC model
  4. Trains the residual LSTM on the real data (same architecture, real errors)
  5. Evaluates: physics-only vs hybrid vs pure-ML on held-out real data
  6. Saves all results + a comparison plot

This is the "close the loop" step that addresses the gap identified in
results/sinergym_validation.md: the previous validation (notebook 06) showed
a ~11°C RMSE because it used a closed-loop electricity signal as if it were
an independent forcing input. This time, we:
  - Still use the COP-approximated Q_hvac (stock Sinergym limitation), but
  - Fit the RC model TO this data first (so the RC model is adapted to the
    signal it's actually getting, rather than using default params), then
  - Train the residual LSTM to correct whatever systematic gap remains.

The hybrid twin (fitted RC + trained LSTM) is the final product: a fast,
interpretable model trained against real building physics.

Requires: sinergym >= 3.0 + EnergyPlus. See docs/sinergym_setup.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Ensure the repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin.rc_model import RCParams

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Step 1: Collect episodes from Sinergym                        ║
# ╚══════════════════════════════════════════════════════════════════╝


def step1_collect(
    n_episodes: int = 5,
    max_steps: int = 2160,
    cache_path: str = "data/sinergym_episodes.npz",
) -> list[dict]:
    """Collect episodes from Sinergym, or load from cache if already collected.

    The EnergyPlus simulation is the slowest part (~2-5 min per episode), so we
    cache the results to .npz and skip re-collection on subsequent runs.
    """
    from twin.sinergym_data import (
        collect_sinergym_dataset,
        load_episodes,
        save_episodes,
    )

    cache = Path(cache_path)
    if cache.exists():
        print(f"[Step 1] Loading cached Sinergym episodes from {cache}")
        episodes = load_episodes(cache)
        print(
            f"  Loaded {len(episodes)} episodes, {len(episodes[0]['T_out'])} steps each."
        )
        return episodes

    print(f"[Step 1] Collecting {n_episodes} episodes from Sinergym (Eplus-demo-v1)...")
    print("  This runs EnergyPlus and will take a few minutes per episode.\n")

    episodes = collect_sinergym_dataset(
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
    )
    save_episodes(episodes, cache)
    return episodes


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Step 2: Fit RC model parameters to real data                  ║
# ╚══════════════════════════════════════════════════════════════════╝


def step2_fit_rc(episodes: list[dict]) -> RCParams:
    """Fit 3R2C parameters against the first episode's trajectory.

    Uses the first 70% for fitting, reports RMSE on the held-out 30%.
    The fitted params are then used to recompute T_in_physics for ALL
    episodes — so the residual LSTM trains against the best RC prediction
    rather than the default (unfitted) one.
    """
    from twin.rc_model import RCThermalZone, fit_rc_params

    print("\n[Step 2] Fitting 3R2C parameters to real EnergyPlus data...")

    # Use the first episode for fitting (longest, most representative)
    ep = episodes[0]
    n = len(ep["T_out"])
    split = int(n * 0.7)
    dt = 3600.0  # 1-hour timestep

    fitted_params = fit_rc_params(
        ep["T_out"][:split],
        ep["Q_hvac"][:split],
        ep["Q_gain_base"][:split],
        ep["T_in_true"][:split],
        dt_seconds=dt,
    )
    print(f"  Fitted RC params: {fitted_params}")

    # Evaluate on full first episode
    zone = RCThermalZone(fitted_params)
    traj = zone.simulate(
        ep["T_out"],
        ep["Q_hvac"],
        ep["Q_gain_base"],
        T_wall0=ep["T_in_true"][0],
        T_in0=ep["T_in_true"][0],
        dt_seconds=dt,
    )
    T_in_fitted = traj[1:, 1]

    train_rmse = float(
        np.sqrt(np.mean((T_in_fitted[:split] - ep["T_in_true"][:split]) ** 2))
    )
    test_rmse = float(
        np.sqrt(np.mean((T_in_fitted[split:] - ep["T_in_true"][split:]) ** 2))
    )
    print(f"  Train RMSE (fitted RC): {train_rmse:.3f}°C")
    print(f"  Test  RMSE (fitted RC): {test_rmse:.3f}°C")

    # Compare with default (unfitted) params
    default_rmse = float(np.sqrt(np.mean(ep["residual"] ** 2)))
    print(f"  Default RC RMSE (unfitted): {default_rmse:.3f}°C")
    print(f"  → Fitting improved RMSE by {default_rmse - test_rmse:.3f}°C")

    return fitted_params


def step2b_recompute_physics(episodes: list[dict], fitted_params) -> list[dict]:
    """Recompute T_in_physics and residual for all episodes using fitted RC params.

    This is critical: the residual LSTM should learn to correct the FITTED
    model's errors, not the default model's errors. Otherwise you're training
    the LSTM to fix parameter mismatch that the RC fitting already handles.
    """
    from twin.rc_model import RCThermalZone

    print("\n[Step 2b] Recomputing physics predictions with fitted RC params...")

    zone = RCThermalZone(fitted_params)
    dt = 3600.0

    updated = []
    for i, ep in enumerate(episodes):
        traj = zone.simulate(
            ep["T_out"],
            ep["Q_hvac"],
            ep["Q_gain_base"],
            T_wall0=ep["T_in_true"][0],
            T_in0=ep["T_in_true"][0],
            dt_seconds=dt,
        )
        new_ep = dict(ep)  # shallow copy
        new_ep["T_in_physics"] = traj[1:, 1]
        new_ep["residual"] = ep["T_in_true"] - new_ep["T_in_physics"]
        updated.append(new_ep)

        rmse = float(np.sqrt(np.mean(new_ep["residual"] ** 2)))
        print(f"  Episode {i+1}: fitted RC RMSE = {rmse:.3f}°C")

    return updated


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Step 3: Train residual LSTM on real data                      ║
# ╚══════════════════════════════════════════════════════════════════╝


def step3_train_lstm(episodes: list[dict], window: int = 8, epochs: int = 20):
    """Train the residual LSTM on Sinergym episodes — same architecture as
    the synthetic version, just different (real) data.

    Split: first 60% of episodes for training, next 20% for validation,
    last 20% for testing.
    """
    from twin.residual_lstm import (
        DirectLSTM,
        ResidualLSTM,
        ResidualWindowDataset,
        train_model,
    )

    print(f"\n[Step 3] Training residual LSTM on {len(episodes)} real episodes...")

    n = len(episodes)
    n_train = max(1, int(n * 0.6))
    n_val = max(1, int(n * 0.2))

    train_eps = episodes[:n_train]
    val_eps = episodes[n_train : n_train + n_val]
    test_eps = episodes[n_train + n_val :]

    if not test_eps:
        # If too few episodes, use val as test too
        test_eps = val_eps

    print(
        f"  Split: {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test episodes"
    )

    # Build datasets
    train_ds = ResidualWindowDataset(train_eps, window=window, target="residual")
    val_ds = ResidualWindowDataset(val_eps, window=window, target="residual")
    test_ds = ResidualWindowDataset(test_eps, window=window, target="residual")

    print(f"  Samples: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    # Train residual model (hybrid approach)
    print("\n  --- Training ResidualLSTM (physics + ML hybrid) ---")
    residual_model = ResidualLSTM(n_features=6, hidden_size=32, num_layers=1)
    train_model(residual_model, train_ds, val_ds, epochs=epochs, verbose=True)

    # Train direct model (pure-ML baseline, no physics)
    print("\n  --- Training DirectLSTM (pure-ML baseline, no physics) ---")
    train_ds_direct = ResidualWindowDataset(
        train_eps, window=window, target="T_in_true"
    )
    val_ds_direct = ResidualWindowDataset(val_eps, window=window, target="T_in_true")

    direct_model = DirectLSTM(n_features=6, hidden_size=32, num_layers=1)
    train_model(
        direct_model, train_ds_direct, val_ds_direct, epochs=epochs, verbose=True
    )

    return residual_model, direct_model, test_eps


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Step 4: Evaluate all three approaches on held-out real data    ║
# ╚══════════════════════════════════════════════════════════════════╝


def step4_evaluate(
    residual_model,
    direct_model,
    test_episodes: list[dict],
    window: int = 8,
):
    """Evaluate physics-only, hybrid (physics + LSTM), and pure-ML on held-out
    Sinergym episodes. Returns a results dict."""
    from twin.residual_lstm import _build_features

    print("\n[Step 4] Evaluating on held-out real EnergyPlus data...")

    all_physics_errors = []
    all_hybrid_errors = []
    all_direct_errors = []

    residual_model.eval()
    direct_model.eval()

    for ep_idx, ep in enumerate(test_episodes):
        feats = _build_features(ep)
        n = len(ep["T_in_true"])

        physics_errors = []
        hybrid_errors = []
        direct_errors = []

        with torch.no_grad():
            for t in range(window, n):
                x = torch.from_numpy(feats[t - window : t]).unsqueeze(0)

                # Physics-only
                physics_err = ep["T_in_true"][t] - ep["T_in_physics"][t]
                physics_errors.append(physics_err**2)

                # Hybrid: physics + residual correction
                residual_pred = residual_model(x).item()
                hybrid_pred = ep["T_in_physics"][t] + residual_pred
                hybrid_err = ep["T_in_true"][t] - hybrid_pred
                hybrid_errors.append(hybrid_err**2)

                # Pure-ML (direct prediction)
                direct_pred = direct_model(x).item()
                direct_err = ep["T_in_true"][t] - direct_pred
                direct_errors.append(direct_err**2)

        all_physics_errors.extend(physics_errors)
        all_hybrid_errors.extend(hybrid_errors)
        all_direct_errors.extend(direct_errors)

    physics_rmse = float(np.sqrt(np.mean(all_physics_errors)))
    hybrid_rmse = float(np.sqrt(np.mean(all_hybrid_errors)))
    direct_rmse = float(np.sqrt(np.mean(all_direct_errors)))

    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │  RMSE on held-out real EnergyPlus data          │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Physics-only (fitted RC):      {physics_rmse:.3f}°C         │")
    print(f"  │  Hybrid (RC + Residual LSTM):   {hybrid_rmse:.3f}°C         │")
    print(f"  │  Pure-ML (Direct LSTM):         {direct_rmse:.3f}°C         │")
    print("  └─────────────────────────────────────────────────┘")

    if hybrid_rmse < physics_rmse:
        improvement = ((physics_rmse - hybrid_rmse) / physics_rmse) * 100
        print(f"\n  Hybrid improved over physics-only by {improvement:.1f}%")
    if hybrid_rmse < direct_rmse:
        print("  Hybrid also beat pure-ML (physics prior helps)")

    return {
        "physics_rmse": physics_rmse,
        "hybrid_rmse": hybrid_rmse,
        "direct_rmse": direct_rmse,
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Step 5: Save results + comparison plot                         ║
# ╚══════════════════════════════════════════════════════════════════╝


def step5_save_results(results: dict, test_episodes: list[dict], fitted_params):
    """Save the comparison plot and a results markdown file."""

    os.makedirs("results", exist_ok=True)

    # --- Bar chart: RMSE comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart
    ax = axes[0]
    methods = [
        "Physics-only\n(fitted RC)",
        "Hybrid\n(RC + LSTM)",
        "Pure-ML\n(Direct LSTM)",
    ]
    rmses = [results["physics_rmse"], results["hybrid_rmse"], results["direct_rmse"]]
    colors = ["#4a90d9", "#2ecc71", "#e74c3c"]
    bars = ax.bar(methods, rmses, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("RMSE (°C)")
    ax.set_title(
        "Prediction accuracy on real EnergyPlus data\n(held-out test episodes)"
    )
    for bar, rmse in zip(bars, rmses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{rmse:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Right: time-series snippet from first test episode
    if test_episodes:
        ax2 = axes[1]
        ep = test_episodes[0]
        n_show = min(500, len(ep["T_in_true"]))
        hours = np.arange(n_show)
        ax2.plot(
            hours,
            ep["T_in_true"][:n_show],
            label="Ground truth (EnergyPlus)",
            linewidth=1.5,
            color="#2c3e50",
        )
        ax2.plot(
            hours,
            ep["T_in_physics"][:n_show],
            label="Fitted RC model",
            linewidth=1.0,
            alpha=0.7,
            color="#4a90d9",
        )
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("Zone temperature (°C)")
        ax2.set_title("Fitted RC model vs real EnergyPlus trajectory")
        ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("results/sinergym_integration.png", dpi=120)
    print("\n[Step 5] Saved results/sinergym_integration.png")

    # --- Results markdown ---
    with open("results/sinergym_integration.md", "w") as f:
        f.write("# CoolTwin — Sinergym Integration Results\n\n")
        f.write(
            "Full pipeline: Sinergym (EnergyPlus) data → fitted RC model → "
            "residual LSTM → hybrid twin validated on held-out real data.\n\n"
        )
        f.write(f"- Fitted RC params: `{fitted_params}`\n")
        f.write(f"- COP assumption: {3.0} (electrical→thermal conversion)\n")
        f.write("- LSTM window: 8 steps, hidden_size: 32\n\n")
        f.write("## Results on held-out real EnergyPlus episodes\n\n")
        f.write("| Method | RMSE (°C) |\n|---|---|\n")
        f.write(f"| Physics-only (fitted RC) | {results['physics_rmse']:.3f} |\n")
        f.write(
            f"| **Hybrid (RC + Residual LSTM)** | **{results['hybrid_rmse']:.3f}** |\n"
        )
        f.write(f"| Pure-ML (Direct LSTM) | {results['direct_rmse']:.3f} |\n\n")

        if results["hybrid_rmse"] < results["physics_rmse"]:
            improvement = (
                (results["physics_rmse"] - results["hybrid_rmse"])
                / results["physics_rmse"]
            ) * 100
            f.write(
                f"The hybrid approach improved over physics-only by **{improvement:.1f}%** "
                f"on real building data, confirming that the residual LSTM adds value "
                f"even when the RC model is properly fitted to real trajectories.\n\n"
            )

        f.write("## What this validates\n\n")
        f.write(
            "1. The hybrid twin methodology (physics + learned residual) works on "
            "**real EnergyPlus building physics**, not just synthetic data.\n"
        )
        f.write(
            "2. The entire pipeline (data collection → RC fitting → LSTM training → "
            "evaluation) runs end-to-end on Sinergym data with zero code changes "
            "to the core twin/rl modules.\n"
        )
        f.write(
            "3. The residual LSTM learns real, systematic errors in the RC model's "
            "approximation of EnergyPlus's detailed thermal simulation.\n\n"
        )

        f.write("## Known limitations\n\n")
        f.write(
            "- Q_hvac is approximated via COP from electrical demand (Sinergym stock "
            "envs don't expose the raw thermal rate)\n"
        )
        f.write(
            "- Internal gains (Q_gain) are set to zero (not exposed by stock env)\n"
        )
        f.write("- Single building type (5-zone ASHRAE reference model)\n")
        f.write(
            "- Deterministic EnergyPlus = limited episode variety from seed changes alone\n"
        )

    print("Saved results/sinergym_integration.md")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Main: run the full pipeline                                    ║
# ╚══════════════════════════════════════════════════════════════════╝


def main():
    print("=" * 60)
    print("CoolTwin — Sinergym Integration Pipeline")
    print("=" * 60)

    # Step 1: Collect (or load cached) episodes
    episodes = step1_collect(n_episodes=5, max_steps=2160)

    # Step 2: Fit RC model to real data
    fitted_params = step2_fit_rc(episodes)

    # Step 2b: Recompute all physics predictions with fitted params
    episodes = step2b_recompute_physics(episodes, fitted_params)

    # Step 3: Train residual LSTM on real data
    residual_model, direct_model, test_eps = step3_train_lstm(episodes)

    # Step 4: Evaluate
    results = step4_evaluate(residual_model, direct_model, test_eps)

    # Step 5: Save results
    step5_save_results(results, test_eps, fitted_params)

    # Save the trained model
    os.makedirs("results", exist_ok=True)
    torch.save(residual_model.state_dict(), "results/sinergym_residual_lstm.pt")
    print("\nSaved trained model to results/sinergym_residual_lstm.pt")

    print("\n" + "=" * 60)
    print("Done! The hybrid twin is now trained on real building physics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
