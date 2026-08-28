"""
03_uncertainty_quantification.py

Phase 4 deliverable:
  1. Train MC Dropout and Deep Ensemble uncertainty models on the same
     synthetic dataset used in Phase 2.
  2. Evaluate both on held-out test episodes, compute reliability diagrams
     and Expected Calibration Error (ECE) for each -- an honest comparison,
     not just "we implemented two methods."
  3. Demonstrate the uncertainty-gated safety layer: run it on a normal
     in-distribution episode (low fallback rate expected) and on an
     out-of-distribution "heatwave shock" episode (higher fallback rate
     expected) to show the fallback actually responds to genuine
     uncertainty, not just firing at a constant rate.

Usage:
    PYTHONPATH=. python notebooks/03_uncertainty_quantification.py
"""

from __future__ import annotations

import numpy as np
import torch

from twin.data_gen import generate_dataset, generate_episode, RCThermalZone, RCParams, TRUE_PARAMS
from twin.residual_lstm import ResidualWindowDataset, build_feature_vector
from twin.env import CoolTwinEnv
from rl.train_ppo import train_ppo
from uncertainty.mc_dropout import MCDropoutResidualLSTM, predict_with_uncertainty
from uncertainty.ensembles import DeepEnsemble
from uncertainty.calibration import (
    compute_reliability_diagram,
    expected_calibration_error,
    plot_reliability_diagram,
    fit_variance_scale,
)
from uncertainty.safety_layer import UncertaintyGatedController


WINDOW = 8


def train_mc_dropout(train_ds, val_ds, epochs=15):
    from twin.residual_lstm import train_model

    torch.manual_seed(0)
    model = MCDropoutResidualLSTM(n_features=6, hidden_size=32, num_layers=2, dropout=0.2)
    train_model(model, train_ds, val_ds, epochs=epochs, verbose=False)
    model.eval()
    return model


def evaluate_uncertainty_method(predict_fn, test_episodes, window=WINDOW):
    """predict_fn(x: torch.Tensor) -> (mean, std) numpy arrays, batched over
    the whole test set at once for speed."""
    all_true, all_mean, all_std = [], [], []
    for ep in test_episodes:
        feats = np.stack(
            [
                build_feature_vector(ep["T_out"][t], ep["hour"][t], ep["Q_hvac"][t], ep["Q_gain_base"][t], ep["T_in_physics"][t])
                for t in range(len(ep["T_out"]))
            ]
        )
        n = len(feats)
        windows = np.stack([feats[t - window : t] for t in range(window, n)])
        x = torch.from_numpy(windows)
        mean, std = predict_fn(x)

        all_true.append(ep["residual"][window:])
        all_mean.append(mean)
        all_std.append(std)

    return np.concatenate(all_true), np.concatenate(all_mean), np.concatenate(all_std)


def generate_ood_episode(n_steps: int = 96, seed: int = 999):
    """A 'heatwave shock' episode with outdoor temperatures well outside the
    training distribution (which ranges roughly 15-33C) -- used to check
    whether the uncertainty estimators (and therefore the safety layer)
    actually respond to genuinely out-of-distribution conditions."""
    rng = np.random.default_rng(seed)
    hours = (np.arange(n_steps) * 900 / 3600.0) % 24
    days = np.arange(n_steps) * 900 / 3600.0 / 24.0

    T_out = 55 + 3 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.5, n_steps)  # extreme heat
    occ = ((hours >= 9) & (hours < 18)).astype(float)
    Q_gain_base = np.where(occ > 0, 300.0, 50.0)
    Q_hvac = np.full(n_steps, -1800.0) + rng.normal(0, 50, n_steps)  # near-max cooling

    from twin.data_gen import _solar_gain

    solar = np.array([_solar_gain(hours[i], days[i], rng) for i in range(n_steps)])
    Q_gain_true = Q_gain_base + solar

    true_zone = RCThermalZone(TRUE_PARAMS)
    traj_true = true_zone.simulate(T_out, Q_hvac, Q_gain_true, T_wall0=30, T_in0=30, dt_seconds=900)
    T_in_true = traj_true[1:, 1] + rng.normal(0, 0.15, n_steps)

    nominal_zone = RCThermalZone(RCParams())
    traj_phys = nominal_zone.simulate(T_out, Q_hvac, Q_gain_base, T_wall0=30, T_in0=30, dt_seconds=900)
    T_in_physics = traj_phys[1:, 1]

    return {
        "T_out": T_out, "hour": hours, "day_frac": days, "Q_hvac": Q_hvac,
        "Q_gain_base": Q_gain_base, "T_in_true": T_in_true, "T_in_physics": T_in_physics,
        "residual": T_in_true - T_in_physics,
    }


def main():
    print("Generating synthetic dataset...")
    train_episodes = generate_dataset(n_episodes=24, n_steps=672, seed=1)
    val_episodes = generate_dataset(n_episodes=6, n_steps=672, seed=2)
    test_episodes = generate_dataset(n_episodes=6, n_steps=672, seed=3)

    train_ds = ResidualWindowDataset(train_episodes, window=WINDOW, target="residual")
    val_ds = ResidualWindowDataset(val_episodes, window=WINDOW, target="residual")

    print("\nTraining MC Dropout residual model...")
    mc_model = train_mc_dropout(train_ds, val_ds, epochs=15)

    print("Training Deep Ensemble (4 members)...")
    ensemble = DeepEnsemble(n_models=4, n_features=6, hidden_size=32)
    ensemble.fit(train_ds, val_ds, epochs=15, verbose=True)

    print("\nEvaluating calibration on held-out in-distribution test episodes...")
    # Fit the post-hoc variance-scaling factor on VALIDATION episodes, then
    # apply it to the TEST episodes below -- keeps the reported test-set ECE
    # honest (not fit and evaluated on the same data).
    mc_val_true, mc_val_mean, mc_val_std = evaluate_uncertainty_method(
        lambda x: predict_with_uncertainty(mc_model, x, n_samples=30), val_episodes
    )
    ens_val_true, ens_val_mean, ens_val_std = evaluate_uncertainty_method(
        ensemble.predict_with_uncertainty, val_episodes
    )
    mc_scale = fit_variance_scale(mc_val_true, mc_val_mean, mc_val_std)
    ens_scale = fit_variance_scale(ens_val_true, ens_val_mean, ens_val_std)
    print(f"Fitted variance-scaling factors (on validation set): MC Dropout x{mc_scale:.2f}, Ensemble x{ens_scale:.2f}")

    mc_true, mc_mean, mc_std = evaluate_uncertainty_method(
        lambda x: predict_with_uncertainty(mc_model, x, n_samples=30), test_episodes
    )
    ens_true, ens_mean, ens_std = evaluate_uncertainty_method(
        ensemble.predict_with_uncertainty, test_episodes
    )

    mc_nominal, mc_observed = compute_reliability_diagram(mc_true, mc_mean, mc_std)
    ens_nominal, ens_observed = compute_reliability_diagram(ens_true, ens_mean, ens_std)
    mc_ece = expected_calibration_error(mc_nominal, mc_observed)
    ens_ece = expected_calibration_error(ens_nominal, ens_observed)

    # recalibrated (scaled) versions, evaluated on the same held-out test set
    mc_nominal_cal, mc_observed_cal = compute_reliability_diagram(mc_true, mc_mean, mc_std * mc_scale)
    ens_nominal_cal, ens_observed_cal = compute_reliability_diagram(ens_true, ens_mean, ens_std * ens_scale)
    mc_ece_cal = expected_calibration_error(mc_nominal_cal, mc_observed_cal)
    ens_ece_cal = expected_calibration_error(ens_nominal_cal, ens_observed_cal)

    plot_reliability_diagram(
        {
            "MC Dropout (raw)": (mc_nominal, mc_observed),
            "Deep Ensemble (raw)": (ens_nominal, ens_observed),
            "MC Dropout (calibrated)": (mc_nominal_cal, mc_observed_cal),
            "Deep Ensemble (calibrated)": (ens_nominal_cal, ens_observed_cal),
        },
        save_path="results/calibration_reliability_diagram.png",
    )

    print("\n" + "=" * 68)
    print("Phase 4 result: calibration comparison (test set, before/after scaling)")
    print("=" * 68)
    print(f"{'Method':<20}{'Mean pred std (C)':>20}{'Raw ECE':>12}{'Calibrated ECE':>16}")
    print(f"{'MC Dropout':<20}{mc_std.mean():>20.3f}{mc_ece:>12.4f}{mc_ece_cal:>16.4f}")
    print(f"{'Deep Ensemble':<20}{ens_std.mean():>20.3f}{ens_ece:>12.4f}{ens_ece_cal:>16.4f}")
    print("=" * 68)

    # --- OOD check: does predictive uncertainty rise on the heatwave episode? ---
    # Checked for BOTH methods (not just one) since the safety layer's choice
    # of uncertainty source below is decided from these actual numbers, not
    # from an assumption about which method "should" be more sensitive.
    print("\nChecking predictive uncertainty on an out-of-distribution 'heatwave shock' episode...")
    ood_ep = generate_ood_episode(n_steps=96)

    _, _, mc_id_std = evaluate_uncertainty_method(
        lambda x: predict_with_uncertainty(mc_model, x, n_samples=30), [test_episodes[0]]
    )
    _, _, mc_ood_std = evaluate_uncertainty_method(
        lambda x: predict_with_uncertainty(mc_model, x, n_samples=30), [ood_ep]
    )
    _, _, ens_id_std = evaluate_uncertainty_method(ensemble.predict_with_uncertainty, [test_episodes[0]])
    _, _, ens_ood_std = evaluate_uncertainty_method(ensemble.predict_with_uncertainty, [ood_ep])

    mc_ratio = mc_ood_std.mean() / max(mc_id_std.mean(), 1e-6)
    ens_ratio = ens_ood_std.mean() / max(ens_id_std.mean(), 1e-6)

    print(f"  MC Dropout    -- in-dist std: {mc_id_std.mean():.4f} C, OOD std: {mc_ood_std.mean():.4f} C ({mc_ratio:.2f}x)")
    print(f"  Deep Ensemble -- in-dist std: {ens_id_std.mean():.4f} C, OOD std: {ens_ood_std.mean():.4f} C ({ens_ratio:.2f}x)")

    if max(mc_ratio, ens_ratio) > 1.0:
        print("  -> At least one method's uncertainty rises under distribution shift.")
    else:
        print("  -> WARNING: neither method's uncertainty rose under distribution shift (see README caveat).")

    # --- Safety layer demonstration ---
    # Use whichever method showed the larger OOD sensitivity above, decided
    # from the numbers just printed rather than assumed in advance.
    if ens_ratio >= mc_ratio:
        chosen_name, chosen_id_std, chosen_scale = "Deep Ensemble", ens_id_std, ens_scale
        uncertainty_fn = lambda feature_window: float(
            ensemble.predict_with_uncertainty(torch.from_numpy(feature_window).unsqueeze(0))[1][0] * ens_scale
        )
    else:
        chosen_name, chosen_id_std, chosen_scale = "MC Dropout", mc_id_std, mc_scale
        uncertainty_fn = lambda feature_window: float(
            predict_with_uncertainty(mc_model, torch.from_numpy(feature_window).unsqueeze(0), n_samples=30)[1][0] * mc_scale
        )
    print(f"\nUsing {chosen_name} (calibrated, x{chosen_scale:.2f}) for the safety layer.")

    print("Training a small PPO policy to demonstrate the safety layer...")
    ppo_model = train_ppo(total_timesteps=10_000, seed=0)

    threshold = float(np.percentile(chosen_id_std * chosen_scale, 90))
    print(f"Using fallback threshold (90th percentile of calibrated in-distribution std): {threshold:.4f} C")

    for label, env in [
        ("In-distribution week", CoolTwinEnv(episode_hours=24 * 7, seed=100)),
        ("Heatwave-shock week", CoolTwinEnv(episode_hours=24 * 4, seed=999)),
    ]:
        controller = UncertaintyGatedController(ppo_model, uncertainty_fn, std_threshold=threshold)
        obs, _ = env.reset(seed=100 if "In-dist" in label else 999)
        if "Heatwave" in label:
            # monkey-patch the outdoor temp generator to simulate an OOD shock:
            # sustained ~42C outdoor temp, well outside the ~15-33C training range
            env._outdoor_temp = lambda step: 55 + 3 * np.sin(
                ((step * env.dt_seconds / 3600.0) % 24 - 9) / 24 * 2 * np.pi
            )

        feature_history = []  # built from REAL past steps only, matching
                               # training-time semantics (features[t-window:t]
                               # predict residual[t] -- never includes the
                               # action about to be taken)
        terminated = truncated = False
        while not (terminated or truncated):
            window = np.stack(feature_history[-WINDOW:]) if len(feature_history) >= WINDOW else None
            action, used_fallback = controller.act(obs, window)
            obs, reward, terminated, truncated, info = env.step(action)

            hour = (env.t * env.dt_seconds / 3600.0) % 24  # env.t already advanced past this step
            feat = build_feature_vector(
                info["T_out"], hour, info["Q_hvac"], info["Q_gain"], info["T_in_physics"]
            )
            feature_history.append(feat)

        print(f"  {label}: fallback triggered on {controller.fallback_rate*100:.1f}% of steps")

    return {"mc_ece": mc_ece, "ens_ece": ens_ece}


if __name__ == "__main__":
    main()
