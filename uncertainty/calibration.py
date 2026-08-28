"""
calibration.py

Calibration diagnostics for the residual model's uncertainty estimates.
For regression with Gaussian predictive uncertainty (mean, std), a
well-calibrated model should have: for a nominal confidence level p (e.g.
90%), approximately p% of true values fall inside the model's p%
predictive interval. We check this across a range of confidence levels
and report:

  1. A reliability diagram: nominal confidence vs. observed (empirical)
     coverage -- a perfectly calibrated model traces the diagonal.
  2. Expected Calibration Error (regression variant): the average absolute
     gap between nominal and observed coverage across confidence levels.

This is the mechanism used in notebooks/03_uncertainty_quantification.py to
compare MC Dropout vs Deep Ensembles honestly, rather than just eyeballing
which spread "looks about right."
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def compute_reliability_diagram(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    confidence_levels: np.ndarray | None = None,
):
    """Returns (nominal_levels, observed_coverage) arrays for plotting.

    For each nominal confidence level p, the predictive interval is
    [mean - z*std, mean + z*std] where z = norm.ppf(0.5 + p/2). Observed
    coverage is the fraction of y_true that actually falls inside that
    interval.
    """
    if confidence_levels is None:
        confidence_levels = np.linspace(0.05, 0.95, 10)

    observed = []
    y_std_safe = np.clip(y_std, 1e-6, None)  # avoid div-by-zero for near-zero uncertainty
    for p in confidence_levels:
        z = norm.ppf(0.5 + p / 2)
        lower = y_mean - z * y_std_safe
        upper = y_mean + z * y_std_safe
        inside = (y_true >= lower) & (y_true <= upper)
        observed.append(inside.mean())

    return confidence_levels, np.array(observed)


def expected_calibration_error(nominal: np.ndarray, observed: np.ndarray) -> float:
    """Mean absolute gap between nominal confidence and observed coverage.
    0 = perfectly calibrated. This is the regression analogue of the
    classification ECE metric."""
    return float(np.mean(np.abs(nominal - observed)))


def fit_variance_scale(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, scale_range=(0.1, 20.0), n_grid=200) -> float:
    """Post-hoc variance scaling (a standard, cheap recalibration technique --
    see Kuleshov et al. 2018): find a single scalar s such that using
    (y_mean, s * y_std) as the predictive distribution minimizes calibration
    error on a held-out calibration set. Applied AFTER training, using
    validation data disjoint from what's used to report final test-set ECE,
    so this isn't circular.
    """
    scales = np.linspace(scale_range[0], scale_range[1], n_grid)
    best_scale, best_ece = 1.0, np.inf
    for s in scales:
        nominal, observed = compute_reliability_diagram(y_true, y_mean, y_std * s)
        ece = expected_calibration_error(nominal, observed)
        if ece < best_ece:
            best_ece, best_scale = ece, s
    return float(best_scale)


def plot_reliability_diagram(results: dict[str, tuple[np.ndarray, np.ndarray]], save_path: str):
    """results: {method_name: (nominal_levels, observed_coverage)}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    for name, (nominal, observed) in results.items():
        ax.plot(nominal, observed, marker="o", label=name)

    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Observed coverage")
    ax.set_title("CoolTwin: Uncertainty Calibration (Reliability Diagram)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return save_path
