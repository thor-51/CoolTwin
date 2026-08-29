import numpy as np

from evaluation.metrics import compute_comfort_hours, benchmark_inference_latency
from evaluation.baselines import RuleBasedThermostat


def test_compute_comfort_hours_conversion():
    # 4 K*step accumulated at 15-min (900s) steps -> 1 C-hour
    assert np.isclose(compute_comfort_hours(4.0, dt_seconds=900.0), 1.0)


def test_compute_comfort_hours_scales_with_dt():
    # same K*step total at a coarser control interval (1 hour) -> larger C-hours
    hours_15min = compute_comfort_hours(4.0, dt_seconds=900.0)
    hours_1hr = compute_comfort_hours(4.0, dt_seconds=3600.0)
    assert hours_1hr == 4 * hours_15min


def test_compute_comfort_hours_zero():
    assert compute_comfort_hours(0.0, dt_seconds=900.0) == 0.0


def test_benchmark_inference_latency_returns_expected_keys():
    ctrl = RuleBasedThermostat()
    obs = np.zeros(7, dtype=np.float32)
    result = benchmark_inference_latency(ctrl.act, obs, n_calls=20)
    for key in ["mean_ms", "p50_ms", "p95_ms"]:
        assert key in result
        assert result[key] >= 0.0


def test_benchmark_inference_latency_p95_at_least_p50():
    ctrl = RuleBasedThermostat()
    obs = np.zeros(7, dtype=np.float32)
    result = benchmark_inference_latency(ctrl.act, obs, n_calls=50)
    assert result["p95_ms"] >= result["p50_ms"]
