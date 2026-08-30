"""
Tests for the Sinergym baseline (notebooks/00b_sinergym_baseline.py).

These are SKIPPED, not failed, when EnergyPlus isn't installed or
EPLUS_PATH isn't set -- Sinergym is an optional, higher-fidelity validation
path (see docs/sinergym_setup.md), not a required dependency for the rest of
the repo's tests, training, or CI. This mirrors the project's explicit scope
decision: everything else in CoolTwin runs on the lightweight RC-network
environment (twin/env.py) with no EnergyPlus install required.

Uses importlib rather than a normal `import` statement because the script's
filename starts with a digit (`00b_...`), which isn't a valid Python module
identifier -- consistent with the rest of the numbered notebooks/ scripts in
this repo, none of which are imported as modules elsewhere either.
"""

import importlib.util
import os

import pytest

_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "00b_sinergym_baseline.py",
)


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location("sinergym_baseline", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sinergym_baseline_runs_or_skips():
    module = _load_baseline_module()
    try:
        result = module.run_sinergym_baseline()
    except RuntimeError as e:
        pytest.skip(f"EnergyPlus/Sinergym not available -- see docs/sinergym_setup.md ({e})")
        return

    assert result["steps"] > 0
    assert "total_reward" in result
    assert result["observation_shape"][0] > 0
