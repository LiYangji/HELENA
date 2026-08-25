"""Focused P2 checks for the public runtime and parameter registry."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helena.fitting import CANONICAL_PARAMETER_NAMES
from helena.parameters import (
    DIFFERENTIABLE_PARAMETER_NAMES,
    DIFFERENTIABLE_PARAMETERS,
    PARAMETERS_BY_NAME,
    ParameterSpec,
)

EXPECTED_NAMES = (
    "log_Le",
    "log_Lp",
    "log_B",
    "log_R",
    "p_e",
    "p_p",
    "log_gamma_e_max",
    "log_gamma_p_max",
)
ROOT = Path(__file__).resolve().parents[1]


def _run(script, **environment):
    env = os.environ.copy()
    env.update(environment)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "src"), env.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _runtime(device, *, skip_unavailable=False):
    script = f"""
import json
from helena import configure
runtime = configure(device={device!r})
import helena.models
import jax.numpy as jnp
x = jnp.asarray([0.125, 0.5, 1.25], dtype=jnp.float64)
y = jnp.sin(x) + jnp.log1p(x * x)
print(json.dumps({{"runtime": dict(runtime), "dtype": str(y.dtype), "value": y.tolist()}}))
"""
    result = _run(script)
    if result.returncode:
        if (
            skip_unavailable
            and device == "gpu"
            and "HELENA could not initialize requested device 'gpu'" in result.stderr
        ):
            pytest.skip("no JAX GPU is available")
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_registry_is_the_single_complete_validated_source():
    assert DIFFERENTIABLE_PARAMETER_NAMES == EXPECTED_NAMES
    assert CANONICAL_PARAMETER_NAMES is DIFFERENTIABLE_PARAMETER_NAMES
    assert len(DIFFERENTIABLE_PARAMETERS) == len(set(EXPECTED_NAMES)) == 8
    assert tuple(PARAMETERS_BY_NAME) == EXPECTED_NAMES
    assert all(
        isinstance(parameter, ParameterSpec) for parameter in DIFFERENTIABLE_PARAMETERS
    )
    assert all(
        parameter.name
        and parameter.quantity
        and parameter.coordinate
        and parameter.unit
        and parameter.differentiable
        and parameter.validated
        for parameter in DIFFERENTIABLE_PARAMETERS
    )
    assert (
        not {
            "enable_pp",
            "enable_ssc",
            "enable_ec",
            "bethe_heitler_mode",
            "ebl_model",
            "grid_size",
            "solver_tolerance",
            "physics_only",
        }
        & PARAMETERS_BY_NAME.keys()
    )
    with pytest.raises(TypeError):
        PARAMETERS_BY_NAME["extra"] = DIFFERENTIABLE_PARAMETERS[0]


def test_registry_coordinates_match_the_validated_mappers():
    coordinates = {
        parameter.name: parameter.coordinate for parameter in DIFFERENTIABLE_PARAMETERS
    }
    assert all(
        "natural-log reference offset" in coordinates[name]
        for name in EXPECTED_NAMES[:4]
    )
    assert all(
        "additive reference offset" in coordinates[name] for name in EXPECTED_NAMES[4:6]
    )
    assert all(
        "natural-log reference offset" in coordinates[name]
        for name in EXPECTED_NAMES[6:]
    )


def test_fresh_auto_and_cpu_runtime_contracts():
    auto = _runtime("auto")
    cpu = _runtime("cpu")

    assert auto["runtime"]["resolved_platform"] in ("cpu", "gpu", "tpu")
    assert cpu["runtime"]["resolved_platform"] == "cpu"
    assert all(result["runtime"]["x64"] for result in (auto, cpu))
    assert all(result["dtype"] == "float64" for result in (auto, cpu))


def test_fresh_gpu_runtime_and_cpu_parity():
    cpu = _runtime("cpu")
    gpu = _runtime("gpu", skip_unavailable=True)

    assert gpu["runtime"]["resolved_platform"] == "gpu"
    assert gpu["runtime"]["x64"]
    assert gpu["dtype"] == "float64"
    np.testing.assert_allclose(cpu["value"], gpu["value"], rtol=1.0e-13, atol=1.0e-15)


def test_invalid_unavailable_repeated_and_late_requests_fail_safely():
    invalid = _run("from helena import configure; configure(device='banana')")
    assert invalid.returncode != 0
    assert "device must be 'auto', 'cpu', or 'gpu'" in invalid.stderr

    unavailable = _run(
        "from helena import configure; configure(device='gpu')",
        CUDA_VISIBLE_DEVICES="",
    )
    assert unavailable.returncode != 0
    assert "no fallback or backend reset was attempted" in unavailable.stderr

    repeated = _run(
        "from helena import configure; "
        "print(dict(configure(device='cpu')) == dict(configure(device='cpu')))"
    )
    assert repeated.returncode == 0
    assert repeated.stdout.strip() == "True"

    late = _run(
        "import jax; jax.config.update('jax_platforms', 'cpu'); "
        "jax.default_backend(); from helena import configure; configure(device='gpu')"
    )
    assert late.returncode != 0
    assert "JAX had already resolved platform 'cpu'" in late.stderr
