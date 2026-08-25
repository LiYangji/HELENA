"""Focused checks for named subsets of the canonical fitting vector."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helena.fitting import CANONICAL_PARAMETER_NAMES, ParameterSelection
from helena.models import (
    c513_configuration_from_injection_shape_offsets,
    c513_configuration_from_log_offsets,
)

jax.config.update("jax_enable_x64", True)


def _bounds():
    return {
        name: (-10.0 + index, 10.0 + index)
        for index, name in enumerate(CANONICAL_PARAMETER_NAMES)
    }


def _selection(free_names, baseline=None):
    return ParameterSelection(
        free_names=free_names,
        baseline_values=np.arange(8.0) if baseline is None else baseline,
        bounds_by_name=_bounds(),
    )


def test_canonical_order_subset_round_trip_fixed_values_and_bounds():
    assert CANONICAL_PARAMETER_NAMES == (
        "log_Le",
        "log_Lp",
        "log_B",
        "log_R",
        "p_e",
        "p_p",
        "log_gamma_e_max",
        "log_gamma_p_max",
    )
    selection = _selection(("p_p", "log_Le", "log_gamma_e_max"))
    full = np.arange(8.0) + 0.25
    packed = selection.pack(full)
    reconstructed = np.asarray(selection.unpack(packed))

    np.testing.assert_array_equal(packed, full[[5, 0, 6]])
    np.testing.assert_array_equal(reconstructed[[5, 0, 6]], full[[5, 0, 6]])
    np.testing.assert_array_equal(reconstructed[[1, 2, 3, 4, 7]], [1, 2, 3, 4, 7])
    assert selection.fixed_names == (
        "log_Lp",
        "log_B",
        "log_R",
        "p_e",
        "log_gamma_p_max",
    )
    assert selection.free_bounds == ((-5.0, 15.0), (-10.0, 10.0), (-4.0, 16.0))
    with pytest.raises(ValueError):
        selection.baseline_values[0] = 99.0


@pytest.mark.parametrize(
    "free_names",
    [
        ("log_Le", "log_B", "log_R", "p_e"),
        ("log_Le", "log_B", "log_R", "p_e", "log_gamma_e_max"),
        ("log_Le", "log_B", "log_R", "p_e", "log_gamma_e_max", "log_Lp"),
        CANONICAL_PARAMETER_NAMES,
    ],
)
def test_staged_dimensions(free_names):
    selection = _selection(free_names)
    assert selection.pack(np.zeros(8)).shape == (len(free_names),)
    assert selection.unpack(np.zeros(len(free_names))).shape == (8,)


@pytest.mark.parametrize(
    ("free_names", "bounds", "message"),
    [
        (("not_a_parameter",), _bounds(), "unknown parameter"),
        (("log_Le", "log_Le"), _bounds(), "unique"),
        (("log_Le",), {}, "missing bounds"),
        (("log_Le",), {"log_Le": (1.0, 1.0)}, "finite and increasing"),
        (("log_Le",), {"not_a_parameter": (0.0, 1.0)}, "unknown bound"),
    ],
)
def test_invalid_names_and_bounds_fail_clearly(free_names, bounds, message):
    with pytest.raises(ValueError, match=message):
        ParameterSelection(free_names, np.zeros(8), bounds)


def test_unpack_is_finite_through_jit_and_value_and_grad():
    selection = _selection(("log_R", "p_e"), baseline=np.zeros(8))

    @jax.jit
    def loss(values):
        full = selection.unpack(values)
        return jnp.sum(jnp.exp(0.1 * full))

    value, gradient = jax.value_and_grad(loss)(jnp.asarray([0.2, -0.3]))
    assert np.isfinite(value)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.asarray(gradient).shape == (2,)


def test_all_free_selection_preserves_existing_eight_parameter_transform():
    gamma = jnp.asarray([1.000001, 10.0, 100.0], dtype=jnp.float64)
    configuration = {
        "primary_electron_injection_dgamma_cm3_s": jnp.ones(3),
        "primary_proton_injection_dgamma_cm3_s": 2.0 * jnp.ones(3),
        "magnetic_field_G": 0.1,
        "radius_cm": 1.0e16,
        "electron_escape_rate_s_inv": 1.0e-6,
        "proton_escape_rate_s_inv": 1.0e-6,
        "electron_injection_normalization_cm3_s": 1.0,
        "proton_injection_normalization_cm3_s": 2.0,
        "electron_index": 2.0,
        "proton_index": 2.0,
        "electron_gamma_cutoff": 3.0e5,
        "proton_gamma_cutoff": 1.0e7,
        "electron_gamma": gamma,
        "proton_gamma": gamma,
    }
    theta = jnp.asarray([0.1, -0.2, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08])
    selection = _selection(CANONICAL_PARAMETER_NAMES, baseline=np.zeros(8))

    def transform(values):
        dynamic = c513_configuration_from_log_offsets(configuration, values[:4])
        return c513_configuration_from_injection_shape_offsets(dynamic, values[4:])

    direct = transform(theta)
    selected = transform(selection.unpack(selection.pack(theta)))
    assert direct.keys() == selected.keys()
    for name in direct:
        np.testing.assert_array_equal(
            np.asarray(direct[name]), np.asarray(selected[name])
        )
