"""Focused P6 checks for the public sensitivity facade."""

import os
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helena import ModelConfig, OneZoneModel, sensitivity
from helena.fitting import public as public_fitting


def _config():
    return ModelConfig(
        magnetic_field_G=0.1,
        radius_cm=1.0e16,
        redshift=0.1,
        doppler_factor=30.0,
        luminosity_distance_cm=1.0e27,
        electron_kinetic_injection_luminosity_erg_s=3.7e40,
        electron_injection_index=2.0,
        electron_gamma_max=3.0e5,
        proton_kinetic_injection_luminosity_erg_s=2.8e46,
        proton_injection_index=2.0,
        proton_gamma_max=1.0e7,
    )


@pytest.fixture
def linear_target(monkeypatch):
    columns = {
        "log_Le": (1.0, 0.0, -1.0),
        "log_Lp": (0.0, 2.0, 1.0),
        "log_B": (3.0, -1.0, 0.5),
        "log_R": (0.2, 0.4, 0.6),
        "p_e": (-0.5, 0.7, 1.2),
        "p_p": (1.1, -0.3, 0.8),
        "log_gamma_e_max": (0.9, 0.1, -0.2),
        "log_gamma_p_max": (-0.4, 0.5, 0.3),
    }

    def build(_model, names):
        design = jnp.asarray(np.column_stack([columns[name] for name in names]))
        coordinates = jnp.arange(len(names), dtype=jnp.float64) / 10.0

        def target(values):
            return jnp.asarray((2.0, 3.0, 4.0)) + design @ values

        return (
            np.asarray((1.0, 2.0, 4.0)),
            np.asarray((0, 2, 5)),
            coordinates,
            target,
        )

    monkeypatch.setattr(sensitivity, "_earth_photon_log_sed_target", build)
    return columns


def test_import_surface_order_jvp_and_repeated_determinism(linear_target):
    assert sensitivity.__all__ == ["SensitivityResult", "jacobian", "jvp"]
    assert not hasattr(sensitivity, "gradient")
    assert not hasattr(sensitivity, "jacobian_8d")

    names = ("log_B", "log_Le")
    expected = np.column_stack([linear_target[name] for name in names])
    jacobian = sensitivity.jacobian(object(), parameters=names)
    repeated = sensitivity.jacobian(object(), parameters=names)
    directional = sensitivity.jvp(object(), parameters=names, direction=(0.3, -0.2))

    assert isinstance(jacobian, sensitivity.SensitivityResult)
    assert jacobian.parameters == names
    assert jacobian.orientation == "energy, parameter"
    assert directional.orientation == "energy"
    assert "E F_E^Earth" in jacobian.target
    np.testing.assert_array_equal(jacobian.values, expected)
    np.testing.assert_array_equal(repeated.values, jacobian.values)
    np.testing.assert_allclose(directional.values, expected @ (0.3, -0.2))
    np.testing.assert_array_equal(jacobian.energy_erg, (1.0, 2.0, 4.0))
    np.testing.assert_array_equal(jacobian.support_indices, (0, 2, 5))
    assert not jacobian.values.flags.writeable
    assert not jacobian.energy_erg.flags.writeable
    with pytest.raises(FrozenInstanceError):
        jacobian.values = np.zeros_like(jacobian.values)


def test_one_parameter_arbitrary_subset_validation_and_cpu_warning(
    monkeypatch, linear_target
):
    one = sensitivity.jacobian(object(), parameters=["log_B"])
    np.testing.assert_array_equal(one.values[:, 0], linear_target["log_B"])

    names = ("p_p", "log_R", "log_gamma_e_max")
    monkeypatch.setattr(sensitivity.jax, "default_backend", lambda: "cpu")
    with pytest.warns(RuntimeWarning, match="more than two"):
        arbitrary = sensitivity.jacobian(object(), parameters=names)
    assert arbitrary.parameters == names
    np.testing.assert_array_equal(
        arbitrary.values, np.column_stack([linear_target[name] for name in names])
    )

    for parameters, error, message in (
        ((), ValueError, "at least one"),
        (("log_B", "log_B"), ValueError, "unique"),
        (("not_a_parameter",), ValueError, "unknown parameter"),
        (("enable_ssc",), ValueError, "unknown parameter"),
    ):
        with pytest.raises(error, match=message):
            sensitivity.jacobian(object(), parameters=parameters)
    with pytest.raises(TypeError, match="sequence"):
        sensitivity.jacobian(object(), parameters="log_B")
    with pytest.raises(ValueError, match="direction"):
        sensitivity.jvp(object(), parameters=("log_B", "log_Le"), direction=(1.0,))


@pytest.mark.skipif(
    not os.environ.get("HELENA_BH_CACHE_DIR"),
    reason="requires the existing strict prepared-response cache",
)
def test_real_public_owner_equivalence_finite_difference_and_jvp_contraction():
    model = OneZoneModel(_config(), cache_dir=os.environ["HELENA_BH_CACHE_DIR"])

    public_1d = sensitivity.jacobian(model, parameters=("log_B",))
    energy, support, coordinates, target = public_fitting._earth_photon_log_sed_target(
        model, ("log_B",)
    )
    internal_1d = jax.jacfwd(target)(coordinates)
    internal_1d.block_until_ready()
    np.testing.assert_array_equal(public_1d.energy_erg, energy)
    np.testing.assert_array_equal(public_1d.support_indices, support)
    np.testing.assert_allclose(public_1d.values, internal_1d, rtol=2.0e-9, atol=1.0e-10)

    step = jnp.float64(1.0e-4)
    finite_difference = (target(coordinates + step) - target(coordinates - step)) / (
        2.0 * step
    )
    finite_difference.block_until_ready()
    np.testing.assert_allclose(
        public_1d.values[:, 0], finite_difference, rtol=2.0e-2, atol=2.0e-5
    )

    names = ("log_B", "log_Le")
    public_2d = sensitivity.jacobian(model, parameters=names)
    public_jvp = sensitivity.jvp(model, parameters=names, direction=(0.3, -0.2))
    np.testing.assert_array_equal(public_2d.parameters, names)
    np.testing.assert_array_equal(public_2d.support_indices, public_1d.support_indices)
    np.testing.assert_allclose(
        public_2d.values[:, 0], public_1d.values[:, 0], rtol=2.0e-9, atol=1.0e-10
    )
    np.testing.assert_allclose(
        public_jvp.values,
        public_2d.values @ np.asarray((0.3, -0.2)),
        rtol=2.0e-9,
        atol=1.0e-10,
    )
