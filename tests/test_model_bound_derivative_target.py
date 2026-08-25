"""Focused checks for the internal model-bound derivative target."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helena import ModelConfig, OneZoneModel
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


def test_fixed_positive_support_and_parameter_order(monkeypatch):
    model = OneZoneModel(_config())

    def model_prediction(_model, _selection):
        def prediction(values):
            return jnp.asarray(
                (0.0, 2.0 * jnp.exp(values[0]), jnp.nan, 3.0 + values[1])
            )

        return np.asarray((1.0, 2.0, 3.0, 4.0)), prediction

    monkeypatch.setattr(public_fitting, "_model_prediction", model_prediction)
    energy, support, coordinates, target = public_fitting._earth_photon_log_sed_target(
        model, ("log_B", "log_Le")
    )

    np.testing.assert_array_equal(support, (1, 3))
    np.testing.assert_array_equal(energy, (2.0, 4.0))
    np.testing.assert_allclose(target(coordinates), np.log((2.0, 3.0)))
    jacobian = jax.jacfwd(target)(coordinates)
    np.testing.assert_allclose(jacobian, ((1.0, 0.0), (0.0, 1.0 / 3.0)))
    assert not np.asarray(energy).flags.writeable
    assert not np.asarray(support).flags.writeable
    assert not np.all(np.isfinite(np.asarray(target(coordinates.at[1].set(-4.0)))))

    with pytest.raises(ValueError, match="at least one"):
        public_fitting._earth_photon_log_sed_target(model, ())
    with pytest.raises(ValueError, match="unique"):
        public_fitting._earth_photon_log_sed_target(model, ("log_B", "log_B"))
    with pytest.raises(ValueError, match="unknown parameter"):
        public_fitting._earth_photon_log_sed_target(model, ("enable_ssc",))


@pytest.mark.skipif(
    not os.environ.get("HELENA_BH_CACHE_DIR"),
    reason="requires the existing strict prepared-response cache",
)
def test_real_target_equivalence_and_lowdim_derivatives():
    model = OneZoneModel(_config(), cache_dir=os.environ["HELENA_BH_CACHE_DIR"])

    energy_1d, support_1d, coordinates_1d, target_1d = (
        public_fitting._earth_photon_log_sed_target(model, ("log_B",))
    )
    baseline_log_flux = target_1d(coordinates_1d)
    baseline_log_flux.block_until_ready()
    forward = model.forward()
    forward_flux = np.asarray(
        forward.sed[
            "broadband_intrinsic_energy_flux_observer_erg_cm2_s"
        ]
    )
    np.testing.assert_array_equal(support_1d, np.arange(forward_flux.size))
    np.testing.assert_array_equal(
        energy_1d, np.asarray(forward.sed["broadband_energy_observer_erg"])
    )
    np.testing.assert_allclose(
        np.exp(np.asarray(baseline_log_flux)),
        forward_flux,
        rtol=1.0e-12,
        atol=np.finfo(np.float64).tiny,
    )

    direction_1d = jnp.ones(1, dtype=jnp.float64)
    _, jvp_1d = jax.jvp(target_1d, (coordinates_1d,), (direction_1d,))
    jacobian_1d = jax.jacfwd(target_1d)(coordinates_1d)
    jvp_1d.block_until_ready()
    jacobian_1d.block_until_ready()
    np.testing.assert_allclose(jvp_1d, jacobian_1d[:, 0], rtol=2.0e-9, atol=1.0e-10)

    step = jnp.float64(1.0e-4)
    plus = target_1d(coordinates_1d + step)
    minus = target_1d(coordinates_1d - step)
    plus.block_until_ready()
    minus.block_until_ready()
    finite_difference = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        finite_difference, jacobian_1d[:, 0], rtol=2.0e-2, atol=2.0e-5
    )

    _, support_2d, coordinates_2d, target_2d = (
        public_fitting._earth_photon_log_sed_target(model, ("log_B", "log_Le"))
    )
    compiled_jacobian = jax.jit(jax.jacfwd(target_2d))
    jacobian_2d = compiled_jacobian(coordinates_2d)
    repeated = compiled_jacobian(coordinates_2d)
    direction_2d = jnp.asarray((0.3, -0.2), dtype=jnp.float64)
    _, jvp_2d = jax.jvp(target_2d, (coordinates_2d,), (direction_2d,))
    jax.block_until_ready((jacobian_2d, repeated, jvp_2d))

    np.testing.assert_array_equal(support_2d, support_1d)
    np.testing.assert_allclose(
        jacobian_2d[:, 0], jacobian_1d[:, 0], rtol=2.0e-9, atol=1.0e-10
    )
    np.testing.assert_allclose(
        jvp_2d, jacobian_2d @ direction_2d, rtol=2.0e-9, atol=1.0e-10
    )
    np.testing.assert_array_equal(repeated, jacobian_2d)
    assert jacobian_2d.shape == (support_2d.size, 2)
    assert jacobian_2d.dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(jacobian_2d)))
    assert np.all(np.max(np.abs(np.asarray(jacobian_2d)), axis=0) > 0.0)
