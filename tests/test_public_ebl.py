"""Focused shared-Earth-prediction and component-closure regressions."""

import os
from dataclasses import replace

import jax
import numpy as np
import pytest

from helena import ModelConfig, OneZoneModel
from helena.fitting import public as public_fitting
from helena.model import _observer_ebl_support

GROUPS = (
    "transported_lepton_synchrotron",
    "transported_lepton_ssc",
    "proton_synchrotron",
    "neutral_pion_decay",
    "charged_pion_synchrotron",
    "muon_synchrotron",
)


def _config(**updates):
    config = ModelConfig(
        magnetic_field_G=0.4,
        radius_cm=5.0e15,
        redshift=0.3365,
        doppler_factor=40.0,
        luminosity_distance_cm=5.5e27,
        electron_kinetic_injection_luminosity_erg_s=3.0e41,
        electron_injection_index=2.0,
        electron_gamma_max=1.2e4,
        proton_kinetic_injection_luminosity_erg_s=6.0e45,
        proton_injection_index=2.0,
        proton_gamma_max=5.0e7,
    )
    return replace(config, **updates)


def test_ebl_switch_validation_and_non_extrapolated_support():
    with pytest.raises(TypeError, match="enable_ebl must be boolean"):
        _config(enable_ebl="yes")
    energy = np.asarray((1.0e-6, 1.0e-3, 1.0, 200.0)) * 1.602176634
    indices, transmission = _observer_ebl_support(
        _config(enable_ebl=True), energy
    )
    np.testing.assert_array_equal(indices, (0, 1, 2))
    assert transmission[0] == 1.0
    assert np.all((transmission > 0.0) & (transmission <= 1.0))
    off_indices, off_transmission = _observer_ebl_support(
        _config(), energy
    )
    np.testing.assert_array_equal(off_indices, np.arange(4))
    np.testing.assert_array_equal(off_transmission, np.ones(4))


@pytest.mark.skipif(
    not os.environ.get("HELENA_BH_CACHE_DIR"),
    reason="requires the existing strict prepared-response cache",
)
def test_real_forward_fit_target_jacobian_and_component_closure():
    cache = os.environ["HELENA_BH_CACHE_DIR"]
    enabled = OneZoneModel(_config(enable_ebl=True), cache_dir=cache)
    result = enabled.forward()
    earth_energy = result.sed["broadband_energy_earth_erg"]
    earth_flux = result.sed[
        "broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s"
    ]
    source_indices = result.sed["broadband_earth_source_indices"]
    transmission = result.sed["broadband_ebl_transmission"]
    intrinsic = result.sed[
        "broadband_intrinsic_energy_flux_observer_erg_cm2_s"
    ]
    np.testing.assert_array_equal(
        earth_flux, intrinsic[source_indices] * transmission
    )
    assert np.all(np.diff(earth_energy) > 0.0)
    assert np.all(transmission > 0.0)
    assert earth_energy[-1] / 1.602176634 <= 100.0

    component_sum = sum(
        result.components[
            f"{name}_broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s"
        ]
        for name in GROUPS
    )
    support = earth_flux > np.max(earth_flux) * 1.0e-20
    np.testing.assert_allclose(
        component_sum[support], earth_flux[support], rtol=2.0e-13, atol=0.0
    )

    energy, target_support, coordinates, target = (
        public_fitting._earth_photon_log_sed_target(
            enabled, ("log_Le", "log_Lp", "log_B", "log_R")
        )
    )
    jacobian, target_value = jax.jacfwd(
        lambda values: (target(values), target(values)), has_aux=True
    )(coordinates)
    target_value.block_until_ready()
    jacobian.block_until_ready()
    np.testing.assert_array_equal(energy, earth_energy[target_support])
    np.testing.assert_allclose(
        np.exp(np.asarray(target_value)), earth_flux[target_support],
        rtol=2.0e-11, atol=0.0,
    )
    assert jacobian.shape == (energy.size, 4)
    assert np.all(np.isfinite(jacobian))
    assert np.all(np.any(np.asarray(jacobian) != 0.0, axis=0))

    disabled = OneZoneModel(_config(enable_ebl=False), cache_dir=cache).forward()
    assert ModelConfig.__dataclass_fields__["enable_ebl"].default is False
    np.testing.assert_array_equal(
        disabled.sed["broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s"],
        disabled.sed["broadband_intrinsic_energy_flux_observer_erg_cm2_s"],
    )
