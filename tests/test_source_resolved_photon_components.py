"""Focused post-convergence source-ownership and closure regression."""

import jax.numpy as jnp
import numpy as np

from helena.core.jax_backend.radiation.inverse_compton import inverse_compton_sed_jax
from helena.core.jax_backend.radiation.synchrotron import electron_synchrotron_sed_jax
from helena.core.jax_backend.transport.electron_positron_transport import (
    electron_positron_steady_state_transport_jax,
)
from helena.models.c513 import (
    _SOURCE_RESOLVED_ADDITIVE_COMPONENTS,
    _canonical_broadband_photon_support,
    _source_resolved_broadband_components,
)


def test_frozen_source_resolved_transport_radiation_and_escape_close():
    gamma = jnp.geomspace(1.01, 1.0e4, 17)
    low_energy = jnp.geomspace(1.0e-16, 1.0e-6, 13)
    high_energy = jnp.geomspace(1.0e-6, 1.0, 15)
    source_shape = gamma**-2 * jnp.exp(-gamma / 3.0e3)
    sources = {
        "primary": (1.0e-3 * source_shape, 0.1e-3 * source_shape),
        "bethe_heitler": (0.3e-3 * source_shape, 0.4e-3 * source_shape),
        "photomeson": (0.2e-3 * source_shape, 0.25e-3 * source_shape),
        "gamma_gamma": (0.05e-3 * source_shape, 0.07e-3 * source_shape),
    }
    configuration = {
        "electron_gamma": gamma,
        "synchrotron_photon_energy_erg": low_energy,
        "gamma_ray_energy_erg": high_energy,
        "synchrotron_split_energy_erg": low_energy[-1],
        "primary_electron_injection_dgamma_cm3_s": sources["primary"][0],
        "primary_positron_injection_dgamma_cm3_s": sources["primary"][1],
        "magnetic_field_G": 0.5,
        "radius_cm": 1.0e14,
        "electron_adiabatic_rate_s_inv": jnp.zeros_like(gamma),
        "electron_escape_rate_s_inv": jnp.full_like(gamma, 1.0e-5),
        "additional_continuous_loss_rate_dgamma_s": jnp.zeros_like(gamma),
    }
    total_electron = sum(source[0] for source in sources.values())
    total_positron = sum(source[1] for source in sources.values())
    electron_density, positron_density, _, _ = (
        electron_positron_steady_state_transport_jax(
            gamma,
            total_electron,
            total_positron,
            configuration["magnetic_field_G"],
            configuration["electron_adiabatic_rate_s_inv"],
            configuration["electron_escape_rate_s_inv"],
        )
    )
    total_density = electron_density + positron_density
    target_density = jnp.geomspace(1.0e15, 1.0e5, low_energy.size)
    low_synchrotron = electron_synchrotron_sed_jax(
        low_energy,
        gamma,
        total_density,
        magnetic_field_G=configuration["magnetic_field_G"],
        radius_cm=configuration["radius_cm"],
    )
    high_synchrotron = jnp.where(
        high_energy > low_energy[-1],
        electron_synchrotron_sed_jax(
            high_energy,
            gamma,
            total_density,
            magnetic_field_G=configuration["magnetic_field_G"],
            radius_cm=configuration["radius_cm"],
        ),
        0.0,
    )
    _, low_stop, _ = _canonical_broadband_photon_support(configuration)
    low_inverse_compton = jnp.zeros_like(low_energy).at[:low_stop].set(
        inverse_compton_sed_jax(
            low_energy[:low_stop],
            gamma,
            total_density,
            low_energy,
            target_density,
            radius_cm=configuration["radius_cm"],
        )
    )
    high_inverse_compton = jnp.where(
        high_energy > low_energy[-1],
        inverse_compton_sed_jax(
            high_energy,
            gamma,
            total_density,
            low_energy,
            target_density,
            radius_cm=configuration["radius_cm"],
        ),
        0.0,
    )
    zeros_low = jnp.zeros_like(low_energy)
    zeros_high = jnp.zeros_like(high_energy)
    inner = {
        "external_compton_intrinsic_sed_erg_s": zeros_high,
        "total_inverse_compton_loss_rate_dgamma_s": jnp.zeros_like(gamma),
        "enable_ssc": jnp.asarray(True),
        "electron_pair_injection_dgamma_cm3_s": sources["gamma_gamma"][0],
        "positron_pair_injection_dgamma_cm3_s": sources["gamma_gamma"][1],
        "electron_density_dgamma_cm3": electron_density,
        "positron_density_dgamma_cm3": positron_density,
        "internal_synchrotron_photon_density_cm3_erg": target_density,
        "electron_synchrotron_intrinsic_sed_erg_s": low_synchrotron,
        "electron_synchrotron_high_energy_intrinsic_sed_erg_s": high_synchrotron,
        "ssc_low_energy_intrinsic_sed_erg_s": low_inverse_compton,
        "ssc_high_energy_intrinsic_sed_erg_s": high_inverse_compton,
        "electron_synchrotron_absorption_coefficient_cm1": zeros_low,
        "gamma_gamma_transfer_factor": jnp.ones_like(high_energy),
        "proton_synchrotron_low_energy_intrinsic_sed_erg_s": zeros_low,
        "proton_synchrotron_high_energy_intrinsic_sed_erg_s": zeros_high,
        "pi0_intrinsic_sed_erg_s": zeros_high,
        "total_low_energy_photon_escaped_sed_erg_s": (
            low_synchrotron + low_inverse_compton
        ),
        "escaped_gamma_sed_erg_s": high_synchrotron + high_inverse_compton,
    }
    final = {
        "inner_electromagnetic_cascade": inner,
        "q_bh_electron_dgamma_cm3_s": sources["bethe_heitler"][0],
        "q_bh_positron_dgamma_cm3_s": sources["bethe_heitler"][1],
        "photomeson_chain": {
            "electron": sources["photomeson"][0],
            "positron": sources["photomeson"][1],
        },
        "meson_synchrotron_low_energy_intrinsic_seds_erg_s": {
            "pion_total": zeros_low,
            "muon_total": zeros_low,
        },
        "meson_synchrotron_intrinsic_seds_erg_s": {
            "pion_total": zeros_high,
            "muon_total": zeros_high,
        },
    }

    components, diagnostics = _source_resolved_broadband_components(
        configuration, final
    )

    assert tuple(components)[: len(_SOURCE_RESOLVED_ADDITIVE_COMPONENTS)] == (
        _SOURCE_RESOLVED_ADDITIVE_COMPONENTS
    )
    assert diagnostics["source_resolved_gamma_gamma_parent_ownership"] == "aggregate"
    for name, value in diagnostics.items():
        if name.endswith(("maximum_relative_error", "weighted_error")):
            assert value < 2.0e-13
    expected = np.concatenate(
        (
            np.asarray(low_synchrotron + low_inverse_compton)[:low_stop],
            np.asarray(high_synchrotron + high_inverse_compton)[1:],
        )
    )
    np.testing.assert_allclose(
        sum(
            np.asarray(components[name])
            for name in _SOURCE_RESOLVED_ADDITIVE_COMPONENTS
        ),
        expected,
        rtol=2.0e-13,
        atol=0.0,
    )
    assert np.any(np.asarray(components["bethe_heitler_secondary_radiation"]) > 0.0)
    assert np.any(np.asarray(components["photomeson_secondary_radiation"]) > 0.0)
    assert np.any(np.asarray(components["gamma_gamma_reprocessed_radiation"]) > 0.0)
    np.testing.assert_array_equal(
        components["primary_electron_inverse_compton"],
        components["primary_electron_ssc"],
    )
    for source, aggregate in (
        ("bethe_heitler_pair", "bethe_heitler_secondary_radiation"),
        ("photomeson_secondary_pair", "photomeson_secondary_radiation"),
        ("gamma_gamma_reprocessed_pair", "gamma_gamma_reprocessed_radiation"),
    ):
        np.testing.assert_allclose(
            components[f"{source}_synchrotron"]
            + components[f"{source}_inverse_compton"],
            components[aggregate],
            rtol=2.0e-13,
            atol=0.0,
        )
