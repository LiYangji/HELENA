"""Focused post-convergence source-ownership and closure regression."""

import jax.numpy as jnp
import numpy as np
import pytest

from helena.core.jax_backend.kernels.inverse_compton_cooling import (
    electron_inverse_compton_loss_rate_dgamma_s_jax,
)
from helena.core.jax_backend.radiation.inverse_compton import inverse_compton_sed_jax
from helena.core.jax_backend.radiation.synchrotron import electron_synchrotron_sed_jax
from helena.core.jax_backend.transport.electron_positron_transport import (
    electron_positron_steady_state_transport_jax,
)
from helena.models.c513 import (
    _SOURCE_RESOLVED_ADDITIVE_COMPONENTS,
    _source_resolved_broadband_components,
)


@pytest.mark.parametrize("enable_ec", [False, True])
@pytest.mark.parametrize("enable_internal_ic", [False, True])
def test_frozen_source_resolved_transport_radiation_and_escape_close(
    enable_ec, enable_internal_ic
):
    gamma = jnp.geomspace(1.01, 1.0e4, 17)
    photon_energy = jnp.geomspace(1.0e-16, 1.0, 27)
    external_energy = jnp.geomspace(1.0e-13, 1.0e-10, 7)
    external_density = jnp.geomspace(1.0e16, 1.0e12, 7)
    source_shape = gamma**-2 * jnp.exp(-gamma / 3.0e3)
    sources = {
        "primary": (1.0e-3 * source_shape, 0.1e-3 * source_shape),
        "bethe_heitler": (0.3e-3 * source_shape, 0.4e-3 * source_shape),
        "photomeson": (0.2e-3 * source_shape, 0.25e-3 * source_shape),
        "gamma_gamma": (0.05e-3 * source_shape, 0.07e-3 * source_shape),
    }
    configuration = {
        "electron_gamma": gamma,
        "synchrotron_photon_energy_erg": photon_energy,
        "gamma_ray_energy_erg": photon_energy,
        "primary_electron_injection_dgamma_cm3_s": sources["primary"][0],
        "primary_positron_injection_dgamma_cm3_s": sources["primary"][1],
        "magnetic_field_G": 0.5,
        "radius_cm": 1.0e14,
        "electron_adiabatic_rate_s_inv": jnp.zeros_like(gamma),
        "electron_escape_rate_s_inv": jnp.full_like(gamma, 1.0e-5),
        "additional_continuous_loss_rate_dgamma_s": jnp.zeros_like(gamma),
    }
    synchrotron_seed = jnp.geomspace(1.0e15, 1.0e5, photon_energy.size)
    # A distinct non-synchrotron contribution makes a stale SSC-only path fail.
    total_seed = synchrotron_seed + 2.0 * jnp.geomspace(
        1.0e14, 1.0e6, photon_energy.size
    )
    internal_cooling = (
        electron_inverse_compton_loss_rate_dgamma_s_jax(
            gamma, photon_energy, total_seed
        )
        if enable_internal_ic
        else jnp.zeros_like(gamma)
    )
    external_cooling = (
        electron_inverse_compton_loss_rate_dgamma_s_jax(
            gamma, external_energy, external_density
        )
        if enable_ec
        else jnp.zeros_like(gamma)
    )
    cooling = internal_cooling + external_cooling
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
            additional_continuous_loss_rate_dgamma_s=cooling,
        )
    )
    total_density = electron_density + positron_density
    synchrotron = electron_synchrotron_sed_jax(
        photon_energy,
        gamma,
        total_density,
        magnetic_field_G=configuration["magnetic_field_G"],
        radius_cm=configuration["radius_cm"],
    )
    internal_ic = (
        inverse_compton_sed_jax(
            photon_energy,
            gamma,
            total_density,
            photon_energy,
            total_seed,
            radius_cm=configuration["radius_cm"],
        )
        if enable_internal_ic
        else jnp.zeros_like(photon_energy)
    )
    external_compton = (
        inverse_compton_sed_jax(
            photon_energy,
            gamma,
            total_density,
            external_energy,
            external_density,
            radius_cm=configuration["radius_cm"],
        )
        if enable_ec
        else jnp.zeros_like(photon_energy)
    )
    zeros = jnp.zeros_like(photon_energy)
    components = {
        "electron_synchrotron": synchrotron,
        "internal_ic": internal_ic,
        "external_compton": external_compton,
        "proton_synchrotron": zeros,
        "pion_synchrotron": zeros,
        "muon_synchrotron": zeros,
        "pi0": zeros,
    }
    # This is a frozen linear-decomposition test, not a self-consistent source.
    # A nontrivial accepted transfer makes accidental retransfer observable.
    transfer = jnp.geomspace(0.2, 1.0, photon_energy.size)
    inner = {
        "photon_energy_erg": photon_energy,
        "photon_components_intrinsic_sed_erg_s": components,
        "photon_components_escaped_sed_erg_s": {
            name: values * transfer for name, values in components.items()
        },
        "photon_transfer_factor": transfer,
        "escaped_photon_sed_erg_s": sum(components.values()) * transfer,
        "external_photon_energy_erg": external_energy,
        "external_photon_density_cm3_erg": external_density,
        "total_inverse_compton_loss_rate_dgamma_s": cooling,
        "enable_internal_ic": jnp.asarray(enable_internal_ic),
        "enable_ec": jnp.asarray(enable_ec),
        "electron_pair_injection_dgamma_cm3_s": sources["gamma_gamma"][0],
        "positron_pair_injection_dgamma_cm3_s": sources["gamma_gamma"][1],
        "electron_density_dgamma_cm3": electron_density,
        "positron_density_dgamma_cm3": positron_density,
        "internal_synchrotron_photon_density_cm3_erg": synchrotron_seed,
        "internal_ic_seed_photon_density_cm3_erg": total_seed,
        "internal_photon_density_cm3_erg": total_seed,
    }
    final = {
        "inner_electromagnetic_cascade": inner,
        "q_bh_electron_dgamma_cm3_s": sources["bethe_heitler"][0],
        "q_bh_positron_dgamma_cm3_s": sources["bethe_heitler"][1],
        "photomeson_chain": {
            "electron": sources["photomeson"][0],
            "positron": sources["photomeson"][1],
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
    expected = np.asarray((synchrotron + internal_ic + external_compton) * transfer)
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
    primary_density = sum(
        electron_positron_steady_state_transport_jax(
            gamma,
            *sources["primary"],
            configuration["magnetic_field_G"],
            configuration["electron_adiabatic_rate_s_inv"],
            configuration["electron_escape_rate_s_inv"],
            additional_continuous_loss_rate_dgamma_s=cooling,
        )[:2]
    )
    expected_primary_ssc = (
        inverse_compton_sed_jax(
            photon_energy,
            gamma,
            primary_density,
            photon_energy,
            synchrotron_seed,
            radius_cm=configuration["radius_cm"],
        )
        * transfer
        if enable_internal_ic
        else jnp.zeros_like(photon_energy)
    )
    np.testing.assert_allclose(
        components["primary_electron_ssc"], expected_primary_ssc, rtol=2e-13, atol=0
    )
    assert "primary_electron_ssc" not in _SOURCE_RESOLVED_ADDITIVE_COMPONENTS
    if enable_internal_ic or enable_ec:
        assert np.any(
            np.asarray(
                components["primary_electron_inverse_compton"]
                - components["primary_electron_ssc"]
            )
            > 0
        )
    else:
        np.testing.assert_array_equal(components["primary_electron_inverse_compton"], 0)
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
