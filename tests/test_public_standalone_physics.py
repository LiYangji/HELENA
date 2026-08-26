"""Focused equivalence and transparency checks for the P5 L2 facade."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import helena
import helena.cascade as public_cascade
from helena import physics, transport
from helena.core.constants import (
    CHARGED_PION_MASS_CGS,
    ELECTRON_MASS_CGS,
    ELEMENTARY_CHARGE_ESU,
    MUON_MASS_CGS,
    PLANCK_CONSTANT_CGS,
    PROTON_MASS_CGS,
)
from helena.core.jax_backend.chains.steady_state_electromagnetic_cascade import (
    steady_state_electromagnetic_cascade_jax,
)
from helena.core.jax_backend.kernels.bethe_heitler import (
    bethe_heitler_proton_energy_loss_rate_jax,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    gamma_gamma_absorbed_sed_jax,
    gamma_gamma_absorption_coefficient_jax,
    gamma_gamma_optical_depth_jax,
)
from helena.core.jax_backend.kernels.muon_decay import (
    muon_decay_electron_positron_injection_jax,
    muon_decay_neutrino_injection_jax,
)
from helena.core.jax_backend.kernels.photomeson import (
    photomeson_energy_loss_rate_jax,
    photomeson_interaction_rate_jax,
    photomeson_pion_injection_jax,
)
from helena.core.jax_backend.kernels.pion_decay import (
    charged_pion_decay_muon_injection_jax,
    charged_pion_decay_neutrino_injection_jax,
    neutral_pion_decay_photon_injection_jax,
)
from helena.core.jax_backend.kernels.proton_proton import (
    proton_proton_pion_injection_and_loss_jax,
)
from helena.core.jax_backend.radiation.inverse_compton import (
    inverse_compton_sed_jax,
)
from helena.core.jax_backend.radiation.photon_fields import (
    produced_sed_to_internal_photon_density_jax,
)
from helena.core.jax_backend.radiation.ssa import (
    electron_synchrotron_self_absorbed_sed_jax,
    synchrotron_absorption_coefficient_jax,
)
from helena.core.jax_backend.radiation.synchrotron import (
    electron_synchrotron_sed_jax,
    synchrotron_sed_jax,
)
from helena.core.jax_backend.transport.steady_state_finite_volume import (
    steady_state_finite_volume_solution_jax,
)

jax.config.update("jax_enable_x64", True)


def _assert_exact_tree(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def test_l2_modules_leave_the_root_l1_namespace_unchanged():
    assert helena.__all__ == [
        "FitResult",
        "ForwardResult",
        "ModelConfig",
        "OneZoneModel",
        "SEDDataset",
        "configure",
        "fit",
    ]
    assert physics.__all__
    assert transport.__all__ == ["solve_steady_state"]
    assert public_cascade.__all__ == ["EMCascadeResult", "solve_em"]
    assert physics.__all__.count("electron_synchrotron_self_absorbed_sed") == 1
    assert not any(name.endswith("_jax") for name in physics.__all__)


def test_radiation_and_gamma_gamma_facades_match_production_owners_exactly():
    photon_energy = jnp.geomspace(1.0e-14, 1.0e-6, 5)
    gamma = jnp.geomspace(1.01, 1.0e5, 9)
    density = 3.0e2 * gamma**-2.1
    common = {"magnetic_field_G": 1.7, "radius_cm": 3.0e15}

    np.testing.assert_array_equal(
        physics.electron_synchrotron(photon_energy, gamma, density, **common),
        electron_synchrotron_sed_jax(photon_energy, gamma, density, **common),
    )
    for public, mass in (
        (physics.proton_synchrotron, PROTON_MASS_CGS),
        (physics.charged_pion_synchrotron, CHARGED_PION_MASS_CGS),
        (physics.muon_synchrotron, MUON_MASS_CGS),
    ):
        np.testing.assert_array_equal(
            public(photon_energy, gamma, density, **common),
            synchrotron_sed_jax(
                photon_energy,
                gamma,
                density,
                **common,
                particle_mass_g=mass,
                charge_esu=ELEMENTARY_CHARGE_ESU,
            ),
        )

    np.testing.assert_array_equal(
        physics.electron_synchrotron_absorption_coefficient(
            photon_energy,
            gamma,
            density,
            magnetic_field_G=common["magnetic_field_G"],
        ),
        synchrotron_absorption_coefficient_jax(
            photon_energy,
            gamma,
            density,
            magnetic_field_G=common["magnetic_field_G"],
            particle_mass_g=ELECTRON_MASS_CGS,
            charge_esu=ELEMENTARY_CHARGE_ESU,
        ),
    )
    np.testing.assert_array_equal(
        physics.electron_synchrotron_self_absorbed_sed(
            photon_energy,
            gamma,
            density,
            **common,
        ),
        electron_synchrotron_self_absorbed_sed_jax(
            photon_energy,
            gamma,
            density,
            **common,
        ),
    )

    target_energy = jnp.geomspace(1.0e-14, 1.0e-7, 7)
    target_density = 1.0e22 * jnp.exp(-((jnp.log(target_energy / 1.0e-10) / 1.5) ** 2))
    np.testing.assert_array_equal(
        physics.inverse_compton(
            photon_energy,
            gamma,
            density,
            target_energy,
            target_density,
            radius_cm=common["radius_cm"],
        ),
        inverse_compton_sed_jax(
            photon_energy,
            gamma,
            density,
            target_energy,
            target_density,
            radius_cm=common["radius_cm"],
        ),
    )
    thin_sed = physics.electron_synchrotron(photon_energy, gamma, density, **common)
    absorption = jnp.maximum(
        physics.electron_synchrotron_absorption_coefficient(
            photon_energy,
            gamma,
            density,
            magnetic_field_G=common["magnetic_field_G"],
        ),
        0.0,
    )
    np.testing.assert_array_equal(
        physics.internal_photon_density(
            photon_energy,
            thin_sed,
            absorption,
            radius_cm=common["radius_cm"],
        ),
        produced_sed_to_internal_photon_density_jax(
            photon_energy,
            thin_sed,
            absorption,
            radius_cm=common["radius_cm"],
        ),
    )

    gamma_ray_energy = jnp.geomspace(1.0e-8, 1.0e-2, 6)
    coefficient = physics.gamma_gamma_absorption_coefficient(
        gamma_ray_energy, target_energy, target_density
    )
    np.testing.assert_array_equal(
        coefficient,
        gamma_gamma_absorption_coefficient_jax(
            gamma_ray_energy, target_energy, target_density
        ),
    )
    np.testing.assert_array_equal(
        physics.gamma_gamma_optical_depth(
            gamma_ray_energy,
            target_energy,
            target_density,
            radius_cm=common["radius_cm"],
        ),
        gamma_gamma_optical_depth_jax(
            gamma_ray_energy,
            target_energy,
            target_density,
            radius_cm=common["radius_cm"],
        ),
    )
    intrinsic = jnp.geomspace(1.0e30, 1.0e36, gamma_ray_energy.size)
    np.testing.assert_array_equal(
        physics.gamma_gamma_absorbed_sed(
            intrinsic, coefficient, radius_cm=common["radius_cm"]
        ),
        gamma_gamma_absorbed_sed_jax(
            intrinsic, coefficient, radius_cm=common["radius_cm"]
        ),
    )


def test_public_escaped_ssa_accepts_full_quickstart_arrays_and_keeps_maser_guard():
    for magnetic_field_G, gamma_cutoff in ((0.1, 1.0e4), (0.01, 1.0e6)):
        gamma = np.geomspace(1.0 + 1.0e-8, 100.0 * gamma_cutoff, 240)
        density = 1.0e4 * gamma**-1.9 * np.exp(-gamma / gamma_cutoff)
        photon_energy = PLANCK_CONSTANT_CGS * np.geomspace(
            1.0e6 / 30.0, 1.0e33 / 30.0, 240
        )
        keywords = {
            "magnetic_field_G": magnetic_field_G,
            "radius_cm": 1.0e15,
        }

        actual = physics.electron_synchrotron_self_absorbed_sed(
            photon_energy, gamma, density, **keywords
        )
        expected = electron_synchrotron_self_absorbed_sed_jax(
            photon_energy, gamma, density, **keywords
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == photon_energy.shape
        assert actual.dtype == jnp.float64
        assert np.all(np.isfinite(actual))
        assert np.all(np.asarray(actual) >= 0.0)

    inversion_energy = np.geomspace(1.0e-13, 1.0e-9, 8)
    inversion_gamma = np.geomspace(1.0e2, 1.0e6, 512)
    inversion_density = (inversion_gamma / inversion_gamma[0]) ** 3
    with pytest.raises(ValueError, match="negative.*maser"):
        physics.electron_synchrotron_self_absorbed_sed(
            inversion_energy,
            inversion_gamma,
            inversion_density,
            magnetic_field_G=1.0,
            radius_cm=1.0e16,
        )


def test_hadronic_and_decay_facades_match_production_owners_exactly():
    proton_gamma = jnp.geomspace(1.0e5, 1.0e10, 9)
    proton_density = 1.0e3 * (proton_gamma / proton_gamma[0]) ** -2.0
    target_energy = jnp.geomspace(1.0e-12, 1.0e-7, 9)
    target_density = 1.0e24 * jnp.exp(-((jnp.log(target_energy / 1.0e-9) / 0.8) ** 2))
    pion_gamma = jnp.geomspace(1.0e3, 2.0e10, 11)

    for public, owner in (
        (physics.photomeson_interaction_rate, photomeson_interaction_rate_jax),
        (physics.photomeson_energy_loss_rate, photomeson_energy_loss_rate_jax),
        (
            physics.bethe_heitler_proton_energy_loss_rate,
            bethe_heitler_proton_energy_loss_rate_jax,
        ),
    ):
        actual = public(proton_gamma, target_energy, target_density)
        expected = owner(proton_gamma, target_energy, target_density)
        np.testing.assert_array_equal(actual, expected)
        assert np.any(np.asarray(actual) > 0.0)

    _assert_exact_tree(
        physics.photomeson_pion_injection(
            pion_gamma,
            proton_gamma,
            proton_density,
            target_energy,
            target_density,
        ),
        physics.PionInjection(
            *photomeson_pion_injection_jax(
                pion_gamma,
                proton_gamma,
                proton_density,
                target_energy,
                target_density,
            )
        ),
    )

    pp_proton_gamma = jnp.geomspace(1.000001, 1.0e7, 9)
    pp_pion_gamma = jnp.geomspace(1.000001, 1.0e7, 11)
    pp_density = 4.0e2 * pp_proton_gamma**-1.8
    _assert_exact_tree(
        physics.proton_proton_pion_injection_and_loss(
            pp_pion_gamma, pp_proton_gamma, pp_density, 30.0
        ),
        physics.ProtonProtonResult(
            *proton_proton_pion_injection_and_loss_jax(
                pp_pion_gamma, pp_proton_gamma, pp_density, 30.0
            )
        ),
    )

    decay_pion_gamma = jnp.geomspace(1.0e2, 1.0e6, 13)
    positive_decay = jnp.exp(-((jnp.log(decay_pion_gamma / 2.0e4) / 0.65) ** 2))
    negative_decay = 0.7 * jnp.exp(-((jnp.log(decay_pion_gamma / 7.0e4) / 0.8) ** 2))
    photon_energy = jnp.geomspace(1.0e-10, 3.0e2, 15)
    muon_gamma = jnp.geomspace(1.0e2, 1.0e6, 14)
    neutrino_energy = jnp.geomspace(1.0e-10, 3.0e2, 15)

    np.testing.assert_array_equal(
        physics.neutral_pion_decay_photon_injection(
            photon_energy, decay_pion_gamma, positive_decay
        ),
        neutral_pion_decay_photon_injection_jax(
            photon_energy, decay_pion_gamma, positive_decay
        ),
    )
    public_muons = physics.charged_pion_decay_muon_injection(
        muon_gamma, decay_pion_gamma, positive_decay, negative_decay
    )
    _assert_exact_tree(
        public_muons,
        physics.ChargedPionMuonInjection(
            *charged_pion_decay_muon_injection_jax(
                muon_gamma, decay_pion_gamma, positive_decay, negative_decay
            )
        ),
    )
    _assert_exact_tree(
        physics.charged_pion_decay_neutrino_injection(
            neutrino_energy, decay_pion_gamma, positive_decay, negative_decay
        ),
        physics.PionDecayNeutrinoInjection(
            *charged_pion_decay_neutrino_injection_jax(
                neutrino_energy,
                decay_pion_gamma,
                positive_decay,
                negative_decay,
            )
        ),
    )

    electron_gamma = jnp.geomspace(1.01, 2.0e8, 15)
    muon_decay_rates = tuple(public_muons)
    _assert_exact_tree(
        physics.muon_decay_electron_positron_injection(
            electron_gamma, muon_gamma, *muon_decay_rates
        ),
        physics.MuonDecayElectronPositronInjection(
            *muon_decay_electron_positron_injection_jax(
                electron_gamma, muon_gamma, *muon_decay_rates
            )
        ),
    )
    _assert_exact_tree(
        physics.muon_decay_neutrino_injection(
            neutrino_energy, muon_gamma, *muon_decay_rates
        ),
        physics.MuonDecayNeutrinoInjection(
            *muon_decay_neutrino_injection_jax(
                neutrino_energy, muon_gamma, *muon_decay_rates
            )
        ),
    )


def test_transport_facade_matches_owner_and_is_jit_ad_transparent():
    gamma = jnp.geomspace(1.01, 1.0e5, 17)
    injection = 2.0e-8 * gamma**-1.7
    loss = 4.0e-6 * gamma
    sink = jnp.full_like(gamma, 3.0e-7)

    expected = steady_state_finite_volume_solution_jax(gamma, injection, loss, sink)
    actual = transport.solve_steady_state(gamma, injection, loss, sink)
    np.testing.assert_array_equal(actual, expected)

    compiled = jax.jit(transport.solve_steady_state)(gamma, injection, loss, sink)
    # JIT may reorder FP64 operations; require strict numerical, not bitwise, agreement.
    np.testing.assert_allclose(compiled, actual, rtol=2.0e-15, atol=0.0)
    value, tangent = jax.jvp(
        lambda scale: transport.solve_steady_state(
            gamma, scale * injection, loss, sink
        ),
        (jnp.float64(1.0),),
        (jnp.float64(0.1),),
    )
    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(tangent))
    assert np.any(np.asarray(tangent) != 0.0)


def test_radiation_and_hadronic_facades_are_jit_ad_transparent():
    photon_energy = jnp.geomspace(1.0e-14, 1.0e-7, 4)
    gamma = jnp.geomspace(1.01, 1.0e5, 7)
    density = 3.0e2 * gamma**-2.0

    def radiation(scale):
        return physics.electron_synchrotron(
            photon_energy,
            gamma,
            scale * density,
            magnetic_field_G=1.2,
            radius_cm=2.0e15,
        )

    # The absolute term applies only to the numerically negligible far-tail value.
    np.testing.assert_allclose(
        jax.jit(radiation)(1.0), radiation(1.0), rtol=2.0e-15, atol=1.0e-100
    )
    radiation_value, radiation_tangent = jax.jvp(
        radiation, (jnp.float64(1.0),), (jnp.float64(0.1),)
    )

    proton_gamma = jnp.geomspace(7.0e4, 2.0e7, 7)
    target_energy = jnp.geomspace(1.0e-12, 1.0e-7, 8)
    target_density = 1.0e24 * jnp.exp(-((jnp.log(target_energy / 1.0e-9) / 0.8) ** 2))

    def hadronic(scale):
        return physics.bethe_heitler_proton_energy_loss_rate(
            proton_gamma, target_energy, scale * target_density
        )

    np.testing.assert_allclose(
        jax.jit(hadronic)(1.0), hadronic(1.0), rtol=2.0e-15, atol=0.0
    )
    hadronic_value, hadronic_tangent = jax.jvp(
        hadronic, (jnp.float64(1.0),), (jnp.float64(0.1),)
    )
    for values in (
        radiation_value,
        radiation_tangent,
        hadronic_value,
        hadronic_tangent,
    ):
        assert np.all(np.isfinite(values))
        assert np.any(np.asarray(values) != 0.0)


def test_em_cascade_facade_matches_selected_owner_outputs_exactly():
    electron_gamma = jnp.geomspace(1.000001, 1.0e5, 9)
    synchrotron_energy = jnp.geomspace(1.0e-14, 1.0e-9, 4)
    gamma_energy = jnp.geomspace(1.0e-9, 1.0e-4, 4)
    proton_gamma = jnp.geomspace(1.01e2, 1.0e6, 3)
    external_energy = jnp.geomspace(1.0e-14, 1.0e-10, 3)
    electron_source = 1.0e-12 * jnp.exp(-((jnp.log(electron_gamma / 1.0e3) / 1.0) ** 2))
    positron_source = 0.2 * electron_source
    zero_electron = jnp.zeros_like(electron_gamma)
    zero_proton = jnp.zeros_like(proton_gamma)
    zero_gamma = jnp.zeros_like(gamma_energy)
    zero_external = jnp.zeros_like(external_energy)
    keywords = {
        "magnetic_field_G": 0.0,
        "radius_cm": 2.0e15,
        "adiabatic_rate_s_inv": 2.0e-7,
        "escape_rate_s_inv": 3.0e-6,
    }

    actual = public_cascade.solve_em(
        synchrotron_energy,
        gamma_energy,
        electron_gamma,
        electron_source,
        positron_source,
        proton_gamma,
        zero_proton,
        zero_gamma,
        external_energy,
        zero_external,
        **keywords,
    )
    owner = steady_state_electromagnetic_cascade_jax(
        synchrotron_energy,
        gamma_energy,
        electron_gamma,
        electron_source,
        positron_source,
        zero_electron,
        zero_electron,
        zero_electron,
        zero_electron,
        proton_gamma,
        zero_proton,
        zero_gamma,
        external_energy,
        zero_external,
        keywords["magnetic_field_G"],
        keywords["radius_cm"],
        keywords["adiabatic_rate_s_inv"],
        keywords["escape_rate_s_inv"],
    )
    expected = public_cascade.EMCascadeResult(
        electron_density_dgamma_cm3=owner["electron_density_dgamma_cm3"],
        positron_density_dgamma_cm3=owner["positron_density_dgamma_cm3"],
        intrinsic_gamma_sed_erg_s=owner["total_gamma_intrinsic_sed_erg_s"],
        escaped_gamma_sed_erg_s=owner["escaped_gamma_sed_erg_s"],
        gamma_gamma_optical_depth=owner["gamma_gamma_optical_depth"],
        converged=owner["converged"],
        iterations=owner["iteration_count"],
        residual=owner["final_cascade_residual"],
    )
    _assert_exact_tree(actual, expected)
    assert bool(actual.converged)
    assert actual.electron_density_dgamma_cm3.dtype == jnp.float64
