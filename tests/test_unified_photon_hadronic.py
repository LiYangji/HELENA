"""The full photon state replaces physical low/high target selection."""

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from helena.core.constants import ELECTRON_REST_ENERGY_CGS as MEC2

hadronic = importlib.import_module(
    "helena.core.jax_backend.chains.steady_state_hadronic_cascade"
)


def test_hadronic_targets_retain_the_whole_accepted_population():
    energy = MEC2 * np.array([0.01, 0.9, 1.8, 3.0, 100.0])
    density = np.array([2.0, 3.0, 5.0, 7.0, 11.0])
    internal = {"energy_erg": energy, "density_cm3_erg": density}
    external = {"energy_erg": energy[:2], "density_cm3_erg": density[:2]}
    targets = hadronic._target_components(internal, external)
    assert set(targets) == {"external", "internal"}
    np.testing.assert_array_equal(targets["internal"]["energy_erg"], energy)
    np.testing.assert_array_equal(targets["internal"]["density_cm3_erg"], density)
    # Differently transported legacy low/high fields cannot define a total.
    with pytest.raises(KeyError):
        hadronic._target_components(
            {"low_energy_erg": energy, "low_density_cm3_erg": density}, external
        )
    accepted = hadronic._next_internal_state(
        {"photon_energy_erg": energy, "internal_photon_density_cm3_erg": density}
    )
    np.testing.assert_array_equal(accepted["density_cm3_erg"], density)


@pytest.mark.parametrize("enabled", [False, True])
def test_meson_synchrotron_has_one_full_grid_and_no_energy_mask(monkeypatch, enabled):
    energy = MEC2 * np.array([0.01, 0.9, 1.8, 3.0, 100.0])
    gamma = np.array([1.01, 2.0, 10.0])
    calls = []

    def emit(photon_energy, particle_gamma, density, **kwargs):
        calls.append(np.asarray(photon_energy))
        return jnp.asarray(photon_energy) * jnp.sum(density)

    monkeypatch.setattr(hadronic, "synchrotron_sed_jax", emit)
    configuration = {
        "gamma_ray_energy_erg": energy,
        "pion_gamma": gamma,
        "muon_gamma": gamma,
        "magnetic_field_G": 1.0,
        "radius_cm": 3.0,
        "enable_pion_synchrotron_photons": enabled,
        "enable_muon_synchrotron_photons": enabled,
    }
    chain = {
        name: jnp.ones_like(gamma)
        for name in (
            "n_pi_plus",
            "n_pi_minus",
            "n_mu_plus_left",
            "n_mu_plus_right",
            "n_mu_minus_left",
            "n_mu_minus_right",
        )
    }
    result = hadronic._meson_synchrotron_sources(configuration, chain)
    assert len(calls) == (4 if enabled else 0)
    for grid in calls:
        np.testing.assert_array_equal(grid, energy)
    seds = result["intrinsic_seds_erg_s"]
    injections = result["photon_injections_cm3_s_erg"]
    volume = 4.0 * np.pi * configuration["radius_cm"] ** 3 / 3.0
    for name, sed in seds.items():
        np.testing.assert_allclose(sed, volume * energy**2 * injections[name])
        assert np.all(np.asarray(sed) > 0) if enabled else np.all(np.asarray(sed) == 0)
    np.testing.assert_array_equal(
        seds["meson_total"], seds["pion_total"] + seds["muon_total"]
    )
    assert not any("low" in name or "high" in name for name in result)


def test_retired_split_radiation_assemblers_are_absent():
    blob = importlib.import_module(
        "helena.core.jax_backend.chains.blob_intrinsic_radiation"
    )
    leptonic = importlib.import_module(
        "helena.core.jax_backend.chains.leptonic_radiation"
    )
    assert not hasattr(blob, "_blob_intrinsic_radiation_with_high_synchrotron_jax")
    assert not hasattr(leptonic, "_leptonic_radiation_with_high_synchrotron_jax")


def test_shared_outer_defect_matches_host_and_jit():
    import jax

    coordinate = np.array([1.0, 2.0, 4.0])
    old = np.array([0.8, 0.2, 0.01])
    new = np.array([0.9, 0.1, 0.02])
    expected = np.trapezoid(coordinate * np.abs(new - old), x=coordinate) / max(
        np.trapezoid(coordinate * new, x=coordinate),
        np.trapezoid(coordinate * old, x=coordinate),
    )
    evaluate = jax.jit(hadronic.energy_weighted_spectral_defect_jax)
    np.testing.assert_allclose(evaluate(new, old, coordinate), expected, rtol=1e-14)
    host = hadronic._energy_weighted_spectral_residual(new, old, coordinate)
    assert host["relative_residual"] == float(evaluate(new, old, coordinate))
    assert float(evaluate(np.zeros(3), np.zeros(3), coordinate)) == 0.0
    assert np.isinf(evaluate(np.array([np.nan, 0.0, 0.0]), old, coordinate))


def test_shared_em_acceptance_rejects_signed_and_nonfinite_defects():
    import jax

    tolerance = 1.0e-8
    limits = {
        **hadronic._EM_LEDGER_LIMITS,
        "photon_map_number_residual": tolerance,
        "photon_map_energy_residual": tolerance,
    }
    values = {name: jnp.asarray(0.0) for name in limits}
    evaluate = jax.jit(hadronic.electromagnetic_ledger_passed_jax)
    inner = {"ledger_diagnostics": values}
    assert bool(evaluate(inner, tolerance))
    assert hadronic._em_ledger(inner, tolerance)["passed"]
    for bad in (-2.0 * limits["pair_energy_relative_residual"], np.nan):
        inner = {
            "ledger_diagnostics": {
                **values,
                "pair_energy_relative_residual": jnp.asarray(bad),
            }
        }
        assert not bool(evaluate(inner, tolerance))
        assert not hadronic._em_ledger(inner, tolerance)["passed"]


def test_actual_pp_and_meson_photons_enter_the_common_em_state(record_property):
    """Small source-routing integration test, not a paper-state accuracy test.

    The pp grids/scales follow the existing standalone pp/decay facade test.
    BH is explicitly off to isolate pp routing without constructing its cache.
    Gamma-gamma and SSC remain enabled; zero targets isolate the pp injection.
    """
    import jax

    from helena.core.jax_backend.chains.gamma_gamma_pair_source import (
        prepare_gamma_gamma_total_field_response_jax,
    )

    photon = np.geomspace(1.0e-14, 3.0e2, 15)
    proton = np.geomspace(1.000001, 1.0e7, 9)
    pion = np.geomspace(1.000001, 1.0e7, 11)
    electron = np.geomspace(1.000001, 1.0e9, 65)
    external_energy = photon[:3]
    state = {"energy_erg": photon, "density_cm3_erg": np.zeros_like(photon)}
    external = {
        "energy_erg": external_energy,
        "density_cm3_erg": np.zeros_like(external_energy),
    }
    response = prepare_gamma_gamma_total_field_response_jax(photon, electron)
    configuration = {
        "synchrotron_photon_energy_erg": photon,
        "gamma_ray_energy_erg": photon,
        "proton_gamma": proton,
        "pion_gamma": pion,
        "muon_gamma": pion,
        "electron_gamma": electron,
        "neutrino_energy_erg": photon,
        "primary_proton_injection_dgamma_cm3_s": 1.0e-12 * proton**-1.8,
        "primary_electron_injection_dgamma_cm3_s": np.zeros_like(electron),
        "primary_positron_injection_dgamma_cm3_s": np.zeros_like(electron),
        "magnetic_field_G": 1.0,
        "radius_cm": 3.0e15,
        "proton_adiabatic_rate_s_inv": 1.0e-6,
        "proton_escape_rate_s_inv": 1.0e-5,
        "electron_adiabatic_rate_s_inv": 1.0e-6,
        "electron_escape_rate_s_inv": 1.0e-5,
        "bethe_heitler_mode": "off",
        "cold_proton_density_cm3": 30.0,
        "enable_ssc": True,
        "enable_ec": False,
        "enable_internal_gamma_gamma": True,
        "inner_relative_tolerance": 1.0e-6,
        "inner_relaxation_omega": 0.7,
        "inner_maximum_iterations": 96,
    }
    for pp, meson in [(False, False), (True, False), (True, True)]:
        current = {
            **configuration,
            "enable_pp": pp,
            "enable_pion_synchrotron_photons": meson,
            "enable_muon_synchrotron_photons": meson,
        }
        result = hadronic.outer_hadronic_picard_step(
            state,
            external,
            current,
            gamma_gamma_pair_response=response,
            physics_only=True,
        )
        jax.block_until_ready(result)
        inner = result["inner_electromagnetic_cascade"]
        sources = inner["photon_components_intrinsic_sed_erg_s"]
        densities = inner["photon_components_internal_density_cm3_erg"]
        assert bool(inner["converged"])
        volume = 4.0 * np.pi * current["radius_cm"] ** 3 / 3.0
        np.testing.assert_allclose(
            sources["pi0"],
            volume * photon**2 * result["photomeson_chain"]["pi0_decay_photons"],
            rtol=2e-15,
        )
        assert bool(np.any(np.asarray(sources["pi0"]) > 0)) == pp
        assert bool(np.any(np.asarray(densities["pi0"]) > 0)) == pp
        assert (
            bool(np.any(np.asarray(result["photomeson_chain"]["electron"]) > 0)) == pp
        )
        for name in ("pion_synchrotron", "muon_synchrotron"):
            assert bool(np.any(np.asarray(sources[name]) > 0)) == meson
        np.testing.assert_allclose(
            sum(densities.values()),
            result["next_internal_photon_state"]["density_cm3_erg"],
            rtol=3e-15,
        )
        record_property(
            f"pp_{pp}_meson_{meson}_iterations", int(inner["iteration_count"])
        )
        record_property(
            f"pp_{pp}_meson_{meson}_gg_events_cm3_s",
            float(inner["internal_gamma_gamma_budget"]["event_rate_cm3_s"]),
        )
        if pp:
            assert float(inner["internal_gamma_gamma_budget"]["event_rate_cm3_s"]) > 0.0
