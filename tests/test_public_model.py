"""Focused P3 checks for the public one-zone forward boundary."""

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

from helena import ForwardResult, ModelConfig, OneZoneModel, configure
from helena.models import c513


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


def _fake_raw(configuration):
    values = np.asarray([1.0, 2.0, 3.0])
    high_values = np.ones_like(configuration["gamma_ray_energy_erg"])
    broadband_energy, _, _ = c513._canonical_broadband_photon_support(configuration)
    broadband_values = np.ones_like(broadband_energy)
    flavors = {
        "nu_e": values,
        "anti_nu_e": values,
        "nu_mu": values,
        "anti_nu_mu": values,
    }
    mixed = {**flavors, "nu_tau": values, "anti_nu_tau": values}
    mixed["all_flavor"] = 6.0 * values
    unmixed = {**flavors, "all_flavor": 4.0 * values}
    particles = {}
    for species in ("electron", "positron", "proton"):
        particles[f"{species}_gamma_total_energy_over_rest_energy"] = values
        particles[f"{species}_primary_injection_blob_dgamma_cm3_s"] = values
        particles[f"{species}_density_blob_dgamma_cm3"] = values
    photon = {
        "energy_blob_erg": configuration["gamma_ray_energy_erg"],
        "escaped_energy_luminosity_blob_erg_s": high_values,
        "energy_observer_erg": configuration["gamma_ray_energy_erg"],
        "intrinsic_energy_flux_observer_erg_cm2_s": high_values,
        "ebl_attenuated_energy_flux_earth_erg_cm2_s": high_values,
        "broadband_energy_blob_erg": broadband_energy,
        "broadband_escaped_energy_luminosity_blob_erg_s": broadband_values,
        "broadband_energy_observer_erg": broadband_energy,
        "broadband_intrinsic_energy_flux_observer_erg_cm2_s": broadband_values,
    }
    inner = {
        "iteration_count": np.int32(3),
        "electron_synchrotron_high_energy_intrinsic_sed_erg_s": high_values,
        "ssc_intrinsic_sed_erg_s": high_values,
        "external_compton_intrinsic_sed_erg_s": high_values,
        "proton_synchrotron_intrinsic_sed_erg_s": high_values,
        "pi0_intrinsic_sed_erg_s": high_values,
        "total_gamma_intrinsic_sed_erg_s": high_values,
    }
    return {
        "converged": True,
        "outer_iteration_count": 4,
        "iterations": (
            {
                "inner_electromagnetic_cascade": inner,
                "meson_synchrotron_intrinsic_seds_erg_s": {
                    "pion_total": high_values,
                    "muon_total": high_values,
                },
                "outer_residuals": {"maximum_enabled_relative_residual": 1.0e-5},
                "checks": {
                    "inner_converged": True,
                    "finite_nonnegative": True,
                    "ledgers_passed": True,
                },
            },
        ),
        "canonical_quantities": {
            "particles": particles,
            "photon": photon,
            "photon_components": {
                **{
                    name: broadband_values
                    for name in (
                        "transported_lepton_synchrotron",
                        "transported_lepton_ssc",
                    )
                },
                **{
                    name: broadband_values / 9.0
                    for name in c513._SOURCE_RESOLVED_ADDITIVE_COMPONENTS
                },
                **{
                    name: broadband_values / 12.0
                    for name in (
                        "primary_electron_inverse_compton",
                        "bethe_heitler_pair_synchrotron",
                        "bethe_heitler_pair_inverse_compton",
                        "photomeson_secondary_pair_synchrotron",
                        "photomeson_secondary_pair_inverse_compton",
                        "gamma_gamma_reprocessed_pair_synchrotron",
                        "gamma_gamma_reprocessed_pair_inverse_compton",
                    )
                },
            },
            "source_resolved_photon_component_diagnostics": {
                "source_resolved_particle_electron_maximum_relative_error": 0.0,
                "source_resolved_gamma_gamma_parent_ownership": "aggregate",
            },
            "neutrino": {
                "energy_blob_erg": values,
                "source_injection_blob_denergy_cm3_s": flavors,
                "energy_observer_erg": values,
                "unmixed_flavor_energy_flux_observer_erg_cm2_s": unmixed,
                "mixed_flavor_energy_flux_earth_erg_cm2_s": mixed,
            },
        },
    }


def test_root_imports_config_validation_and_immutability():
    assert configure and ModelConfig and OneZoneModel and ForwardResult
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.radius_cm = 2.0e16
    with pytest.raises(ValueError, match="radius_cm must be positive"):
        replace(config, radius_cm=0.0)
    with pytest.raises(ValueError, match="electron_gamma_max"):
        replace(config, electron_gamma_max=1.0)
    with pytest.raises(ValueError, match="cold_proton_density_cm3 must be positive"):
        replace(config, enable_proton_proton=True)
    with pytest.raises(TypeError, match="enable_ssc must be boolean"):
        replace(config, enable_ssc="yes")
    with pytest.raises(TypeError, match="enable_internal_gamma_gamma must be boolean"):
        replace(config, enable_internal_gamma_gamma="yes")


def test_absolute_reference_config_maps_exactly_to_canonical_configuration():
    config = _config()
    model = OneZoneModel(config)
    assert model.config is config
    with pytest.raises(AttributeError):
        model.config = replace(config, radius_cm=2.0e16)
    reference = c513.build_c513_configuration()["production_configuration"]
    actual = model._setup["production_configuration"]
    for name in (
        "primary_electron_injection_dgamma_cm3_s",
        "primary_proton_injection_dgamma_cm3_s",
        "magnetic_field_G",
        "radius_cm",
        "electron_index",
        "proton_index",
        "electron_gamma_cutoff",
        "proton_gamma_cutoff",
    ):
        np.testing.assert_array_equal(actual[name], reference[name])
    assert actual["bethe_heitler_mode"] == "dynamic"
    assert actual["enable_ssc"] is True
    assert actual["enable_internal_gamma_gamma"] is True
    assert actual["enable_pp"] is False


def test_nonreference_absolute_config_uses_validated_mappers():
    config = replace(
        _config(),
        magnetic_field_G=0.2,
        electron_injection_index=2.2,
        proton_gamma_max=2.0e7,
        enable_internal_gamma_gamma=False,
        enable_bethe_heitler=False,
        enable_pion_synchrotron_photons=False,
    )
    model = OneZoneModel(config)
    setup = c513.build_c513_configuration()
    expected = c513.c513_configuration_from_log_offsets(
        setup["production_configuration"], (0.0, 0.0, np.log(2.0), 0.0)
    )
    expected = c513.c513_configuration_from_injection_shape_offsets(
        expected, (0.2, 0.0, 0.0, np.log(2.0))
    )
    actual = model._setup["production_configuration"]
    for name in (
        "primary_electron_injection_dgamma_cm3_s",
        "primary_proton_injection_dgamma_cm3_s",
        "magnetic_field_G",
        "electron_index",
        "proton_gamma_cutoff",
    ):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert actual["bethe_heitler_mode"] == "off"
    assert actual["enable_internal_gamma_gamma"] is False
    assert actual["enable_pion_synchrotron_photons"] is False


def test_forward_wraps_canonical_result_and_reuses_responses(monkeypatch):
    calls = {"prepare": 0, "forward": 0}
    responses = object()

    def prepare(setup, *, cache_dir, allow_build):
        calls["prepare"] += 1
        assert setup["production_configuration"]
        assert cache_dir == "cache"
        assert allow_build is False
        return responses

    def forward(*, setup, prepared_responses, observer_options):
        calls["forward"] += 1
        assert setup["production_configuration"]
        assert prepared_responses is responses
        assert observer_options["ebl_transmission"].shape == (113,)
        np.testing.assert_array_equal(observer_options["ebl_transmission"], 1.0)
        return _fake_raw(setup["production_configuration"])

    monkeypatch.setattr(c513, "prepare_c513_responses", prepare)
    monkeypatch.setattr(c513, "run_c513_forward", forward)
    model = OneZoneModel(_config(), cache_dir="cache")
    first = model.forward()
    second = model.forward()

    assert isinstance(first, ForwardResult)
    assert calls == {"prepare": 1, "forward": 2}
    np.testing.assert_array_equal(
        first.sed["escaped_energy_luminosity_blob_erg_s"],
        second.sed["escaped_energy_luminosity_blob_erg_s"],
    )
    assert tuple(ForwardResult.__dataclass_fields__) == (
        "sed",
        "components",
        "particles",
        "neutrinos",
        "diagnostics",
    )
    assert isinstance(first.sed, MappingProxyType)
    assert not first.sed["energy_blob_erg"].flags.writeable
    assert not first.sed["broadband_energy_blob_erg"].flags.writeable
    assert np.all(np.diff(first.sed["broadband_energy_blob_erg"]) > 0.0)
    assert first.diagnostics["converged"] is True
    assert first.diagnostics["outer_iterations"] == 4
    assert first.diagnostics["maximum_outer_relative_residual"] == 1.0e-5
    assert first.diagnostics[
        "source_resolved_gamma_gamma_parent_ownership"
    ] == "aggregate"
    assert np.isfinite(
        first.diagnostics[
            "source_resolved_escaped_earth_maximum_relative_error"
        ]
    )
    assert (
        "primary_electron_synchrotron_broadband_ebl_attenuated_"
        "energy_flux_earth_erg_cm2_s"
    ) in first.components
    for name in (
        "primary_electron_inverse_compton",
        "bethe_heitler_pair_synchrotron",
        "bethe_heitler_pair_inverse_compton",
        "photomeson_secondary_pair_synchrotron",
        "photomeson_secondary_pair_inverse_compton",
        "gamma_gamma_reprocessed_pair_synchrotron",
        "gamma_gamma_reprocessed_pair_inverse_compton",
    ):
        field = (
            f"{name}_broadband_ebl_attenuated_"
            "energy_flux_earth_erg_cm2_s"
        )
        assert field in first.components
        assert not first.components[field].flags.writeable
    assert "all_flavor" in first.neutrinos["source_injection_blob_denergy_cm3_s"]
    assert not hasattr(first, "_raw")
    assert not any("c513" in name or "response" in name for name in first.__slots__)
