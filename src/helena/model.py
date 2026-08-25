"""Public one-zone model configuration and forward-result boundary."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log
from numbers import Real
from types import MappingProxyType

import numpy as np

__all__ = ["ForwardResult", "ModelConfig", "OneZoneModel"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Physical one-zone inputs in cgs units.

    Electron and proton ``gamma_min`` remain the validated fixed value 1.0.
    The current public model has no external photon field. ``enable_ebl=True``
    applies the validated Dominguez-2011 table to the Earth-observed photon
    output. The default is ``False`` and preserves identity propagation.

    ``enable_internal_gamma_gamma`` defaults to ``True``. Setting it to
    ``False`` disables internal gamma-gamma absorption and its pair injection
    only; the synchrotron target, photomeson interactions, and independent
    external gamma-gamma semantics remain unchanged.
    """

    magnetic_field_G: float
    radius_cm: float
    redshift: float
    doppler_factor: float
    luminosity_distance_cm: float
    electron_kinetic_injection_luminosity_erg_s: float
    electron_injection_index: float
    electron_gamma_max: float
    proton_kinetic_injection_luminosity_erg_s: float
    proton_injection_index: float
    proton_gamma_max: float
    enable_ssc: bool = True
    enable_bethe_heitler: bool = True
    enable_proton_proton: bool = False
    cold_proton_density_cm3: float = 0.0
    enable_pion_synchrotron_photons: bool = True
    enable_muon_synchrotron_photons: bool = True
    enable_internal_gamma_gamma: bool = True
    enable_ebl: bool = False

    def __post_init__(self):
        numeric_fields = (
            "magnetic_field_G",
            "radius_cm",
            "redshift",
            "doppler_factor",
            "luminosity_distance_cm",
            "electron_kinetic_injection_luminosity_erg_s",
            "electron_injection_index",
            "electron_gamma_max",
            "proton_kinetic_injection_luminosity_erg_s",
            "proton_injection_index",
            "proton_gamma_max",
            "cold_proton_density_cm3",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")

        for name in (
            "magnetic_field_G",
            "radius_cm",
            "doppler_factor",
            "luminosity_distance_cm",
            "electron_kinetic_injection_luminosity_erg_s",
            "proton_kinetic_injection_luminosity_erg_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.redshift < 0.0:
            raise ValueError("redshift must be non-negative")
        if self.electron_gamma_max <= 1.0:
            raise ValueError("electron_gamma_max must be greater than gamma_min=1")
        if self.proton_gamma_max <= 1.0:
            raise ValueError("proton_gamma_max must be greater than gamma_min=1")
        if self.cold_proton_density_cm3 < 0.0:
            raise ValueError("cold_proton_density_cm3 must be non-negative")

        for name in (
            "enable_ssc",
            "enable_internal_gamma_gamma",
            "enable_bethe_heitler",
            "enable_proton_proton",
            "enable_pion_synchrotron_photons",
            "enable_muon_synchrotron_photons",
            "enable_ebl",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.enable_proton_proton and self.cold_proton_density_cm3 == 0.0:
            raise ValueError(
                "cold_proton_density_cm3 must be positive when proton-proton "
                "interactions are enabled"
            )
        if not self.enable_proton_proton and self.cold_proton_density_cm3 != 0.0:
            raise ValueError(
                "cold_proton_density_cm3 must be zero when proton-proton "
                "interactions are disabled"
            )


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Stable scientific view of one canonical full-forward evaluation.

    ``sed`` retains the existing high-energy fields and additionally provides
    ``broadband_*`` blob, observer, and Earth fields for the canonical escaped
    low-plus-high photon SED. ``components`` includes additive escaped
    broadband process groups in the same frames. Source-resolved lepton groups
    are post-convergence diagnostics with the final transport, target, and
    transfer state frozen. All mappings and host arrays are read-only.
    """

    sed: Mapping[str, object]
    components: Mapping[str, object]
    particles: Mapping[str, object]
    neutrinos: Mapping[str, object]
    diagnostics: Mapping[str, object]


def _internal_configuration(config, setup):
    from helena.models.c513 import (
        c513_configuration_from_injection_shape_offsets,
        c513_configuration_from_log_offsets,
    )

    reference = setup["metadata"]["physical_parameters"]
    log_offsets = (
        log(
            config.electron_kinetic_injection_luminosity_erg_s
            / reference["electron_kinetic_injection_luminosity_erg_s"]
        ),
        log(
            config.proton_kinetic_injection_luminosity_erg_s
            / reference["proton_kinetic_injection_luminosity_erg_s"]
        ),
        log(config.magnetic_field_G / reference["magnetic_field_G"]),
        log(config.radius_cm / reference["radius_cm"]),
    )
    configuration = setup["production_configuration"]
    if any(log_offsets):
        configuration = c513_configuration_from_log_offsets(configuration, log_offsets)
    shape_offsets = (
        config.electron_injection_index - reference["electron_index"],
        config.proton_injection_index - reference["proton_index"],
        log(config.electron_gamma_max / reference["electron_gamma_cutoff"]),
        log(config.proton_gamma_max / reference["proton_gamma_cutoff"]),
    )
    if any(shape_offsets):
        configuration = c513_configuration_from_injection_shape_offsets(
            configuration, shape_offsets
        )
    return {
        **configuration,
        "bethe_heitler_mode": ("dynamic" if config.enable_bethe_heitler else "off"),
        "enable_ssc": config.enable_ssc,
        "enable_internal_gamma_gamma": config.enable_internal_gamma_gamma,
        "enable_pp": config.enable_proton_proton,
        "cold_proton_density_cm3": config.cold_proton_density_cm3,
        "enable_pion_synchrotron_photons": (config.enable_pion_synchrotron_photons),
        "enable_muon_synchrotron_photons": (config.enable_muon_synchrotron_photons),
    }


def _readonly_tree(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: _readonly_tree(item) for name, item in value.items()}
        )
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _observer_ebl_support(config, observer_energy_erg):
    energy = np.asarray(observer_energy_erg, dtype=np.float64)
    if not config.enable_ebl:
        return np.arange(energy.size), np.ones_like(energy)
    from helena.core.propagation import _prepare_ebl_supported_transmission

    indices, transmission = _prepare_ebl_supported_transmission(
        energy, config.redshift
    )
    positive = transmission > 0.0
    if not np.any(positive):
        raise ValueError("observer photon grid has no positive EBL support")
    return indices[positive], transmission[positive]


def _earth_broadband_photon_support(config, configuration):
    from helena.core.observer import (
        sed_luminosity_to_energy_flux,
        transform_comoving_sed,
    )
    from helena.models.c513 import _canonical_broadband_photon_support

    energy_blob, _, _ = _canonical_broadband_photon_support(configuration)
    energy_observer, unit_luminosity = transform_comoving_sed(
        energy_blob,
        np.ones_like(energy_blob),
        doppler_factor=config.doppler_factor,
        redshift=config.redshift,
    )
    flux_scale = sed_luminosity_to_energy_flux(
        unit_luminosity,
        luminosity_distance_cm=config.luminosity_distance_cm,
    )
    indices, transmission = _observer_ebl_support(config, energy_observer)
    return energy_blob, energy_observer, flux_scale, indices, transmission


def _forward_result(raw, config, configuration):
    from helena.core.propagation import apply_ebl_attenuation
    from helena.models.c513 import _SOURCE_RESOLVED_ADDITIVE_COMPONENTS

    canonical = raw["canonical_quantities"]
    final = raw["iterations"][-1]
    inner = final["inner_electromagnetic_cascade"]
    meson = final["meson_synchrotron_intrinsic_seds_erg_s"]
    particle_values = canonical["particles"]
    particles = {
        species: {
            "gamma_total_energy_over_rest_energy": particle_values[
                f"{species}_gamma_total_energy_over_rest_energy"
            ],
            "primary_injection_blob_dgamma_cm3_s": particle_values[
                f"{species}_primary_injection_blob_dgamma_cm3_s"
            ],
            "density_blob_dgamma_cm3": particle_values[
                f"{species}_density_blob_dgamma_cm3"
            ],
        }
        for species in ("electron", "positron", "proton")
    }
    source_neutrinos = dict(
        canonical["neutrino"]["source_injection_blob_denergy_cm3_s"]
    )
    source_neutrinos["all_flavor"] = sum(source_neutrinos.values())
    neutrinos = {
        **canonical["neutrino"],
        "source_injection_blob_denergy_cm3_s": source_neutrinos,
    }
    outer_residuals = final["outer_residuals"]
    diagnostics = {
        "converged": bool(raw["converged"]),
        "outer_iterations": int(raw["outer_iteration_count"]),
        "inner_converged": bool(final["checks"]["inner_converged"]),
        "inner_iterations": int(np.asarray(inner["iteration_count"])),
        "maximum_outer_relative_residual": (
            None
            if outer_residuals is None
            else float(outer_residuals["maximum_enabled_relative_residual"])
        ),
        "finite_nonnegative": bool(final["checks"]["finite_nonnegative"]),
        "conservation_checks_passed": bool(final["checks"]["ledgers_passed"]),
        **canonical.get("source_resolved_photon_component_diagnostics", {}),
    }
    _, broadband_energy_observer, flux_scale, earth_indices, transmission = (
        _earth_broadband_photon_support(config, configuration)
    )
    photon = {
        **canonical["photon"],
        "broadband_earth_source_indices": earth_indices,
        "broadband_energy_earth_erg": broadband_energy_observer[earth_indices],
        "broadband_ebl_transmission": transmission,
        "broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s": np.asarray(
            apply_ebl_attenuation(
                np.asarray(
                    canonical["photon"][
                        "broadband_intrinsic_energy_flux_observer_erg_cm2_s"
                    ]
                )[earth_indices],
                transmission,
            )
        ),
    }
    component_values = {
        "energy_blob_erg": canonical["photon"]["energy_blob_erg"],
        "electron_synchrotron_energy_luminosity_blob_erg_s": inner[
            "electron_synchrotron_high_energy_intrinsic_sed_erg_s"
        ],
        "ssc_energy_luminosity_blob_erg_s": inner["ssc_intrinsic_sed_erg_s"],
        "external_compton_energy_luminosity_blob_erg_s": inner[
            "external_compton_intrinsic_sed_erg_s"
        ],
        "proton_synchrotron_energy_luminosity_blob_erg_s": inner[
            "proton_synchrotron_intrinsic_sed_erg_s"
        ],
        "neutral_pion_decay_energy_luminosity_blob_erg_s": inner[
            "pi0_intrinsic_sed_erg_s"
        ],
        "pion_synchrotron_energy_luminosity_blob_erg_s": meson["pion_total"],
        "muon_synchrotron_energy_luminosity_blob_erg_s": meson["muon_total"],
        "total_intrinsic_energy_luminosity_blob_erg_s": inner[
            "total_gamma_intrinsic_sed_erg_s"
        ],
        "broadband_energy_blob_erg": canonical["photon"][
            "broadband_energy_blob_erg"
        ],
        "broadband_energy_observer_erg": broadband_energy_observer,
        "broadband_earth_source_indices": earth_indices,
        "broadband_energy_earth_erg": broadband_energy_observer[earth_indices],
    }
    for name, luminosity in canonical["photon_components"].items():
        luminosity = np.asarray(luminosity)
        observer_flux = luminosity * flux_scale
        component_values[
            f"{name}_broadband_escaped_energy_luminosity_blob_erg_s"
        ] = luminosity
        component_values[
            f"{name}_broadband_escaped_energy_flux_observer_erg_cm2_s"
        ] = observer_flux
        component_values[
            f"{name}_broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s"
        ] = np.asarray(
            apply_ebl_attenuation(observer_flux[earth_indices], transmission)
        )
    if all(
        f"{name}_broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s"
        in component_values
        for name in _SOURCE_RESOLVED_ADDITIVE_COMPONENTS
    ):
        earth_total = photon[
            "broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s"
        ]
        earth_sum = sum(
            component_values[
                f"{name}_broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s"
            ]
            for name in _SOURCE_RESOLVED_ADDITIVE_COMPONENTS
        )
        support = earth_total > np.max(earth_total) * 1.0e-20
        diagnostics["source_resolved_escaped_earth_maximum_relative_error"] = float(
            np.max(
                np.abs(earth_sum[support] - earth_total[support])
                / earth_total[support]
            )
        )
        diagnostics["source_resolved_escaped_earth_weighted_error"] = float(
            np.trapezoid(
                np.abs(earth_sum - earth_total),
                x=np.log(photon["broadband_energy_earth_erg"]),
            )
            / np.trapezoid(
                earth_total,
                x=np.log(photon["broadband_energy_earth_erg"]),
            )
        )
    return ForwardResult(
        sed=_readonly_tree(photon),
        components=_readonly_tree(
            component_values
        ),
        particles=_readonly_tree(particles),
        neutrinos=_readonly_tree(neutrinos),
        diagnostics=MappingProxyType(diagnostics),
    )


class OneZoneModel:
    """Thin state-free physics wrapper around the canonical full forward."""

    __slots__ = ("_cache_dir", "_config", "_prepared_responses", "_setup")

    def __init__(self, config, *, cache_dir=None):
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        from helena.models.c513 import build_c513_configuration

        setup = build_c513_configuration()
        self._config = config
        self._cache_dir = cache_dir
        self._prepared_responses = None
        self._setup = {
            **setup,
            "production_configuration": _internal_configuration(config, setup),
        }

    @property
    def config(self):
        return self._config

    def _get_prepared_responses(self):
        from helena.models.c513 import prepare_c513_responses

        if self._prepared_responses is None:
            self._prepared_responses = prepare_c513_responses(
                self._setup,
                cache_dir=self._cache_dir,
                allow_build=False,
            )
        return self._prepared_responses

    def forward(self):
        """Run the canonical source-to-Earth model and return its public view."""
        from helena.models.c513 import run_c513_forward

        prepared_responses = self._get_prepared_responses()
        configuration = self._setup["production_configuration"]
        from helena.core.observer import transform_comoving_sed

        high_observer_energy, _ = transform_comoving_sed(
            np.asarray(configuration["gamma_ray_energy_erg"]),
            np.ones_like(configuration["gamma_ray_energy_erg"]),
            doppler_factor=self.config.doppler_factor,
            redshift=self.config.redshift,
        )
        gamma_indices, gamma_transmission = _observer_ebl_support(
            self.config, high_observer_energy
        )
        raw = run_c513_forward(
            setup=self._setup,
            prepared_responses=prepared_responses,
            observer_options={
                "doppler_factor": self.config.doppler_factor,
                "redshift": self.config.redshift,
                "luminosity_distance_cm": self.config.luminosity_distance_cm,
                "ebl_transmission": gamma_transmission,
                "gamma_indices": gamma_indices,
            },
        )
        return _forward_result(raw, self.config, configuration)
