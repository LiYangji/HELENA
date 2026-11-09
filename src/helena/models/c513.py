"""Canonical blob-comoving C513 scenario and response preparation.

Importing this canonical model intentionally enables global JAX x64 before any
scientific arrays are built; backend kernels also fail fast when x64 is absent.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from helena.core.gamma_gamma import _MEC2_CGS
from helena.core.jax_backend.chains.steady_state_hadronic_cascade import (
    _target_components,
    solve_steady_state_hadronic_cascade,
)
from helena.core.jax_backend.radiation.inverse_compton import inverse_compton_sed_jax
from helena.core.jax_backend.radiation.synchrotron import electron_synchrotron_sed_jax
from helena.core.jax_backend.transport.electron_positron_transport import (
    electron_positron_steady_state_transport_jax,
)
from helena.core.observer import (
    sed_luminosity_to_energy_flux,
    transform_comoving_sed,
)
from helena.core.photomeson import _PROTON_REST_ENERGY_CGS
from helena.core.propagation import with_observer_propagation
from helena.core.synchrotron import _C_CGS
from helena.runtime_cache import (
    bethe_heitler_response_fingerprint,
    load_or_prepare_bethe_heitler_pair_response_jax,
    load_or_prepare_gamma_gamma_response_jax,
)

jax.config.update("jax_enable_x64", True)

__all__ = [
    "build_c513_configuration",
    "c513_configuration_from_escape_rate_offset",
    "c513_configuration_from_injection_shape_offsets",
    "c513_configuration_from_log_offsets",
    "prepare_c513_responses",
    "run_c513_forward",
]

SCENARIO_NAME = "c513_preflight"
REFERENCE = (
    "Cerruti et al., A Comprehensive Hadronic Code Comparison for Active "
    "Galactic Nuclei, arXiv:2411.14218v3, Table 3 and Section 4"
)
REFERENCE_URL = "https://arxiv.org/abs/2411.14218"

EV_TO_ERG = 1.602176634e-12
RADIUS_CM = 1.0e16
MAGNETIC_FIELD_G = 0.1
DOPPLER_FACTOR = 30.0
ELECTRON_CUTOFF_GAMMA = 3.0e5
PROTON_CUTOFF_GAMMA = 1.0e7
ELECTRON_KINETIC_INJECTION_LUMINOSITY_ERG_S = 3.7e40
PROTON_KINETIC_INJECTION_LUMINOSITY_ERG_S = 2.8e46
OUTER_RTOL = 1.0e-3
MAX_OUTER_ITERATIONS = 30
REQUIRED_CONSECUTIVE_PASSES = 2
INNER_RTOL = 1.0e-4
INNER_MAXIMUM_ITERATIONS = 32

_INJECTION_QUADRATURE_NODES, _INJECTION_QUADRATURE_WEIGHTS = (
    np.polynomial.legendre.leggauss(64)
)


def c513_configuration_from_log_offsets(configuration, offsets):
    """Apply dimensionless ``(log L_e,kin, log L_p,kin, log B, log R)`` offsets.

    The fixed blob-frame production grids are unchanged. Injection densities
    remain in cm^-3 s^-1 per Lorentz factor, field and radius remain in G and
    cm, and ``L_e,kin``/``L_p,kin`` retain the builder's kinetic-energy
    normalization ``(gamma - 1) m c^2``. Lorentz factor remains total particle
    energy divided by rest energy. The production escape prescription remains
    ``t_esc = R/c``.
    Adiabatic rates are unchanged because the C513 scenario sets them to exact
    zero independently of radius.
    """
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    if offsets.shape != (4,):
        raise ValueError("offsets must have shape (4,)")
    electron_scale, proton_scale, magnetic_scale, radius_scale = jnp.exp(offsets)
    inverse_volume_scale = radius_scale**-3
    inverse_radius_scale = radius_scale**-1
    dynamic = {
        **configuration,
        "primary_electron_injection_dgamma_cm3_s": (
            electron_scale
            * inverse_volume_scale
            * configuration["primary_electron_injection_dgamma_cm3_s"]
        ),
        "primary_proton_injection_dgamma_cm3_s": (
            proton_scale
            * inverse_volume_scale
            * configuration["primary_proton_injection_dgamma_cm3_s"]
        ),
        "magnetic_field_G": magnetic_scale * configuration["magnetic_field_G"],
        "radius_cm": radius_scale * configuration["radius_cm"],
        "electron_escape_rate_s_inv": (
            inverse_radius_scale * configuration["electron_escape_rate_s_inv"]
        ),
        "proton_escape_rate_s_inv": (
            inverse_radius_scale * configuration["proton_escape_rate_s_inv"]
        ),
    }
    for name, scale in (
        ("electron_injection_normalization_cm3_s", electron_scale),
        ("proton_injection_normalization_cm3_s", proton_scale),
    ):
        if name in configuration:
            dynamic[name] = scale * inverse_volume_scale * configuration[name]
    return dynamic


def _cutoff_injection_nominal_integral_jax(index, cutoff_gamma):
    index = jnp.asarray(index, dtype=jnp.float64)
    cutoff = jnp.asarray(cutoff_gamma, dtype=jnp.float64)
    upper = jnp.log(cutoff)
    log_gamma = (
        0.5
        * (jnp.asarray(_INJECTION_QUADRATURE_NODES, dtype=jnp.float64) + 1.0)
        * upper
    )
    gamma = jnp.exp(log_gamma)
    integrand = (gamma - 1.0) * gamma ** (1.0 - index) * jnp.exp(-gamma / cutoff)
    return (
        0.5
        * upper
        * jnp.sum(
            jnp.asarray(_INJECTION_QUADRATURE_WEIGHTS, dtype=jnp.float64) * integrand
        )
    )


def _rescaled_cutoff_injection(configuration, particle, index, cutoff):
    baseline_index = configuration[f"{particle}_index"]
    baseline_cutoff = configuration[f"{particle}_gamma_cutoff"]
    baseline_normalization = configuration[f"{particle}_injection_normalization_cm3_s"]
    grid_name = "electron_gamma" if particle == "electron" else "proton_gamma"
    grid = configuration[grid_name]
    normalization = baseline_normalization * (
        _cutoff_injection_nominal_integral_jax(baseline_index, baseline_cutoff)
        / _cutoff_injection_nominal_integral_jax(index, cutoff)
    )
    return (
        normalization * grid ** (-index) * jnp.exp(-grid / cutoff),
        normalization,
    )


def c513_configuration_from_injection_shape_offsets(configuration, offsets):
    """Apply ``(Delta p_e, Delta p_p, Delta log gamma_e,max, Delta log gamma_p,max)``.

    Offsets are finite dimensionless local perturbations. The C513 blob-frame
    particle grids remain fixed. Both injections retain the production smooth
    exponential cutoff and the nominal luminosity integral from gamma=1 to
    gamma_max; injection arrays remain in cm^-3 s^-1 per Lorentz factor.
    """
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    if offsets.shape != (4,):
        raise ValueError("offsets must have shape (4,)")
    electron_index = configuration["electron_index"] + offsets[0]
    proton_index = configuration["proton_index"] + offsets[1]
    electron_cutoff = configuration["electron_gamma_cutoff"] * jnp.exp(offsets[2])
    proton_cutoff = configuration["proton_gamma_cutoff"] * jnp.exp(offsets[3])
    electron_injection, electron_normalization = _rescaled_cutoff_injection(
        configuration,
        "electron",
        electron_index,
        electron_cutoff,
    )
    proton_injection, proton_normalization = _rescaled_cutoff_injection(
        configuration,
        "proton",
        proton_index,
        proton_cutoff,
    )
    return {
        **configuration,
        "electron_index": electron_index,
        "proton_index": proton_index,
        "electron_gamma_cutoff": electron_cutoff,
        "proton_gamma_cutoff": proton_cutoff,
        "electron_injection_normalization_cm3_s": electron_normalization,
        "proton_injection_normalization_cm3_s": proton_normalization,
        "primary_electron_injection_dgamma_cm3_s": electron_injection,
        "primary_proton_injection_dgamma_cm3_s": proton_injection,
    }


def c513_configuration_from_escape_rate_offset(configuration, offsets):
    """Apply one shared dimensionless ``Delta log f_esc`` offset.

    The blob-comoving C513 escape sinks remain in s^-1 and follow
    ``lambda_e,esc = lambda_p,esc = f_esc c/R`` around the production
    baseline ``f_esc = 1``. Electron and proton rates share the same factor,
    as in the existing builder; all particle and photon grids remain fixed.
    Finite local offsets are valid while the resulting float64 rates remain
    finite and positive. No injection, cooling, interaction, or frame input is
    changed.
    """
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    if offsets.shape != (1,):
        raise ValueError("offsets must have shape (1,)")
    scale = jnp.exp(offsets[0])
    return {
        **configuration,
        "electron_escape_rate_s_inv": (
            scale * configuration["electron_escape_rate_s_inv"]
        ),
        "proton_escape_rate_s_inv": (scale * configuration["proton_escape_rate_s_inv"]),
    }


def _integral(values, coordinate):
    return float(np.trapezoid(np.asarray(values), x=np.asarray(coordinate)))


def _target_grid():
    """Historical sampling nodes, with no collision or source selection role."""
    grid = np.geomspace(1.0e-9, 1.0e6, 96) * EV_TO_ERG
    points_per_decade = (grid.size - 1) / np.log10(grid[-1] / grid[0])
    for maximum in (6.1e-2, 6.1e-1, 6.1):
        intervals = round(points_per_decade * np.log10(maximum / grid[-1]))
        grid = np.concatenate(
            (grid, np.geomspace(grid[-1], maximum, intervals + 1)[1:])
        )
    if grid.size != 137 or grid[-1] != 6.1:
        raise RuntimeError(
            "C overlap target grid no longer has 137 nodes through 6.1 erg"
        )
    return grid


def _aligned_lepton_grid(photon_energy):
    # Each daughter is bounded by the sum of both incoming photon energies.
    maximum = 2.0 * photon_energy[-1] / _MEC2_CGS
    f4097 = np.geomspace(1.000001, maximum, 4097)
    return f4097[::8], f4097[::2], f4097


def _cutoff_injection(
    gamma_total_energy_over_rest_energy,
    cutoff,
    kinetic_injection_luminosity_erg_s,
    rest_energy_erg,
    volume_cm3,
):
    """Normalize ``Q(gamma)`` by kinetic energy without changing its shape."""

    grid = gamma_total_energy_over_rest_energy

    def logarithmic_integrand(log_gamma):
        gamma = np.exp(log_gamma)
        return (gamma - 1.0) / gamma * np.exp(-gamma / cutoff)

    nominal_integral = quad(
        logarithmic_integrand,
        0.0,
        np.log(cutoff),
        epsabs=0.0,
        epsrel=1.0e-12,
        limit=200,
    )[0]
    normalization = kinetic_injection_luminosity_erg_s / (
        volume_cm3 * rest_energy_erg * nominal_integral
    )
    injection = normalization * grid**-2.0 * np.exp(-grid / cutoff)
    recovered = volume_cm3 * rest_energy_erg * normalization * nominal_integral
    continuous_full = (
        volume_cm3
        * rest_energy_erg
        * normalization
        * quad(
            logarithmic_integrand,
            0.0,
            np.log(grid[-1]),
            epsabs=0.0,
            epsrel=1.0e-12,
            limit=300,
        )[0]
    )
    discrete_full = (
        volume_cm3
        * rest_energy_erg
        * _integral(
            grid * (grid - 1.0) * injection,
            np.log(grid),
        )
    )
    return injection, {
        "requested_kinetic_injection_luminosity_erg_s": (
            kinetic_injection_luminosity_erg_s
        ),
        "normalization_cm3_s": normalization,
        "recovered_nominal_kinetic_injection_luminosity_erg_s": recovered,
        "continuous_full_grid_kinetic_injection_luminosity_erg_s": continuous_full,
        "discrete_full_grid_kinetic_injection_luminosity_erg_s": discrete_full,
        "tail_above_cutoff_kinetic_injection_luminosity_erg_s": (
            continuous_full - recovered
        ),
        "finite_grid_kinetic_injection_quadrature_difference_erg_s": (
            discrete_full - continuous_full
        ),
        "normalization_upper_gamma": cutoff,
        "evaluated_grid_upper_gamma": float(grid[-1]),
    }


def _grid_metadata(grid, source):
    grid = np.asarray(grid)
    return {
        "nodes": int(grid.size),
        "minimum": float(grid[0]),
        "maximum": float(grid[-1]),
        "dtype": str(grid.dtype),
        "source": source,
    }


def build_c513_configuration():
    """Build blob-comoving C513 quantities on a shared photon grid.

    Particle ``gamma`` is total energy divided by rest energy. Particle
    injections and densities are per unit gamma; photon and neutrino fields
    and sources are per unit energy. Electron/proton injection luminosities
    are unchanged kinetic-energy integrals weighted by ``(gamma - 1) m c^2``.
    The fixed external photon state is already blob-comoving; this builder
    performs no host/AGN-to-blob transformation.
    """
    # Historical sampling nodes are retained, but no node selects a source or
    # collision side. All radiation processes use this one complete grid.
    photon_energy = np.unique(
        np.concatenate((_target_grid(), np.geomspace(1.0e2, 1.0e19, 112) * EV_TO_ERG))
    )
    synchrotron_energy = gamma_ray_energy = photon_energy
    electron_gamma, c2049, f4097 = _aligned_lepton_grid(photon_energy)
    # Resolve the near-rest pair-source shape without changing particle transport.
    pair_daughter_gamma = np.unique(
        np.r_[
            electron_gamma,
            1.0
            + np.geomspace(electron_gamma[0] - 1.0, electron_gamma[1] - 1.0, 26)[1:-1],
        ]
    )

    # Match the validated 40 intervals/decade proton resolution while extending
    # the existing 1e5..1e11 range down to the benchmark gamma_min=1 boundary.
    proton_gamma = np.geomspace(1.000001, 1.0e11, 441)
    pion_gamma = np.geomspace(1.0e3, 3.0e11, 2400)
    muon_gamma = np.geomspace(1.0e2, 4.0e11, 2700)
    neutrino_energy = np.geomspace(1.0e-11, 7.0e7, 2400)
    external_energy = np.geomspace(1.0e-4, 1.0e2, 64) * EV_TO_ERG
    volume = 4.0 * np.pi * RADIUS_CM**3 / 3.0

    electron_injection, electron_audit = _cutoff_injection(
        electron_gamma,
        ELECTRON_CUTOFF_GAMMA,
        ELECTRON_KINETIC_INJECTION_LUMINOSITY_ERG_S,
        _MEC2_CGS,
        volume,
    )
    proton_injection, proton_audit = _cutoff_injection(
        proton_gamma,
        PROTON_CUTOFF_GAMMA,
        PROTON_KINETIC_INJECTION_LUMINOSITY_ERG_S,
        _PROTON_REST_ENERGY_CGS,
        volume,
    )
    escape_rate = _C_CGS / RADIUS_CM
    initial_internal = {
        "energy_erg": jnp.asarray(photon_energy),
        "density_cm3_erg": jnp.zeros(photon_energy.size, dtype=jnp.float64),
    }
    fixed_external = {
        "energy_erg": jnp.asarray(external_energy),
        "density_cm3_erg": jnp.zeros(external_energy.size, dtype=jnp.float64),
    }
    production = {
        "proton_gamma": jnp.asarray(proton_gamma),
        "primary_proton_injection_dgamma_cm3_s": jnp.asarray(proton_injection),
        "pion_gamma": jnp.asarray(pion_gamma),
        "muon_gamma": jnp.asarray(muon_gamma),
        "synchrotron_photon_energy_erg": jnp.asarray(synchrotron_energy),
        "gamma_ray_energy_erg": jnp.asarray(gamma_ray_energy),
        "electron_gamma": jnp.asarray(electron_gamma),
        "neutrino_energy_erg": jnp.asarray(neutrino_energy),
        "primary_electron_injection_dgamma_cm3_s": jnp.asarray(electron_injection),
        "primary_positron_injection_dgamma_cm3_s": jnp.zeros(
            electron_gamma.size, dtype=jnp.float64
        ),
        "electron_index": jnp.float64(2.0),
        "proton_index": jnp.float64(2.0),
        "electron_gamma_cutoff": jnp.float64(ELECTRON_CUTOFF_GAMMA),
        "proton_gamma_cutoff": jnp.float64(PROTON_CUTOFF_GAMMA),
        "electron_injection_normalization_cm3_s": jnp.float64(
            electron_audit["normalization_cm3_s"]
        ),
        "proton_injection_normalization_cm3_s": jnp.float64(
            proton_audit["normalization_cm3_s"]
        ),
        "magnetic_field_G": jnp.float64(MAGNETIC_FIELD_G),
        "radius_cm": jnp.float64(RADIUS_CM),
        "proton_adiabatic_rate_s_inv": jnp.float64(0.0),
        "proton_escape_rate_s_inv": jnp.float64(escape_rate),
        "fractional_bethe_heitler_energy_loss_rate_s_inv": jnp.zeros(
            proton_gamma.size, dtype=jnp.float64
        ),
        "electron_adiabatic_rate_s_inv": jnp.float64(0.0),
        "electron_escape_rate_s_inv": jnp.float64(escape_rate),
        "additional_fixed_inverse_compton_loss_rate_dgamma_s": jnp.float64(0.0),
        "additional_continuous_loss_rate_dgamma_s": jnp.float64(0.0),
        "include_external_gamma_gamma_target": False,
        "pair_daughter_gamma": jnp.asarray(pair_daughter_gamma, dtype=jnp.float64),
        "inner_relaxation_omega": 1.0,
        "inner_relative_tolerance": INNER_RTOL,
        "inner_maximum_iterations": INNER_MAXIMUM_ITERATIONS,
        "bethe_heitler_mode": "dynamic",
        "enable_ssc": True,
        "enable_internal_gamma_gamma": True,
        "enable_ec": False,
        "enable_pion_synchrotron_photons": True,
        "enable_muon_synchrotron_photons": True,
    }
    metadata = {
        "scenario_name": SCENARIO_NAME,
        "reference": REFERENCE,
        "reference_url": REFERENCE_URL,
        "physical_parameters": {
            "radius_cm": RADIUS_CM,
            "magnetic_field_G": MAGNETIC_FIELD_G,
            "doppler_factor": DOPPLER_FACTOR,
            "external_photon_field_blob": "none",
            "electron_gamma_min": 1.0,
            "electron_gamma_cutoff": ELECTRON_CUTOFF_GAMMA,
            "electron_index": 2.0,
            "electron_kinetic_injection_luminosity_erg_s": (
                ELECTRON_KINETIC_INJECTION_LUMINOSITY_ERG_S
            ),
            "electron_escape_time_s": RADIUS_CM / _C_CGS,
            "proton_gamma_min": 1.0,
            "proton_gamma_cutoff": PROTON_CUTOFF_GAMMA,
            "proton_index": 2.0,
            "proton_kinetic_injection_luminosity_erg_s": (
                PROTON_KINETIC_INJECTION_LUMINOSITY_ERG_S
            ),
            "proton_escape_time_s": RADIUS_CM / _C_CGS,
            "primary_positron_injection": "exact_zero",
            "electron_adiabatic_loss": "exact_zero",
            "proton_adiabatic_loss": "exact_zero",
        },
        "quantity_contracts": {
            "particle_gamma": "total particle energy divided by rest energy",
            "particle_injection": "Q(gamma), cm^-3 s^-1 per unit gamma",
            "particle_density": "N(gamma), cm^-3 per unit gamma",
            "injection_luminosity": (
                "kinetic: V m c^2 integral[(gamma - 1) Q(gamma) dgamma]"
            ),
            "photon_neutrino_measure": "per energy, never per gamma",
            "fixed_external_photon_frame": (
                "blob-comoving; host/AGN fields require an explicit upstream transform"
            ),
        },
        "injection_audits": {
            "electron": electron_audit,
            "proton": proton_audit,
        },
        "grids": {
            "electron_lepton_transport": _grid_metadata(
                electron_gamma, "C513=F4097[::8]=C2049[::4]"
            ),
            "pair_daughter": _grid_metadata(
                pair_daughter_gamma,
                "C513 plus 24 log(gamma-1) nodes in first cell; number-conserving remap to transport",
            ),
            "proton": _grid_metadata(
                proton_gamma,
                "photomeson_decay_chain 1e11 upper range at 40 intervals/decade",
            ),
            "pion": _grid_metadata(pion_gamma, "examples/photomeson_decay_chain.py"),
            "muon": _grid_metadata(muon_gamma, "examples/photomeson_decay_chain.py"),
            "neutrino": _grid_metadata(
                neutrino_energy, "examples/photomeson_decay_chain.py"
            ),
            "internal_photon": _grid_metadata(
                photon_energy,
                "complete union of historical sampling nodes; no physical split",
            ),
            "external_zero_placeholder": _grid_metadata(
                external_energy, "historical 64-node external grid, density zero"
            ),
        },
        "grid_checks": {
            "c513_exactly_513": electron_gamma.size == 513,
            "c513_equals_c2049_every_four": np.array_equal(electron_gamma, c2049[::4]),
            "c513_equals_f4097_every_eight": np.array_equal(electron_gamma, f4097[::8]),
            "pair_daughter_exactly_537": pair_daughter_gamma.size == 537,
            "pair_daughter_contains_transport": np.all(
                np.isin(electron_gamma, pair_daughter_gamma)
            ),
            "pair_daughter_same_endpoints": np.array_equal(
                pair_daughter_gamma[[0, -1]], electron_gamma[[0, -1]]
            ),
            "all_float64": all(
                np.asarray(value).dtype == np.float64
                for value in (
                    electron_gamma,
                    pair_daughter_gamma,
                    proton_gamma,
                    pion_gamma,
                    muon_gamma,
                    neutrino_energy,
                    synchrotron_energy,
                    gamma_ray_energy,
                )
            ),
        },
        "numerical_strategy": {
            "outer_method": "direct_picard",
            "outer_rtol": OUTER_RTOL,
            "max_outer_iterations": MAX_OUTER_ITERATIONS,
            "required_consecutive_passes": REQUIRED_CONSECUTIVE_PASSES,
            "inner_relaxation_omega": 1.0,
            "inner_relative_tolerance": INNER_RTOL,
            "inner_maximum_iterations": INNER_MAXIMUM_ITERATIONS,
            "parameter_scan": False,
            "outer_mixing": False,
        },
        "observer_output": {
            "doppler_factor": DOPPLER_FACTOR,
            "redshift": "not supplied; no cosmological shift applied",
            "luminosity_distance": "not supplied; no Earth flux produced",
        },
    }
    return {
        "initial_internal_photon_state_blob": initial_internal,
        "fixed_external_photon_state_blob": fixed_external,
        "production_configuration": production,
        "metadata": metadata,
    }


def _validate_common_photon_grid(setup):
    """Reject stale source/response axes at the concrete setup boundary."""
    configuration = setup["production_configuration"]
    grids = (
        setup["initial_internal_photon_state_blob"]["energy_erg"],
        configuration["synchrotron_photon_energy_erg"],
        configuration["gamma_ray_energy_erg"],
    )
    if any(
        isinstance(value, jax.core.Tracer) for value in jax.tree_util.tree_leaves(grids)
    ):
        raise ValueError("photon grids must be concrete; prepare the setup before JIT")
    initial, synchrotron, gamma = (np.asarray(grid) for grid in grids)
    if not np.array_equal(initial, gamma) or not np.array_equal(synchrotron, gamma):
        raise ValueError(
            "initial and configured photon grids must match exactly; rebuild the "
            "setup when changing photon-grid values"
        )


def prepare_c513_responses(setup, *, cache_dir, allow_build=False):
    """Return fixed C513 BH and gamma-gamma responses for the internal solver.

    ``setup`` is returned by :func:`build_c513_configuration`.
    ``cache_dir`` selects the strict persistent BH and gamma-gamma cache;
    ``allow_build=False`` requires exact cache hits. Response units, grids,
    compatibility checks, and physical assumptions are inherited unchanged
    from the existing response builders.
    """
    _validate_common_photon_grid(setup)
    initial = setup["initial_internal_photon_state_blob"]
    external = setup["fixed_external_photon_state_blob"]
    configuration = setup["production_configuration"]
    components = _target_components(initial, external)
    responses = {
        name: load_or_prepare_bethe_heitler_pair_response_jax(
            configuration["electron_gamma"],
            configuration["proton_gamma"],
            component["energy_erg"],
            cache_dir=cache_dir,
            allow_build=allow_build,
        )
        for name, component in components.items()
    }
    fingerprints = {
        name: bethe_heitler_response_fingerprint(
            configuration["electron_gamma"],
            configuration["proton_gamma"],
            component["energy_erg"],
        )
        for name, component in components.items()
    }
    pair_gamma = configuration.get("pair_daughter_gamma")
    if pair_gamma is None:
        pair_gamma = configuration["electron_gamma"]
    gamma_gamma_response = load_or_prepare_gamma_gamma_response_jax(
        configuration["gamma_ray_energy_erg"],
        pair_gamma,
        cache_dir=cache_dir,
        allow_build=allow_build,
    )
    if configuration.get("include_external_gamma_gamma_target", False):
        external_response = load_or_prepare_gamma_gamma_response_jax(
            configuration["gamma_ray_energy_erg"],
            pair_gamma,
            external["energy_erg"],
            cache_dir=cache_dir,
            allow_build=allow_build,
        )
        gamma_gamma_response["external_target_photon_energy_erg"] = external_response[
            "target_photon_energy_erg"
        ]
        gamma_gamma_response["external_pair_response"] = external_response[
            "pair_response"
        ]
    return {
        "bethe_heitler_pair_responses": responses,
        "bethe_heitler_response_fingerprints": fingerprints,
        "gamma_gamma_pair_response": gamma_gamma_response,
    }


def _canonical_broadband_photon_support(configuration):
    """Return the common blob-comoving photon axis without source selection."""
    return np.asarray(configuration["gamma_ray_energy_erg"])


def _canonical_broadband_escaped_photon_sed(configuration, inner):
    return inner["photon_energy_erg"], inner["escaped_photon_sed_erg_s"]


def _canonical_broadband_escaped_photon_components(configuration, final):
    """Read accepted escaped components; never reconstruct their transfer."""
    inner = final["inner_electromagnetic_cascade"]
    names = {
        "transported_lepton_synchrotron": "electron_synchrotron",
        "transported_lepton_internal_inverse_compton": "internal_ic",
        "proton_synchrotron": "proton_synchrotron",
        "neutral_pion_decay": "pi0",
        "charged_pion_synchrotron": "pion_synchrotron",
        "muon_synchrotron": "muon_synchrotron",
        "external_compton": "external_compton",
    }
    accepted = inner["photon_components_escaped_sed_erg_s"]
    return inner["photon_energy_erg"], {
        public: accepted[physical] for public, physical in names.items()
    }


_SOURCE_RESOLVED_ADDITIVE_COMPONENTS = (
    "primary_electron_synchrotron",
    "primary_electron_inverse_compton",
    "bethe_heitler_secondary_radiation",
    "photomeson_secondary_radiation",
    "gamma_gamma_reprocessed_radiation",
    "proton_synchrotron",
    "neutral_pion_decay",
    "charged_pion_synchrotron",
    "muon_synchrotron",
)


def _closure_errors(actual, expected, coordinate):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    coordinate = np.asarray(coordinate, dtype=np.float64)
    scale = float(np.max(np.abs(expected)))
    if scale == 0.0:
        return (0.0, 0.0) if np.array_equal(actual, expected) else (np.inf, np.inf)
    support = np.abs(expected) > scale * 1.0e-20
    maximum = float(
        np.max(np.abs(actual[support] - expected[support]) / np.abs(expected[support]))
    )
    denominator = float(np.trapezoid(np.abs(expected), x=coordinate))
    weighted = float(
        np.trapezoid(np.abs(actual - expected), x=coordinate) / denominator
    )
    return maximum, weighted


def _source_resolved_broadband_components(configuration, final):
    """Decompose one converged solution with its final linear lepton operator.

    The production cascade is not iterated here. Final cooling, escape, target
    photon density, SSA opacity, and gamma-gamma transfer are frozen; the same
    transport and radiation owners are then evaluated once per mutually
    exclusive injection class. Gamma-gamma pairs retain aggregate reprocessed
    ownership because their mixed parent-photon ancestry is not separable.
    Internal IC uses the entire accepted internal photon density, and EC uses
    the separate fixed external reservoir once. ``primary_electron_ssc`` is a
    synchrotron-seeded diagnostic, not an additional additive component.
    """
    inner = final["inner_electromagnetic_cascade"]
    electron_gamma = jnp.asarray(configuration["electron_gamma"], dtype=jnp.float64)
    photomeson = final["photomeson_chain"]
    sources = {
        "primary": (
            configuration["primary_electron_injection_dgamma_cm3_s"],
            configuration["primary_positron_injection_dgamma_cm3_s"],
        ),
        "bethe_heitler": (
            final["q_bh_electron_dgamma_cm3_s"],
            final["q_bh_positron_dgamma_cm3_s"],
        ),
        "photomeson": (photomeson["electron"], photomeson["positron"]),
        "gamma_gamma": (
            inner["electron_pair_injection_dgamma_cm3_s"],
            inner["positron_pair_injection_dgamma_cm3_s"],
        ),
    }
    total_loss = jnp.asarray(
        inner["total_inverse_compton_loss_rate_dgamma_s"], dtype=jnp.float64
    ) + jnp.asarray(
        configuration.get("additional_continuous_loss_rate_dgamma_s", 0.0),
        dtype=jnp.float64,
    )
    populations = {
        name: electron_positron_steady_state_transport_jax(
            electron_gamma,
            electron_source,
            positron_source,
            configuration["magnetic_field_G"],
            configuration["electron_adiabatic_rate_s_inv"],
            configuration["electron_escape_rate_s_inv"],
            additional_continuous_loss_rate_dgamma_s=total_loss,
        )[:2]
        for name, (electron_source, positron_source) in sources.items()
    }

    photon_energy = inner["photon_energy_erg"]
    broadband_energy = photon_energy
    target_density = inner["internal_ic_seed_photon_density_cm3_erg"]
    internal_ic_enabled = bool(np.asarray(inner["enable_internal_ic"]))
    synchrotron = {}
    inverse_compton = {}
    internal_ic = {}
    ec_enabled = bool(np.asarray(inner["enable_ec"]))
    for name, (electron_density, positron_density) in populations.items():
        lepton_density = electron_density + positron_density
        synchrotron[name] = electron_synchrotron_sed_jax(
            photon_energy,
            electron_gamma,
            lepton_density,
            magnetic_field_G=configuration["magnetic_field_G"],
            radius_cm=configuration["radius_cm"],
        )
        internal_ic[name] = (
            inverse_compton_sed_jax(
                photon_energy,
                electron_gamma,
                lepton_density,
                photon_energy,
                target_density,
                radius_cm=configuration["radius_cm"],
            )
            if internal_ic_enabled
            else jnp.zeros_like(photon_energy)
        )
        external_compton = (
            inverse_compton_sed_jax(
                photon_energy,
                electron_gamma,
                lepton_density,
                inner["external_photon_energy_erg"],
                inner["external_photon_density_cm3_erg"],
                radius_cm=configuration["radius_cm"],
            )
            if ec_enabled
            else jnp.zeros_like(photon_energy)
        )
        inverse_compton[name] = internal_ic[name] + external_compton

    def escaped(intrinsic):
        return intrinsic * inner["photon_transfer_factor"]

    primary_ssc = (
        inverse_compton_sed_jax(
            photon_energy,
            electron_gamma,
            sum(populations["primary"]),
            photon_energy,
            inner["internal_synchrotron_photon_density_cm3_erg"],
            radius_cm=configuration["radius_cm"],
        )
        if internal_ic_enabled
        else jnp.zeros_like(photon_energy)
    )
    _, aggregate = _canonical_broadband_escaped_photon_components(configuration, final)
    mechanism_components = {
        "primary_electron_synchrotron": escaped(synchrotron["primary"]),
        "primary_electron_inverse_compton": escaped(inverse_compton["primary"]),
        "bethe_heitler_pair_synchrotron": escaped(synchrotron["bethe_heitler"]),
        "bethe_heitler_pair_inverse_compton": escaped(inverse_compton["bethe_heitler"]),
        "photomeson_secondary_pair_synchrotron": escaped(synchrotron["photomeson"]),
        "photomeson_secondary_pair_inverse_compton": escaped(
            inverse_compton["photomeson"]
        ),
        "gamma_gamma_reprocessed_pair_synchrotron": escaped(synchrotron["gamma_gamma"]),
        "gamma_gamma_reprocessed_pair_inverse_compton": escaped(
            inverse_compton["gamma_gamma"]
        ),
    }
    components = {
        "primary_electron_synchrotron": mechanism_components[
            "primary_electron_synchrotron"
        ],
        "primary_electron_inverse_compton": mechanism_components[
            "primary_electron_inverse_compton"
        ],
        "bethe_heitler_secondary_radiation": escaped(
            synchrotron["bethe_heitler"] + inverse_compton["bethe_heitler"]
        ),
        "photomeson_secondary_radiation": escaped(
            synchrotron["photomeson"] + inverse_compton["photomeson"]
        ),
        "gamma_gamma_reprocessed_radiation": escaped(
            synchrotron["gamma_gamma"] + inverse_compton["gamma_gamma"]
        ),
        **{
            name: aggregate[name]
            for name in (
                "proton_synchrotron",
                "neutral_pion_decay",
                "charged_pion_synchrotron",
                "muon_synchrotron",
            )
        },
        **mechanism_components,
        "primary_electron_ssc": escaped(primary_ssc),
    }
    electron_total = sum(population[0] for population in populations.values())
    positron_total = sum(population[1] for population in populations.values())
    synchrotron_actual = sum(synchrotron.values())
    synchrotron_expected = inner["photon_components_intrinsic_sed_erg_s"][
        "electron_synchrotron"
    ]
    inverse_compton_actual = sum(inverse_compton.values())
    inverse_compton_expected = (
        inner["photon_components_intrinsic_sed_erg_s"]["internal_ic"]
        + inner["photon_components_intrinsic_sed_erg_s"]["external_compton"]
    )
    _, total_escaped = _canonical_broadband_escaped_photon_sed(configuration, inner)
    metrics = {}
    for prefix, actual, expected, coordinate in (
        (
            "particle_electron",
            electron_total,
            inner["electron_density_dgamma_cm3"],
            electron_gamma,
        ),
        (
            "particle_positron",
            positron_total,
            inner["positron_density_dgamma_cm3"],
            electron_gamma,
        ),
        (
            "synchrotron",
            synchrotron_actual,
            synchrotron_expected,
            np.log(broadband_energy),
        ),
        (
            "inverse_compton",
            inverse_compton_actual,
            inverse_compton_expected,
            np.log(broadband_energy),
        ),
        (
            "escaped_blob",
            sum(components[name] for name in _SOURCE_RESOLVED_ADDITIVE_COMPONENTS),
            total_escaped,
            np.log(broadband_energy),
        ),
    ):
        maximum, weighted = _closure_errors(actual, expected, coordinate)
        metrics[f"source_resolved_{prefix}_maximum_relative_error"] = maximum
        metrics[f"source_resolved_{prefix}_weighted_error"] = weighted
    metrics["source_resolved_gamma_gamma_parent_ownership"] = "aggregate"
    return components, metrics


def run_c513_forward(
    *,
    observer_options,
    setup=None,
    production_configuration=None,
    prepared_responses=None,
    cache_dir=None,
    allow_response_build=False,
    solver_options=None,
):
    """Run the canonical internal C513 source-to-Earth full forward.

    setup defaults to build_c513_configuration.
    production_configuration may replace its configuration while retaining the
    scenario states and metadata. Fixed responses may be supplied from
    prepare_c513_responses; otherwise cache_dir is required and they
    are prepared once. solver_options overrides the setup numerical strategy,
    and observer_options is passed unchanged to with_observer_propagation.

    Input units and the blob-comoving frame are inherited from the existing
    builder, solver, and propagation functions. ``canonical_quantities`` is
    the supported quantity-aware handoff: every key names its frame, measure,
    and ``E L_E`` luminosity versus ``E F_E`` flux semantics. Its broadband
    photon fields read the common accepted internal photon population and its
    escaped components, without split masks or an independent transfer call. The returned
    mapping is the complete solver result with its comoving particles,
    components, ledgers, diagnostics, and convergence state, plus the existing
    observer gamma/EBL and source/Earth neutrino outputs. No new radiation or
    transfer physics is evaluated in this wrapper.
    """
    setup = build_c513_configuration() if setup is None else setup
    if production_configuration is not None:
        setup = {**setup, "production_configuration": production_configuration}
    _validate_common_photon_grid(setup)
    configuration = setup["production_configuration"]
    if prepared_responses is None:
        if cache_dir is None:
            raise ValueError(
                "cache_dir is required when prepared_responses is not supplied"
            )
        prepared_responses = prepare_c513_responses(
            setup,
            cache_dir=cache_dir,
            allow_build=allow_response_build,
        )

    strategy = setup["metadata"]["numerical_strategy"]
    options = {
        "max_outer_iterations": strategy["max_outer_iterations"],
        "outer_rtol": strategy["outer_rtol"],
        "required_consecutive_passes": strategy["required_consecutive_passes"],
    }
    options.update(solver_options or {})
    result = solve_steady_state_hadronic_cascade(
        setup["initial_internal_photon_state_blob"],
        setup["fixed_external_photon_state_blob"],
        configuration,
        bethe_heitler_pair_responses=prepared_responses["bethe_heitler_pair_responses"],
        gamma_gamma_pair_response=prepared_responses["gamma_gamma_pair_response"],
        **options,
    )
    propagated = with_observer_propagation(result, configuration, **observer_options)
    final = propagated["iterations"][-1]
    inner = final["inner_electromagnetic_cascade"]
    observer = propagated["observer_propagation"]
    broadband_energy, broadband_escaped = _canonical_broadband_escaped_photon_sed(
        configuration, inner
    )
    broadband_escaped = np.asarray(broadband_escaped)
    broadband_observer_energy, broadband_observer_luminosity = transform_comoving_sed(
        broadband_energy,
        broadband_escaped,
        doppler_factor=observer["doppler_factor"],
        redshift=observer["redshift"],
    )
    broadband_observer_flux = sed_luminosity_to_energy_flux(
        broadband_observer_luminosity,
        luminosity_distance_cm=observer["luminosity_distance_cm"],
    )
    _, broadband_components = _canonical_broadband_escaped_photon_components(
        configuration, final
    )
    source_resolved_components = {}
    source_resolved_diagnostics = {}
    if (
        not bool(np.asarray(configuration.get("enable_pp", False)))
        and "electron_pair_injection_dgamma_cm3_s" in inner
        and "photomeson_chain" in final
    ):
        source_resolved_components, source_resolved_diagnostics = (
            _source_resolved_broadband_components(configuration, final)
        )
        broadband_components = {
            **broadband_components,
            **source_resolved_components,
        }
    propagated["canonical_quantities"] = {
        "particles": {
            "electron_gamma_total_energy_over_rest_energy": configuration[
                "electron_gamma"
            ],
            "electron_primary_injection_blob_dgamma_cm3_s": configuration[
                "primary_electron_injection_dgamma_cm3_s"
            ],
            "electron_density_blob_dgamma_cm3": inner["electron_density_dgamma_cm3"],
            "positron_gamma_total_energy_over_rest_energy": configuration[
                "electron_gamma"
            ],
            "positron_primary_injection_blob_dgamma_cm3_s": configuration[
                "primary_positron_injection_dgamma_cm3_s"
            ],
            "positron_density_blob_dgamma_cm3": inner["positron_density_dgamma_cm3"],
            "proton_gamma_total_energy_over_rest_energy": configuration["proton_gamma"],
            "proton_primary_injection_blob_dgamma_cm3_s": configuration[
                "primary_proton_injection_dgamma_cm3_s"
            ],
            "proton_density_blob_dgamma_cm3": final["proton_density_dgamma_cm3"],
        },
        "photon": {
            "energy_blob_erg": configuration["gamma_ray_energy_erg"],
            "escaped_energy_luminosity_blob_erg_s": inner["escaped_gamma_sed_erg_s"],
            "energy_observer_erg": observer["gamma"]["observer_energy_erg"],
            "intrinsic_energy_flux_observer_erg_cm2_s": observer["gamma"][
                "observer_intrinsic_energy_flux_erg_cm2_s"
            ],
            "ebl_attenuated_energy_flux_earth_erg_cm2_s": observer["gamma"][
                "earth_ebl_attenuated_energy_flux_erg_cm2_s"
            ],
            "broadband_energy_blob_erg": broadband_energy,
            "broadband_escaped_energy_luminosity_blob_erg_s": (broadband_escaped),
            "broadband_energy_observer_erg": broadband_observer_energy,
            "broadband_intrinsic_energy_flux_observer_erg_cm2_s": (
                broadband_observer_flux
            ),
        },
        "photon_components": broadband_components,
        "source_resolved_photon_component_diagnostics": (source_resolved_diagnostics),
        "neutrino": {
            "energy_blob_erg": configuration["neutrino_energy_erg"],
            "source_injection_blob_denergy_cm3_s": observer["neutrino"][
                "source_injection_blob_denergy_cm3_s"
            ],
            "energy_observer_erg": observer["neutrino"]["observer_energy_erg"],
            "unmixed_flavor_energy_flux_observer_erg_cm2_s": observer["neutrino"][
                "observer_unmixed_flavor_energy_flux_erg_cm2_s"
            ],
            "mixed_flavor_energy_flux_earth_erg_cm2_s": observer["neutrino"][
                "earth_mixed_flavor_energy_flux_erg_cm2_s"
            ],
        },
    }
    return propagated
