"""Direct outer Picard coupling of photomeson and electromagnetic cascades."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    CHARGED_PION_MASS_CGS as _CHARGED_PION_MASS_G,
)
from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
)
from helena.core.constants import (
    ELEMENTARY_CHARGE_ESU as _E_ESU,
)
from helena.core.constants import (
    MUON_MASS_CGS as _MUON_MASS_G,
)
from helena.core.constants import (
    PROTON_REST_ENERGY_CGS as _PROTON_REST_ENERGY_CGS,
)
from helena.core.jax_backend.chains.photomeson_chain import (
    photomeson_secondary_chain_jax,
)
from helena.core.jax_backend.chains.steady_state_electromagnetic_cascade import (
    steady_state_electromagnetic_cascade_jax,
)
from helena.core.jax_backend.kernels.bethe_heitler import (
    bethe_heitler_pair_injection_jax,
    bethe_heitler_proton_energy_loss_rate_jax,
)
from helena.core.jax_backend.kernels.photomeson import (
    photomeson_energy_loss_rate_jax,
)
from helena.core.jax_backend.kernels.proton_proton import (
    proton_proton_fractional_energy_loss_rate_jax,
    proton_proton_pion_injection_and_loss_jax,
)
from helena.core.jax_backend.radiation.synchrotron import synchrotron_sed_jax
from helena.core.jax_backend.transport.proton_transport import (
    proton_steady_state_transport_jax,
)
from helena.core.tables import PION_REST_ENERGIES_CGS as _PION_REST_ENERGY_CGS
from helena.diagnostics import (
    hadronic_cascade_meson_synchrotron_power_ledgers as _meson_synchrotron_power_ledgers,
)
from helena.diagnostics import (
    hadronic_cascade_particle_ledgers as _particle_ledgers,
)

__all__ = [
    "electromagnetic_ledger_passed_jax",
    "energy_weighted_spectral_defect_jax",
    "outer_hadronic_picard_step",
    "solve_steady_state_hadronic_cascade",
]

_TARGET_NAMES = ("external", "internal")
_BETHE_HEITLER_MODES = ("off", "dynamic", "legacy_fixed_loss")
_BH_PAIR_PROVIDER_DOMAIN = "born_omega_lt_600"
_BH_PROTON_LOSS_PROVIDER = "existing_continuous_loss"
_NEUTRINO_KEYS = (
    "muon_decay_nu_e",
    "muon_decay_anti_nu_e",
    "total_nu_mu",
    "total_anti_nu_mu",
)
_EM_LEDGER_LIMITS = {
    "photon_transfer_residual": 1.0e-12,
    "internal_photon_absorption_residual": 2.0e-10,
    "pair_number_relative_residual": 5.0e-3,
    "pair_energy_relative_residual": 5.0e-3,
    "electron_pair_remap_number_residual": 1.0e-12,
    "positron_pair_remap_number_residual": 1.0e-12,
    "electron_pair_remap_energy_residual": 1.0e-3,
    "positron_pair_remap_energy_residual": 1.0e-3,
}


def _value(configuration, name, default=None):
    if name in configuration:
        return configuration[name]
    if default is not None:
        return default
    raise ValueError(f"production_configuration is missing {name!r}")


def _target(energy, density):
    energy = jnp.asarray(energy, dtype=jnp.float64)
    density = jnp.asarray(density, dtype=jnp.float64)
    if energy.ndim != 1 or energy.shape[0] < 2 or density.shape != energy.shape:
        raise ValueError("photon targets require matching one-dimensional grids")
    if not any(isinstance(value, jax.core.Tracer) for value in (energy, density)):
        energy_array = np.asarray(energy)
        density_array = np.asarray(density)
        if (
            not np.all(np.isfinite(energy_array))
            or np.any(energy_array <= 0.0)
            or np.any(np.diff(energy_array) <= 0.0)
            or not np.all(np.isfinite(density_array))
            or np.any(density_array < 0.0)
        ):
            raise ValueError("photon targets must be finite, ordered, and non-negative")
    return {"energy_erg": energy, "density_cm3_erg": density}


def _target_components(previous_internal_photon_state, fixed_external_photon_state):
    """One accepted internal population and a separate fixed external reservoir.

    Both states use comoving erg and dn/dE in cm^-3 erg^-1. Old low/high
    dictionaries cannot be converted without losing their differing transfer
    semantics and are deliberately rejected by the required-key lookup.
    """
    return {
        "external": _target(
            fixed_external_photon_state["energy_erg"],
            fixed_external_photon_state["density_cm3_erg"],
        ),
        "internal": _target(
            previous_internal_photon_state["energy_erg"],
            previous_internal_photon_state["density_cm3_erg"],
        ),
    }


def _sum_component_chains(component_chains, proton_density):
    first = component_chains["external"]
    total = {"proton_density": proton_density}
    for name in first:
        if name != "proton_density":
            total[name] = sum(
                component_chains[target][name] for target in _TARGET_NAMES
            )
    return total


def _configuration_flag(configuration, name, default=False):
    value = configuration.get(name, default)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _bethe_heitler_mode(configuration):
    has_mode = "bethe_heitler_mode" in configuration
    has_legacy_flag = "enable_bethe_heitler" in configuration
    if has_mode and has_legacy_flag:
        raise ValueError(
            "bethe_heitler_mode and enable_bethe_heitler cannot both be supplied"
        )
    if has_mode:
        mode = configuration["bethe_heitler_mode"]
        if not isinstance(mode, str) or mode not in _BETHE_HEITLER_MODES:
            raise ValueError(
                f"bethe_heitler_mode must be one of {_BETHE_HEITLER_MODES!r}"
            )
        return mode
    if has_legacy_flag:
        enabled = configuration["enable_bethe_heitler"]
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enable_bethe_heitler must be boolean when supplied")
        return "dynamic" if bool(enabled) else "off"
    raise ValueError(
        "production_configuration requires explicit bethe_heitler_mode; "
        "use 'legacy_fixed_loss' for the former omitted-key behavior"
    )


def _meson_synchrotron_sources(configuration, chain):
    photon_energy = jnp.asarray(
        _value(configuration, "gamma_ray_energy_erg"), dtype=jnp.float64
    )
    pion_gamma = _value(configuration, "pion_gamma")
    muon_gamma = _value(configuration, "muon_gamma")
    magnetic_field = _value(configuration, "magnetic_field_G")
    radius = _value(configuration, "radius_cm")
    pion_enabled = _configuration_flag(configuration, "enable_pion_synchrotron_photons")
    muon_enabled = _configuration_flag(configuration, "enable_muon_synchrotron_photons")

    def emit(photon_energy, gamma, density, mass, enabled):
        if not enabled:
            return jnp.zeros_like(photon_energy)
        return synchrotron_sed_jax(
            photon_energy,
            gamma,
            density,
            magnetic_field_G=magnetic_field,
            radius_cm=radius,
            particle_mass_g=mass,
            charge_esu=_E_ESU,
        )

    def spectra(photon_energy):
        seds = {
            "pion_plus": emit(
                photon_energy,
                pion_gamma,
                chain["n_pi_plus"],
                _CHARGED_PION_MASS_G,
                pion_enabled,
            ),
            "pion_minus": emit(
                photon_energy,
                pion_gamma,
                chain["n_pi_minus"],
                _CHARGED_PION_MASS_G,
                pion_enabled,
            ),
            "muon_plus": emit(
                photon_energy,
                muon_gamma,
                chain["n_mu_plus_left"] + chain["n_mu_plus_right"],
                _MUON_MASS_G,
                muon_enabled,
            ),
            "muon_minus": emit(
                photon_energy,
                muon_gamma,
                chain["n_mu_minus_left"] + chain["n_mu_minus_right"],
                _MUON_MASS_G,
                muon_enabled,
            ),
        }
        seds["pion_total"] = seds["pion_plus"] + seds["pion_minus"]
        seds["muon_total"] = seds["muon_plus"] + seds["muon_minus"]
        seds["meson_total"] = seds["pion_total"] + seds["muon_total"]
        return seds

    seds = spectra(photon_energy)
    volume = 4.0 * jnp.pi * jnp.asarray(radius, dtype=jnp.float64) ** 3 / 3.0

    def injections(seds, energy):
        result = {
            name: seds[name] / (volume * energy**2)
            for name in ("pion_plus", "pion_minus", "muon_plus", "muon_minus")
        }
        result["pion_total"] = result["pion_plus"] + result["pion_minus"]
        result["muon_total"] = result["muon_plus"] + result["muon_minus"]
        result["meson_total"] = result["pion_total"] + result["muon_total"]
        return result

    return {
        "pion_enabled": pion_enabled,
        "muon_enabled": muon_enabled,
        "intrinsic_seds_erg_s": seds,
        "photon_injections_cm3_s_erg": injections(seds, photon_energy),
    }


def electromagnetic_ledger_passed_jax(inner, relative_tolerance):
    """Shared finite-program acceptance gate for the accepted EM state.

    The scalar bool uses the same dimensionless residual limits as the host
    diagnostic ledger; it introduces no alternate iteration or physical mode.
    """
    values = inner["ledger_diagnostics"]
    limits = {
        **_EM_LEDGER_LIMITS,
        "photon_map_number_residual": relative_tolerance,
        "photon_map_energy_residual": relative_tolerance,
    }
    return jnp.all(
        jnp.stack(
            [
                jnp.isfinite(values[name]) & (jnp.abs(values[name]) <= limit)
                for name, limit in limits.items()
            ]
        )
    )


def _em_ledger(inner, relative_tolerance):
    """Check the accepted total-field budgets, keeping signed errors explicit."""
    values = inner["ledger_diagnostics"]
    limits = {
        **_EM_LEDGER_LIMITS,
        "photon_map_number_residual": relative_tolerance,
        "photon_map_energy_residual": relative_tolerance,
    }
    checks = {
        name: np.isfinite(float(values[name])) and abs(float(values[name])) <= limit
        for name, limit in limits.items()
    }
    return {
        "values": values,
        "limits": limits,
        "checks": checks,
        "passed": bool(electromagnetic_ledger_passed_jax(inner, relative_tolerance)),
    }


def _finite_nonnegative(values):
    for value in jax.tree_util.tree_leaves(values):
        array = np.asarray(value)
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            return False
    return True


def _integral(values, coordinate):
    return float(np.trapezoid(np.asarray(values), x=np.asarray(coordinate)))


def _step_metrics(
    configuration,
    components,
    next_internal_state,
    loss_rate,
    bethe_heitler_loss_rate,
    proton_proton_loss_rate,
    proton_proton_pion_injection,
    chain,
    bh_electron,
    bh_positron,
):
    proton_gamma = np.asarray(_value(configuration, "proton_gamma"))
    pion_gamma = np.asarray(_value(configuration, "pion_gamma"))
    photon_energy = np.asarray(_value(configuration, "gamma_ray_energy_erg"))
    electron_gamma = np.asarray(_value(configuration, "electron_gamma"))
    neutrino_energy = np.asarray(_value(configuration, "neutrino_energy_erg"))
    radius = float(_value(configuration, "radius_cm"))
    volume = 4.0 * np.pi * radius**3 / 3.0
    neutrinos = sum(chain[name] for name in _NEUTRINO_KEYS)
    charged_pions = (
        np.asarray(chain["q_pi_plus"]) * _PION_REST_ENERGY_CGS[1]
        + np.asarray(chain["q_pi_minus"]) * _PION_REST_ENERGY_CGS[2]
    )
    bh_pairs = np.asarray(bh_electron) + np.asarray(bh_positron)
    bh_pair_number = _integral(bh_pairs, electron_gamma)
    bh_pair_energy = _integral(
        (electron_gamma - 1.0) * bh_pairs * _MEC2_CGS,
        electron_gamma,
    )
    bh_proton_loss = _integral(
        proton_gamma
        * np.asarray(chain["proton_density"])
        * np.asarray(bethe_heitler_loss_rate)
        * _PROTON_REST_ENERGY_CGS,
        proton_gamma,
    )
    pp_proton_loss = _integral(
        proton_gamma
        * np.asarray(chain["proton_density"])
        * np.asarray(proton_proton_loss_rate)
        * _PROTON_REST_ENERGY_CGS,
        proton_gamma,
    )
    pp_pions = sum(
        np.asarray(injection) * rest_energy
        for injection, rest_energy in zip(
            proton_proton_pion_injection, _PION_REST_ENERGY_CGS, strict=True
        )
    )
    pp_pion_power = _integral(pion_gamma * pp_pions, pion_gamma)
    pp_scale = max(abs(pp_proton_loss), abs(pp_pion_power), np.finfo(np.float64).tiny)
    bh_scale = max(abs(bh_pair_energy), abs(bh_proton_loss), np.finfo(np.float64).tiny)
    return {
        "target_internal_photon_energy_density_erg_cm3": _integral(
            components["internal"]["energy_erg"]
            * components["internal"]["density_cm3_erg"],
            components["internal"]["energy_erg"],
        ),
        "internal_photon_energy_density_erg_cm3": _integral(
            next_internal_state["energy_erg"] * next_internal_state["density_cm3_erg"],
            next_internal_state["energy_erg"],
        ),
        "proton_energy_density_erg_cm3": _integral(
            proton_gamma
            * np.asarray(chain["proton_density"])
            * _PROTON_REST_ENERGY_CGS,
            proton_gamma,
        ),
        "photomeson_loss_power_erg_s": volume
        * _integral(
            proton_gamma
            * np.asarray(chain["proton_density"])
            * np.asarray(loss_rate)
            * _PROTON_REST_ENERGY_CGS,
            proton_gamma,
        ),
        "proton_proton_loss_power_erg_s": volume * pp_proton_loss,
        "proton_proton_pion_injected_power_erg_s": volume * pp_pion_power,
        "proton_proton_energy_residual": ((pp_pion_power - pp_proton_loss) / pp_scale),
        "charged_pion_injected_power_erg_s": volume
        * _integral(pion_gamma * charged_pions, pion_gamma),
        "pi0_gamma_power_erg_s": volume
        * _integral(
            photon_energy * np.asarray(chain["pi0_decay_photons"]),
            photon_energy,
        ),
        "secondary_electron_power_erg_s": volume
        * _integral(
            (electron_gamma - 1.0) * np.asarray(chain["electron"]) * _MEC2_CGS,
            electron_gamma,
        ),
        "secondary_positron_power_erg_s": volume
        * _integral(
            (electron_gamma - 1.0) * np.asarray(chain["positron"]) * _MEC2_CGS,
            electron_gamma,
        ),
        "secondary_lepton_power_erg_s": volume
        * _integral(
            (electron_gamma - 1.0)
            * (np.asarray(chain["electron"]) + np.asarray(chain["positron"]))
            * _MEC2_CGS,
            electron_gamma,
        ),
        "neutrino_power_erg_s": volume
        * _integral(neutrino_energy * np.asarray(neutrinos), neutrino_energy),
        "bh_pair_number_injection_cm3_s": bh_pair_number,
        "bh_pair_energy_injection_erg_cm3_s": bh_pair_energy,
        "bh_pair_injected_power_erg_s": volume * bh_pair_energy,
        "bh_proton_loss_power_erg_s": volume * bh_proton_loss,
        "bh_pair_loss_diagnostic_residual": (
            (bh_pair_energy - bh_proton_loss) / bh_scale
        ),
    }


def _next_internal_state(inner):
    return {
        "energy_erg": inner["photon_energy_erg"],
        "density_cm3_erg": inner["internal_photon_density_cm3_erg"],
    }


def outer_hadronic_picard_step(
    previous_internal_photon_state,
    fixed_external_photon_state,
    production_configuration,
    *,
    bethe_heitler_pair_responses=None,
    gamma_gamma_pair_response=None,
    external_gamma_gamma_pair_response=None,
    physics_only=False,
):
    """Run one blob-comoving outer hadronic Picard update.

    Photon-state energies are in erg and densities are local isotropic or
    angle-averaged ``dn/dE`` in cm^-3 erg^-1. The internal state has
    ``energy_erg`` and ``density_cm3_erg``. The fixed external state has ``energy_erg`` and
    ``density_cm3_erg``.

    ``production_configuration`` supplies the existing proton, pion, muon,
    electron, photon, and neutrino grids; proton and primary-lepton sources;
    magnetic field, radius, and transport rates. All internal photon sources
    and hadronic targets use the same complete internal photon grid.
    ``bethe_heitler_mode`` is required and is one of ``"off"``, ``"dynamic"``,
    or ``"legacy_fixed_loss"``. Dynamic mode evaluates the existing continuous
    proton loss and Born-domain pair provider independently on the fixed
    external target and the complete accepted internal target. The pair provider covers only
    ``2 <= omega < 600`` and returns positrons before electrons; its domain is
    not identical to the unchanged continuous-loss provider. A non-zero fixed
    ``fractional_bethe_heitler_energy_loss_rate_s_inv`` is valid only in
    legacy-fixed-loss mode, which injects no pairs. ``enable_bethe_heitler`` is
    accepted only as a two-state compatibility alias for dynamic/off; omitting
    both keys is an error rather than an implicit third state.
    ``enable_pion_synchrotron_photons`` and
    ``enable_muon_synchrotron_photons`` independently add charge-separated
    synchrotron emission from the current steady pion and muon densities to
    the intrinsic gamma source. They do not disable the synchrotron cooling
    already used by pion/muon transport. Both default to ``False``. The generic
    kernel returns comoving ``E L_E``; reported local photon injections use the exact
    inverse production adapter ``Q_gamma = (E L_E)/(V E^2)`` on the native
    gamma-ray grid. These sources enter gamma-gamma transfer before escape.
    ``enable_pp=True`` adds cold-target proton cooling and charge-resolved pion
    injection using ``cold_proton_density_cm3``; the default is ``False``.
    pp and photomeson sources are independent and may be enabled together.
    ``enable_ssc`` and ``enable_ec`` are static composition switches. The former
    is a compatibility name for IC on the full accepted internal photon field;
    the latter controls IC on the separate fixed external seed reservoir. Each
    disables both the corresponding emission and electron/positron cooling;
    internal and external photon targets remain available to gamma-gamma and
    hadronic processes. Both default to ``True`` for compatibility.
    ``enable_internal_gamma_gamma=False`` disables only internal gamma-gamma
    absorption and pair injection; it leaves the synchrotron target, external
    gamma-gamma, and all hadronic interactions unchanged. Its default is
    ``True``.
    The fixed external photon state must already be blob-comoving ``dn/dE``;
    host/AGN-frame fields are not accepted or transformed at this boundary.
    Observer transforms, escaped photons, and luminosity-to-density conversions
    are not included.

    Returns a dictionary retaining all three target components, their
    photomeson losses and secondary chains, the summed proton/pion/muon
    forward result, the inner electromagnetic result, the next local photon
    state, residual metrics, and ledgers. Particle grids are
    dimensionless Lorentz factors; injections, densities, rates, field,
    radius, and daughter energies use the units documented by the existing
    production functions. Their finite, ordered, non-negative, blob-frame
    validity requirements are inherited unchanged.

    Component and total proton loss spectra are fractional rates in s^-1.
    Proton, pion, and muon densities are differential in Lorentz factor in
    cm^-3; their injections are cm^-3 s^-1 per Lorentz factor. Secondary
    electron/positron injections use the same convention, while photon and
    neutrino injections are cm^-3 s^-1 erg^-1. Integrated metrics are local
    energy densities in erg cm^-3 or homogeneous-sphere powers in erg s^-1.
    BH pair number and kinetic-energy injection metrics sum both charges and
    use cm^-3 s^-1 and erg cm^-3 s^-1; their comparison with continuous proton
    loss is a finite domain-mismatch diagnostic, not an energy-closure test.
    The whole accepted internal population supplies both hadronic losses and
    secondaries. The external field remains a fixed separate reservoir.
    Both grids require at least two nodes, and prepared responses must match
    their exact energy values, not just their array shapes.

    Fixed BH responses are keyed by ``external`` and ``internal``.
    Fixed gamma-gamma responses are forwarded to the inner
    cascade. ``physics_only=True`` returns the production particle, secondary,
    EM-cascade, and photon-state results before NumPy host ledgers/checks; it
    changes no numerical physics and is intended for finite one-step tracing.
    """
    if not isinstance(physics_only, (bool, np.bool_)):
        raise ValueError("physics_only must be boolean")
    physics_only = bool(physics_only)
    components = _target_components(
        previous_internal_photon_state,
        fixed_external_photon_state,
    )
    proton_gamma = jnp.asarray(
        _value(production_configuration, "proton_gamma"), dtype=jnp.float64
    )
    pion_gamma = jnp.asarray(
        _value(production_configuration, "pion_gamma"), dtype=jnp.float64
    )
    pp_enabled = _configuration_flag(production_configuration, "enable_pp")
    cold_proton_density = _value(
        production_configuration, "cold_proton_density_cm3", 0.0
    )
    bh_mode = _bethe_heitler_mode(production_configuration)
    bh_enabled = bh_mode == "dynamic"
    configured_bh_loss = jnp.broadcast_to(
        jnp.asarray(
            _value(
                production_configuration,
                "fractional_bethe_heitler_energy_loss_rate_s_inv",
                0.0,
            ),
            dtype=jnp.float64,
        ),
        proton_gamma.shape,
    )
    if bh_mode != "legacy_fixed_loss" and np.any(np.asarray(configured_bh_loss) != 0.0):
        raise ValueError(
            f"Bethe-Heitler mode {bh_mode!r} cannot also use a fixed proton loss"
        )
    component_losses = {
        name: photomeson_energy_loss_rate_jax(
            proton_gamma,
            component["energy_erg"],
            component["density_cm3_erg"],
        )
        for name, component in components.items()
    }
    total_loss = sum(component_losses.values())
    zero_proton_rate = jnp.zeros_like(proton_gamma)
    zero_pion_source = jnp.zeros_like(pion_gamma)
    if pp_enabled:
        total_pp_loss = proton_proton_fractional_energy_loss_rate_jax(
            pion_gamma, proton_gamma, cold_proton_density
        )
    else:
        total_pp_loss = zero_proton_rate
    if bh_enabled:
        component_bh_losses = {
            name: bethe_heitler_proton_energy_loss_rate_jax(
                proton_gamma,
                component["energy_erg"],
                component["density_cm3_erg"],
            )
            for name, component in components.items()
        }
        total_bh_loss = sum(component_bh_losses.values())
        bh_loss_source = "current_target_components"
    else:
        component_bh_losses = {name: zero_proton_rate for name in _TARGET_NAMES}
        total_bh_loss = (
            configured_bh_loss if bh_mode == "legacy_fixed_loss" else zero_proton_rate
        )
        bh_loss_source = (
            "legacy_configured" if bh_mode == "legacy_fixed_loss" else "disabled"
        )
    proton_density, proton_escape = proton_steady_state_transport_jax(
        proton_gamma,
        _value(
            production_configuration,
            "primary_proton_injection_dgamma_cm3_s",
        ),
        _value(production_configuration, "magnetic_field_G"),
        total_loss,
        _value(production_configuration, "proton_adiabatic_rate_s_inv"),
        _value(production_configuration, "proton_escape_rate_s_inv"),
        fractional_bethe_heitler_energy_loss_rate_s_inv=total_bh_loss,
        fractional_proton_proton_energy_loss_rate_s_inv=total_pp_loss,
    )
    component_chains = {
        name: photomeson_secondary_chain_jax(
            proton_gamma,
            _value(
                production_configuration,
                "primary_proton_injection_dgamma_cm3_s",
            ),
            component["energy_erg"],
            component["density_cm3_erg"],
            _value(production_configuration, "pion_gamma"),
            _value(production_configuration, "muon_gamma"),
            _value(production_configuration, "gamma_ray_energy_erg"),
            _value(production_configuration, "electron_gamma"),
            _value(production_configuration, "neutrino_energy_erg"),
            _value(production_configuration, "magnetic_field_G"),
            _value(production_configuration, "proton_adiabatic_rate_s_inv"),
            _value(production_configuration, "proton_escape_rate_s_inv"),
            proton_density_dgamma_cm3=proton_density,
        )
        for name, component in components.items()
    }
    chain = _sum_component_chains(component_chains, proton_density)
    pp_pion_injection = (zero_pion_source, zero_pion_source, zero_pion_source)
    pp_chain = None
    if pp_enabled:
        checked_pp_loss, *pp_pion_injection = proton_proton_pion_injection_and_loss_jax(
            pion_gamma,
            proton_gamma,
            proton_density,
            cold_proton_density,
        )
        if not np.array_equal(np.asarray(checked_pp_loss), np.asarray(total_pp_loss)):
            raise RuntimeError(
                "proton-proton loss changed between transport and source"
            )
        pp_chain = photomeson_secondary_chain_jax(
            proton_gamma,
            _value(
                production_configuration,
                "primary_proton_injection_dgamma_cm3_s",
            ),
            components["external"]["energy_erg"],
            jnp.zeros_like(components["external"]["density_cm3_erg"]),
            pion_gamma,
            _value(production_configuration, "muon_gamma"),
            _value(production_configuration, "gamma_ray_energy_erg"),
            _value(production_configuration, "electron_gamma"),
            _value(production_configuration, "neutrino_energy_erg"),
            _value(production_configuration, "magnetic_field_G"),
            _value(production_configuration, "proton_adiabatic_rate_s_inv"),
            _value(production_configuration, "proton_escape_rate_s_inv"),
            proton_density_dgamma_cm3=proton_density,
            additional_pion_injection_dgamma_cm3_s=pp_pion_injection,
        )
        for name in chain:
            if name != "proton_density":
                chain[name] = chain[name] + pp_chain[name]
    meson_synchrotron = _meson_synchrotron_sources(
        production_configuration,
        chain,
    )
    electron_gamma = jnp.asarray(
        _value(production_configuration, "electron_gamma"), dtype=jnp.float64
    )
    zeros = jnp.zeros_like(jnp.asarray(electron_gamma, dtype=jnp.float64))
    if bh_enabled:
        component_bh_pairs = {}
        for name, component in components.items():
            positron, electron = bethe_heitler_pair_injection_jax(
                electron_gamma,
                proton_gamma,
                proton_density,
                component["energy_erg"],
                component["density_cm3_erg"],
                pair_response=(
                    None
                    if bethe_heitler_pair_responses is None
                    else bethe_heitler_pair_responses[name]
                ),
            )
            component_bh_pairs[name] = {
                "electron": electron,
                "positron": positron,
            }
    else:
        component_bh_pairs = {
            name: {"electron": zeros, "positron": zeros} for name in _TARGET_NAMES
        }
    bh_electron = sum(component_bh_pairs[name]["electron"] for name in _TARGET_NAMES)
    bh_positron = sum(component_bh_pairs[name]["positron"] for name in _TARGET_NAMES)
    inner = steady_state_electromagnetic_cascade_jax(
        _value(production_configuration, "synchrotron_photon_energy_erg"),
        _value(production_configuration, "gamma_ray_energy_erg"),
        electron_gamma,
        _value(
            production_configuration,
            "primary_electron_injection_dgamma_cm3_s",
        ),
        _value(
            production_configuration,
            "primary_positron_injection_dgamma_cm3_s",
        ),
        chain["electron"],
        chain["positron"],
        bh_electron,
        bh_positron,
        proton_gamma,
        proton_density,
        chain["pi0_decay_photons"],
        fixed_external_photon_state["energy_erg"],
        fixed_external_photon_state["density_cm3_erg"],
        _value(production_configuration, "magnetic_field_G"),
        _value(production_configuration, "radius_cm"),
        _value(production_configuration, "electron_adiabatic_rate_s_inv"),
        _value(production_configuration, "electron_escape_rate_s_inv"),
        include_external_gamma_gamma_target=_value(
            production_configuration,
            "include_external_gamma_gamma_target",
            False,
        ),
        enable_internal_gamma_gamma=_configuration_flag(
            production_configuration,
            "enable_internal_gamma_gamma",
            True,
        ),
        enable_ssc=_configuration_flag(
            production_configuration,
            "enable_ssc",
            True,
        ),
        enable_ec=_configuration_flag(
            production_configuration,
            "enable_ec",
            True,
        ),
        meson_synchrotron_intrinsic_seds_erg_s=meson_synchrotron[
            "intrinsic_seds_erg_s"
        ],
        additional_fixed_inverse_compton_loss_rate_dgamma_s=_value(
            production_configuration,
            "additional_fixed_inverse_compton_loss_rate_dgamma_s",
            0.0,
        ),
        additional_continuous_loss_rate_dgamma_s=_value(
            production_configuration,
            "additional_continuous_loss_rate_dgamma_s",
            0.0,
        ),
        pair_daughter_gamma=production_configuration.get("pair_daughter_gamma"),
        gamma_gamma_pair_response=gamma_gamma_pair_response,
        external_gamma_gamma_pair_response=(external_gamma_gamma_pair_response),
        relaxation_omega=_value(
            production_configuration, "inner_relaxation_omega", 1.0
        ),
        relative_tolerance=_value(
            production_configuration, "inner_relative_tolerance", 1.0e-4
        ),
        maximum_iterations=_value(
            production_configuration, "inner_maximum_iterations", 32
        ),
    )
    next_state = _next_internal_state(inner)
    if physics_only:
        return {
            "target_components": components,
            "proton_density_dgamma_cm3": proton_density,
            "proton_escape_dgamma_cm3_s": proton_escape,
            "photomeson_chain": chain,
            "component_bethe_heitler_pair_injections": component_bh_pairs,
            "q_bh_electron_dgamma_cm3_s": bh_electron,
            "q_bh_positron_dgamma_cm3_s": bh_positron,
            "inner_electromagnetic_cascade": inner,
            "next_internal_photon_state": next_state,
        }
    particle_ledgers = _particle_ledgers(
        production_configuration,
        total_loss,
        total_bh_loss,
        total_pp_loss,
        proton_density,
        proton_escape,
        chain,
    )
    em_ledger = _em_ledger(
        inner, _value(production_configuration, "inner_relative_tolerance", 1.0e-4)
    )
    meson_synchrotron_ledgers = _meson_synchrotron_power_ledgers(
        production_configuration,
        chain,
        meson_synchrotron,
    )
    metrics = _step_metrics(
        production_configuration,
        components,
        next_state,
        total_loss,
        total_bh_loss,
        total_pp_loss,
        pp_pion_injection,
        chain,
        bh_electron,
        bh_positron,
    )
    for name in ("pion_plus", "pion_minus", "muon_plus", "muon_minus"):
        metrics[f"{name}_synchrotron_loss_power_erg_s"] = meson_synchrotron_ledgers[
            name
        ]["transport_synchrotron_loss_power_erg_s"]
        metrics[f"{name}_synchrotron_photon_power_erg_s"] = meson_synchrotron_ledgers[
            name
        ]["photon_injection_power_erg_s"]
    for name in ("pion_total", "muon_total"):
        metrics[f"{name}_synchrotron_loss_power_erg_s"] = meson_synchrotron_ledgers[
            name
        ]["transport_synchrotron_loss_power_erg_s"]
        metrics[f"{name}_synchrotron_photon_power_erg_s"] = meson_synchrotron_ledgers[
            name
        ]["photon_injection_power_erg_s"]
    metrics["meson_synchrotron_photon_power_erg_s"] = (
        metrics["pion_total_synchrotron_photon_power_erg_s"]
        + metrics["muon_total_synchrotron_photon_power_erg_s"]
    )
    bh_component_sums_passed = all(
        np.allclose(
            np.asarray(total),
            np.asarray(
                sum(component[name] for component in component_bh_pairs.values())
            ),
            rtol=1.0e-12,
            atol=0.0,
        )
        for name, total in (("electron", bh_electron), ("positron", bh_positron))
    ) and (
        bh_mode == "legacy_fixed_loss"
        or np.allclose(
            np.asarray(total_bh_loss),
            np.asarray(sum(component_bh_losses.values())),
            rtol=1.0e-12,
            atol=0.0,
        )
    )
    meson_injections = meson_synchrotron["photon_injections_cm3_s_erg"]
    meson_seds = meson_synchrotron["intrinsic_seds_erg_s"]
    meson_component_sums_passed = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in (
            (
                meson_injections["pion_total"],
                meson_injections["pion_plus"] + meson_injections["pion_minus"],
            ),
            (
                meson_injections["muon_total"],
                meson_injections["muon_plus"] + meson_injections["muon_minus"],
            ),
            (
                meson_injections["meson_total"],
                meson_injections["pion_total"] + meson_injections["muon_total"],
            ),
        )
    )
    source_components = inner["photon_components_intrinsic_sed_erg_s"]
    meson_injected_once = all(
        np.array_equal(
            np.asarray(source_components[name + "_synchrotron"]),
            np.asarray(meson_seds[name + "_total"]),
        )
        for name in ("pion", "muon")
    )
    finite_nonnegative = _finite_nonnegative(
        {
            "components": components,
            "component_losses": component_losses,
            "component_chains": component_chains,
            "proton_proton_loss": total_pp_loss,
            "proton_proton_chain": {} if pp_chain is None else pp_chain,
            "chain": chain,
            "component_bethe_heitler_loss_rates": component_bh_losses,
            "component_bethe_heitler_pairs": component_bh_pairs,
            "meson_synchrotron": meson_synchrotron,
            "proton_escape": proton_escape,
            "inner": {
                "photon_density": inner["internal_photon_density_cm3_erg"],
                "photon_sources": source_components,
                "escaped_sources": inner["photon_components_escaped_sed_erg_s"],
                "electron_density": inner["electron_density_dgamma_cm3"],
                "positron_density": inner["positron_density_dgamma_cm3"],
            },
            "next_internal_photon_state": next_state,
        }
    )
    ledgers_passed = (
        all(ledger["passed"] for ledger in particle_ledgers.values())
        and all(ledger["passed"] for ledger in meson_synchrotron_ledgers.values())
        and em_ledger["passed"]
    )
    return {
        "fixed_external_photon_state": fixed_external_photon_state,
        "target_components": components,
        "component_photomeson_loss_rates_s_inv": component_losses,
        "total_photomeson_loss_rate_s_inv": total_loss,
        "total_proton_proton_loss_rate_s_inv": total_pp_loss,
        "proton_proton_pion_injection_dgamma_cm3_s": {
            "q_pi0": pp_pion_injection[0],
            "q_pi_plus": pp_pion_injection[1],
            "q_pi_minus": pp_pion_injection[2],
        },
        "proton_proton_chain": pp_chain,
        "proton_density_dgamma_cm3": proton_density,
        "proton_escape_dgamma_cm3_s": proton_escape,
        "component_photomeson_chains": component_chains,
        "photomeson_chain": chain,
        "component_bethe_heitler_loss_rates_s_inv": component_bh_losses,
        "total_bethe_heitler_loss_rate_s_inv": total_bh_loss,
        "component_bethe_heitler_pair_injections": component_bh_pairs,
        "q_bh_electron_dgamma_cm3_s": bh_electron,
        "q_bh_positron_dgamma_cm3_s": bh_positron,
        "q_gamma_pion_plus_syn_cm3_s_erg": meson_injections["pion_plus"],
        "q_gamma_pion_minus_syn_cm3_s_erg": meson_injections["pion_minus"],
        "q_gamma_muon_plus_syn_cm3_s_erg": meson_injections["muon_plus"],
        "q_gamma_muon_minus_syn_cm3_s_erg": meson_injections["muon_minus"],
        "q_gamma_pion_syn_total_cm3_s_erg": meson_injections["pion_total"],
        "q_gamma_muon_syn_total_cm3_s_erg": meson_injections["muon_total"],
        "q_gamma_meson_syn_total_cm3_s_erg": meson_injections["meson_total"],
        "meson_synchrotron_intrinsic_seds_erg_s": meson_seds,
        "meson_synchrotron_switches": {
            "pion": meson_synchrotron["pion_enabled"],
            "muon": meson_synchrotron["muon_enabled"],
        },
        "bethe_heitler_domain_diagnostics": {
            "bethe_heitler_mode": bh_mode,
            "bh_pair_provider_domain": _BH_PAIR_PROVIDER_DOMAIN,
            "bh_proton_loss_provider": _BH_PROTON_LOSS_PROVIDER,
            "bh_domains_identical": False,
            "bh_high_omega_pair_contribution_modeled": False,
            "bh_loss_source": bh_loss_source,
        },
        "inner_electromagnetic_cascade": inner,
        "next_internal_photon_state": next_state,
        "metrics": metrics,
        "ledgers": {
            "particles": particle_ledgers,
            "meson_synchrotron": meson_synchrotron_ledgers,
            "electromagnetic": em_ledger,
            "passed": ledgers_passed,
        },
        "checks": {
            "finite_nonnegative": finite_nonnegative,
            "inner_converged": bool(inner["converged"]),
            "ledgers_passed": ledgers_passed,
            "bh_secondary_pair_feedback_included": bh_enabled,
            "proton_proton_included": pp_enabled,
            "bh_component_sums_passed": bh_component_sums_passed,
            "bh_pair_loss_diagnostic_residual_finite": bool(
                np.isfinite(metrics["bh_pair_loss_diagnostic_residual"])
            ),
            "meson_synchrotron_component_sums_passed": (meson_component_sums_passed),
            "meson_synchrotron_injected_once": bool(meson_injected_once),
        },
    }


def _scalar_residual(new, old):
    denominator = max(abs(new), abs(old))
    if denominator <= np.finfo(np.float64).tiny:
        return {
            "near_zero": True,
            "relative_residual": None,
            "absolute_residual": abs(new - old),
        }
    return {
        "near_zero": False,
        "relative_residual": abs(new - old) / denominator,
        "absolute_residual": abs(new - old),
    }


def energy_weighted_spectral_defect_jax(new, old, coordinate):
    """Return the host outer-loop energy-weighted L1 defect as a JAX scalar.

    Integrates coordinate*|spectrum| with respect to coordinate (dE for
    photons, dgamma for particles). A vanishing pair is zero; invalid arrays
    or unresolved differences at a subnormal norm return infinity.
    """
    coordinate = jnp.asarray(coordinate, dtype=jnp.float64)
    new = jnp.asarray(new, dtype=jnp.float64)
    old = jnp.asarray(old, dtype=jnp.float64)
    new_norm = jnp.trapezoid(coordinate * jnp.abs(new), x=coordinate)
    old_norm = jnp.trapezoid(coordinate * jnp.abs(old), x=coordinate)
    denominator = jnp.maximum(new_norm, old_norm)
    absolute = jnp.trapezoid(coordinate * jnp.abs(new - old), x=coordinate)
    tiny = jnp.finfo(jnp.float64).tiny
    near_zero = denominator <= tiny
    relative = absolute / jnp.where(near_zero, 1.0, denominator)
    relative = jnp.where(near_zero, jnp.where(absolute <= tiny, 0.0, jnp.inf), relative)
    finite = (
        jnp.all(jnp.isfinite(new))
        & jnp.all(jnp.isfinite(old))
        & jnp.all(jnp.isfinite(coordinate))
    )
    return jnp.where(finite & jnp.isfinite(relative), relative, jnp.inf)


def _energy_weighted_spectral_residual(new, old, coordinate):
    coordinate = np.asarray(coordinate)
    new = np.asarray(new)
    old = np.asarray(old)
    new_norm = _integral(coordinate * np.abs(new), coordinate)
    old_norm = _integral(coordinate * np.abs(old), coordinate)
    denominator = max(new_norm, old_norm)
    absolute = _integral(coordinate * np.abs(new - old), coordinate)
    near_zero = denominator <= np.finfo(np.float64).tiny
    return {
        "near_zero": near_zero,
        "relative_residual": (
            None
            if near_zero
            else float(energy_weighted_spectral_defect_jax(new, old, coordinate))
        ),
        "absolute_residual": absolute,
    }


def _outer_residuals(previous, current, production_configuration, outer_rtol):
    metric_names = (
        "internal_photon_energy_density_erg_cm3",
        "photomeson_loss_power_erg_s",
        "proton_proton_loss_power_erg_s",
        "proton_proton_pion_injected_power_erg_s",
        "charged_pion_injected_power_erg_s",
        "pi0_gamma_power_erg_s",
        "secondary_electron_power_erg_s",
        "secondary_positron_power_erg_s",
        "secondary_lepton_power_erg_s",
        "neutrino_power_erg_s",
        "pion_total_synchrotron_photon_power_erg_s",
        "muon_total_synchrotron_photon_power_erg_s",
        "meson_synchrotron_photon_power_erg_s",
    )
    scalar_residuals = {
        name: _scalar_residual(
            current["metrics"][name],
            previous["metrics"][name],
        )
        for name in metric_names
    }
    spectral_residuals = {
        "proton_spectrum_energy_weighted": _energy_weighted_spectral_residual(
            current["proton_density_dgamma_cm3"],
            previous["proton_density_dgamma_cm3"],
            _value(production_configuration, "proton_gamma"),
        ),
        "internal_photon_spectrum_energy_weighted": _energy_weighted_spectral_residual(
            current["next_internal_photon_state"]["density_cm3_erg"],
            previous["next_internal_photon_state"]["density_cm3_erg"],
            current["next_internal_photon_state"]["energy_erg"],
        ),
    }
    residuals = {**scalar_residuals, **spectral_residuals}
    enabled = [
        row["relative_residual"] for row in residuals.values() if not row["near_zero"]
    ]
    maximum = max(enabled) if enabled else 0.0
    enabled_spectral = [
        row["relative_residual"]
        for row in spectral_residuals.values()
        if not row["near_zero"]
    ]
    maximum_spectral = max(enabled_spectral) if enabled_spectral else 0.0
    residuals_finite = all(
        np.isfinite(row["absolute_residual"])
        and (row["relative_residual"] is None or np.isfinite(row["relative_residual"]))
        for row in residuals.values()
    )
    spectral_passed = residuals_finite and all(
        (
            row["absolute_residual"] <= np.finfo(np.float64).tiny
            if row["near_zero"]
            else row["relative_residual"] < outer_rtol
        )
        for row in spectral_residuals.values()
    )
    passed = (
        spectral_passed
        and current["checks"]["inner_converged"]
        and current["checks"]["ledgers_passed"]
        and current["checks"]["finite_nonnegative"]
        and current["checks"]["bh_component_sums_passed"]
        and current["checks"]["bh_pair_loss_diagnostic_residual_finite"]
        and current["checks"]["meson_synchrotron_component_sums_passed"]
        and current["checks"]["meson_synchrotron_injected_once"]
    )
    return {
        "fields": residuals,
        "maximum_enabled_relative_residual": maximum,
        "maximum_spectral_relative_residual": maximum_spectral,
        "spectral_passed": spectral_passed,
        "residuals_finite": residuals_finite,
        "outer_rtol": outer_rtol,
        "passed": passed,
    }


def solve_steady_state_hadronic_cascade(
    initial_internal_photon_state,
    fixed_external_photon_state,
    production_configuration,
    *,
    max_outer_iterations,
    outer_rtol,
    dynamic_internal_target=True,
    required_consecutive_passes=2,
    bethe_heitler_pair_responses=None,
    gamma_gamma_pair_response=None,
    external_gamma_gamma_pair_response=None,
):
    """Run a direct Python Picard loop around the existing production chains.

    ``dynamic_internal_target=False`` keeps the initial internal target fixed
    for every p-gamma evaluation while still returning each inner cascade's
    updated local photon state. At least two outer iterations are required so
    adjacent feedback residuals can be evaluated. ``outer_rtol`` must be
    finite and positive. ``required_consecutive_passes`` defaults to two;
    convergence requires that many adjacent updates to pass the proton,
    internal-photon spectral checks together.
    State/configuration units, frame, and physical assumptions are those of
    :func:`outer_hadronic_picard_step`.

    Optional prepared response arguments are passed unchanged to every outer
    step. They are prepared outside this solver and do not alter the Picard
    iteration or convergence criteria. Omitting them preserves the original
    response behavior.

    Returns a dictionary containing the ordered outer-step results, fixed and
    initial states, final local photon state, iteration count, tolerance, mode,
    and convergence flag. This is direct unmixed Picard iteration; it adds no
    observer transform or pp process. Bethe--Heitler behavior is selected by
    the production configuration as documented by the single-step function.
    A step whose inner cascade, ledger, or finite/non-negative check fails is
    retained for diagnostics, is not fed back, and terminates the loop with
    ``converged=False``. Frozen-target mode retains its compatibility behavior
    but cannot report convergence of the dynamic outer fixed point.
    """
    if not isinstance(dynamic_internal_target, (bool, np.bool_)):
        raise ValueError("dynamic_internal_target must be boolean")
    if (
        not isinstance(required_consecutive_passes, int)
        or isinstance(required_consecutive_passes, bool)
        or required_consecutive_passes < 1
    ):
        raise ValueError("required_consecutive_passes must be a positive integer")
    if (
        not isinstance(max_outer_iterations, int)
        or max_outer_iterations < 2
        or not np.isfinite(outer_rtol)
        or outer_rtol <= 0.0
    ):
        raise ValueError(
            "outer loop requires max_outer_iterations >= 2 and outer_rtol > 0"
        )
    iterations = []
    target_state = initial_internal_photon_state
    consecutive_passes = 0
    prepared_responses = {
        name: value
        for name, value in (
            ("bethe_heitler_pair_responses", bethe_heitler_pair_responses),
            ("gamma_gamma_pair_response", gamma_gamma_pair_response),
            (
                "external_gamma_gamma_pair_response",
                external_gamma_gamma_pair_response,
            ),
        )
        if value is not None
    }
    for index in range(max_outer_iterations):
        step = outer_hadronic_picard_step(
            target_state,
            fixed_external_photon_state,
            production_configuration,
            **prepared_responses,
        )
        step["outer_iteration"] = index
        step["outer_residuals"] = (
            None
            if not iterations
            else _outer_residuals(
                iterations[-1],
                step,
                production_configuration,
                outer_rtol,
            )
        )
        consecutive_passes = (
            consecutive_passes + 1
            if step["outer_residuals"] is not None and step["outer_residuals"]["passed"]
            else 0
        )
        step["consecutive_outer_passes"] = consecutive_passes
        iterations.append(step)
        if not (
            step["checks"]["inner_converged"]
            and step["checks"]["ledgers_passed"]
            and step["checks"]["finite_nonnegative"]
            and step["checks"]["bh_component_sums_passed"]
            and step["checks"]["bh_pair_loss_diagnostic_residual_finite"]
            and step["checks"]["meson_synchrotron_component_sums_passed"]
            and step["checks"]["meson_synchrotron_injected_once"]
        ):
            break
        if (
            dynamic_internal_target
            and consecutive_passes >= required_consecutive_passes
        ):
            break
        target_state = (
            step["next_internal_photon_state"]
            if dynamic_internal_target
            else initial_internal_photon_state
        )
    return {
        "iterations": tuple(iterations),
        "fixed_external_photon_state": fixed_external_photon_state,
        "initial_internal_photon_state": initial_internal_photon_state,
        "final_internal_photon_state": iterations[-1]["next_internal_photon_state"],
        "dynamic_internal_target": dynamic_internal_target,
        "outer_iteration_count": len(iterations),
        "outer_rtol": outer_rtol,
        "maximum_outer_iterations": max_outer_iterations,
        "required_consecutive_passes": required_consecutive_passes,
        "final_consecutive_outer_passes": consecutive_passes,
        "converged": bool(
            dynamic_internal_target
            and consecutive_passes >= required_consecutive_passes
        ),
    }
