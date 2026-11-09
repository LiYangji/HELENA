"""Steady particles and one internal photon population on a common energy grid."""

from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_MASS_CGS as _M_E_CGS,
)
from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
)
from helena.core.constants import (
    ELEMENTARY_CHARGE_ESU as _E_ESU,
)
from helena.core.constants import (
    PROTON_MASS_CGS as _M_P_CGS,
)
from helena.core.constants import (
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.jax_backend.chains.gamma_gamma_pair_source import (
    gamma_gamma_total_field_jax,
    prepare_gamma_gamma_total_field_response_jax,
)
from helena.core.jax_backend.chains.total_lepton_transport import (
    total_lepton_transport_chain_jax,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    _angle_averaged_cross_section_jax,
    _contains_tracer,
)
from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
    _contract_pair_response,
    prepare_gamma_gamma_pair_response_jax,
)
from helena.core.jax_backend.kernels.inverse_compton_cooling import (
    electron_inverse_compton_loss_rate_dgamma_s_jax,
)
from helena.core.jax_backend.radiation.inverse_compton import inverse_compton_sed_jax
from helena.core.jax_backend.radiation.photon_fields import (
    _internal_photon_density_kernel,
)
from helena.core.jax_backend.radiation.ssa import (
    _sphere_attenuation_factor_jax,
    synchrotron_absorption_coefficient_jax,
)
from helena.core.jax_backend.radiation.synchrotron import (
    electron_synchrotron_sed_jax,
    synchrotron_sed_jax,
)

__all__ = ["steady_state_electromagnetic_cascade_jax"]
_PHOTON_COMPONENTS = (
    "electron_synchrotron",
    "proton_synchrotron",
    "pion_synchrotron",
    "muon_synchrotron",
    "internal_ic",
    "external_compton",
    "pi0",
)


def _positive_iteration_scalar(name, value, *, upper=None):
    if not _contains_tracer(value):
        array = np.asarray(value)
        if array.ndim != 0 or array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a finite real scalar")
        scalar = float(array)
        if (
            not np.isfinite(scalar)
            or scalar <= 0.0
            or (upper is not None and scalar > upper)
        ):
            bound = f" <= {upper}" if upper is not None else ""
            raise ValueError(f"{name} must satisfy 0 < {name}{bound}")
    return jnp.asarray(value, dtype=jnp.float64)


def _trapezoid_weights(coordinate):
    coordinate = jnp.asarray(coordinate, dtype=jnp.float64)
    if coordinate.shape[0] == 1:
        return jnp.ones_like(coordinate)
    widths = jnp.diff(coordinate)
    return jnp.concatenate(
        (
            widths[:1] / 2.0,
            (widths[:-1] + widths[1:]) / 2.0,
            widths[-1:] / 2.0,
        )
    )


def _validated_pair_daughter_gamma(pair_daughter_gamma, electron_gamma):
    daughter = jnp.asarray(pair_daughter_gamma, dtype=jnp.float64)
    if daughter.ndim != 1 or daughter.shape[0] < 2:
        raise ValueError(
            "pair_daughter_gamma must be one-dimensional with at least two points"
        )
    if not _contains_tracer((pair_daughter_gamma, electron_gamma)):
        supplied = np.asarray(pair_daughter_gamma)
        transport = np.asarray(electron_gamma)
        if supplied.dtype != np.dtype(np.float64):
            raise ValueError("pair_daughter_gamma must have dtype float64")
        if not np.all(np.isfinite(supplied)):
            raise ValueError("pair_daughter_gamma must contain only finite values")
        if np.any(supplied <= 1.0) or np.any(np.diff(supplied) <= 0.0):
            raise ValueError(
                "pair_daughter_gamma must contain strictly increasing values > 1"
            )
        if transport.ndim != 1 or transport.size < 2:
            raise ValueError(
                "electron_gamma must have at least two points when pair_daughter_gamma "
                "is supplied"
            )
        if supplied[0] != transport[0] or supplied[-1] != transport[-1]:
            raise ValueError(
                "pair_daughter_gamma must have the same endpoints as electron_gamma"
            )
    return daughter


def _pair_grids_are_identical(pair_daughter_gamma, electron_gamma):
    if pair_daughter_gamma.shape != electron_gamma.shape:
        return False
    if pair_daughter_gamma is electron_gamma:
        return True
    if _contains_tracer((pair_daughter_gamma, electron_gamma)):
        return False
    return np.array_equal(pair_daughter_gamma, electron_gamma)


def _pair_source_remap_operator_jax(pair_daughter_gamma, transport_gamma):
    """Precompute a piecewise-constant dual-cell overlap operator."""
    source = jnp.asarray(pair_daughter_gamma, dtype=jnp.float64)
    target = jnp.asarray(transport_gamma, dtype=jnp.float64)
    source_faces = jnp.concatenate(
        (source[:1], (source[:-1] + source[1:]) / 2.0, source[-1:])
    )
    target_faces = jnp.concatenate(
        (target[:1], (target[:-1] + target[1:]) / 2.0, target[-1:])
    )
    overlap = jnp.maximum(
        0.0,
        jnp.minimum(target_faces[1:, None], source_faces[None, 1:])
        - jnp.maximum(target_faces[:-1, None], source_faces[None, :-1]),
    )
    return overlap / _trapezoid_weights(target)[:, None]


def _conservative_pair_source_remap_jax(pair_source_dgamma, operator):
    """Apply a fixed number-conserving dual-cell overlap operator."""
    source = jnp.asarray(pair_source_dgamma, dtype=jnp.float64)
    return operator @ source


def _pair_source_moments(pair_source_dgamma, gamma):
    weights = _trapezoid_weights(gamma)
    source = jnp.asarray(pair_source_dgamma, dtype=jnp.float64)
    coordinate = jnp.asarray(gamma, dtype=jnp.float64)
    return jnp.sum(weights * source), jnp.sum(weights * coordinate * source)


def _symmetric_relative_l1(new, old, coordinate):
    weights = _trapezoid_weights(coordinate)
    new_norm = jnp.sum(weights * jnp.abs(new))
    old_norm = jnp.sum(weights * jnp.abs(old))
    denominator = jnp.maximum(new_norm, old_norm)
    safe_denominator = jnp.where(denominator == 0.0, 1.0, denominator)
    residual = jnp.sum(weights * jnp.abs(new - old)) / safe_denominator
    return jnp.where(denominator == 0.0, 0.0, residual)


def _symmetric_scalar_residual(left, right):
    denominator = jnp.maximum(jnp.abs(left), jnp.abs(right))
    safe_denominator = jnp.where(denominator == 0.0, 1.0, denominator)
    residual = jnp.abs(left - right) / safe_denominator
    return jnp.where(denominator == 0.0, 0.0, residual)


def _safe_ratio(numerator, denominator):
    safe_denominator = jnp.where(denominator == 0.0, 1.0, denominator)
    return jnp.where(denominator == 0.0, 0.0, numerator / safe_denominator)


def steady_state_electromagnetic_cascade_jax(
    synchrotron_photon_energy_erg,
    gamma_ray_energy_erg,
    electron_gamma,
    primary_electron_injection_dgamma_cm3_s,
    primary_positron_injection_dgamma_cm3_s,
    muon_decay_electron_injection_dgamma_cm3_s,
    muon_decay_positron_injection_dgamma_cm3_s,
    bethe_heitler_electron_injection_dgamma_cm3_s,
    bethe_heitler_positron_injection_dgamma_cm3_s,
    proton_gamma,
    proton_density_dgamma_cm3,
    pi0_decay_photon_injection_cm3_s_erg,
    external_seed_photon_energy_erg,
    dn_external_seed_photon_denergy_cm3_erg,
    magnetic_field_G,
    radius_cm,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
    *,
    include_external_gamma_gamma_target=False,
    enable_internal_gamma_gamma=True,
    enable_ssc=True,
    enable_ec=True,
    meson_synchrotron_intrinsic_seds_erg_s=None,
    additional_fixed_inverse_compton_loss_rate_dgamma_s=0.0,
    additional_continuous_loss_rate_dgamma_s=0.0,
    pair_daughter_gamma=None,
    initial_gamma_gamma_electron_injection_dgamma_cm3_s=None,
    initial_gamma_gamma_positron_injection_dgamma_cm3_s=None,
    initial_ssc_loss_rate_dgamma_s=None,
    gamma_gamma_pair_response=None,
    external_gamma_gamma_pair_response=None,
    relaxation_omega=1.0,
    relative_tolerance=1.0e-4,
    maximum_iterations=32,
):
    """Return the finite-scan steady electromagnetic state (float64, blob frame).

    The two positional photon grids must now be identical full-support grids,
    in erg. Historical different/split grids are rejected: their independent
    photon populations cannot be converted without source information. Input
    lepton sources are cm^-3 s^-1 dgamma^-1, pi0 source cm^-3 s^-1 erg^-1;
    meson dictionary pion_total/muon_total contains intrinsic E L_E (erg/s).
    n=dn/dE is cm^-3 erg^-1; B,R are G,cm. The seven source populations share
    electron SSA plus internal/external gamma-gamma transfer at every energy.
    Internal IC emission and cooling read the same accepted total internal n,
    including internally produced IC/EC/pi0 photons. The historical enable_ssc
    switch now controls this internal IC process. SSC outputs are nonadditive
    diagnostics of the four-synchrotron-seed subset, not a second population.
    Fixed external seeds enter EC once; photons emitted by EC may subsequently
    scatter internally. No second SSA factor is applied to any seed density.
    The existing ultrarelativistic upscattering kernels are retained; this is
    not an arbitrary-energy bidirectional Compton/heating solver. Seed-photon
    removal is neglected in the optically thin scattering regime (requiring a separate
    scattering-opacity check), not included in the SSA/gamma-gamma photon loss ledger.

    Internal gamma-gamma uses the same accepted n on BOTH sides; prescribed
    external targets are a separate nondepleting reservoir, without a half.
    Photon and particle raw map defects control freezing. At an exhausted
    iteration limit converged=False and iteration_limit_reached=True; callers
    must reject that result. This is finite-program AD, not implicit steady AD.
    No relaxed step size is substituted for the true photon map residual.
    """
    if (
        not isinstance(maximum_iterations, Integral)
        or isinstance(maximum_iterations, bool)
        or maximum_iterations < 1
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    for name, value in (
        ("include_external_gamma_gamma_target", include_external_gamma_gamma_target),
        ("enable_internal_gamma_gamma", enable_internal_gamma_gamma),
        ("enable_ssc", enable_ssc),
        ("enable_ec", enable_ec),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    if initial_ssc_loss_rate_dgamma_s is not None:
        raise ValueError(
            "initial_ssc_loss_rate_dgamma_s is retired: internal IC cooling is derived from the common photon state"
        )
    if not _contains_tracer(
        (synchrotron_photon_energy_erg, gamma_ray_energy_erg)
    ) and not np.array_equal(synchrotron_photon_energy_erg, gamma_ray_energy_erg):
        raise ValueError(
            "EM requires identical full photon grids; split source inputs must be rebuilt before calling"
        )
    if np.shape(synchrotron_photon_energy_erg) != np.shape(gamma_ray_energy_erg):
        raise ValueError("EM requires identical full photon-grid shapes")
    energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
    gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    daughter = (
        gamma
        if pair_daughter_gamma is None
        else _validated_pair_daughter_gamma(pair_daughter_gamma, electron_gamma)
    )
    remap = (
        None
        if _pair_grids_are_identical(daughter, gamma)
        else _pair_source_remap_operator_jax(daughter, gamma)
    )
    if gamma_gamma_pair_response is None:
        response = prepare_gamma_gamma_total_field_response_jax(
            gamma_ray_energy_erg, daughter
        )
    elif not isinstance(gamma_gamma_pair_response, dict):
        raise ValueError(
            "gamma_gamma_pair_response must be a prepared total-field response, not the retired labelled tensor"
        )
    else:
        response = gamma_gamma_pair_response
    if response["pair_response"].shape != (daughter.size, energy.size, energy.size):
        raise ValueError(
            "total-field response shape does not match photon/daughter grids"
        )
    if not _contains_tracer((energy, daughter, response)) and (
        not np.array_equal(energy, response["photon_energy_erg"])
        or not np.array_equal(daughter, response["lepton_gamma"])
    ):
        raise ValueError("total-field response belongs to different grids")
    omega = _positive_iteration_scalar("relaxation_omega", relaxation_omega, upper=1.0)
    tolerance = _positive_iteration_scalar("relative_tolerance", relative_tolerance)
    volume = 4.0 * jnp.pi * radius_cm**3 / 3.0
    weights = response["photon_integration_weights_erg"]
    ext_energy = jnp.asarray(external_seed_photon_energy_erg, dtype=jnp.float64)
    ext_density = jnp.asarray(
        dn_external_seed_photon_denergy_cm3_erg, dtype=jnp.float64
    )
    ec_density = ext_density if enable_ec else jnp.zeros_like(ext_density)
    ext_cooling = electron_inverse_compton_loss_rate_dgamma_s_jax(
        gamma, ext_energy, ec_density
    )
    fixed_cooling = jnp.broadcast_to(
        jnp.asarray(
            additional_fixed_inverse_compton_loss_rate_dgamma_s, dtype=jnp.float64
        ),
        gamma.shape,
    )
    if not _contains_tracer(fixed_cooling):
        fixed_values = np.asarray(fixed_cooling)
        if not np.all(np.isfinite(fixed_values)) or np.any(fixed_values < 0):
            raise ValueError(
                "additional_fixed_inverse_compton_loss_rate_dgamma_s must be finite and non-negative"
            )
    ext_weights = ext_energy * _trapezoid_weights(jnp.log(ext_energy))
    ext_target = (
        ext_density
        if include_external_gamma_gamma_target
        else jnp.zeros_like(ext_density)
    )
    ext_kernel = _angle_averaged_cross_section_jax(
        energy[:, None] * ext_energy[None, :] / _MEC2_CGS**2
    )
    ext_alpha = ext_kernel @ (ext_target * ext_weights)
    if include_external_gamma_gamma_target:
        if external_gamma_gamma_pair_response is None:
            external_gamma_gamma_pair_response = response.get("external_pair_response")
        if external_gamma_gamma_pair_response is None:
            external_gamma_gamma_pair_response = prepare_gamma_gamma_pair_response_jax(
                daughter, gamma_ray_energy_erg, external_seed_photon_energy_erg
            )
        else:
            if "external_target_photon_energy_erg" not in response:
                raise ValueError(
                    "Cached external pair response requires exact external_target_photon_energy_erg metadata in the prepared response"
                )
            if external_gamma_gamma_pair_response.shape != (
                daughter.size,
                energy.size,
                ext_energy.size,
            ):
                raise ValueError(
                    "External pair response shape does not match its grids"
                )
            if not _contains_tracer((ext_energy, response)) and not np.array_equal(
                ext_energy, response["external_target_photon_energy_erg"]
            ):
                raise ValueError(
                    "External pair response belongs to a different target grid"
                )
    # All-angle daughter support for the distinct internal/external populations.
    x, y = energy[:, None] / _MEC2_CGS, ext_energy[None, :] / _MEC2_CGS
    product = x * y
    total = x + y
    invariant = jnp.minimum(jnp.maximum(total / 2, 1.0), jnp.maximum(product, 1.0))
    width = jnp.sqrt(1 - 1 / invariant) * jnp.sqrt(
        jnp.maximum((x - y) ** 2 + 4 * (product - invariant), 0.0)
    )
    lower = jnp.maximum(
        (4 * invariant + total**2 / invariant - 4) / (2 * (total + width)), 1.0
    )
    upper = (total + width) / 2
    padding = 64 * jnp.finfo(jnp.float64).eps
    ext_lower_mask = (product > 1) & (
        lower <= daughter[0] + padding * jnp.maximum(lower, daughter[0])
    )
    ext_upper_mask = (product > 1) & (
        upper >= daughter[-1] - padding * jnp.maximum(total, daughter[-1])
    )
    proton_sed = synchrotron_sed_jax(
        energy,
        proton_gamma,
        proton_density_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_P_CGS,
        charge_esu=_E_ESU,
    )
    meson = (
        {}
        if meson_synchrotron_intrinsic_seds_erg_s is None
        else meson_synchrotron_intrinsic_seds_erg_s
    )
    pion_sed = jnp.asarray(meson.get("pion_total", jnp.zeros_like(energy)))
    muon_sed = jnp.asarray(meson.get("muon_total", jnp.zeros_like(energy)))
    pi0_sed = jnp.asarray(pi0_decay_photon_injection_cm3_s_erg) * volume * energy**2
    zero_pair = jnp.zeros_like(gamma)
    initial_e = (
        zero_pair
        if initial_gamma_gamma_electron_injection_dgamma_cm3_s is None
        else jnp.asarray(initial_gamma_gamma_electron_injection_dgamma_cm3_s)
    )
    initial_p = (
        zero_pair
        if initial_gamma_gamma_positron_injection_dgamma_cm3_s is None
        else jnp.asarray(initial_gamma_gamma_positron_injection_dgamma_cm3_s)
    )

    def photon_residual(new, old, energy_weight):
        measure = weights * energy_weight
        numerator = jnp.sum(jnp.abs(new - old) * measure, axis=-1)
        denominator = jnp.maximum(
            jnp.sum(jnp.abs(new) * measure, axis=-1),
            jnp.sum(jnp.abs(old) * measure, axis=-1),
        )
        return jnp.max(
            jnp.where(
                denominator > 0.0,
                numerator / jnp.where(denominator > 0.0, denominator, 1.0),
                0.0,
            )
        )

    def evaluate(pair_e, pair_p, components):
        density = jnp.sum(components, axis=0)
        internal_ic_cooling = (
            electron_inverse_compton_loss_rate_dgamma_s_jax(gamma, energy, density)
            if enable_ssc
            else zero_pair
        )
        cooling = ext_cooling + fixed_cooling + internal_ic_cooling
        transport = total_lepton_transport_chain_jax(
            gamma,
            primary_electron_injection_dgamma_cm3_s,
            primary_positron_injection_dgamma_cm3_s,
            muon_decay_electron_injection_dgamma_cm3_s,
            muon_decay_positron_injection_dgamma_cm3_s,
            bethe_heitler_electron_injection_dgamma_cm3_s,
            bethe_heitler_positron_injection_dgamma_cm3_s,
            pair_e,
            pair_p,
            magnetic_field_G,
            adiabatic_rate_s_inv,
            escape_rate_s_inv,
            inverse_compton_loss_rate_dgamma_s=cooling,
            additional_continuous_loss_rate_dgamma_s=additional_continuous_loss_rate_dgamma_s,
        )
        leptons = (
            transport["electron_density_dgamma_cm3"]
            + transport["positron_density_dgamma_cm3"]
        )
        electron_sed = electron_synchrotron_sed_jax(
            energy,
            gamma,
            leptons,
            magnetic_field_G=magnetic_field_G,
            radius_cm=radius_cm,
        )
        ssa = synchrotron_absorption_coefficient_jax(
            energy,
            gamma,
            leptons,
            magnetic_field_G=magnetic_field_G,
            particle_mass_g=_M_E_CGS,
            charge_esu=_E_ESU,
        )
        internal_ic = (
            inverse_compton_sed_jax(
                energy, gamma, leptons, energy, density, radius_cm=radius_cm
            )
            if enable_ssc
            else jnp.zeros_like(energy)
        )
        ec = (
            inverse_compton_sed_jax(
                energy, gamma, leptons, ext_energy, ec_density, radius_cm=radius_cm
            )
            if enable_ec
            else jnp.zeros_like(energy)
        )
        sources = jnp.stack(
            (electron_sed, proton_sed, pion_sed, muon_sed, internal_ic, ec, pi0_sed)
        )
        collisions = gamma_gamma_total_field_jax(
            density if enable_internal_gamma_gamma else jnp.zeros_like(density),
            response,
        )
        ext_q = (
            _contract_pair_response(
                external_gamma_gamma_pair_response, density, ext_target
            )[0]
            if include_external_gamma_gamma_target
            else jnp.zeros_like(daughter)
        )
        high_q = collisions["electron_pair_injection_dgamma_cm3_s"] + ext_q
        raw_pair = (
            high_q
            if remap is None
            else _conservative_pair_source_remap_jax(high_q, remap)
        )
        gg_alpha = collisions["absorption_coefficient_cm1"] + ext_alpha
        total_alpha = ssa + gg_alpha
        mapped = _internal_photon_density_kernel(
            energy, sources, total_alpha, radius_cm
        )
        transfer = _sphere_attenuation_factor_jax(2.0 * radius_cm * total_alpha)
        escaped = sources * transfer
        ext_loss = _C_CGS * ext_alpha * density
        ext_events = jnp.sum(ext_loss * weights)
        ext_reservoir_power = _C_CGS * jnp.sum(
            density * weights * (ext_kernel @ (ext_target * ext_weights * ext_energy))
        )
        number_residual = photon_residual(mapped, components, 1.0)
        energy_residual = photon_residual(mapped, components, energy)
        pair_residual = jnp.maximum(
            _symmetric_relative_l1(raw_pair, pair_e, gamma),
            _symmetric_relative_l1(raw_pair, pair_p, gamma),
        )
        finite = (
            jnp.all(jnp.isfinite(mapped))
            & jnp.all(mapped >= 0)
            & jnp.all(jnp.isfinite(raw_pair))
            & jnp.all(jnp.isfinite(leptons))
            # A positive gamma-gamma opacity must not hide unsupported maser SSA.
            & jnp.all(jnp.isfinite(ssa))
            & jnp.all(ssa >= 0)
            & jnp.all(total_alpha >= 0)
        )
        return {
            "transport": transport,
            "components": components,
            "mapped_components": mapped,
            "sources": sources,
            "escaped_components": escaped,
            "total_density": density,
            "ssa": ssa,
            "gg_alpha": gg_alpha,
            "total_alpha": total_alpha,
            "transfer": transfer,
            "collisions": collisions,
            "external_q": ext_q,
            "external_loss": ext_loss,
            "external_events": ext_events,
            "external_reservoir_power": ext_reservoir_power,
            "pair_e": pair_e,
            "pair_p": pair_p,
            "raw_pair": raw_pair,
            "high_q": high_q,
            "internal_ic_cooling": internal_ic_cooling,
            "total_cooling": cooling,
            "photon_number_residual": number_residual,
            "photon_energy_residual": energy_residual,
            "pair_residual": pair_residual,
            "finite": finite,
        }

    def residuals(state, previous):
        electron = _symmetric_relative_l1(
            state["transport"]["electron_density_dgamma_cm3"],
            previous["transport"]["electron_density_dgamma_cm3"],
            gamma,
        )
        positron = _symmetric_relative_l1(
            state["transport"]["positron_density_dgamma_cm3"],
            previous["transport"]["positron_density_dgamma_cm3"],
            gamma,
        )
        cooling = _symmetric_relative_l1(
            state["internal_ic_cooling"], previous["internal_ic_cooling"], gamma
        )
        maximum = jnp.max(
            jnp.stack(
                (
                    electron,
                    positron,
                    cooling,
                    state["pair_residual"],
                    state["photon_number_residual"],
                    state["photon_energy_residual"],
                )
            )
        )
        return {
            "cascade": jnp.where(state["finite"], maximum, jnp.inf),
            "electron_density": electron,
            "positron_density": positron,
            "pair_fixed_point": state["pair_residual"],
            "photon_map_number": state["photon_number_residual"],
            "photon_map_energy": state["photon_energy_residual"],
            "internal_ic_cooling": cooling,
        }

    first = evaluate(
        initial_e,
        initial_p,
        jnp.zeros((len(_PHOTON_COMPONENTS), energy.size), dtype=jnp.float64),
    )
    first_residual = residuals(first, jax.tree_util.tree_map(jnp.zeros_like, first))
    carry = (
        first,
        first_residual,
        first_residual["cascade"] < tolerance,
        jnp.asarray(1, dtype=jnp.int32),
    )

    def step(carry, _):
        def active(previous):
            state, _, _, count = previous
            new = evaluate(
                (1.0 - omega) * state["pair_e"] + omega * state["raw_pair"],
                (1.0 - omega) * state["pair_p"] + omega * state["raw_pair"],
                (1.0 - omega) * state["components"]
                + omega * state["mapped_components"],
            )
            defect = residuals(new, state)
            return (new, defect, defect["cascade"] < tolerance, count + 1)

        following = jax.lax.cond(carry[2], lambda x: x, active, carry)
        return following, following[1]

    (state, defects, converged, count), history_tail = jax.lax.scan(
        step, carry, None, length=int(maximum_iterations) - 1
    )
    history = {
        k: jnp.concatenate((first_residual[k][None], history_tail[k])) for k in defects
    }
    transport = state["transport"]
    # Synchrotron-seed diagnostics only: never add these to internal_ic again.
    synchrotron_seed = jnp.sum(state["components"][:4], axis=0)
    final_leptons = (
        transport["electron_density_dgamma_cm3"]
        + transport["positron_density_dgamma_cm3"]
    )
    ssc_diagnostic = (
        inverse_compton_sed_jax(
            energy, gamma, final_leptons, energy, synchrotron_seed, radius_cm=radius_cm
        )
        if enable_ssc
        else jnp.zeros_like(energy)
    )
    ssc_cooling_diagnostic = (
        electron_inverse_compton_loss_rate_dgamma_s_jax(gamma, energy, synchrotron_seed)
        if enable_ssc
        else zero_pair
    )
    collision = state["collisions"]
    sources = state["sources"]
    escaped = state["escaped_components"]
    total_source = jnp.sum(sources, axis=0)
    total_escape = jnp.sum(escaped, axis=0)
    photon_loss = collision["photon_loss_rate_cm3_s_erg"] + state["external_loss"]
    qnumber, qmoment = _pair_source_moments(state["high_q"], daughter)
    mapped_number, mapped_moment = _pair_source_moments(state["raw_pair"], gamma)
    events = collision["event_rate_cm3_s"] + state["external_events"]
    external_internal_power = jnp.sum(state["external_loss"] * weights * energy)
    incident_power = (
        collision["photon_loss_power_erg_cm3_s"]
        + external_internal_power
        + state["external_reservoir_power"]
    )
    pair_power = 2.0 * _MEC2_CGS * qmoment
    transfer_balance = (
        total_source
        - total_escape
        - volume
        * energy**2
        * _C_CGS
        * state["total_alpha"]
        * jnp.sum(state["mapped_components"], axis=0)
    )
    ledger = {
        "photon_transfer_residual": _safe_ratio(
            jnp.sum(jnp.abs(transfer_balance)), jnp.sum(jnp.abs(total_source))
        ),
        "photon_map_number_residual": defects["photon_map_number"],
        "photon_map_energy_residual": defects["photon_map_energy"],
        "internal_photon_absorption_residual": _symmetric_relative_l1(
            photon_loss, _C_CGS * state["gg_alpha"] * state["total_density"], energy
        ),
        "pair_number_relative_residual": _safe_ratio(qnumber - events, events),
        "pair_energy_relative_residual": _safe_ratio(
            pair_power - incident_power, incident_power
        ),
        "internal_event_rate_cm3_s": collision["event_rate_cm3_s"],
        "external_event_rate_cm3_s": state["external_events"],
        "photon_loss_power_erg_cm3_s": collision["photon_loss_power_erg_cm3_s"]
        + external_internal_power,
        "external_reservoir_loss_power_erg_cm3_s": state["external_reservoir_power"],
        "pair_injected_energy_rate_erg_cm3_s": pair_power,
        "electron_pair_remap_number_residual": _symmetric_scalar_residual(
            mapped_number, qnumber
        ),
        "positron_pair_remap_number_residual": _symmetric_scalar_residual(
            mapped_number, qnumber
        ),
        "electron_pair_remap_energy_residual": _symmetric_scalar_residual(
            mapped_moment, qmoment
        ),
        "positron_pair_remap_energy_residual": _symmetric_scalar_residual(
            mapped_moment, qmoment
        ),
    }
    ext_cells = (
        _C_CGS
        * ext_kernel
        * (state["total_density"] * weights)[:, None]
        * (ext_target * ext_weights)[None, :]
    )
    ext_outside_n = jnp.sum(ext_cells * (ext_lower_mask | ext_upper_mask))
    ext_incident_energy = energy[:, None] + ext_energy[None, :]
    ext_outside_power = jnp.sum(
        ext_cells
        * jnp.minimum(
            ext_lower_mask
            * jnp.minimum(2 * _MEC2_CGS * daughter[0], ext_incident_energy)
            + ext_upper_mask * ext_incident_energy,
            ext_incident_energy,
        )
    )
    ext_pair_number, ext_pair_moment = _pair_source_moments(
        state["external_q"], daughter
    )
    external_reservoir_loss = (
        _C_CGS * ext_target * (ext_kernel.T @ (state["total_density"] * weights))
    )
    external_budget = {
        "event_rate_cm3_s": state["external_events"],
        "reservoir_photon_loss_rate_cm3_s_erg": external_reservoir_loss,
        "reservoir_photon_loss_number_rate_cm3_s": jnp.sum(
            external_reservoir_loss * ext_weights
        ),
        "internal_photon_loss_power_erg_cm3_s": external_internal_power,
        "reservoir_photon_loss_power_erg_cm3_s": state["external_reservoir_power"],
        "pair_injection_power_erg_cm3_s": 2 * _MEC2_CGS * ext_pair_moment,
        "outside_per_charge_number_rate_bound_cm3_s": ext_outside_n,
        "outside_pair_power_bound_erg_cm3_s": ext_outside_power,
        "outside_number_fraction_bound": _safe_ratio(
            ext_outside_n, state["external_events"]
        ),
        "outside_energy_fraction_bound": _safe_ratio(
            ext_outside_power,
            external_internal_power + state["external_reservoir_power"],
        ),
        "pair_number_relative_residual": _safe_ratio(
            ext_pair_number - state["external_events"], state["external_events"]
        ),
        "pair_energy_relative_residual": _safe_ratio(
            2 * _MEC2_CGS * ext_pair_moment
            - external_internal_power
            - state["external_reservoir_power"],
            external_internal_power + state["external_reservoir_power"],
        ),
    }
    component_dict = lambda array: {
        name: array[i] for i, name in enumerate(_PHOTON_COMPONENTS)
    }
    result = {
        "photon_energy_erg": energy,
        "internal_photon_density_cm3_erg": state["total_density"],
        "external_photon_energy_erg": ext_energy,
        "external_photon_density_cm3_erg": ext_density,
        "photon_components_internal_density_cm3_erg": component_dict(
            state["components"]
        ),
        "photon_components_intrinsic_sed_erg_s": component_dict(sources),
        "photon_components_escaped_sed_erg_s": component_dict(escaped),
        "photon_transfer_factor": state["transfer"],
        "photon_absorption_coefficient_cm1": state["total_alpha"],
        "intrinsic_photon_sed_erg_s": total_source,
        "escaped_photon_sed_erg_s": total_escape,
        "photon_loss_rate_cm3_s_erg": photon_loss,
        "ssa_photon_loss_rate_cm3_s_erg": _C_CGS
        * state["ssa"]
        * state["total_density"],
        "internal_synchrotron_photon_density_cm3_erg": synchrotron_seed,
        "internal_ic_seed_photon_density_cm3_erg": state["total_density"],
        "ssc_intrinsic_sed_erg_s": ssc_diagnostic,
        "total_synchrotron_intrinsic_sed_erg_s": jnp.sum(sources[:4], axis=0),
        "electron_synchrotron_absorption_coefficient_cm1": state["ssa"],
        "total_lepton_density_dgamma_cm3": transport["electron_density_dgamma_cm3"]
        + transport["positron_density_dgamma_cm3"],
        "electron_density_dgamma_cm3": transport["electron_density_dgamma_cm3"],
        "positron_density_dgamma_cm3": transport["positron_density_dgamma_cm3"],
        "electron_total_injection_dgamma_cm3_s": transport[
            "total_electron_injection_dgamma_cm3_s"
        ],
        "positron_total_injection_dgamma_cm3_s": transport[
            "total_positron_injection_dgamma_cm3_s"
        ],
        "electron_escape_dgamma_cm3_s": transport["electron_escape_dgamma_cm3_s"],
        "positron_escape_dgamma_cm3_s": transport["positron_escape_dgamma_cm3_s"],
        "electron_pair_injection_dgamma_cm3_s": state["pair_e"],
        "positron_pair_injection_dgamma_cm3_s": state["pair_p"],
        "raw_next_electron_pair_injection_dgamma_cm3_s": state["raw_pair"],
        "raw_next_positron_pair_injection_dgamma_cm3_s": state["raw_pair"],
        "high_resolution_electron_pair_injection_dgamma_cm3_s": state["high_q"],
        "high_resolution_positron_pair_injection_dgamma_cm3_s": state["high_q"],
        "high_resolution_total_pair_injection_dgamma_cm3_s": 2 * state["high_q"],
        "pair_daughter_gamma": daughter,
        "internal_gamma_gamma_budget": collision,
        "external_gamma_gamma_budget": external_budget,
        "external_gamma_gamma_event_rate_cm3_s": state["external_events"],
        "external_reservoir_loss_power_erg_cm3_s": state["external_reservoir_power"],
        "external_target_electron_pair_injection_dgamma_cm3_s": state["external_q"],
        "external_target_positron_pair_injection_dgamma_cm3_s": state["external_q"],
        "internal_target_electron_pair_injection_dgamma_cm3_s": collision[
            "electron_pair_injection_dgamma_cm3_s"
        ],
        "internal_target_positron_pair_injection_dgamma_cm3_s": collision[
            "positron_pair_injection_dgamma_cm3_s"
        ],
        "gamma_gamma_absorption_coefficient_cm1": state["gg_alpha"],
        "gamma_gamma_internal_target_absorption_coefficient_cm1": collision[
            "absorption_coefficient_cm1"
        ],
        "gamma_gamma_external_target_absorption_coefficient_cm1": ext_alpha,
        "gamma_gamma_total_absorption_coefficient_cm1": state["gg_alpha"],
        "gamma_gamma_optical_depth": 2 * radius_cm * state["gg_alpha"],
        "escaped_gamma_sed_erg_s": total_escape,
        "total_gamma_intrinsic_sed_erg_s": total_source,
        "intrinsic_bolometric_luminosity_erg_s": jnp.trapezoid(
            total_source, x=jnp.log(energy)
        ),
        "internal_inverse_compton_loss_rate_dgamma_s": state["internal_ic_cooling"],
        "ssc_inverse_compton_loss_rate_dgamma_s": ssc_cooling_diagnostic,
        "external_inverse_compton_loss_rate_dgamma_s": ext_cooling,
        "additional_fixed_inverse_compton_loss_rate_dgamma_s": fixed_cooling,
        "total_inverse_compton_loss_rate_dgamma_s": state["total_cooling"],
        "include_external_gamma_gamma_target": jnp.asarray(
            include_external_gamma_gamma_target
        ),
        "enable_internal_gamma_gamma": jnp.asarray(enable_internal_gamma_gamma),
        "enable_ssc": jnp.asarray(enable_ssc),
        "enable_internal_ic": jnp.asarray(enable_ssc),
        "enable_ec": jnp.asarray(enable_ec),
        "converged": converged,
        "iteration_limit_reached": ~converged,
        "iteration_count": count,
        "maximum_iterations": jnp.asarray(maximum_iterations),
        "relative_tolerance": tolerance,
        "relaxation_omega": omega,
        "final_cascade_residual": defects["cascade"],
        "final_electron_density_residual": defects["electron_density"],
        "final_positron_density_residual": defects["positron_density"],
        "final_pair_fixed_point_residual": defects["pair_fixed_point"],
        "final_photon_map_number_residual": defects["photon_map_number"],
        "final_photon_map_energy_residual": defects["photon_map_energy"],
        "residual_history": history["cascade"],
        "residual_histories": history,
        "ledger_diagnostics": ledger,
    }
    for i, name in enumerate(_PHOTON_COMPONENTS):
        result[name + "_intrinsic_sed_erg_s"] = sources[i]
    result.update({key: value for key, value in ledger.items() if "remap" in key})
    return result
