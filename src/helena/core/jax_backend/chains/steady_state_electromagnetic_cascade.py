"""Minimal steady internal electron-photon cascade closure."""

from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
)
from helena.core.constants import (
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.contracts import real_1d as _real_1d
from helena.core.jax_backend.chains.blob_intrinsic_radiation import (
    _blob_intrinsic_radiation_with_high_synchrotron_jax,
)
from helena.core.jax_backend.chains.gamma_gamma_pair_source import (
    gamma_gamma_pair_source_chain_jax,
)
from helena.core.jax_backend.chains.total_lepton_transport import (
    total_lepton_transport_chain_jax,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    _angle_averaged_cross_section_jax,
    _contains_tracer,
)
from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
    prepare_gamma_gamma_pair_response_jax,
)
from helena.core.jax_backend.kernels.inverse_compton_cooling import (
    electron_inverse_compton_loss_rate_dgamma_s_jax,
)
from helena.core.jax_backend.radiation.inverse_compton import inverse_compton_sed_jax
from helena.core.jax_backend.radiation.photon_fields import (
    synchrotron_sed_to_internal_photon_density_jax,
)
from helena.core.jax_backend.radiation.ssa import _transfer_from_quantities

__all__ = ["steady_state_electromagnetic_cascade_jax"]


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


def _exact_collision_energy_rate(
    gamma_ray_energy_erg,
    internal_gamma_photon_density_cm3_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    gamma_ray_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    target_density = jnp.asarray(
        dn_target_photon_denergy_cm3_erg,
        dtype=jnp.float64,
    )
    y = gamma_ray_energy[:, None] * target_energy[None, :] / _MEC2_CGS**2
    exact_inner = jnp.trapezoid(
        target_energy[None, :]
        * target_density[None, :]
        * _angle_averaged_cross_section_jax(y)
        * (gamma_ray_energy[:, None] + target_energy[None, :]),
        x=jnp.log(target_energy),
        axis=1,
    )
    return _C_CGS * jnp.trapezoid(
        gamma_ray_energy * internal_gamma_photon_density_cm3_erg * exact_inner,
        x=jnp.log(gamma_ray_energy),
    )


def _iteration_map(
    electron_gamma,
    pair_daughter_gamma,
    pair_source_remap_operator,
    synchrotron_photon_energy_erg,
    gamma_ray_energy_erg,
    synchrotron_primary_split_energy_erg,
    primary_electron_injection_dgamma_cm3_s,
    primary_positron_injection_dgamma_cm3_s,
    muon_decay_electron_injection_dgamma_cm3_s,
    muon_decay_positron_injection_dgamma_cm3_s,
    bethe_heitler_electron_injection_dgamma_cm3_s,
    bethe_heitler_positron_injection_dgamma_cm3_s,
    electron_pair_injection_dgamma_cm3_s,
    positron_pair_injection_dgamma_cm3_s,
    ssc_inverse_compton_loss_rate_dgamma_s,
    proton_gamma,
    proton_density_dgamma_cm3,
    pi0_decay_photon_injection_cm3_s_erg,
    external_seed_photon_energy_erg,
    dn_external_seed_photon_denergy_cm3_erg,
    dn_external_compton_seed_photon_denergy_cm3_erg,
    include_external_gamma_gamma_target,
    enable_internal_gamma_gamma,
    enable_ssc,
    compose_proton_synchrotron_low_energy,
    magnetic_field_G,
    radius_cm,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
    external_inverse_compton_loss_rate_dgamma_s,
    additional_fixed_inverse_compton_loss_rate_dgamma_s,
    additional_continuous_loss_rate_dgamma_s,
    relaxation_omega,
    meson_synchrotron_low_energy_intrinsic_sed_erg_s=None,
    meson_synchrotron_intrinsic_sed_erg_s=None,
    gamma_gamma_pair_response=None,
    external_gamma_gamma_pair_response=None,
):
    if meson_synchrotron_low_energy_intrinsic_sed_erg_s is None:
        meson_synchrotron_low_energy_intrinsic_sed_erg_s = jnp.zeros_like(
            jnp.asarray(synchrotron_photon_energy_erg, dtype=jnp.float64)
        )
    if meson_synchrotron_intrinsic_sed_erg_s is None:
        meson_synchrotron_intrinsic_sed_erg_s = jnp.zeros_like(
            jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
        )
    ssc_inverse_compton_loss_rate_dgamma_s = (
        ssc_inverse_compton_loss_rate_dgamma_s
        if enable_ssc
        else jnp.zeros_like(ssc_inverse_compton_loss_rate_dgamma_s)
    )
    total_inverse_compton_loss = (
        external_inverse_compton_loss_rate_dgamma_s
        + additional_fixed_inverse_compton_loss_rate_dgamma_s
        + ssc_inverse_compton_loss_rate_dgamma_s
    )
    transport = total_lepton_transport_chain_jax(
        electron_gamma,
        primary_electron_injection_dgamma_cm3_s,
        primary_positron_injection_dgamma_cm3_s,
        muon_decay_electron_injection_dgamma_cm3_s,
        muon_decay_positron_injection_dgamma_cm3_s,
        bethe_heitler_electron_injection_dgamma_cm3_s,
        bethe_heitler_positron_injection_dgamma_cm3_s,
        electron_pair_injection_dgamma_cm3_s,
        positron_pair_injection_dgamma_cm3_s,
        magnetic_field_G,
        adiabatic_rate_s_inv,
        escape_rate_s_inv,
        inverse_compton_loss_rate_dgamma_s=total_inverse_compton_loss,
        additional_continuous_loss_rate_dgamma_s=(
            additional_continuous_loss_rate_dgamma_s
        ),
    )
    radiation = _blob_intrinsic_radiation_with_high_synchrotron_jax(
        synchrotron_photon_energy_erg,
        gamma_ray_energy_erg,
        electron_gamma,
        transport["electron_density_dgamma_cm3"],
        transport["positron_density_dgamma_cm3"],
        proton_gamma,
        proton_density_dgamma_cm3,
        pi0_decay_photon_injection_cm3_s_erg,
        external_seed_photon_energy_erg,
        dn_external_compton_seed_photon_denergy_cm3_erg,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        synchrotron_primary_split_energy_erg=(synchrotron_primary_split_energy_erg),
        compose_proton_synchrotron_low_energy=(compose_proton_synchrotron_low_energy),
    )
    low_energy_intrinsic_sed = (
        radiation["electron_synchrotron_intrinsic_sed_erg_s"]
        + meson_synchrotron_low_energy_intrinsic_sed_erg_s
        + (
            radiation["proton_synchrotron_low_energy_intrinsic_sed_erg_s"]
            if compose_proton_synchrotron_low_energy
            else 0.0
        )
    )
    internal_low_energy_density = synchrotron_sed_to_internal_photon_density_jax(
        synchrotron_photon_energy_erg,
        low_energy_intrinsic_sed,
        radiation["electron_synchrotron_absorption_coefficient_cm1"],
        radius_cm=radius_cm,
    )
    low_energy_escaped_sed = _transfer_from_quantities(
        radiation["electron_synchrotron_absorption_coefficient_cm1"],
        low_energy_intrinsic_sed,
        radius_cm,
    )
    ssc_intrinsic_sed = (
        inverse_compton_sed_jax(
            gamma_ray_energy_erg,
            electron_gamma,
            radiation["total_lepton_density_dgamma_cm3"],
            synchrotron_photon_energy_erg,
            internal_low_energy_density,
            radius_cm=radius_cm,
        )
        if enable_ssc
        else jnp.zeros_like(gamma_ray_energy_erg)
    )
    # Keep the full-grid component diagnostic, but let the high composition own
    # only nodes strictly above the canonical split; the low grid owns the split.
    ssc_high_energy_intrinsic_sed = jnp.where(
        jnp.asarray(gamma_ray_energy_erg)
        > radiation["synchrotron_split_energy_erg"],
        ssc_intrinsic_sed,
        0.0,
    )
    ssc_delta = (
        ssc_high_energy_intrinsic_sed - radiation["ssc_intrinsic_sed_erg_s"]
    )
    radiation = {
        **radiation,
        "meson_synchrotron_low_energy_intrinsic_sed_erg_s": (
            meson_synchrotron_low_energy_intrinsic_sed_erg_s
        ),
        "total_low_energy_synchrotron_intrinsic_sed_erg_s": (low_energy_intrinsic_sed),
        "total_low_energy_synchrotron_escaped_sed_erg_s": (low_energy_escaped_sed),
        "total_low_energy_synchrotron_primary_intrinsic_sed_erg_s": (
            radiation["electron_synchrotron_target_band_intrinsic_sed_erg_s"]
            + meson_synchrotron_low_energy_intrinsic_sed_erg_s
            + (
                radiation["proton_synchrotron_low_energy_intrinsic_sed_erg_s"]
                if compose_proton_synchrotron_low_energy
                else 0.0
            )
        ),
        "internal_synchrotron_photon_density_cm3_erg": (internal_low_energy_density),
        "ssc_intrinsic_sed_erg_s": ssc_intrinsic_sed,
        "ssc_high_energy_intrinsic_sed_erg_s": ssc_high_energy_intrinsic_sed,
        "total_leptonic_gamma_intrinsic_sed_erg_s": (
            radiation["total_leptonic_gamma_intrinsic_sed_erg_s"] + ssc_delta
        ),
        "meson_synchrotron_intrinsic_sed_erg_s": (
            meson_synchrotron_intrinsic_sed_erg_s
        ),
        "total_hadronic_gamma_intrinsic_sed_erg_s": (
            radiation["total_hadronic_gamma_intrinsic_sed_erg_s"]
            + meson_synchrotron_intrinsic_sed_erg_s
        ),
        "total_gamma_intrinsic_sed_erg_s": (
            radiation["total_gamma_intrinsic_sed_erg_s"]
            + ssc_delta
            + meson_synchrotron_intrinsic_sed_erg_s
        ),
    }
    pair_source = gamma_gamma_pair_source_chain_jax(
        gamma_ray_energy_erg,
        radiation["total_gamma_intrinsic_sed_erg_s"],
        synchrotron_photon_energy_erg,
        (
            radiation["internal_synchrotron_photon_density_cm3_erg"]
            if enable_internal_gamma_gamma
            else jnp.zeros_like(
                radiation["internal_synchrotron_photon_density_cm3_erg"]
            )
        ),
        pair_daughter_gamma,
        radius_cm=radius_cm,
        external_target_photon_energy_erg=external_seed_photon_energy_erg,
        dn_external_target_photon_denergy_cm3_erg=(
            dn_external_seed_photon_denergy_cm3_erg
        ),
        include_external_gamma_gamma_target=(include_external_gamma_gamma_target),
        internal_pair_response=gamma_gamma_pair_response,
        external_pair_response=external_gamma_gamma_pair_response,
    )
    high_resolution_raw_electron = pair_source["electron_pair_injection_dgamma_cm3_s"]
    high_resolution_raw_positron = pair_source["positron_pair_injection_dgamma_cm3_s"]
    if pair_source_remap_operator is None:
        raw_electron = high_resolution_raw_electron
        raw_positron = high_resolution_raw_positron
    else:
        raw_electron = _conservative_pair_source_remap_jax(
            high_resolution_raw_electron,
            pair_source_remap_operator,
        )
        raw_positron = _conservative_pair_source_remap_jax(
            high_resolution_raw_positron,
            pair_source_remap_operator,
        )
    raw_ssc = (
        electron_inverse_compton_loss_rate_dgamma_s_jax(
            electron_gamma,
            synchrotron_photon_energy_erg,
            radiation["internal_synchrotron_photon_density_cm3_erg"],
        )
        if enable_ssc
        else jnp.zeros_like(electron_gamma)
    )
    relaxed_electron = (
        1.0 - relaxation_omega
    ) * electron_pair_injection_dgamma_cm3_s + relaxation_omega * raw_electron
    relaxed_positron = (
        1.0 - relaxation_omega
    ) * positron_pair_injection_dgamma_cm3_s + relaxation_omega * raw_positron
    relaxed_ssc = (
        1.0 - relaxation_omega
    ) * ssc_inverse_compton_loss_rate_dgamma_s + relaxation_omega * raw_ssc
    return {
        "transport": transport,
        "radiation": radiation,
        "pair_source": pair_source,
        "electron_pair_injection_dgamma_cm3_s": (electron_pair_injection_dgamma_cm3_s),
        "positron_pair_injection_dgamma_cm3_s": (positron_pair_injection_dgamma_cm3_s),
        "external_inverse_compton_loss_rate_dgamma_s": (
            external_inverse_compton_loss_rate_dgamma_s
        ),
        "additional_fixed_inverse_compton_loss_rate_dgamma_s": (
            additional_fixed_inverse_compton_loss_rate_dgamma_s
        ),
        "ssc_inverse_compton_loss_rate_dgamma_s": (
            ssc_inverse_compton_loss_rate_dgamma_s
        ),
        "total_inverse_compton_loss_rate_dgamma_s": total_inverse_compton_loss,
        "raw_next_high_resolution_electron_pair_injection_dgamma_cm3_s": (
            high_resolution_raw_electron
        ),
        "raw_next_high_resolution_positron_pair_injection_dgamma_cm3_s": (
            high_resolution_raw_positron
        ),
        "raw_next_electron_pair_injection_dgamma_cm3_s": raw_electron,
        "raw_next_positron_pair_injection_dgamma_cm3_s": raw_positron,
        "mapped_raw_next_electron_pair_injection_dgamma_cm3_s": raw_electron,
        "mapped_raw_next_positron_pair_injection_dgamma_cm3_s": raw_positron,
        "raw_next_ssc_inverse_compton_loss_rate_dgamma_s": raw_ssc,
        "relaxed_next_electron_pair_injection_dgamma_cm3_s": (relaxed_electron),
        "relaxed_next_positron_pair_injection_dgamma_cm3_s": (relaxed_positron),
        "relaxed_next_ssc_inverse_compton_loss_rate_dgamma_s": relaxed_ssc,
    }


def _monitored_residuals(
    state,
    previous_state,
    electron_gamma,
    synchrotron_photon_energy_erg,
    gamma_ray_energy_erg,
):
    electron_density = _symmetric_relative_l1(
        state["transport"]["electron_density_dgamma_cm3"],
        previous_state["transport"]["electron_density_dgamma_cm3"],
        electron_gamma,
    )
    positron_density = _symmetric_relative_l1(
        state["transport"]["positron_density_dgamma_cm3"],
        previous_state["transport"]["positron_density_dgamma_cm3"],
        electron_gamma,
    )
    electron_pair_fixed_point = _symmetric_relative_l1(
        state["raw_next_electron_pair_injection_dgamma_cm3_s"],
        state["electron_pair_injection_dgamma_cm3_s"],
        electron_gamma,
    )
    positron_pair_fixed_point = _symmetric_relative_l1(
        state["raw_next_positron_pair_injection_dgamma_cm3_s"],
        state["positron_pair_injection_dgamma_cm3_s"],
        electron_gamma,
    )
    pair_fixed_point = jnp.maximum(
        electron_pair_fixed_point,
        positron_pair_fixed_point,
    )
    electron_pair_update = _symmetric_relative_l1(
        state["relaxed_next_electron_pair_injection_dgamma_cm3_s"],
        state["electron_pair_injection_dgamma_cm3_s"],
        electron_gamma,
    )
    positron_pair_update = _symmetric_relative_l1(
        state["relaxed_next_positron_pair_injection_dgamma_cm3_s"],
        state["positron_pair_injection_dgamma_cm3_s"],
        electron_gamma,
    )
    pair_update = jnp.maximum(electron_pair_update, positron_pair_update)
    ssc_cooling_fixed_point = _symmetric_relative_l1(
        state["raw_next_ssc_inverse_compton_loss_rate_dgamma_s"],
        state["ssc_inverse_compton_loss_rate_dgamma_s"],
        electron_gamma,
    )
    ssc_cooling_update = _symmetric_relative_l1(
        state["relaxed_next_ssc_inverse_compton_loss_rate_dgamma_s"],
        state["ssc_inverse_compton_loss_rate_dgamma_s"],
        electron_gamma,
    )
    internal_synchrotron = _symmetric_relative_l1(
        state["radiation"]["internal_synchrotron_photon_density_cm3_erg"],
        previous_state["radiation"]["internal_synchrotron_photon_density_cm3_erg"],
        synchrotron_photon_energy_erg,
    )
    escaped_gamma = _symmetric_relative_l1(
        state["pair_source"]["escaped_sed_erg_s"],
        previous_state["pair_source"]["escaped_sed_erg_s"],
        jnp.log(gamma_ray_energy_erg),
    )
    cascade = jnp.maximum(
        jnp.maximum(electron_density, positron_density),
        jnp.maximum(
            jnp.maximum(pair_fixed_point, ssc_cooling_fixed_point),
            jnp.maximum(internal_synchrotron, escaped_gamma),
        ),
    )
    return {
        "electron_density": electron_density,
        "positron_density": positron_density,
        "electron_pair_fixed_point": electron_pair_fixed_point,
        "positron_pair_fixed_point": positron_pair_fixed_point,
        "pair_fixed_point": pair_fixed_point,
        "electron_pair_update": electron_pair_update,
        "positron_pair_update": positron_pair_update,
        "pair_update": pair_update,
        "ssc_cooling_fixed_point": ssc_cooling_fixed_point,
        "ssc_cooling_update": ssc_cooling_update,
        "electron_pair_injection": electron_pair_fixed_point,
        "positron_pair_injection": positron_pair_fixed_point,
        "pair_injection": pair_fixed_point,
        "internal_synchrotron_photon_density": internal_synchrotron,
        "escaped_gamma_sed": escaped_gamma,
        "cascade": cascade,
    }


def _ledger_diagnostics(
    state,
    electron_gamma,
    pair_daughter_gamma,
    synchrotron_photon_energy_erg,
    gamma_ray_energy_erg,
    external_seed_photon_energy_erg,
    dn_external_seed_photon_denergy_cm3_erg,
    enable_internal_gamma_gamma,
    include_external_gamma_gamma_target,
    compose_proton_synchrotron_low_energy,
):
    transport = state["transport"]
    radiation = state["radiation"]
    pair_source = state["pair_source"]
    explicit_electron_source = (
        transport["primary_electron_injection_dgamma_cm3_s"]
        + transport["muon_decay_electron_injection_dgamma_cm3_s"]
        + transport["bethe_heitler_electron_injection_dgamma_cm3_s"]
        + state["electron_pair_injection_dgamma_cm3_s"]
    )
    explicit_positron_source = (
        transport["primary_positron_injection_dgamma_cm3_s"]
        + transport["muon_decay_positron_injection_dgamma_cm3_s"]
        + transport["bethe_heitler_positron_injection_dgamma_cm3_s"]
        + state["positron_pair_injection_dgamma_cm3_s"]
    )
    electron_source_residual = _symmetric_relative_l1(
        transport["total_electron_injection_dgamma_cm3_s"],
        explicit_electron_source,
        electron_gamma,
    )
    positron_source_residual = _symmetric_relative_l1(
        transport["total_positron_injection_dgamma_cm3_s"],
        explicit_positron_source,
        electron_gamma,
    )

    gamma_coordinate = jnp.log(gamma_ray_energy_erg)
    explicit_intrinsic_gamma = (
        radiation["electron_synchrotron_high_energy_intrinsic_sed_erg_s"]
        + radiation["ssc_high_energy_intrinsic_sed_erg_s"]
        + radiation["external_compton_intrinsic_sed_erg_s"]
        + (
            (
                radiation["proton_synchrotron_high_energy_intrinsic_sed_erg_s"]
                if compose_proton_synchrotron_low_energy
                else radiation["proton_synchrotron_intrinsic_sed_erg_s"]
            )
            + radiation["pi0_intrinsic_sed_erg_s"]
            + radiation["meson_synchrotron_intrinsic_sed_erg_s"]
        )
    )
    intrinsic_gamma_residual = _symmetric_relative_l1(
        radiation["total_gamma_intrinsic_sed_erg_s"],
        explicit_intrinsic_gamma,
        gamma_coordinate,
    )
    transfer_residual = _symmetric_relative_l1(
        pair_source["intrinsic_sed_erg_s"],
        pair_source["escaped_sed_erg_s"] + pair_source["removed_sed_erg_s"],
        gamma_coordinate,
    )

    volume = pair_source["volume_cm3"]
    gamma_ray_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
    denominator = volume * gamma_ray_energy**2
    production_rate = pair_source["intrinsic_sed_erg_s"] / denominator
    escaped_rate = pair_source["escaped_sed_erg_s"] / denominator
    absorbed_rate = pair_source["absorbed_photon_rate_cm3_s_erg"]
    internal_production_residual = _symmetric_relative_l1(
        production_rate,
        escaped_rate + absorbed_rate,
        gamma_ray_energy,
    )
    local_absorption_rate = (
        _C_CGS
        * pair_source["absorption_coefficient_cm1"]
        * pair_source["internal_gamma_photon_density_cm3_erg"]
    )
    internal_absorption_residual = _symmetric_relative_l1(
        absorbed_rate,
        local_absorption_rate,
        gamma_ray_energy,
    )

    high_resolution_electron_pair = state[
        "raw_next_high_resolution_electron_pair_injection_dgamma_cm3_s"
    ]
    high_resolution_positron_pair = state[
        "raw_next_high_resolution_positron_pair_injection_dgamma_cm3_s"
    ]
    mapped_electron_pair = state["mapped_raw_next_electron_pair_injection_dgamma_cm3_s"]
    mapped_positron_pair = state["mapped_raw_next_positron_pair_injection_dgamma_cm3_s"]
    electron_event_rate, electron_energy_moment = _pair_source_moments(
        high_resolution_electron_pair,
        pair_daughter_gamma,
    )
    positron_event_rate, positron_energy_moment = _pair_source_moments(
        high_resolution_positron_pair,
        pair_daughter_gamma,
    )
    mapped_electron_event_rate, mapped_electron_energy_moment = _pair_source_moments(
        mapped_electron_pair,
        electron_gamma,
    )
    mapped_positron_event_rate, mapped_positron_energy_moment = _pair_source_moments(
        mapped_positron_pair,
        electron_gamma,
    )
    electron_pair_remap_number_residual = _symmetric_scalar_residual(
        mapped_electron_event_rate,
        electron_event_rate,
    )
    positron_pair_remap_number_residual = _symmetric_scalar_residual(
        mapped_positron_event_rate,
        positron_event_rate,
    )
    electron_pair_remap_energy_residual = _symmetric_scalar_residual(
        mapped_electron_energy_moment,
        electron_energy_moment,
    )
    positron_pair_remap_energy_residual = _symmetric_scalar_residual(
        mapped_positron_energy_moment,
        positron_energy_moment,
    )
    absorbed_event_rate = jnp.trapezoid(
        gamma_ray_energy * absorbed_rate,
        x=gamma_coordinate,
    )
    internal_target_absorbed_event_rate = _C_CGS * jnp.trapezoid(
        gamma_ray_energy
        * pair_source["internal_gamma_photon_density_cm3_erg"]
        * pair_source["gamma_gamma_internal_target_absorption_coefficient_cm1"],
        x=gamma_coordinate,
    )
    external_target_absorbed_event_rate = _C_CGS * jnp.trapezoid(
        gamma_ray_energy
        * pair_source["internal_gamma_photon_density_cm3_erg"]
        * pair_source["gamma_gamma_external_target_absorption_coefficient_cm1"],
        x=gamma_coordinate,
    )
    absorption_decomposition_absolute_residual = jnp.abs(
        absorbed_event_rate
        - internal_target_absorbed_event_rate
        - external_target_absorbed_event_rate
    )
    absorption_decomposition_scale = jnp.maximum(
        jnp.maximum(
            jnp.abs(absorbed_event_rate),
            jnp.abs(
                internal_target_absorbed_event_rate
                + external_target_absorbed_event_rate
            ),
        ),
        jnp.finfo(jnp.float64).tiny,
    )
    absorption_decomposition_relative_residual = (
        absorption_decomposition_absolute_residual / absorption_decomposition_scale
    )
    electron_event_residual = _symmetric_scalar_residual(
        electron_event_rate,
        absorbed_event_rate,
    )
    positron_event_residual = _symmetric_scalar_residual(
        positron_event_rate,
        absorbed_event_rate,
    )
    pair_multiplicity_residual = _symmetric_scalar_residual(
        electron_event_rate + positron_event_rate,
        2.0 * absorbed_event_rate,
    )

    internal_electron_event_rate, internal_electron_energy_moment = (
        _pair_source_moments(
            pair_source["internal_target_electron_pair_injection_dgamma_cm3_s"],
            pair_daughter_gamma,
        )
    )
    internal_positron_event_rate, internal_positron_energy_moment = (
        _pair_source_moments(
            pair_source["internal_target_positron_pair_injection_dgamma_cm3_s"],
            pair_daughter_gamma,
        )
    )
    external_electron_event_rate, external_electron_energy_moment = (
        _pair_source_moments(
            pair_source["external_target_electron_pair_injection_dgamma_cm3_s"],
            pair_daughter_gamma,
        )
    )
    external_positron_event_rate, external_positron_energy_moment = (
        _pair_source_moments(
            pair_source["external_target_positron_pair_injection_dgamma_cm3_s"],
            pair_daughter_gamma,
        )
    )
    internal_pair_injected_energy_rate = _MEC2_CGS * (
        internal_electron_energy_moment + internal_positron_energy_moment
    )
    external_pair_injected_energy_rate = _MEC2_CGS * (
        external_electron_energy_moment + external_positron_energy_moment
    )
    internal_exact_collision_energy_rate = _exact_collision_energy_rate(
        gamma_ray_energy,
        pair_source["internal_gamma_photon_density_cm3_erg"],
        synchrotron_photon_energy_erg,
        (
            radiation["internal_synchrotron_photon_density_cm3_erg"]
            if enable_internal_gamma_gamma
            else jnp.zeros_like(
                radiation["internal_synchrotron_photon_density_cm3_erg"]
            )
        ),
    )
    external_target_density = jnp.where(
        include_external_gamma_gamma_target,
        jnp.asarray(
            dn_external_seed_photon_denergy_cm3_erg,
            dtype=jnp.float64,
        ),
        0.0,
    )
    external_exact_collision_energy_rate = _exact_collision_energy_rate(
        gamma_ray_energy,
        pair_source["internal_gamma_photon_density_cm3_erg"],
        external_seed_photon_energy_erg,
        external_target_density,
    )
    exact_collision_energy_rate = (
        internal_exact_collision_energy_rate + external_exact_collision_energy_rate
    )
    pair_injected_energy_rate = _MEC2_CGS * (
        electron_energy_moment + positron_energy_moment
    )
    mapped_pair_injected_energy_rate = _MEC2_CGS * (
        mapped_electron_energy_moment + mapped_positron_energy_moment
    )
    pair_energy_capture_fraction = _safe_ratio(
        pair_injected_energy_rate,
        exact_collision_energy_rate,
    )
    zero_pair_energy = (pair_injected_energy_rate == 0.0) & (
        exact_collision_energy_rate == 0.0
    )
    pair_energy_deficit_fraction = jnp.where(
        zero_pair_energy,
        0.0,
        jnp.abs(1.0 - pair_energy_capture_fraction),
    )
    pair_energy_symmetric_residual = jnp.where(
        zero_pair_energy,
        0.0,
        2.0
        * jnp.abs(1.0 - pair_energy_capture_fraction)
        / (1.0 + pair_energy_capture_fraction),
    )
    pair_energy_max_relative_residual = _symmetric_scalar_residual(
        pair_injected_energy_rate,
        exact_collision_energy_rate,
    )
    inverse_compton_loss_composition_residual = _symmetric_relative_l1(
        state["total_inverse_compton_loss_rate_dgamma_s"],
        (
            state["external_inverse_compton_loss_rate_dgamma_s"]
            + state["additional_fixed_inverse_compton_loss_rate_dgamma_s"]
            + state["ssc_inverse_compton_loss_rate_dgamma_s"]
        ),
        electron_gamma,
    )
    return {
        "electron_source_composition_residual": electron_source_residual,
        "positron_source_composition_residual": positron_source_residual,
        "intrinsic_gamma_composition_residual": intrinsic_gamma_residual,
        "gamma_gamma_transfer_residual": transfer_residual,
        "internal_photon_production_residual": internal_production_residual,
        "internal_photon_absorption_residual": internal_absorption_residual,
        "absorbed_gamma_event_rate_cm3_s": absorbed_event_rate,
        "internal_target_absorbed_gamma_event_rate_cm3_s": (
            internal_target_absorbed_event_rate
        ),
        "external_target_absorbed_gamma_event_rate_cm3_s": (
            external_target_absorbed_event_rate
        ),
        "absolute_absorption_decomposition_residual_cm3_s": (
            absorption_decomposition_absolute_residual
        ),
        "relative_absorption_decomposition_residual": (
            absorption_decomposition_relative_residual
        ),
        "absorption_decomposition_normalization_scale_cm3_s": (
            absorption_decomposition_scale
        ),
        "internal_target_electron_pair_event_rate_cm3_s": (
            internal_electron_event_rate
        ),
        "internal_target_positron_pair_event_rate_cm3_s": (
            internal_positron_event_rate
        ),
        "external_target_electron_pair_event_rate_cm3_s": (
            external_electron_event_rate
        ),
        "external_target_positron_pair_event_rate_cm3_s": (
            external_positron_event_rate
        ),
        "internal_target_pair_event_rate_cm3_s": (
            0.5 * (internal_electron_event_rate + internal_positron_event_rate)
        ),
        "external_target_pair_event_rate_cm3_s": (
            0.5 * (external_electron_event_rate + external_positron_event_rate)
        ),
        "electron_pair_event_rate_cm3_s": electron_event_rate,
        "positron_pair_event_rate_cm3_s": positron_event_rate,
        "mapped_electron_pair_event_rate_cm3_s": mapped_electron_event_rate,
        "mapped_positron_pair_event_rate_cm3_s": mapped_positron_event_rate,
        "electron_pair_event_residual": electron_event_residual,
        "positron_pair_event_residual": positron_event_residual,
        "pair_multiplicity": _safe_ratio(
            electron_event_rate + positron_event_rate,
            absorbed_event_rate,
        ),
        "pair_multiplicity_residual": pair_multiplicity_residual,
        "pair_injected_energy_rate_erg_cm3_s": pair_injected_energy_rate,
        "internal_target_pair_injected_energy_rate_erg_cm3_s": (
            internal_pair_injected_energy_rate
        ),
        "external_target_pair_injected_energy_rate_erg_cm3_s": (
            external_pair_injected_energy_rate
        ),
        "mapped_pair_injected_energy_rate_erg_cm3_s": (
            mapped_pair_injected_energy_rate
        ),
        "exact_collision_energy_rate_erg_cm3_s": (exact_collision_energy_rate),
        "internal_target_exact_collision_energy_rate_erg_cm3_s": (
            internal_exact_collision_energy_rate
        ),
        "external_target_exact_collision_energy_rate_erg_cm3_s": (
            external_exact_collision_energy_rate
        ),
        "pair_energy_capture_fraction": pair_energy_capture_fraction,
        "pair_energy_deficit_fraction": pair_energy_deficit_fraction,
        "pair_energy_symmetric_residual": pair_energy_symmetric_residual,
        "pair_energy_max_relative_residual": pair_energy_max_relative_residual,
        "pair_energy_residual": pair_energy_max_relative_residual,
        "electron_pair_remap_number_residual": (electron_pair_remap_number_residual),
        "positron_pair_remap_number_residual": (positron_pair_remap_number_residual),
        "electron_pair_remap_energy_residual": (electron_pair_remap_energy_residual),
        "positron_pair_remap_energy_residual": (positron_pair_remap_energy_residual),
        "inverse_compton_loss_composition_residual": (
            inverse_compton_loss_composition_residual
        ),
    }


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
    synchrotron_primary_split_energy_erg=None,
    compose_proton_synchrotron_low_energy=False,
    meson_synchrotron_low_energy_intrinsic_sed_erg_s=None,
    meson_synchrotron_intrinsic_sed_erg_s=None,
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
    """Return a fixed-proton steady internal electromagnetic cascade.

    Photon energies are strictly increasing blob-comoving grids in erg.
    ``electron_gamma`` and ``proton_gamma`` are increasing Lorentz-factor
    grids. Particle densities are ``dn/dgamma`` in cm^-3, lepton sources are
    ``dN/(dV dt dgamma)`` in cm^-3 s^-1, the pi0 photon source is
    ``dN/(dV dt dE)`` in cm^-3 s^-1 erg^-1. The optional meson synchrotron
    sources are blob-comoving intrinsic ``E L_E`` in erg s^-1 on the
    synchrotron and gamma-ray grids; ``None`` is exact zero. The low-grid
    source is added before the existing electron-SSA transfer, while the
    gamma-grid source enters gamma-gamma transfer. Photon densities are
    ``dn/dE`` in cm^-3 erg^-1. Magnetic field, radius, continuous loss, and
    local escape inputs use G, cm, ``b=-dgamma/dt`` in s^-1, and s^-1.

    Primary, muon-decay, Bethe--Heitler, proton, pi0, meson-synchrotron,
    external-field, caller-supplied additional loss, and escape inputs remain fixed. External
    inverse-Compton cooling is evaluated once from the same external photon
    grid and density used by the radiation chain. Each iteration adds it to
    ``additional_fixed_inverse_compton_loss_rate_dgamma_s`` and the current
    SSC cooling, transports electrons and positrons separately, evaluates the
    intrinsic radiation once, and updates both the pair source and SSC cooling
    from the current SSA-corrected synchrotron target. Electron/positron
    synchrotron is evaluated directly on both grids. By default,
    ``synchrotron_primary_split_energy_erg=None`` uses the final target-grid
    energy as before. An explicit finite positive value within the target-grid
    coverage independently sets the boundary above which gamma-grid
    synchrotron is a primary; the complete supplied target grid still
    participates in SSC and gamma-gamma calculations.

    This only removes the erroneous coupling between a numerical target-grid
    upper bound and the physical primary split. If the target grid extends
    above an explicit split, the overlapping primary/target photon population
    remains an approximation: no symmetric high-high photon collision operator
    is included, and regimes where those target photons matter require a
    separate benchmark before science-ready use.

    ``include_external_gamma_gamma_target=False`` preserves the prior external
    gamma-gamma behavior. When true, the prescribed external EC field is also
    treated as an isotropic, homogeneous, co-spatial blob-comoving target.
    Internal and external opacities are summed before one transfer, after which
    charge-separated pair sources are evaluated per target and added. The
    external field is not depleted; anisotropic and line-of-sight geometries
    are outside this local closure.

    ``enable_internal_gamma_gamma=False`` removes only the internal
    synchrotron target from gamma-gamma absorption and pair injection. The
    synchrotron photon field itself remains available to SSC and upstream
    hadronic interactions, and external gamma-gamma retains its independent
    ``include_external_gamma_gamma_target`` setting. The default is true and
    preserves the historical production composition.

    ``enable_ssc`` and ``enable_ec`` are static, non-differentiable composition
    switches. Disabling either process sets both its emitted SED and its
    electron cooling contribution to exact zero. SSC-off retains the internal
    synchrotron target for gamma-gamma transfer. EC-off retains the prescribed
    external target for external gamma-gamma transfer. Both default to true,
    preserving the historical target-driven behavior.

    ``compose_proton_synchrotron_low_energy=True`` is reserved for the outer
    hadronic forward. It adds the existing low-grid proton-synchrotron owner
    output to the low intrinsic/escaped field and internal photon target, while
    the gamma grid owns only energies strictly above the split. The standalone
    electromagnetic owner defaults to its historical fixed-proton composition.

    By default, gamma-gamma daughters are evaluated directly on
    ``electron_gamma``. A supplied finite, strictly increasing float64
    ``pair_daughter_gamma`` with the same endpoints evaluates the differential
    pair source on that fixed grid, then deposits its piecewise-constant
    dual-cell contents by source/target cell overlap onto ``electron_gamma``.
    Particle number is conserved; the discrete first-gamma-moment residual is
    reported but is not artificially corrected. Identical pair and transport
    grids use the exact identity path. The remap operator is built once outside
    the fixed-point scan; initial pair sources, relaxation, and convergence
    remain on the transport grid.

    ``relaxation_omega`` satisfies ``0 < omega <= 1``. The default initial pair
    sources and SSC cooling are exact zeros. A scalar or electron-grid-shaped
    finite non-negative ``initial_ssc_loss_rate_dgamma_s`` only changes the
    initial iterate. Convergence uses weighted symmetric relative L1 residuals
    for both lepton densities, the raw pair-source and SSC-cooling fixed-point
    defects, the internal synchrotron target, and escaped gamma SED. Relaxed
    pair and SSC update residuals are diagnostics only. The legacy
    ``final_*_pair_injection_residual`` outputs remain aliases of the raw
    fixed-point residuals. In the pair-energy ledger,
    ``pair_energy_residual`` remains a compatibility alias of
    ``pair_energy_max_relative_residual = |1-f|/max(1,f)``. The separately
    named deficit and symmetric residuals are ``|1-f|`` and
    ``2|1-f|/(1+f)``, where ``f`` is the high-resolution pair-energy capture
    fraction. The JAX-native fixed-length scan freezes state after convergence.
    Gradients propagate through the finite ``maximum_iterations`` expansion;
    no implicit fixed-point differentiation is used.

    Fixed ``gamma_gamma_pair_response`` and
    ``external_gamma_gamma_pair_response`` arrays may be prepared before JAX
    tracing. When omitted, the existing eager fixed-grid cache is used.
    """
    if not isinstance(maximum_iterations, Integral) or isinstance(
        maximum_iterations, bool
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    maximum_iterations = int(maximum_iterations)
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be a positive integer")
    if not isinstance(include_external_gamma_gamma_target, (bool, np.bool_)):
        raise ValueError("include_external_gamma_gamma_target must be boolean")
    include_external_gamma_gamma_target = bool(include_external_gamma_gamma_target)
    for name, enabled in (
        ("enable_internal_gamma_gamma", enable_internal_gamma_gamma),
        ("enable_ssc", enable_ssc),
        ("enable_ec", enable_ec),
        (
            "compose_proton_synchrotron_low_energy",
            compose_proton_synchrotron_low_energy,
        ),
    ):
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    enable_internal_gamma_gamma = bool(enable_internal_gamma_gamma)
    enable_ssc = bool(enable_ssc)
    enable_ec = bool(enable_ec)
    compose_proton_synchrotron_low_energy = bool(compose_proton_synchrotron_low_energy)
    if not enable_ssc and initial_ssc_loss_rate_dgamma_s is not None:
        raise ValueError(
            "initial_ssc_loss_rate_dgamma_s must be None when enable_ssc=False"
        )
    if synchrotron_primary_split_energy_erg is None:
        synchrotron_primary_split_energy = None
        ssc_low_energy_stop = np.shape(synchrotron_photon_energy_erg)[0]
    else:
        synchrotron_primary_split_energy = _positive_iteration_scalar(
            "synchrotron_primary_split_energy_erg",
            synchrotron_primary_split_energy_erg,
        )
        if not _contains_tracer(
            (
                synchrotron_primary_split_energy_erg,
                synchrotron_photon_energy_erg,
            )
        ):
            target_energy = np.asarray(synchrotron_photon_energy_erg)
            split_value = float(np.asarray(synchrotron_primary_split_energy_erg))
            if split_value < target_energy[0] or split_value > target_energy[-1]:
                raise ValueError(
                    "synchrotron_primary_split_energy_erg must lie within "
                    "the synchrotron target-grid coverage"
                )
            ssc_low_energy_stop = int(
                np.searchsorted(target_energy, split_value, side="right")
            )
        else:
            ssc_low_energy_stop = np.shape(synchrotron_photon_energy_erg)[0]
    omega = _positive_iteration_scalar(
        "relaxation_omega",
        relaxation_omega,
        upper=1.0,
    )
    tolerance = _positive_iteration_scalar(
        "relative_tolerance",
        relative_tolerance,
    )
    electron_gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    if pair_daughter_gamma is None:
        daughter_gamma = electron_gamma
        pair_source_remap_operator = None
    else:
        daughter_gamma = _validated_pair_daughter_gamma(
            pair_daughter_gamma,
            electron_gamma,
        )
        pair_source_remap_operator = (
            None
            if _pair_grids_are_identical(daughter_gamma, electron_gamma)
            else _pair_source_remap_operator_jax(
                daughter_gamma,
                electron_gamma,
            )
        )
    initial_electron = (
        jnp.zeros_like(electron_gamma)
        if initial_gamma_gamma_electron_injection_dgamma_cm3_s is None
        else jnp.asarray(
            initial_gamma_gamma_electron_injection_dgamma_cm3_s,
            dtype=jnp.float64,
        )
    )
    initial_positron = (
        jnp.zeros_like(electron_gamma)
        if initial_gamma_gamma_positron_injection_dgamma_cm3_s is None
        else jnp.asarray(
            initial_gamma_gamma_positron_injection_dgamma_cm3_s,
            dtype=jnp.float64,
        )
    )
    initial_ssc = (
        jnp.zeros_like(electron_gamma)
        if initial_ssc_loss_rate_dgamma_s is None
        else jnp.broadcast_to(
            jnp.asarray(initial_ssc_loss_rate_dgamma_s, dtype=jnp.float64),
            electron_gamma.shape,
        )
    )
    if not _contains_tracer(initial_ssc):
        initial_ssc_array = np.asarray(initial_ssc)
        if not np.all(np.isfinite(initial_ssc_array)) or np.any(
            initial_ssc_array < 0.0
        ):
            raise ValueError(
                "initial_ssc_loss_rate_dgamma_s must be finite and non-negative"
            )
    additional_fixed_inverse_compton_loss = jnp.broadcast_to(
        jnp.asarray(
            additional_fixed_inverse_compton_loss_rate_dgamma_s,
            dtype=jnp.float64,
        ),
        electron_gamma.shape,
    )
    if not _contains_tracer(additional_fixed_inverse_compton_loss):
        additional_fixed_array = np.asarray(additional_fixed_inverse_compton_loss)
        if not np.all(np.isfinite(additional_fixed_array)) or np.any(
            additional_fixed_array < 0.0
        ):
            raise ValueError(
                "additional_fixed_inverse_compton_loss_rate_dgamma_s must be "
                "finite and non-negative"
            )

    if meson_synchrotron_low_energy_intrinsic_sed_erg_s is None:
        meson_synchrotron_low_energy_sed = jnp.zeros_like(
            jnp.asarray(synchrotron_photon_energy_erg, dtype=jnp.float64)
        )
    elif _contains_tracer(
        (
            meson_synchrotron_low_energy_intrinsic_sed_erg_s,
            synchrotron_photon_energy_erg,
        )
    ):
        meson_synchrotron_low_energy_sed = jnp.asarray(
            meson_synchrotron_low_energy_intrinsic_sed_erg_s,
            dtype=jnp.float64,
        )
    else:
        meson_synchrotron_low_energy_sed = jnp.asarray(
            _real_1d(
                "meson_synchrotron_low_energy_intrinsic_sed_erg_s",
                meson_synchrotron_low_energy_intrinsic_sed_erg_s,
                minimum_size=1,
                positive=False,
                shape=np.shape(synchrotron_photon_energy_erg),
            ),
            dtype=jnp.float64,
        )

    if meson_synchrotron_intrinsic_sed_erg_s is None:
        meson_synchrotron_sed = jnp.zeros_like(
            jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
        )
    elif _contains_tracer(
        (meson_synchrotron_intrinsic_sed_erg_s, gamma_ray_energy_erg)
    ):
        meson_synchrotron_sed = jnp.asarray(
            meson_synchrotron_intrinsic_sed_erg_s,
            dtype=jnp.float64,
        )
    else:
        meson_synchrotron_sed = jnp.asarray(
            _real_1d(
                "meson_synchrotron_intrinsic_sed_erg_s",
                meson_synchrotron_intrinsic_sed_erg_s,
                minimum_size=1,
                positive=False,
                shape=np.shape(gamma_ray_energy_erg),
            ),
            dtype=jnp.float64,
        )

    external_compton_density = (
        dn_external_seed_photon_denergy_cm3_erg
        if enable_ec
        else jnp.zeros_like(
            jnp.asarray(
                dn_external_seed_photon_denergy_cm3_erg,
                dtype=jnp.float64,
            )
        )
    )
    external_inverse_compton_loss = electron_inverse_compton_loss_rate_dgamma_s_jax(
        electron_gamma,
        external_seed_photon_energy_erg,
        external_compton_density,
    )
    if gamma_gamma_pair_response is None:
        gamma_gamma_pair_response = prepare_gamma_gamma_pair_response_jax(
            daughter_gamma,
            gamma_ray_energy_erg,
            synchrotron_photon_energy_erg,
        )
    if include_external_gamma_gamma_target:
        if external_gamma_gamma_pair_response is None:
            external_gamma_gamma_pair_response = prepare_gamma_gamma_pair_response_jax(
                daughter_gamma,
                gamma_ray_energy_erg,
                external_seed_photon_energy_erg,
            )
    else:
        external_gamma_gamma_pair_response = None

    fixed_arguments = (
        electron_gamma,
        daughter_gamma,
        pair_source_remap_operator,
        synchrotron_photon_energy_erg,
        gamma_ray_energy_erg,
        synchrotron_primary_split_energy,
        primary_electron_injection_dgamma_cm3_s,
        primary_positron_injection_dgamma_cm3_s,
        muon_decay_electron_injection_dgamma_cm3_s,
        muon_decay_positron_injection_dgamma_cm3_s,
        bethe_heitler_electron_injection_dgamma_cm3_s,
        bethe_heitler_positron_injection_dgamma_cm3_s,
    )
    trailing_arguments = (
        proton_gamma,
        proton_density_dgamma_cm3,
        pi0_decay_photon_injection_cm3_s_erg,
        external_seed_photon_energy_erg,
        dn_external_seed_photon_denergy_cm3_erg,
        external_compton_density,
        include_external_gamma_gamma_target,
        enable_internal_gamma_gamma,
        enable_ssc,
        compose_proton_synchrotron_low_energy,
        magnetic_field_G,
        radius_cm,
        adiabatic_rate_s_inv,
        escape_rate_s_inv,
        external_inverse_compton_loss,
        additional_fixed_inverse_compton_loss,
        additional_continuous_loss_rate_dgamma_s,
        omega,
        meson_synchrotron_low_energy_sed,
        meson_synchrotron_sed,
    )

    def evaluate(electron_pair, positron_pair, ssc_loss):
        return _iteration_map(
            *fixed_arguments,
            electron_pair,
            positron_pair,
            ssc_loss,
            *trailing_arguments,
            gamma_gamma_pair_response=gamma_gamma_pair_response,
            external_gamma_gamma_pair_response=(external_gamma_gamma_pair_response),
        )

    initial_state = evaluate(initial_electron, initial_positron, initial_ssc)
    zero_state = jax.tree_util.tree_map(jnp.zeros_like, initial_state)
    initial_residuals = _monitored_residuals(
        initial_state,
        zero_state,
        electron_gamma,
        synchrotron_photon_energy_erg,
        gamma_ray_energy_erg,
    )
    initially_converged = initial_residuals["cascade"] < tolerance
    next_electron = jnp.where(
        initially_converged,
        initial_electron,
        initial_state["relaxed_next_electron_pair_injection_dgamma_cm3_s"],
    )
    next_positron = jnp.where(
        initially_converged,
        initial_positron,
        initial_state["relaxed_next_positron_pair_injection_dgamma_cm3_s"],
    )
    next_ssc = jnp.where(
        initially_converged,
        initial_ssc,
        initial_state["relaxed_next_ssc_inverse_compton_loss_rate_dgamma_s"],
    )
    initial_carry = (
        next_electron,
        next_positron,
        next_ssc,
        initial_state,
        initial_residuals,
        initially_converged,
        jnp.asarray(1, dtype=jnp.int32),
    )

    def scan_step(carry, _):
        def frozen(previous):
            return previous, previous[4]

        def active(previous):
            electron_pair, positron_pair, ssc_loss, previous_state, _, _, count = (
                previous
            )
            state = evaluate(electron_pair, positron_pair, ssc_loss)
            residuals = _monitored_residuals(
                state,
                previous_state,
                electron_gamma,
                synchrotron_photon_energy_erg,
                gamma_ray_energy_erg,
            )
            converged = residuals["cascade"] < tolerance
            following_electron = jnp.where(
                converged,
                electron_pair,
                state["relaxed_next_electron_pair_injection_dgamma_cm3_s"],
            )
            following_positron = jnp.where(
                converged,
                positron_pair,
                state["relaxed_next_positron_pair_injection_dgamma_cm3_s"],
            )
            following_ssc = jnp.where(
                converged,
                ssc_loss,
                state["relaxed_next_ssc_inverse_compton_loss_rate_dgamma_s"],
            )
            updated = (
                following_electron,
                following_positron,
                following_ssc,
                state,
                residuals,
                converged,
                count + jnp.asarray(1, dtype=jnp.int32),
            )
            return updated, residuals

        return jax.lax.cond(carry[5], frozen, active, carry)

    final_carry, residual_tail = jax.lax.scan(
        scan_step,
        initial_carry,
        xs=None,
        length=maximum_iterations - 1,
    )
    _, _, _, final_state, final_residuals, converged, iteration_count = final_carry
    residual_history = {
        name: jnp.concatenate((initial_residuals[name][None], values))
        for name, values in residual_tail.items()
    }
    transport = final_state["transport"]
    radiation = final_state["radiation"]
    pair_source = final_state["pair_source"]
    ssc_low_energy_owned_sed = (
        inverse_compton_sed_jax(
            jnp.asarray(synchrotron_photon_energy_erg)[:ssc_low_energy_stop],
            electron_gamma,
            radiation["total_lepton_density_dgamma_cm3"],
            synchrotron_photon_energy_erg,
            radiation["internal_synchrotron_photon_density_cm3_erg"],
            radius_cm=radius_cm,
        )
        if enable_ssc
        else jnp.zeros((ssc_low_energy_stop,), dtype=jnp.float64)
    )
    ssc_low_energy_intrinsic_sed = jnp.zeros_like(
        synchrotron_photon_energy_erg
    ).at[:ssc_low_energy_stop].set(
        ssc_low_energy_owned_sed
    )
    total_low_energy_photon_intrinsic_sed = (
        radiation["total_low_energy_synchrotron_intrinsic_sed_erg_s"]
        + ssc_low_energy_intrinsic_sed
    )
    radiation = {
        **radiation,
        "ssc_low_energy_intrinsic_sed_erg_s": ssc_low_energy_intrinsic_sed,
        "total_low_energy_photon_intrinsic_sed_erg_s": (
            total_low_energy_photon_intrinsic_sed
        ),
        "total_low_energy_photon_escaped_sed_erg_s": _transfer_from_quantities(
            radiation["electron_synchrotron_absorption_coefficient_cm1"],
            total_low_energy_photon_intrinsic_sed,
            radius_cm,
        ),
        "total_low_energy_photon_primary_intrinsic_sed_erg_s": (
            radiation["total_low_energy_synchrotron_primary_intrinsic_sed_erg_s"]
            + jnp.where(
                synchrotron_photon_energy_erg
                <= radiation["synchrotron_split_energy_erg"],
                ssc_low_energy_intrinsic_sed,
                0.0,
            )
        ),
        "internal_low_energy_photon_density_cm3_erg": (
            synchrotron_sed_to_internal_photon_density_jax(
                synchrotron_photon_energy_erg,
                total_low_energy_photon_intrinsic_sed,
                radiation["electron_synchrotron_absorption_coefficient_cm1"],
                radius_cm=radius_cm,
            )
        ),
    }
    ledger = _ledger_diagnostics(
        final_state,
        electron_gamma,
        daughter_gamma,
        synchrotron_photon_energy_erg,
        gamma_ray_energy_erg,
        external_seed_photon_energy_erg,
        dn_external_seed_photon_denergy_cm3_erg,
        enable_internal_gamma_gamma,
        include_external_gamma_gamma_target,
        compose_proton_synchrotron_low_energy,
    )
    intrinsic_bolometric_luminosity = jnp.trapezoid(
        radiation["total_low_energy_photon_primary_intrinsic_sed_erg_s"],
        x=jnp.log(synchrotron_photon_energy_erg),
    ) + jnp.trapezoid(
        radiation["total_gamma_intrinsic_sed_erg_s"],
        x=jnp.log(gamma_ray_energy_erg),
    )
    return {
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
        "initial_electron_pair_injection_dgamma_cm3_s": initial_electron,
        "initial_positron_pair_injection_dgamma_cm3_s": initial_positron,
        "initial_ssc_inverse_compton_loss_rate_dgamma_s": initial_ssc,
        "pair_daughter_gamma": daughter_gamma,
        "include_external_gamma_gamma_target": jnp.asarray(
            include_external_gamma_gamma_target
        ),
        "enable_internal_gamma_gamma": jnp.asarray(enable_internal_gamma_gamma),
        "enable_ssc": jnp.asarray(enable_ssc),
        "enable_ec": jnp.asarray(enable_ec),
        "electron_pair_injection_dgamma_cm3_s": final_state[
            "electron_pair_injection_dgamma_cm3_s"
        ],
        "positron_pair_injection_dgamma_cm3_s": final_state[
            "positron_pair_injection_dgamma_cm3_s"
        ],
        "raw_next_electron_pair_injection_dgamma_cm3_s": final_state[
            "raw_next_electron_pair_injection_dgamma_cm3_s"
        ],
        "raw_next_positron_pair_injection_dgamma_cm3_s": final_state[
            "raw_next_positron_pair_injection_dgamma_cm3_s"
        ],
        "high_resolution_electron_pair_injection_dgamma_cm3_s": final_state[
            "raw_next_high_resolution_electron_pair_injection_dgamma_cm3_s"
        ],
        "high_resolution_positron_pair_injection_dgamma_cm3_s": final_state[
            "raw_next_high_resolution_positron_pair_injection_dgamma_cm3_s"
        ],
        "high_resolution_total_pair_injection_dgamma_cm3_s": pair_source[
            "total_pair_injection_dgamma_cm3_s"
        ],
        "raw_next_high_resolution_electron_pair_injection_dgamma_cm3_s": (
            final_state["raw_next_high_resolution_electron_pair_injection_dgamma_cm3_s"]
        ),
        "raw_next_high_resolution_positron_pair_injection_dgamma_cm3_s": (
            final_state["raw_next_high_resolution_positron_pair_injection_dgamma_cm3_s"]
        ),
        "mapped_raw_next_electron_pair_injection_dgamma_cm3_s": final_state[
            "mapped_raw_next_electron_pair_injection_dgamma_cm3_s"
        ],
        "mapped_raw_next_positron_pair_injection_dgamma_cm3_s": final_state[
            "mapped_raw_next_positron_pair_injection_dgamma_cm3_s"
        ],
        "external_inverse_compton_loss_rate_dgamma_s": final_state[
            "external_inverse_compton_loss_rate_dgamma_s"
        ],
        "additional_fixed_inverse_compton_loss_rate_dgamma_s": final_state[
            "additional_fixed_inverse_compton_loss_rate_dgamma_s"
        ],
        "ssc_inverse_compton_loss_rate_dgamma_s": final_state[
            "ssc_inverse_compton_loss_rate_dgamma_s"
        ],
        "total_inverse_compton_loss_rate_dgamma_s": final_state[
            "total_inverse_compton_loss_rate_dgamma_s"
        ],
        "raw_next_ssc_inverse_compton_loss_rate_dgamma_s": final_state[
            "raw_next_ssc_inverse_compton_loss_rate_dgamma_s"
        ],
        "relaxed_next_ssc_inverse_compton_loss_rate_dgamma_s": final_state[
            "relaxed_next_ssc_inverse_compton_loss_rate_dgamma_s"
        ],
        **radiation,
        "gamma_gamma_absorption_coefficient_cm1": pair_source[
            "absorption_coefficient_cm1"
        ],
        "gamma_gamma_internal_target_absorption_coefficient_cm1": pair_source[
            "gamma_gamma_internal_target_absorption_coefficient_cm1"
        ],
        "gamma_gamma_external_target_absorption_coefficient_cm1": pair_source[
            "gamma_gamma_external_target_absorption_coefficient_cm1"
        ],
        "gamma_gamma_total_absorption_coefficient_cm1": pair_source[
            "gamma_gamma_total_absorption_coefficient_cm1"
        ],
        "internal_target_electron_pair_injection_dgamma_cm3_s": pair_source[
            "internal_target_electron_pair_injection_dgamma_cm3_s"
        ],
        "internal_target_positron_pair_injection_dgamma_cm3_s": pair_source[
            "internal_target_positron_pair_injection_dgamma_cm3_s"
        ],
        "external_target_electron_pair_injection_dgamma_cm3_s": pair_source[
            "external_target_electron_pair_injection_dgamma_cm3_s"
        ],
        "external_target_positron_pair_injection_dgamma_cm3_s": pair_source[
            "external_target_positron_pair_injection_dgamma_cm3_s"
        ],
        "gamma_gamma_optical_depth": pair_source["optical_depth"],
        "gamma_gamma_transfer_factor": pair_source["transfer_factor"],
        "escaped_gamma_sed_erg_s": pair_source["escaped_sed_erg_s"],
        "removed_gamma_sed_erg_s": pair_source["removed_sed_erg_s"],
        "internal_high_energy_gamma_photon_density_cm3_erg": pair_source[
            "internal_gamma_photon_density_cm3_erg"
        ],
        "absorbed_gamma_photon_rate_cm3_s_erg": pair_source[
            "absorbed_photon_rate_cm3_s_erg"
        ],
        "intrinsic_bolometric_luminosity_erg_s": (intrinsic_bolometric_luminosity),
        "converged": converged,
        "iteration_count": iteration_count,
        "maximum_iterations": jnp.asarray(
            maximum_iterations,
            dtype=jnp.int32,
        ),
        "relative_tolerance": tolerance,
        "relaxation_omega": omega,
        "final_cascade_residual": final_residuals["cascade"],
        "final_electron_density_residual": final_residuals["electron_density"],
        "final_positron_density_residual": final_residuals["positron_density"],
        "final_pair_fixed_point_residual": final_residuals["pair_fixed_point"],
        "final_electron_pair_fixed_point_residual": final_residuals[
            "electron_pair_fixed_point"
        ],
        "final_positron_pair_fixed_point_residual": final_residuals[
            "positron_pair_fixed_point"
        ],
        "final_pair_update_residual": final_residuals["pair_update"],
        "final_ssc_cooling_fixed_point_residual": final_residuals[
            "ssc_cooling_fixed_point"
        ],
        "final_ssc_cooling_update_residual": final_residuals["ssc_cooling_update"],
        "final_electron_pair_update_residual": final_residuals["electron_pair_update"],
        "final_positron_pair_update_residual": final_residuals["positron_pair_update"],
        "final_pair_injection_residual": final_residuals["pair_fixed_point"],
        "final_electron_pair_injection_residual": final_residuals[
            "electron_pair_fixed_point"
        ],
        "final_positron_pair_injection_residual": final_residuals[
            "positron_pair_fixed_point"
        ],
        "final_internal_synchrotron_photon_density_residual": (
            final_residuals["internal_synchrotron_photon_density"]
        ),
        "final_escaped_gamma_sed_residual": final_residuals["escaped_gamma_sed"],
        "residual_history": residual_history["cascade"],
        "residual_histories": residual_history,
        "inverse_compton_loss_composition_residual": ledger[
            "inverse_compton_loss_composition_residual"
        ],
        "electron_pair_remap_number_residual": ledger[
            "electron_pair_remap_number_residual"
        ],
        "positron_pair_remap_number_residual": ledger[
            "positron_pair_remap_number_residual"
        ],
        "electron_pair_remap_energy_residual": ledger[
            "electron_pair_remap_energy_residual"
        ],
        "positron_pair_remap_energy_residual": ledger[
            "positron_pair_remap_energy_residual"
        ],
        "ledger_diagnostics": ledger,
    }
