"""Steady internal gamma-ray photon density in a homogeneous sphere."""

import jax.numpy as jnp
import numpy as np

from helena.core.constants import SPEED_OF_LIGHT_CGS as _C_CGS
from helena.core.contracts import nonempty_real_1d as _real_1d
from helena.core.jax_backend.chains.gamma_gamma_transfer import (
    _gamma_gamma_transfer_with_absorption_coefficient_jax,
    gamma_gamma_transfer_chain_jax,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import _contains_tracer
from helena.core.jax_backend.radiation.photon_fields import (
    _homogeneous_sphere_effective_path_length_jax,
    produced_sed_to_internal_photon_density_jax,
)

__all__ = ["gamma_gamma_internal_photon_field_chain_jax"]


def gamma_gamma_internal_photon_field_chain_jax(
    gamma_ray_energy_erg,
    intrinsic_gamma_sed_erg_s,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    radius_cm,
):
    """Return the steady internal gamma-ray field of a homogeneous sphere.

    Gamma-ray and target-photon energies are strictly increasing positive
    blob-comoving grids in erg. ``intrinsic_gamma_sed_erg_s`` is matching
    intrinsic ``E L_E`` in erg s^-1, the target field is isotropic
    ``dn_ph/dE`` in cm^-3 erg^-1, and ``radius_cm`` is the sphere radius in cm.

    The closure composes the existing mixed-emitter gamma-gamma transfer with
    the source-independent homogeneous-sphere internal-photon average. It
    returns production, escape, and absorption rates in
    cm^-3 s^-1 erg^-1, effective path length in cm, residence time in s, and
    internal ``dn_gamma/dE`` in cm^-3 erg^-1. No pair injection, particle
    transport, feedback, cascade, interpolation, or frame transform is
    included.
    """
    return _gamma_gamma_internal_photon_field_chain_jax(
        gamma_ray_energy_erg,
        intrinsic_gamma_sed_erg_s,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        radius_cm=radius_cm,
        absorption_coefficient_cm1=None,
    )


def _gamma_gamma_internal_photon_field_with_absorption_coefficient_jax(
    gamma_ray_energy_erg,
    intrinsic_gamma_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return one internal gamma field controlled by a precomputed opacity."""
    return _gamma_gamma_internal_photon_field_chain_jax(
        gamma_ray_energy_erg,
        intrinsic_gamma_sed_erg_s,
        None,
        None,
        radius_cm=radius_cm,
        absorption_coefficient_cm1=absorption_coefficient_cm1,
    )


def _gamma_gamma_internal_photon_field_chain_jax(
    gamma_ray_energy_erg,
    intrinsic_gamma_sed_erg_s,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    radius_cm,
    absorption_coefficient_cm1,
):
    if not _contains_tracer((gamma_ray_energy_erg, intrinsic_gamma_sed_erg_s)):
        gamma_ray_energy = _real_1d(
            "gamma_ray_energy_erg",
            gamma_ray_energy_erg,
            positive=True,
        )
        if gamma_ray_energy.size < 2 or np.any(np.diff(gamma_ray_energy) <= 0.0):
            raise ValueError(
                "gamma_ray_energy_erg must contain at least two strictly "
                "increasing values"
            )

    if absorption_coefficient_cm1 is None:
        transfer = gamma_gamma_transfer_chain_jax(
            gamma_ray_energy_erg,
            intrinsic_gamma_sed_erg_s,
            target_photon_energy_erg,
            dn_target_photon_denergy_cm3_erg,
            radius_cm=radius_cm,
        )
    else:
        transfer = _gamma_gamma_transfer_with_absorption_coefficient_jax(
            gamma_ray_energy_erg,
            intrinsic_gamma_sed_erg_s,
            absorption_coefficient_cm1,
            radius_cm=radius_cm,
        )
    gamma_ray_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
    radius = jnp.asarray(radius_cm, dtype=jnp.float64)
    volume = 4.0 * jnp.pi * radius**3 / 3.0
    denominator = volume * gamma_ray_energy**2
    effective_path_length = _homogeneous_sphere_effective_path_length_jax(
        transfer["absorption_coefficient_cm1"],
        radius,
    )
    internal_photon_density = produced_sed_to_internal_photon_density_jax(
        gamma_ray_energy,
        transfer["intrinsic_sed_erg_s"],
        transfer["absorption_coefficient_cm1"],
        radius_cm=radius,
    )
    absorbed_photon_rate = (
        _C_CGS * transfer["absorption_coefficient_cm1"] * internal_photon_density
    )
    removed_sed = denominator * absorbed_photon_rate
    absorbed_photon_rate = removed_sed / denominator

    return {
        "volume_cm3": volume,
        **transfer,
        "removed_sed_erg_s": removed_sed,
        "photon_production_rate_cm3_s_erg": (
            transfer["intrinsic_sed_erg_s"] / denominator
        ),
        "effective_path_length_cm": effective_path_length,
        "effective_residence_time_s": effective_path_length / _C_CGS,
        "internal_photon_density_cm3_erg": internal_photon_density,
        "escaped_photon_rate_cm3_s_erg": (transfer["escaped_sed_erg_s"] / denominator),
        "absorbed_photon_rate_cm3_s_erg": absorbed_photon_rate,
    }
