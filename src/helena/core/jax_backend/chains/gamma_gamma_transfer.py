"""Source-independent gamma-gamma transfer through a homogeneous blob."""

import jax.numpy as jnp
import numpy as np

from helena.core.contracts import real_1d as _real_1d
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    _contains_tracer,
    gamma_gamma_absorbed_sed_jax,
    gamma_gamma_absorption_coefficient_jax,
    gamma_gamma_optical_depth_jax,
)

__all__ = ["gamma_gamma_transfer_chain_jax"]


def gamma_gamma_transfer_chain_jax(
    gamma_ray_energy_erg,
    intrinsic_gamma_sed_erg_s,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    radius_cm,
):
    """Transfer an intrinsic blob-frame gamma-ray SED through gamma-gamma opacity.

    ``gamma_ray_energy_erg`` and ``target_photon_energy_erg`` are strictly
    increasing positive blob-comoving energy grids in erg.
    ``intrinsic_gamma_sed_erg_s`` is matching non-negative ``E L_E`` in
    erg s^-1, and ``dn_target_photon_denergy_cm3_erg`` is matching isotropic
    ``dn_ph/dE`` in cm^-3 erg^-1. ``radius_cm`` is the positive radius of the
    uniformly emitting sphere.

    The returned coefficient is in cm^-1, ``optical_depth = 2 R alpha`` and
    ``transfer_factor`` are dimensionless, and all three SEDs are intrinsic
    blob-comoving ``E L_E`` in erg s^-1. Escaped radiation uses the existing
    mixed-emitter homogeneous-sphere prescription; removed radiation is
    exactly intrinsic minus escaped. No target construction, interpolation,
    pair injection, particle transport, feedback, cascade, or observer
    transformation is included.
    """
    return _gamma_gamma_transfer_chain_jax(
        gamma_ray_energy_erg,
        intrinsic_gamma_sed_erg_s,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        radius_cm=radius_cm,
        absorption_coefficient_cm1=None,
    )


def _gamma_gamma_transfer_with_absorption_coefficient_jax(
    gamma_ray_energy_erg,
    intrinsic_gamma_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Apply one mixed-sphere transfer using a precomputed total opacity."""
    return _gamma_gamma_transfer_chain_jax(
        gamma_ray_energy_erg,
        intrinsic_gamma_sed_erg_s,
        None,
        None,
        radius_cm=radius_cm,
        absorption_coefficient_cm1=absorption_coefficient_cm1,
    )


def _gamma_gamma_transfer_chain_jax(
    gamma_ray_energy_erg,
    intrinsic_gamma_sed_erg_s,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    radius_cm,
    absorption_coefficient_cm1,
):
    if _contains_tracer((gamma_ray_energy_erg, intrinsic_gamma_sed_erg_s)):
        gamma_ray_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
        intrinsic_sed = jnp.asarray(intrinsic_gamma_sed_erg_s, dtype=jnp.float64)
    else:
        gamma_ray_energy = _real_1d(
            "gamma_ray_energy_erg",
            gamma_ray_energy_erg,
            minimum_size=1,
            positive=True,
        )
        if np.any(np.diff(gamma_ray_energy) <= 0.0):
            raise ValueError("gamma_ray_energy_erg must be strictly increasing")
        intrinsic_sed = _real_1d(
            "intrinsic_gamma_sed_erg_s",
            intrinsic_gamma_sed_erg_s,
            minimum_size=1,
            positive=False,
            shape=gamma_ray_energy.shape,
        )

    if absorption_coefficient_cm1 is None:
        absorption_coefficient = gamma_gamma_absorption_coefficient_jax(
            gamma_ray_energy,
            target_photon_energy_erg,
            dn_target_photon_denergy_cm3_erg,
        )
        optical_depth = gamma_gamma_optical_depth_jax(
            gamma_ray_energy,
            target_photon_energy_erg,
            dn_target_photon_denergy_cm3_erg,
            radius_cm=radius_cm,
        )
    else:
        absorption_coefficient = jnp.asarray(
            absorption_coefficient_cm1,
            dtype=jnp.float64,
        )
        optical_depth = (
            2.0 * jnp.asarray(radius_cm, dtype=jnp.float64) * absorption_coefficient
        )
    transfer_factor = gamma_gamma_absorbed_sed_jax(
        jnp.ones_like(intrinsic_sed),
        absorption_coefficient,
        radius_cm=radius_cm,
    )
    escaped_sed = gamma_gamma_absorbed_sed_jax(
        intrinsic_sed,
        absorption_coefficient,
        radius_cm=radius_cm,
    )
    removed_sed = intrinsic_sed - escaped_sed

    return {
        "absorption_coefficient_cm1": absorption_coefficient,
        "optical_depth": optical_depth,
        "transfer_factor": transfer_factor,
        "intrinsic_sed_erg_s": intrinsic_sed,
        "escaped_sed_erg_s": escaped_sed,
        "removed_sed_erg_s": removed_sed,
    }
