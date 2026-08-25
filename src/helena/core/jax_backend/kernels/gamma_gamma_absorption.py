"""JAX local gamma-gamma absorption by isotropic comoving photon fields."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    THOMSON_CROSS_SECTION_CGS as _SIGMA_T_CGS,
)
from helena.core.contracts import (
    positive_scalar as _positive_scalar,
    real_1d as _real_1d,
)
from helena.core.tables import (
    GAMMA_GAMMA_ANGLE_T as _ANGLE_T,
    GAMMA_GAMMA_ANGLE_WEIGHT as _ANGLE_WEIGHT,
)

__all__ = [
    "gamma_gamma_absorption_coefficient_jax",
    "gamma_gamma_optical_depth_jax",
    "gamma_gamma_absorbed_sed_jax",
]


def _contains_tracer(value):
    return any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )


def _breit_wheeler_cross_section_jax(s):
    s = jnp.asarray(s, dtype=jnp.float64)
    active = s > 1.0
    s_safe = jnp.where(active, s, 1.0)
    inverse_s = 1.0 / s_safe
    beta_squared = jnp.clip(1.0 - inverse_s, 0.0, 1.0)
    beta = jnp.sqrt(beta_squared)
    log_ratio = jnp.log(s_safe) + 2.0 * jnp.log1p(beta)
    sigma_raw = (
        3.0
        * _SIGMA_T_CGS
        / 16.0
        * inverse_s
        * ((3.0 - beta_squared**2) * log_ratio - 2.0 * beta * (2.0 - beta_squared))
    )
    return jnp.where(active, sigma_raw, 0.0)


def _angle_averaged_cross_section_jax(y):
    y = jnp.asarray(y, dtype=jnp.float64)
    active = y > 1.0
    y_safe = jnp.where(active, y, 1.0)
    interval = y_safe - 1.0
    angle_t = jnp.asarray(_ANGLE_T, dtype=jnp.float64)
    angle_weight = jnp.asarray(_ANGLE_WEIGHT, dtype=jnp.float64)
    s = 1.0 + interval[..., None] * angle_t
    integral = jnp.sum(angle_weight * s * _breit_wheeler_cross_section_jax(s), axis=-1)
    sigma_raw = 2.0 * (interval / y_safe) * (integral / y_safe)
    return jnp.where(active, sigma_raw, 0.0)


def _absorption_coefficient_kernel(
    gamma_ray_energy_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    gamma_ray_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    target_density = jnp.asarray(dn_target_photon_denergy_cm3_erg, dtype=jnp.float64)
    y = gamma_ray_energy[:, None] * target_energy[None, :] / _MEC2_CGS**2
    integrand = (
        target_energy[None, :]
        * target_density[None, :]
        * _angle_averaged_cross_section_jax(y)
    )
    return jnp.trapezoid(integrand, x=jnp.log(target_energy), axis=1)


def _optical_depth_kernel(
    gamma_ray_energy_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    radius_cm,
):
    return (
        2.0
        * jnp.asarray(radius_cm, dtype=jnp.float64)
        * (
            _absorption_coefficient_kernel(
                gamma_ray_energy_erg,
                target_photon_energy_erg,
                dn_target_photon_denergy_cm3_erg,
            )
        )
    )


def _sphere_attenuation_factor_jax(tau_diameter):
    tau_diameter = jnp.asarray(tau_diameter, dtype=jnp.float64)
    small = (tau_diameter > 0.0) & (tau_diameter <= 0.1)
    small_tau = jnp.where(small, tau_diameter, 0.0)
    small_result = 1.0 + small_tau * (
        -3.0 / 8.0
        + small_tau
        * (
            1.0 / 10.0
            + small_tau
            * (
                -1.0 / 48.0
                + small_tau
                * (1.0 / 280.0 + small_tau * (-1.0 / 1920.0 + small_tau / 15120.0))
            )
        )
    )

    regular = tau_diameter > 0.1
    regular_tau = jnp.where(regular, tau_diameter, 1.0)
    inverse_tau = 1.0 / regular_tau
    regular_result = (
        3.0
        * inverse_tau
        * (0.5 - (1.0 - (1.0 + regular_tau) * jnp.exp(-regular_tau)) * inverse_tau**2)
    )
    return jnp.where(
        tau_diameter == 0.0,
        1.0,
        jnp.where(small, small_result, regular_result),
    )


def _absorbed_sed_kernel(intrinsic_sed_erg_s, absorption_coefficient_cm1, radius_cm):
    intrinsic_sed = jnp.asarray(intrinsic_sed_erg_s, dtype=jnp.float64)
    coefficient = jnp.asarray(absorption_coefficient_cm1, dtype=jnp.float64)
    optical_depth = 2.0 * jnp.asarray(radius_cm, dtype=jnp.float64) * coefficient
    return intrinsic_sed * _sphere_attenuation_factor_jax(optical_depth)


def _checked_result(result, diagnostic):
    if _contains_tracer(result):
        return result
    values = np.asarray(result)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(f"{diagnostic} exceeds the finite float64 range")
    return result


def gamma_gamma_absorption_coefficient_jax(
    gamma_ray_energy_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return JAX isotropic gamma-gamma absorption in cm^-1.

    Inputs are positive blob-comoving gamma-ray and target-photon energies in
    erg and matching finite non-negative, angle-integrated isotropic
    ``dn_ph/dE`` in cm^-3 erg^-1. Gamma-ray order is preserved; the target grid
    has at least two strictly increasing values.

    The float64 result uses the NumPy reference's exact Breit--Wheeler
    threshold, fixed 128-point isotropic angular average, and trapezoidal
    integration in ``ln(E)``. It includes no frame transform, anisotropic
    geometry, pair injection, transport, cascade, or observer physics. Eager
    calls validate inputs and outputs; traced calls require valid fixed-shape
    inputs.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX gamma-gamma absorption requires jax_enable_x64")

    arguments = (
        gamma_ray_energy_erg,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if _contains_tracer(arguments):
        return _absorption_coefficient_kernel(*arguments)

    gamma_ray_energy = _real_1d(
        "gamma_ray_energy_erg",
        gamma_ray_energy_erg,
        minimum_size=1,
        positive=True,
    )
    target_energy = _real_1d(
        "target_photon_energy_erg",
        target_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(target_energy) <= 0.0):
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    target_density = _real_1d(
        "dn_target_photon_denergy_cm3_erg",
        dn_target_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=target_energy.shape,
    )
    if not np.any(target_density):
        return jnp.zeros(gamma_ray_energy.shape, dtype=jnp.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        maximum_y = np.max(gamma_ray_energy) * np.max(target_energy) / _MEC2_CGS**2
    if not np.isfinite(maximum_y):
        raise FloatingPointError(
            "gamma-ray and target-photon energy product exceeds the finite "
            "float64 range"
        )

    result = _absorption_coefficient_kernel(
        gamma_ray_energy,
        target_energy,
        target_density,
    )
    return _checked_result(result, "gamma-gamma absorption coefficient")


def gamma_gamma_optical_depth_jax(
    gamma_ray_energy_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    radius_cm,
):
    """Return the JAX homogeneous-sphere diameter optical depth.

    Photon inputs and blob-frame units match
    :func:`gamma_gamma_absorption_coefficient_jax`. ``radius_cm`` is a finite
    positive homogeneous-sphere radius in cm. The float64 result is the
    dimensionless center-to-center diameter depth ``tau_d = 2 R alpha`` and
    is not a path depth for another geometry. No pair injection, transport,
    cascade, or observer transformation is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX gamma-gamma absorption requires jax_enable_x64")

    arguments = (
        gamma_ray_energy_erg,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        radius_cm,
    )
    if _contains_tracer(arguments):
        return _optical_depth_kernel(*arguments)

    radius = _positive_scalar("radius_cm", radius_cm)
    coefficient = gamma_gamma_absorption_coefficient_jax(*arguments[:3])
    result = 2.0 * radius * coefficient
    return _checked_result(result, "gamma-gamma diameter optical depth")


def gamma_gamma_absorbed_sed_jax(
    intrinsic_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return a uniformly emitted JAX SED after local gamma-gamma absorption.

    ``intrinsic_sed_erg_s`` is finite non-negative blob-comoving ``E L_E`` in
    erg s^-1. ``absorption_coefficient_cm1`` is a matching one-dimensional
    finite non-negative local coefficient in cm^-1, and ``radius_cm`` is a
    finite positive homogeneous-sphere radius in cm.

    The float64 result is the escaped comoving ``E L_E`` after applying the
    NumPy reference's mixed-emitter homogeneous-sphere attenuation at diameter
    depth ``tau_d = 2 R alpha``; it is not foreground-screen attenuation.
    Apply observer transformations only afterward. No pair injection,
    transport, cascade, or target depletion is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX gamma-gamma absorption requires jax_enable_x64")

    arguments = (
        intrinsic_sed_erg_s,
        absorption_coefficient_cm1,
        radius_cm,
    )
    if _contains_tracer(arguments):
        return _absorbed_sed_kernel(*arguments)

    intrinsic_sed = _real_1d(
        "intrinsic_sed_erg_s",
        intrinsic_sed_erg_s,
        minimum_size=1,
        positive=False,
    )
    coefficient = _real_1d(
        "absorption_coefficient_cm1",
        absorption_coefficient_cm1,
        minimum_size=1,
        positive=False,
        shape=intrinsic_sed.shape,
    )
    radius = _positive_scalar("radius_cm", radius_cm)
    result = _absorbed_sed_kernel(intrinsic_sed, coefficient, radius)
    return _checked_result(result, "gamma-gamma absorbed SED")
