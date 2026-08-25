"""Differential pair injection from isotropic photon-photon collisions."""

import numpy as np
from scipy.integrate import trapezoid

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
    THOMSON_CROSS_SECTION_CGS as _SIGMA_T_CGS,
)
from helena.core.contracts import real_1d as _real_1d
from helena.core.tables import (
    GAMMA_GAMMA_ANGLE_T as _ANGLE_T,
    GAMMA_GAMMA_ANGLE_WEIGHT as _ANGLE_WEIGHT,
)

__all__ = ["gamma_gamma_pair_injection"]


def _differential_cross_section_dgamma(electron_gamma, x, eta, s):
    """Return the exact fixed-collision cross section in cm^2 per gamma."""
    gamma, x, eta, s = np.broadcast_arrays(
        np.asarray(electron_gamma, dtype=float),
        np.asarray(x, dtype=float),
        np.asarray(eta, dtype=float),
        np.asarray(s, dtype=float),
    )
    output_shape = gamma.shape
    gamma = gamma.ravel()
    x = x.ravel()
    eta = eta.ravel()
    s = s.ravel()
    result = np.zeros(gamma.shape, dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        momentum_squared = (x + eta) ** 2 - 4.0 * s
    collision = (s > 1.0) & (momentum_squared > 0.0)
    if not np.any(collision):
        return result.reshape(output_shape)

    s_collision = s[collision]
    beta = np.sqrt(1.0 - 1.0 / s_collision)
    momentum = np.sqrt(momentum_squared[collision])
    total_energy = x[collision] + eta[collision]
    lower = 0.5 * (total_energy - beta * momentum)
    upper = 0.5 * (total_energy + beta * momentum)
    supported = collision.copy()
    supported[collision] = (gamma[collision] >= lower) & (gamma[collision] <= upper)
    if not np.any(supported):
        return result.reshape(output_shape)

    gamma = gamma[supported]
    x = x[supported]
    eta = eta[supported]
    s = s[supported]
    w = np.sqrt(s)
    beta = np.sqrt(1.0 - 1.0 / s)
    momentum = np.sqrt((x + eta) ** 2 - 4.0 * s)
    c_1 = (x - gamma) ** 2 - 1.0
    c_2 = (eta - gamma) ** 2 - 1.0
    d_1 = x**2 + x * eta + gamma * (eta - x)
    d_2 = eta**2 + x * eta - gamma * (eta - x)
    a_1 = x * eta + s * c_1
    a_2 = x * eta + s * c_2

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        g_sum = 1.0 / np.sqrt(a_1) + 1.0 / np.sqrt(a_2)
        f_sum = (d_1 - 2.0 * s) / a_1**1.5 + (d_2 - 2.0 * s) / a_2**1.5
        values = (
            3.0
            * _SIGMA_T_CGS
            / (8.0 * w)
            * (
                -1.0 / (momentum * w)
                + (3.0 - beta**4) * g_sum / 4.0
                - f_sum / (8.0 * s)
            )
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(
            "gamma-gamma differential cross section is not finite and non-negative"
        )
    result[supported] = values
    return result.reshape(output_shape)


def _isotropic_differential_pair_kernel(
    electron_gamma,
    gamma_ray_energy_erg,
    target_photon_energy_erg,
):
    """Return the isotropic response in cm^2 per daughter Lorentz factor."""
    gamma = np.asarray(electron_gamma, dtype=float)
    target_energy = np.asarray(target_photon_energy_erg, dtype=float)
    x = np.asarray(gamma_ray_energy_erg, dtype=float) / _MEC2_CGS
    eta = target_energy / _MEC2_CGS
    with np.errstate(over="ignore", invalid="ignore"):
        y = x * eta
    if not np.all(np.isfinite(y)):
        raise FloatingPointError(
            "gamma-ray and target-photon energy product exceeds float64 range"
        )

    active = y > 1.0
    y_safe = np.where(active, y, 1.0)
    interval = y_safe - 1.0
    s = 1.0 + interval[:, None] * _ANGLE_T[None, :]
    differential = _differential_cross_section_dgamma(
        gamma[:, None, None],
        x,
        eta[None, :, None],
        s[None, :, :],
    )
    integral = np.sum(
        _ANGLE_WEIGHT[None, None, :] * s[None, :, :] * differential,
        axis=-1,
    )
    response = 2.0 * (interval / y_safe)[None, :] * integral / y_safe[None, :]
    return np.where(active[None, :], response, 0.0)


def gamma_gamma_pair_injection(
    electron_gamma,
    gamma_ray_energy_erg,
    dn_gamma_ray_denergy_cm3_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return isotropic gamma-gamma positron and electron injection.

    Parameters
    ----------
    electron_gamma : array_like, shape (n_electron,)
        Finite blob-comoving electron/positron Lorentz factors greater than
        one. Their order is preserved and need not be increasing.
    gamma_ray_energy_erg : array_like, shape (n_gamma_ray,)
        At least two strictly increasing positive finite primary gamma-ray
        energies in the blob comoving frame, in erg.
    dn_gamma_ray_denergy_cm3_erg : array_like, shape (n_gamma_ray,)
        Matching finite non-negative actual primary photon density
        ``dn_gamma/dE_gamma`` in cm^-3 erg^-1. This is not an injection rate.
    target_photon_energy_erg : array_like, shape (n_target,)
        At least two strictly increasing positive finite target-photon
        energies in the blob comoving frame, in erg.
    dn_target_photon_denergy_cm3_erg : array_like, shape (n_target,)
        Matching finite non-negative, angle-integrated isotropic target
        density ``dn_target/dE`` in cm^-3 erg^-1.

    Returns
    -------
    tuple of numpy.ndarray
        ``(Q_positron, Q_electron)`` on ``electron_gamma``, each float64 in
        cm^-3 s^-1 per unit daughter Lorentz factor. Charge symmetry makes the
        two spectra identical.

    Notes
    -----
    This is the full unpolarized differential QED result of Boettcher &
    Schlickeiser (1997), angle averaged for two labelled isotropic photon
    populations with the same fixed 128-point convention as HELENA's total
    gamma-gamma absorption. It retains both incident photon energies and is
    exactly zero outside the pair threshold and daughter support. Log-energy
    trapezoids are applied on the caller's primary and target grids.

    The primary input is a photon number density, not ``Q_gamma`` or a SED.
    No radius, photon residence/escape closure, pair transport, radiation,
    cascade, frame transformation, interpolation, or extrapolation is
    included. The two labelled photon populations are assumed distinct; a
    self-interacting single bath would require an explicit symmetry factor.
    """
    daughter = _real_1d("electron_gamma", electron_gamma, minimum_size=1, positive=True)
    if np.any(daughter <= 1.0):
        raise ValueError("electron_gamma must contain only values > 1")
    gamma_ray_energy = _real_1d(
        "gamma_ray_energy_erg",
        gamma_ray_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(gamma_ray_energy) <= 0.0):
        raise ValueError("gamma_ray_energy_erg must be strictly increasing")
    gamma_ray_density = _real_1d(
        "dn_gamma_ray_denergy_cm3_erg",
        dn_gamma_ray_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=gamma_ray_energy.shape,
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

    result = np.zeros(daughter.shape, dtype=float)
    if not np.any(gamma_ray_density) or not np.any(target_density):
        return result.copy(), result

    log_target_energy = np.log(target_energy)
    target_weight = target_energy * target_density
    primary_integrand = np.empty((daughter.size, gamma_ray_energy.size), dtype=float)
    for index, energy in enumerate(gamma_ray_energy):
        response = _isotropic_differential_pair_kernel(
            daughter,
            energy,
            target_energy,
        )
        target_integral = trapezoid(
            target_weight[None, :] * response,
            x=log_target_energy,
            axis=1,
        )
        primary_integrand[:, index] = (
            energy * gamma_ray_density[index] * target_integral
        )

    result = _C_CGS * trapezoid(
        primary_integrand,
        x=np.log(gamma_ray_energy),
        axis=1,
    )
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "gamma-gamma pair injection exceeds the finite float64 range"
        )
    return result.copy(), result
