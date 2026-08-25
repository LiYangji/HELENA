"""Local gamma-gamma absorption by isotropic comoving photon fields."""

import numpy as np
from scipy.integrate import trapezoid

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    THOMSON_CROSS_SECTION_CGS as _SIGMA_T_CGS,
)
from helena.core.contracts import (
    positive_scalar as _positive_scalar,
    real_1d as _real_1d,
)
from helena.core.ssa import _sphere_attenuation_factor
from helena.core.tables import (
    GAMMA_GAMMA_ANGLE_T as _ANGLE_T,
    GAMMA_GAMMA_ANGLE_WEIGHT as _ANGLE_WEIGHT,
)

__all__ = [
    "gamma_gamma_absorption_coefficient",
    "gamma_gamma_optical_depth",
    "gamma_gamma_absorbed_sed",
]

def _breit_wheeler_cross_section(s):
    """Return the exact Breit--Wheeler cross section for dimensionless ``s``."""
    s = np.asarray(s, dtype=float)
    active = s > 1.0
    s_safe = np.where(active, s, 1.0)
    inverse_s = 1.0 / s_safe
    beta_squared = np.clip(1.0 - inverse_s, 0.0, 1.0)
    beta = np.sqrt(beta_squared)
    log_ratio = np.log(s_safe) + 2.0 * np.log1p(beta)
    sigma_raw = (
        3.0
        * _SIGMA_T_CGS
        / 16.0
        * inverse_s
        * ((3.0 - beta_squared**2) * log_ratio - 2.0 * beta * (2.0 - beta_squared))
    )
    return np.where(active, sigma_raw, 0.0)


def _angle_averaged_cross_section(y):
    """Return the exact isotropic angle-averaged cross section for ``y``."""
    y = np.asarray(y, dtype=float)
    active = y > 1.0
    y_safe = np.where(active, y, 1.0)
    interval = y_safe - 1.0
    s = 1.0 + interval[..., None] * _ANGLE_T
    integral = np.sum(_ANGLE_WEIGHT * s * _breit_wheeler_cross_section(s), axis=-1)
    sigma_raw = 2.0 * (interval / y_safe) * (integral / y_safe)
    return np.where(active, sigma_raw, 0.0)


def gamma_gamma_absorption_coefficient(
    gamma_ray_energy_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return local gamma-gamma absorption in an isotropic comoving field.

    Parameters
    ----------
    gamma_ray_energy_erg : array_like, shape (n_gamma_ray,)
        Positive finite gamma-ray energies in the blob comoving frame, in erg.
        Their order is preserved and need not be increasing.
    target_photon_energy_erg : array_like, shape (n_target,)
        At least two strictly increasing positive finite target-photon energies
        in the blob comoving frame, in erg.
    dn_target_photon_denergy_cm3_erg : array_like, shape (n_target,)
        Finite non-negative, angle-integrated isotropic ``dn_ph/dE`` in
        cm^-3 erg^-1 on the target-energy grid.

    Returns
    -------
    numpy.ndarray, shape (n_gamma_ray,)
        Comoving local absorption coefficient ``alpha_gamma_gamma`` in cm^-1,
        following the input gamma-ray energy order.

    Notes
    -----
    This uses the exact Breit--Wheeler cross section with
    ``s = E_gamma E (1-mu) / [2 (m_e c^2)^2]``. The reaction threshold is
    ``s >= 1``; the cross section is zero at threshold and below, and non-zero
    only for ``s > 1``.
    The exact isotropic angular average is evaluated by fixed 128-point
    Gauss--Legendre quadrature, followed by trapezoidal integration in
    ``ln(E)`` on the caller's target grid. No sorting, interpolation, frame
    transformation, spatial evolution, pair injection, or observer transform
    is performed.
    """
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

    result = np.zeros(gamma_ray_energy.shape, dtype=float)
    if not np.any(target_density):
        return result

    log_target_energy = np.log(target_energy)
    target_weight = target_energy * target_density
    for index, gamma_ray_energy_value in enumerate(gamma_ray_energy):
        with np.errstate(over="ignore", invalid="ignore"):
            y = gamma_ray_energy_value * target_energy / _MEC2_CGS**2
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(
                "gamma-ray and target-photon energy product exceeds the finite "
                "float64 range"
            )
        result[index] = trapezoid(
            target_weight * _angle_averaged_cross_section(y),
            x=log_target_energy,
        )

    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "gamma-gamma absorption coefficient exceeds finite float64 range"
        )
    return result


def gamma_gamma_optical_depth(
    gamma_ray_energy_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    radius_cm,
):
    """Return the homogeneous sphere's diameter gamma-gamma optical depth.

    The photon energies and isotropic ``dn_ph/dE`` have the same blob-comoving
    definitions and units as :func:`gamma_gamma_absorption_coefficient`.
    ``radius_cm`` is the finite positive homogeneous-sphere radius in cm. The
    returned dimensionless quantity is specifically the center-to-center
    diameter optical depth ``tau_d = 2 R alpha_gamma_gamma``; it is not a
    universal path optical depth for other geometries.
    """
    radius = _positive_scalar("radius_cm", radius_cm)
    coefficient = gamma_gamma_absorption_coefficient(
        gamma_ray_energy_erg,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        optical_depth = 2.0 * radius * coefficient
    if not np.all(np.isfinite(optical_depth)):
        raise FloatingPointError(
            "gamma-gamma diameter optical depth exceeds finite float64 range"
        )
    return optical_depth


def gamma_gamma_absorbed_sed(
    intrinsic_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return a uniformly emitted comoving SED after local gamma-gamma absorption.

    ``intrinsic_sed_erg_s`` is finite non-negative comoving ``E L_E`` in
    erg s^-1. ``absorption_coefficient_cm1`` is the matching finite
    non-negative local gamma-gamma coefficient in cm^-1, and ``radius_cm`` is
    the finite positive homogeneous-sphere radius in cm. Arrays must be
    one-dimensional, non-empty, and have identical shapes.

    The returned escaped comoving ``E L_E`` in erg s^-1 is the intrinsic SED
    multiplied by HELENA's homogeneous-sphere attenuation at diameter
    optical depth ``tau_d = 2 R alpha``. This is mixed-emitter transfer, not a
    foreground screen. Apply any observer transformation only afterward.
    """
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
    with np.errstate(over="ignore", invalid="ignore"):
        optical_depth = 2.0 * radius * coefficient
    if not np.all(np.isfinite(optical_depth)):
        raise FloatingPointError(
            "gamma-gamma diameter optical depth exceeds finite float64 range"
        )
    return intrinsic_sed * _sphere_attenuation_factor(optical_depth)
