"""Comoving synchrotron self-absorption for a homogeneous sphere."""

import numpy as np
from scipy.integrate import trapezoid

from helena.core.constants import (
    ELECTRON_MASS_CGS as _M_E_CGS,
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    PLANCK_CONSTANT_CGS as _H_CGS,
    PROTON_MASS_CGS as _M_P_CGS,
    SSA_NEGATIVE_ALPHA_ROUNDOFF_FACTOR as _NEGATIVE_ALPHA_ROUNDOFF_FACTOR,
)
from helena.core.contracts import (
    absorption_inputs as _absorption_inputs,
    finite_scalar as _finite_scalar,
)
from helena.core.synchrotron import synchrotron_kernel, synchrotron_sed

__all__ = [
    "synchrotron_absorption_coefficient",
    "synchrotron_self_absorbed_sed",
    "electron_synchrotron_self_absorbed_sed",
    "proton_synchrotron_self_absorbed_sed",
]


def _absorption_coefficient_and_scale(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
):
    (
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G,
        particle_mass_g,
        charge_esu,
    ) = _absorption_inputs(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=particle_mass_g,
        charge_esu=charge_esu,
    )

    zeros = np.zeros(photon_energy_erg.shape, dtype=float)
    if magnetic_field_G == 0.0 or not np.any(dn_dgamma_cm3):
        return zeros, zeros

    log_gamma = np.log(gamma)
    if np.any(np.diff(log_gamma) <= 0.0):
        raise FloatingPointError("ln(gamma) grid is not numerically resolvable")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower = log_gamma[1:-1] - log_gamma[:-2]
        upper = log_gamma[2:] - log_gamma[1:-1]
        lower_slope = (dn_dgamma_cm3[1:-1] - dn_dgamma_cm3[:-2]) / lower
        upper_slope = (dn_dgamma_cm3[2:] - dn_dgamma_cm3[1:-1]) / upper
        derivative = (upper * lower_slope + lower * upper_slope) / (lower + upper)
        absorption_weight = 2.0 * dn_dgamma_cm3[1:-1] - derivative
    if not np.all(np.isfinite(absorption_weight)):
        raise FloatingPointError("particle-distribution derivative is not finite")

    power_nu = _H_CGS * synchrotron_kernel(
        photon_energy_erg,
        gamma,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=particle_mass_g,
        charge_esu=charge_esu,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        integrand = power_nu[:, 1:-1] * absorption_weight[None, :]
    if not np.all(np.isfinite(integrand)):
        raise FloatingPointError("synchrotron absorption integrand is not finite")

    integral = trapezoid(integrand, x=log_gamma[1:-1], axis=1)
    absolute_integral = trapezoid(np.abs(integrand), x=log_gamma[1:-1], axis=1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        frequency_hz = photon_energy_erg / _H_CGS
        denominator = 8.0 * np.pi * particle_mass_g * frequency_hz**2
        coefficient = integral / denominator
        coefficient_scale = absolute_integral / denominator
    if (
        not np.all(np.isfinite(coefficient))
        or not np.all(np.isfinite(coefficient_scale))
        or np.any(coefficient_scale < 0.0)
    ):
        raise FloatingPointError(
            "synchrotron absorption coefficient exceeds the finite float64 range"
        )
    return coefficient, coefficient_scale


def _sphere_attenuation_factor(tau_diameter):
    """Return the homogeneous-sphere attenuation for diameter optical depth."""
    tau_diameter = np.asarray(tau_diameter)
    if tau_diameter.dtype.kind not in "iuf":
        raise ValueError("tau_diameter must contain finite non-negative values")
    tau_diameter = tau_diameter.astype(float, copy=False)
    if not np.all(np.isfinite(tau_diameter)) or np.any(tau_diameter < 0.0):
        raise ValueError("tau_diameter must contain finite non-negative values")

    result = np.empty_like(tau_diameter)
    zero = tau_diameter == 0.0
    result[zero] = 1.0

    small = (tau_diameter > 0.0) & (tau_diameter <= 0.1)
    tau = tau_diameter[small]
    result[small] = 1.0 + tau * (
        -3.0 / 8.0
        + tau
        * (
            1.0 / 10.0
            + tau
            * (
                -1.0 / 48.0
                + tau * (1.0 / 280.0 + tau * (-1.0 / 1920.0 + tau / 15120.0))
            )
        )
    )

    regular = tau_diameter > 0.1
    tau = tau_diameter[regular]
    inverse_tau = 1.0 / tau
    result[regular] = (
        3.0
        * inverse_tau
        * (0.5 - (1.0 - (1.0 + tau) * np.exp(-tau)) * inverse_tau * inverse_tau)
    )

    if not np.all(np.isfinite(result)) or np.any(result <= 0.0) or np.any(result > 1.0):
        raise FloatingPointError("failed to evaluate the sphere attenuation factor")
    return result


def synchrotron_absorption_coefficient(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
):
    """Return the signed comoving synchrotron absorption coefficient.

    Parameters
    ----------
    photon_energy_erg : array_like, shape (n_energy,)
        Positive finite comoving photon energies in erg; order is preserved.
    gamma : array_like, shape (n_gamma,)
        At least three strictly increasing finite Lorentz factors greater than
        1.
    dn_dgamma_cm3 : array_like, shape (n_gamma,)
        Finite non-negative comoving ``dn/dgamma`` in cm^-3. This is not
        ``dn/dE``, ``dn/dln(gamma)``, or a volume-integrated distribution.
    magnetic_field_G : float
        Finite non-negative comoving magnetic-field magnitude in Gauss.
    particle_mass_g : float
        Finite positive particle mass in grams.
    charge_esu : float
        Finite positive charge magnitude in esu/statC.

    Returns
    -------
    numpy.ndarray, shape (n_energy,)
        Signed comoving ``alpha_nu`` in cm^-1.

    Notes
    -----
    The calculation uses ``nu = E/h``, ``P_nu = h P_E``, a second-order
    numerical derivative in ``ln(gamma)``, and trapezoidal integration on the
    interior of the caller-supplied gamma grid. The electron-grid endpoints do
    not participate in the derivative integral. The coefficient is not
    clipped: a population inversion can produce negative absorption.
    Derivatives make SSA sensitive to grid resolution; callers must check
    convergence for noisy distributions and sharp cutoffs.

    The underlying kernel supplies the exact isotropic pitch-angle average.
    The model is classical, assumes ``gamma >> 1`` and ``chi << 1``, and does
    not include maser transfer, cooling evolution, particle transport,
    observer-frame transformations, or any interpolation or extrapolation.
    """
    coefficient, _ = _absorption_coefficient_and_scale(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=particle_mass_g,
        charge_esu=charge_esu,
    )
    return coefficient


def synchrotron_self_absorbed_sed(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
    particle_mass_g,
    charge_esu,
):
    """Return the comoving SSA synchrotron SED of a homogeneous sphere.

    ``photon_energy_erg`` is a non-empty one-dimensional array of positive
    finite comoving energies in erg. ``gamma`` contains at least three strictly
    increasing finite Lorentz factors greater than 1 and matches the finite,
    non-negative ``dn_dgamma_cm3 = dn/dgamma`` array in cm^-3. The distribution
    is neither per energy nor per log-gamma and has not been multiplied by
    volume. ``magnetic_field_G`` is finite and non-negative in Gauss;
    ``radius_cm`` is the finite positive radius of the uniform comoving sphere;
    mass is in grams and the positive charge magnitude is in esu/statC.

    The returned shape follows ``photon_energy_erg`` and contains comoving
    emergent ``E L_E = nu L_nu`` in erg s^-1. It multiplies the existing thin
    SED by the Dermer homogeneous-sphere factor at full diameter optical depth
    ``tau_d = 2 alpha_nu radius_cm``. The sphere factor is not a slab factor.

    The exact isotropic pitch-angle average, classical ``gamma >> 1`` and
    ``chi << 1`` range, caller-supplied gamma grid, and derivative sensitivity
    of :func:`synchrotron_absorption_coefficient` apply. Clearly negative
    absorption (maser transfer) is rejected; only roundoff-sized negative
    values are treated as zero for transfer. No cooling, transport, observer
    transformation, interpolation, smoothing, or extrapolation is included.
    """
    radius_cm = _finite_scalar("radius_cm", radius_cm, allow_zero=False)
    coefficient, coefficient_scale = _absorption_coefficient_and_scale(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=particle_mass_g,
        charge_esu=charge_esu,
    )
    thin_sed = synchrotron_sed(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=particle_mass_g,
        charge_esu=charge_esu,
    )

    negative_roundoff = _NEGATIVE_ALPHA_ROUNDOFF_FACTOR * coefficient_scale
    if np.any(coefficient < -negative_roundoff):
        raise ValueError("negative synchrotron absorption requires maser transfer")
    transfer_coefficient = np.where(coefficient < 0.0, 0.0, coefficient)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        tau_diameter = 2.0 * transfer_coefficient * radius_cm
    if not np.all(np.isfinite(tau_diameter)):
        raise FloatingPointError(
            "diameter optical depth exceeds the finite float64 range"
        )
    attenuation = _sphere_attenuation_factor(tau_diameter)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = thin_sed * attenuation
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("self-absorbed SED exceeds the finite float64 range")
    return result


def electron_synchrotron_self_absorbed_sed(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return a comoving electron SSA ``E L_E`` in erg s^-1.

    Inputs follow :func:`synchrotron_self_absorbed_sed`: photon energy is in
    erg, ``dn_dgamma_cm3`` is a tabulated comoving ``dn/dgamma`` in cm^-3,
    magnetic field is in Gauss, and ``radius_cm`` is the radius of the uniform
    comoving sphere. The output has the photon-energy shape. This thin wrapper
    fixes the Astropy electron mass and elementary-charge magnitude and inherits
    the exact isotropic, classical ``gamma >> 1`` and ``chi << 1`` assumptions,
    Dermer sphere transfer, derivative sensitivity, negative-absorption policy,
    and exclusions of cooling, transport, and observer-frame transformations.
    """
    return synchrotron_self_absorbed_sed(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_E_CGS,
        charge_esu=_E_ESU,
    )


def proton_synchrotron_self_absorbed_sed(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return a comoving proton SSA ``E L_E`` in erg s^-1.

    Inputs follow :func:`synchrotron_self_absorbed_sed`: photon energy is in
    erg, ``dn_dgamma_cm3`` is a tabulated comoving ``dn/dgamma`` in cm^-3,
    magnetic field is in Gauss, and ``radius_cm`` is the radius of the uniform
    comoving sphere. The output has the photon-energy shape. This thin wrapper
    fixes the Astropy proton mass and elementary-charge magnitude and inherits
    the exact isotropic, classical ``gamma >> 1`` and ``chi << 1`` assumptions,
    Dermer sphere transfer, derivative sensitivity, negative-absorption policy,
    and exclusions of cooling, transport, and observer-frame transformations.
    """
    return synchrotron_self_absorbed_sed(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_P_CGS,
        charge_esu=_E_ESU,
    )
