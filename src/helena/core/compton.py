"""Isotropic inverse-Compton emission in the blob comoving frame."""

import numpy as np
from scipy.integrate import trapezoid

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
    THOMSON_CROSS_SECTION_CGS as _SIGMA_T_CGS,
)
from helena.core.contracts import (
    positive_scalar as _positive_scalar,
    real_1d as _real_1d,
)
from helena.core.photon_fields import transform_isotropic_external_photon_field

__all__ = ["inverse_compton_sed", "external_inverse_compton_sed"]


def inverse_compton_sed(
    scattered_photon_energy_erg,
    electron_gamma,
    dn_electron_dgamma_cm3,
    seed_photon_energy_erg,
    dn_seed_photon_denergy_cm3_erg,
    *,
    radius_cm,
):
    """Return isotropic inverse-Compton ``E_s L_Es`` in the comoving frame.

    Parameters are positive comoving scattered photon energies in erg; at
    least two strictly increasing Lorentz factors greater than one and their
    non-negative ``dn_e/dgamma`` in cm^-3; at least two strictly increasing
    seed energies in erg and their non-negative ``dn_ph/dE`` in
    cm^-3 erg^-1; and the positive homogeneous-sphere radius in cm. The output
    follows the scattered-energy order and is comoving ``E_s L_Es`` in
    erg s^-1.

    The calculation uses the full angle-averaged Klein--Nishina kernel for
    isotropic electrons and photons. A caller-supplied external field must
    already be angle-averaged, isotropic, and transformed to the blob frame;
    this is an angle-averaged EC approximation, not anisotropic EC. The caller
    is responsible for any host-to-blob bulk-Lorentz-factor transformation.
    The isotropic result may subsequently use HELENA's isotropic observer
    transformation. No interpolation, extrapolation, cooling, or transport is
    performed.
    """
    scattered = _real_1d(
        "scattered_photon_energy_erg",
        scattered_photon_energy_erg,
        minimum_size=1,
        positive=True,
    )
    gamma = _real_1d("electron_gamma", electron_gamma, minimum_size=2, positive=True)
    if np.any(gamma <= 1.0) or np.any(np.diff(gamma) <= 0.0):
        raise ValueError("electron_gamma must be strictly increasing and > 1")
    electron_density = _real_1d(
        "dn_electron_dgamma_cm3",
        dn_electron_dgamma_cm3,
        minimum_size=2,
        positive=False,
        shape=gamma.shape,
    )
    seed = _real_1d(
        "seed_photon_energy_erg",
        seed_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(seed) <= 0.0):
        raise ValueError("seed_photon_energy_erg must be strictly increasing")
    seed_density = _real_1d(
        "dn_seed_photon_denergy_cm3_erg",
        dn_seed_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=seed.shape,
    )
    radius = _positive_scalar("radius_cm", radius_cm)

    result = np.zeros(scattered.shape, dtype=float)
    if not np.any(electron_density) or not np.any(seed_density):
        return result

    epsilon = seed / _MEC2_CGS
    log_epsilon = np.log(epsilon)
    log_gamma = np.log(gamma)
    if not np.all(np.isfinite(log_epsilon)) or not np.all(np.isfinite(log_gamma)):
        raise FloatingPointError("logarithmic integration grid is not finite")

    gamma_2d = gamma[:, None]
    gamma_e = 4.0 * gamma_2d * epsilon[None, :]
    weights = electron_density[:, None] / gamma_2d * (_MEC2_CGS * seed_density)[None, :]
    prefactor = 3.0 * _SIGMA_T_CGS * _C_CGS / 4.0

    for index, scattered_energy in enumerate(scattered):
        epsilon_s = scattered_energy / _MEC2_CGS
        allowed_gamma = gamma_2d > epsilon_s
        denominator = gamma_e * (gamma_2d - epsilon_s)
        q = np.zeros_like(denominator)
        np.divide(epsilon_s, denominator, out=q, where=allowed_gamma)
        allowed = allowed_gamma & (q >= 1.0 / (4.0 * gamma_2d**2)) & (q <= 1.0)

        kernel = np.zeros_like(q)
        q_allowed = q[allowed]
        gamma_q = gamma_e[allowed] * q_allowed
        kernel[allowed] = (
            2.0 * q_allowed * np.log(q_allowed)
            + (1.0 + 2.0 * q_allowed) * (1.0 - q_allowed)
            + gamma_q**2 * (1.0 - q_allowed) / (2.0 * (1.0 + gamma_q))
        )
        seed_integral = trapezoid(weights * kernel, x=log_epsilon, axis=1)
        rate = prefactor * trapezoid(seed_integral, x=log_gamma)
        result[index] = 4.0 * np.pi * radius**3 / 3.0 * _MEC2_CGS * epsilon_s**2 * rate

    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("inverse-Compton SED exceeds finite float64 range")
    return result


def external_inverse_compton_sed(
    scattered_photon_energy_comoving_erg,
    electron_gamma,
    dn_electron_dgamma_cm3,
    external_photon_energy_host_erg,
    dn_external_photon_denergy_host_cm3_erg,
    external_photon_energy_comoving_erg,
    *,
    bulk_lorentz_factor,
    radius_cm,
):
    """Return angle-averaged external-field IC emission in the blob frame.

    The external photon energies and angle-integrated isotropic ``dn_ph/dE``
    are supplied in the AGN/host rest frame, in erg and cm^-3 erg^-1.
    ``bulk_lorentz_factor`` is the finite jet bulk Lorentz factor ``Gamma >= 1``.
    Electron Lorentz factors, ``dn_e/dgamma`` in cm^-3, the explicitly supplied
    external-photon output grid, and scattered photon energies are in the blob
    comoving frame; ``radius_cm`` is the homogeneous blob radius in cm.

    The host field is first Lorentz transformed and angle averaged by
    :func:`transform_isotropic_external_photon_field`, then passed unchanged to
    :func:`inverse_compton_sed`. The returned array follows the scattered-energy
    grid and is comoving ``E'_s L'_{E'_s}`` in erg s^-1. This is an isotropic,
    angle-averaged external-field approximation, not a calculation of the
    anisotropic external-Compton angular pattern or observer beaming.
    """
    comoving_density = transform_isotropic_external_photon_field(
        external_photon_energy_host_erg,
        dn_external_photon_denergy_host_cm3_erg,
        external_photon_energy_comoving_erg,
        bulk_lorentz_factor=bulk_lorentz_factor,
    )
    return inverse_compton_sed(
        scattered_photon_energy_erg=scattered_photon_energy_comoving_erg,
        electron_gamma=electron_gamma,
        dn_electron_dgamma_cm3=dn_electron_dgamma_cm3,
        seed_photon_energy_erg=external_photon_energy_comoving_erg,
        dn_seed_photon_denergy_cm3_erg=comoving_density,
        radius_cm=radius_cm,
    )
