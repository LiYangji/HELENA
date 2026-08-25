"""Comoving photon-field helpers for a homogeneous sphere."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from helena.core.constants import SPEED_OF_LIGHT_CGS as _C_CGS
from helena.core.contracts import nonempty_real_1d as _real_1d
from helena.core.ssa import _sphere_attenuation_factor

__all__ = [
    "produced_sed_to_internal_photon_density",
    "synchrotron_sed_to_internal_photon_density",
    "transform_isotropic_external_photon_field",
]

_EXTERNAL_FIELD_MU, _EXTERNAL_FIELD_WEIGHT = leggauss(128)


def transform_isotropic_external_photon_field(
    photon_energy_host_erg,
    dn_photon_denergy_host_cm3_erg,
    photon_energy_comoving_erg,
    *,
    bulk_lorentz_factor,
):
    """Transform a host-frame isotropic photon spectrum to the blob frame.

    ``photon_energy_host_erg`` contains at least two strictly increasing
    positive photon energies in the AGN/host rest frame. The matching
    ``dn_photon_denergy_host_cm3_erg`` is the finite non-negative,
    angle-integrated isotropic ``dn_ph/dE`` in cm^-3 erg^-1.
    ``photon_energy_comoving_erg`` is an independently supplied, strictly
    increasing positive output grid in the blob comoving frame, and
    ``bulk_lorentz_factor`` is the finite bulk Lorentz factor ``Gamma >= 1``.

    The returned array follows the comoving grid and is the angle-averaged
    comoving ``dn'_ph/dE'`` in cm^-3 erg^-1,

    ``n'_E'(E') = 1/2 integral_-1^1 n_E(E'/D(mu)) dmu``,

    where ``D(mu) = Gamma (1 - beta_Gamma mu)``. The angular average uses a
    fixed 128-point Gauss--Legendre quadrature. Values at shifted energies are
    obtained by piecewise-linear interpolation of ``dn_ph/dE`` as a function
    of ``ln(E)``; values outside the host energy range are exactly zero. No
    sorting, extrapolation, smoothing, input mutation, observer transformation,
    or anisotropic external-Compton calculation is performed.
    """
    host_energy = _real_1d(
        "photon_energy_host_erg", photon_energy_host_erg, positive=True
    )
    if host_energy.size < 2 or np.any(np.diff(host_energy) <= 0.0):
        raise ValueError(
            "photon_energy_host_erg must contain at least two strictly "
            "increasing values"
        )
    host_density = _real_1d(
        "dn_photon_denergy_host_cm3_erg",
        dn_photon_denergy_host_cm3_erg,
        positive=False,
        shape=host_energy.shape,
    )
    comoving_energy = _real_1d(
        "photon_energy_comoving_erg", photon_energy_comoving_erg, positive=True
    )
    if comoving_energy.size < 2 or np.any(np.diff(comoving_energy) <= 0.0):
        raise ValueError(
            "photon_energy_comoving_erg must contain at least two strictly "
            "increasing values"
        )

    lorentz_array = np.asarray(bulk_lorentz_factor)
    if lorentz_array.ndim != 0 or lorentz_array.dtype.kind not in "iuf":
        raise ValueError("bulk_lorentz_factor must be a finite real scalar >= 1")
    lorentz_factor = float(lorentz_array)
    if not np.isfinite(lorentz_factor) or lorentz_factor < 1.0:
        raise ValueError("bulk_lorentz_factor must be a finite real scalar >= 1")

    if not np.any(host_density):
        return np.zeros(comoving_energy.shape, dtype=float)

    inverse_lorentz_factor = 1.0 / lorentz_factor
    beta = np.sqrt((1.0 - inverse_lorentz_factor) * (1.0 + inverse_lorentz_factor))
    doppler = lorentz_factor * (1.0 - beta * _EXTERNAL_FIELD_MU)
    shifted_log_energy = np.log(comoving_energy[:, None] / doppler[None, :])
    interpolated = np.interp(
        shifted_log_energy.ravel(),
        np.log(host_energy),
        host_density,
        left=0.0,
        right=0.0,
    ).reshape(shifted_log_energy.shape)
    result = 0.5 * (interpolated @ _EXTERNAL_FIELD_WEIGHT)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "transformed external photon field exceeds finite float64 range"
        )
    return result


def _homogeneous_sphere_effective_path_length(
    absorption_coefficient_cm1,
    radius_cm,
):
    alpha = np.asarray(absorption_coefficient_cm1, dtype=float)
    tau = 2.0 * alpha * radius_cm
    ratio = np.empty_like(tau)
    small = tau <= 0.1
    small_tau = tau[small]
    ratio[small] = (
        2.0
        * radius_cm
        * (
            3.0 / 8.0
            + small_tau
            * (
                -1.0 / 10.0
                + small_tau
                * (
                    1.0 / 48.0
                    + small_tau
                    * (-1.0 / 280.0 + small_tau * (1.0 / 1920.0 - small_tau / 15120.0))
                )
            )
        )
    )
    regular = ~small
    if np.any(regular):
        ratio[regular] = (1.0 - _sphere_attenuation_factor(tau[regular])) / alpha[
            regular
        ]
    return ratio


def produced_sed_to_internal_photon_density(
    photon_energy_erg,
    produced_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return a homogeneous sphere's volume-angle-averaged photon density.

    ``photon_energy_erg`` is a non-empty positive finite comoving energy grid.
    ``produced_sed_erg_s`` is the matching intrinsic comoving ``E L_E`` in
    erg s^-1, ``absorption_coefficient_cm1`` is the matching finite
    non-negative local absorption coefficient in cm^-1, and ``radius_cm`` is
    the finite positive homogeneous-sphere radius in cm.
    Observer-transformed or Doppler-boosted SEDs are not valid inputs.

    The returned ``dn_ph/dE`` has units cm^-3 erg^-1 and uses the Dermer
    homogeneous-sphere volume-and-direction average with effective path
    ``[1 - P_esc(2 R alpha)] / alpha``. Its transparent limit is
    ``9 S_intrinsic / (16 pi R^2 c E^2)``. Negative absorption, ray tracing,
    kinetic evolution, and source-specific radiation physics are outside this
    helper's scope.
    """
    energy = _real_1d("photon_energy_erg", photon_energy_erg, positive=True)
    produced_sed = _real_1d(
        "produced_sed_erg_s",
        produced_sed_erg_s,
        positive=False,
        shape=energy.shape,
    )
    alpha = _real_1d(
        "absorption_coefficient_cm1",
        absorption_coefficient_cm1,
        positive=False,
        shape=energy.shape,
    )
    radius_array = np.asarray(radius_cm)
    if radius_array.ndim != 0 or radius_array.dtype.kind not in "iuf":
        raise ValueError("radius_cm must be a finite positive real scalar")
    radius = float(radius_array)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_cm must be a finite positive real scalar")

    if not np.any(produced_sed):
        return np.zeros(energy.shape, dtype=float)

    ratio = _homogeneous_sphere_effective_path_length(alpha, radius)
    volume = 4.0 * np.pi * radius**3 / 3.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        density = produced_sed * ratio / (volume * _C_CGS * energy**2)
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise FloatingPointError("internal photon density exceeds finite float64 range")
    return density


def synchrotron_sed_to_internal_photon_density(
    photon_energy_erg,
    thin_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return the SSA-aware volume-angle-averaged synchrotron photon density.

    Inputs, units, and homogeneous-sphere assumptions match
    :func:`produced_sed_to_internal_photon_density`. ``thin_sed_erg_s`` is the
    optically thin synchrotron ``E L_E`` in erg s^-1. The returned
    ``dn_syn/dE`` is in cm^-3 erg^-1. Negative absorption (maser transfer),
    ray tracing, and kinetic evolution are outside this helper's scope.
    """
    return produced_sed_to_internal_photon_density(
        photon_energy_erg,
        thin_sed_erg_s,
        absorption_coefficient_cm1,
        radius_cm=radius_cm,
    )
