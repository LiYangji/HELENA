"""Classical single-particle synchrotron radiation in the comoving frame."""

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import kve

from helena.core.constants import (
    ELECTRON_MASS_CGS as _M_E_CGS,
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    PLANCK_CONSTANT_CGS as _H_CGS,
    PROTON_MASS_CGS as _M_P_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
    SYNCHROTRON_LARGE_X as _LARGE_X,
    SYNCHROTRON_LOG_COOLING_COEFFICIENT as _LOG_COOLING_COEFFICIENT,
    SYNCHROTRON_SMALL_X as _SMALL_X,
    SYNCHROTRON_SMALL_X_COEFFICIENT as _SMALL_X_COEFFICIENT,
)
from helena.core.contracts import (
    finite_scalar as _finite_scalar,
    gamma_squared as _gamma_squared,
    nonnegative_1d as _nonnegative_1d,
    positive_1d as _positive_1d,
    positive_scalar_or_1d as _positive_scalar_or_1d,
)

__all__ = [
    "synchrotron_kernel",
    "synchrotron_sed",
    "electron_synchrotron_sed",
    "proton_synchrotron_sed",
    "power_law_distribution",
    "synchrotron_total_power",
    "synchrotron_cooling_time",
    "synchrotron_cooling_lorentz_factor",
    "electron_synchrotron_cooling_time",
    "proton_synchrotron_cooling_time",
]

_LOG_TOTAL_POWER_COEFFICIENT = np.log(4.0 / 9.0) - 3.0 * np.log(_C_CGS)


def _synchrotron_diagnostic_inputs(
    values_name,
    values,
    *,
    lower_bound,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
):
    values = _positive_scalar_or_1d(values_name, values, lower_bound=lower_bound)
    magnetic_field_G = _finite_scalar(
        "magnetic_field_G", magnetic_field_G, allow_zero=True
    )
    particle_mass_g = _finite_scalar(
        "particle_mass_g", particle_mass_g, allow_zero=False
    )
    charge_esu = _finite_scalar("charge_esu", charge_esu, allow_zero=False)
    return values, magnetic_field_G, particle_mass_g, charge_esu


def _checked_positive_power(value, exponent, *, diagnostic):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        powered = np.power(np.float64(value), exponent)
    if not np.isfinite(powered) or powered <= 0.0:
        raise FloatingPointError(
            f"{diagnostic} exceeds the finite float64 numerical range"
        )
    return powered


def _diagnostic_from_log(log_result, *, diagnostic):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.asarray(np.exp(log_result), dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError(
            f"{diagnostic} exceeds the finite float64 numerical range"
        )
    return result


def _isotropic_synchrotron_function(x):
    """Return the exact isotropic pitch-angle synchrotron function R(x)."""
    x = np.asarray(x, dtype=float)
    if np.any(np.isnan(x)) or np.any(x < 0.0):
        raise ValueError("x must be non-negative and not NaN")

    result = np.zeros_like(x)

    small = x <= _SMALL_X
    small_x = x[small]
    result[small] = (
        _SMALL_X_COEFFICIENT * np.cbrt(small_x) - np.pi / np.sqrt(3.0) * small_x
    )

    middle = (x > _SMALL_X) & (x < _LARGE_X)
    middle_x = x[middle]
    z = middle_x / 2.0
    k_4_3 = kve(4.0 / 3.0, z)
    k_1_3 = kve(1.0 / 3.0, z)
    difference_of_squares = (k_4_3 - k_1_3) * (k_4_3 + k_1_3)
    scaled = k_4_3 * k_1_3 - 3.0 * middle_x / 10.0 * difference_of_squares
    result[middle] = middle_x**2 / 2.0 * np.exp(-middle_x) * scaled

    large = x >= _LARGE_X
    large_x = x[large]
    result[large] = np.exp(
        np.log(np.pi / 2.0) + np.log1p(-11.0 / (18.0 * large_x)) - large_x
    )

    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("failed to evaluate a finite non-negative R(x)")
    return result


def synchrotron_kernel(
    photon_energy_erg,
    gamma,
    *,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
):
    """Return the isotropic single-particle synchrotron power per photon energy.

    Parameters
    ----------
    photon_energy_erg : array_like, shape (n_energy,)
        Positive finite comoving-frame photon energies in erg. Input order is
        preserved.
    gamma : array_like, shape (n_gamma,)
        Positive finite particle Lorentz factors, all strictly greater than 1.
        The kernel does not require them to be ordered.
    magnetic_field_G : float
        Finite comoving magnetic-field magnitude in Gauss. Zero returns an
        exact zero matrix after all other inputs are validated.
    particle_mass_g : float
        Finite positive particle mass in grams.
    charge_esu : float
        Finite positive charge magnitude in esu/statC.

    Returns
    -------
    numpy.ndarray, shape (n_energy, n_gamma)
        Angle-integrated single-particle ``P_E`` in
        erg s^-1 erg^-1, averaged exactly over an isotropic pitch-angle
        distribution.

    Notes
    -----
    This is a classical, optically thin, ultrarelativistic continuum kernel in
    the emitting region's comoving frame. Its physical range is ``gamma >> 1``
    and quantum parameter ``chi << 1``. It does not include cyclotron or
    gyrosynchrotron radiation, synchrotron self-absorption, particle-distribution
    integration, cooling, particle transport, or any observer-frame
    transformation.
    """
    photon_energy_erg = _positive_1d(
        "photon_energy_erg", photon_energy_erg, lower_bound=0.0
    )
    gamma = _positive_1d("gamma", gamma, lower_bound=1.0)
    magnetic_field_G = _finite_scalar(
        "magnetic_field_G", magnetic_field_G, allow_zero=True
    )
    particle_mass_g = _finite_scalar(
        "particle_mass_g", particle_mass_g, allow_zero=False
    )
    charge_esu = _finite_scalar("charge_esu", charge_esu, allow_zero=False)

    gamma_squared = _gamma_squared(gamma)

    shape = (photon_energy_erg.size, gamma.size)
    if magnetic_field_G == 0.0:
        return np.zeros(shape, dtype=float)

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        frequency_hz = photon_energy_erg[:, None] / _H_CGS
        critical_frequency_hz = (
            3.0
            * charge_esu
            * magnetic_field_G
            * gamma_squared[None, :]
            / (4.0 * np.pi * particle_mass_g * _C_CGS)
        )
        x = frequency_hz / critical_frequency_hz
        prefactor = (
            np.sqrt(3.0)
            * charge_esu**3
            * magnetic_field_G
            / (particle_mass_g * _C_CGS**2 * _H_CGS)
        )
        result = prefactor * _isotropic_synchrotron_function(x)

    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("inputs exceed the finite float64 kernel range")
    return result


def synchrotron_total_power(
    gamma,
    *,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
) -> np.ndarray:
    """Return isotropic single-particle total synchrotron power.

    Parameters
    ----------
    gamma : float or array_like, shape () or (n_gamma,)
        Finite dimensionless Lorentz factor(s), all strictly greater than 1.
    magnetic_field_G : float
        Finite comoving magnetic-field magnitude in Gauss, greater than or
        equal to zero.
    particle_mass_g : float
        Finite positive particle mass in grams.
    charge_esu : float
        Finite positive charge magnitude in esu/statC.

    Returns
    -------
    numpy.ndarray
        Comoving total power in erg s^-1 with the same shape as ``gamma``.
        A zero magnetic field returns exact zero.

    Notes
    -----
    The exact isotropic pitch-angle average is
    ``P_syn = 4 |q|^4 B^2 gamma^2 / (9 m^2 c^3)``. This is a classical,
    ultrarelativistic (``gamma >> 1``), ``chi << 1`` result. It has no
    ``beta^2`` correction and does not model cyclotron/gyrosynchrotron,
    quantum synchrotron, SSA, cooling evolution, particle transport, or
    observer-frame transformations.
    """
    gamma, magnetic_field_G, particle_mass_g, charge_esu = (
        _synchrotron_diagnostic_inputs(
            "gamma",
            gamma,
            lower_bound=1.0,
            magnetic_field_G=magnetic_field_G,
            particle_mass_g=particle_mass_g,
            charge_esu=charge_esu,
        )
    )
    try:
        _gamma_squared(gamma)
    except FloatingPointError:
        raise FloatingPointError(
            "synchrotron total power exceeds the finite float64 numerical "
            "range because gamma**2 is non-finite"
        ) from None

    if magnetic_field_G == 0.0:
        return np.zeros_like(gamma, dtype=float)

    diagnostic = "synchrotron total power"
    _checked_positive_power(magnetic_field_G, 2, diagnostic=diagnostic)
    _checked_positive_power(particle_mass_g, 2, diagnostic=diagnostic)
    _checked_positive_power(charge_esu, 4, diagnostic=diagnostic)
    log_result = (
        _LOG_TOTAL_POWER_COEFFICIENT
        + 4.0 * np.log(charge_esu)
        + 2.0 * np.log(magnetic_field_G)
        + 2.0 * np.log(gamma)
        - 2.0 * np.log(particle_mass_g)
    )
    return _diagnostic_from_log(log_result, diagnostic=diagnostic)


def synchrotron_cooling_time(
    gamma,
    *,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
) -> np.ndarray:
    """Return the ultrarelativistic synchrotron cooling time.

    Parameters
    ----------
    gamma : float or array_like, shape () or (n_gamma,)
        Finite dimensionless Lorentz factor(s), all strictly greater than 1.
    magnetic_field_G : float
        Finite comoving magnetic-field magnitude in Gauss, greater than or
        equal to zero.
    particle_mass_g : float
        Finite positive particle mass in grams.
    charge_esu : float
        Finite positive charge magnitude in esu/statC.

    Returns
    -------
    numpy.ndarray
        Comoving cooling time in seconds with the same shape as ``gamma``.
        A zero magnetic field returns positive infinity.

    Notes
    -----
    Using particle energy ``gamma m c^2`` and the exact isotropic total power,
    ``t_syn = 9 m^3 c^5 / (4 |q|^4 B^2 gamma)``. This classical result assumes
    ``gamma >> 1`` and ``chi << 1`` and deliberately has no ``beta^2``
    correction. It excludes cyclotron/gyrosynchrotron and quantum corrections,
    SSA, transport, cooling evolution, and observer-frame transformations.
    """
    gamma, magnetic_field_G, particle_mass_g, charge_esu = (
        _synchrotron_diagnostic_inputs(
            "gamma",
            gamma,
            lower_bound=1.0,
            magnetic_field_G=magnetic_field_G,
            particle_mass_g=particle_mass_g,
            charge_esu=charge_esu,
        )
    )
    if magnetic_field_G == 0.0:
        return np.full_like(gamma, np.inf, dtype=float)

    diagnostic = "synchrotron cooling time"
    _checked_positive_power(magnetic_field_G, 2, diagnostic=diagnostic)
    _checked_positive_power(particle_mass_g, 3, diagnostic=diagnostic)
    _checked_positive_power(charge_esu, 4, diagnostic=diagnostic)
    log_result = (
        _LOG_COOLING_COEFFICIENT
        + 3.0 * np.log(particle_mass_g)
        - 4.0 * np.log(charge_esu)
        - 2.0 * np.log(magnetic_field_G)
        - np.log(gamma)
    )
    return _diagnostic_from_log(log_result, diagnostic=diagnostic)


def synchrotron_cooling_lorentz_factor(
    timescale_s,
    *,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
) -> np.ndarray:
    """Return the cooling Lorentz factor for a supplied comoving timescale.

    Parameters
    ----------
    timescale_s : float or array_like, shape () or (n_timescale,)
        Finite positive comoving timescale(s) in seconds.
    magnetic_field_G : float
        Finite comoving magnetic-field magnitude in Gauss, greater than or
        equal to zero.
    particle_mass_g : float
        Finite positive particle mass in grams.
    charge_esu : float
        Finite positive charge magnitude in esu/statC.

    Returns
    -------
    numpy.ndarray
        Dimensionless ``gamma_c`` with the same shape as ``timescale_s``.
        A zero magnetic field returns positive infinity.

    Notes
    -----
    The result solves ``t_syn(gamma_c) = timescale_s`` and is
    ``gamma_c = 9 m^3 c^5 / (4 |q|^4 B^2 timescale_s)``. It shares the exact
    isotropic, classical ``gamma >> 1`` and ``chi << 1`` assumptions of the
    synchrotron kernel and has no ``beta^2`` correction. It excludes
    cyclotron/gyrosynchrotron and quantum corrections, SSA, transport, cooling
    evolution, and observer-frame transformations.
    """
    timescale_s, magnetic_field_G, particle_mass_g, charge_esu = (
        _synchrotron_diagnostic_inputs(
            "timescale_s",
            timescale_s,
            lower_bound=0.0,
            magnetic_field_G=magnetic_field_G,
            particle_mass_g=particle_mass_g,
            charge_esu=charge_esu,
        )
    )
    if magnetic_field_G == 0.0:
        return np.full_like(timescale_s, np.inf, dtype=float)

    diagnostic = "synchrotron cooling Lorentz factor"
    _checked_positive_power(magnetic_field_G, 2, diagnostic=diagnostic)
    _checked_positive_power(particle_mass_g, 3, diagnostic=diagnostic)
    _checked_positive_power(charge_esu, 4, diagnostic=diagnostic)
    log_result = (
        _LOG_COOLING_COEFFICIENT
        + 3.0 * np.log(particle_mass_g)
        - 4.0 * np.log(charge_esu)
        - 2.0 * np.log(magnetic_field_G)
        - np.log(timescale_s)
    )
    return _diagnostic_from_log(log_result, diagnostic=diagnostic)


def electron_synchrotron_cooling_time(
    gamma,
    *,
    magnetic_field_G,
) -> np.ndarray:
    """Return the comoving electron synchrotron cooling time in seconds.

    ``gamma`` is a finite dimensionless scalar or one-dimensional array with
    all values greater than 1; ``magnetic_field_G`` is a finite non-negative
    comoving field in Gauss. The returned NumPy array preserves the input shape,
    fixes ``m=m_e`` and ``|q|=e``, and is positive infinity for ``B=0``.
    The wrapper inherits the generic function's exact isotropic, classical
    ``gamma >> 1`` and ``chi << 1`` assumptions, use of ``gamma m c^2``, and
    exclusions of low-energy, quantum, transport, and observer-frame physics.
    """
    return synchrotron_cooling_time(
        gamma,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=_M_E_CGS,
        charge_esu=_E_ESU,
    )


def proton_synchrotron_cooling_time(
    gamma,
    *,
    magnetic_field_G,
) -> np.ndarray:
    """Return the comoving proton synchrotron cooling time in seconds.

    ``gamma`` is a finite dimensionless scalar or one-dimensional array with
    all values greater than 1; ``magnetic_field_G`` is a finite non-negative
    comoving field in Gauss. The returned NumPy array preserves the input shape,
    fixes ``m=m_p`` and ``|q|=e``, and is positive infinity for ``B=0``.
    The wrapper inherits the generic function's exact isotropic, classical
    ``gamma >> 1`` and ``chi << 1`` assumptions, use of ``gamma m c^2``, and
    exclusions of low-energy, quantum, transport, and observer-frame physics.
    """
    return synchrotron_cooling_time(
        gamma,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=_M_P_CGS,
        charge_esu=_E_ESU,
    )


def synchrotron_sed(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
    particle_mass_g,
    charge_esu,
) -> np.ndarray:
    """Return the comoving synchrotron SED of a tabulated particle distribution.

    Parameters
    ----------
    photon_energy_erg : array_like, shape (n_energy,)
        Positive finite comoving photon energies in erg; input order is kept.
    gamma : array_like, shape (n_gamma,)
        Strictly increasing finite Lorentz factors greater than 1, with at
        least two points.
    dn_dgamma_cm3 : array_like, shape (n_gamma,)
        Finite non-negative differential number density ``dn/dgamma`` in
        cm^-3. It is not ``dn/dE``, ``dn/dln(gamma)``, or a total particle
        number and has not been multiplied by volume.
    magnetic_field_G : float
        Finite comoving magnetic-field magnitude in Gauss, greater than or
        equal to zero.
    radius_cm : float
        Finite positive radius in cm of the homogeneous comoving sphere.
    particle_mass_g : float
        Finite positive particle mass in grams.
    charge_esu : float
        Finite positive charge magnitude in esu/statC.

    Returns
    -------
    numpy.ndarray, shape (n_energy,)
        Comoving ``E L_E = nu L_nu`` in erg s^-1.

    Notes
    -----
    The integral uses only the supplied gamma grid and deterministic
    trapezoidal integration in ``ln(gamma)``. The underlying kernel supplies
    the exact isotropic pitch-angle average. The model is classical, optically
    thin, and valid for ``gamma >> 1`` and ``chi << 1``. It assumes a uniform
    sphere and excludes SSA, cooling, particle transport, and observer-frame
    transformations.
    """
    photon_energy_erg = _positive_1d(
        "photon_energy_erg", photon_energy_erg, lower_bound=0.0
    )
    gamma = _positive_1d("gamma", gamma, lower_bound=1.0)
    if gamma.size < 2:
        raise ValueError("gamma must contain at least two points")
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("gamma must be strictly increasing")
    dn_dgamma_cm3 = _nonnegative_1d("dn_dgamma_cm3", dn_dgamma_cm3, shape=gamma.shape)
    magnetic_field_G = _finite_scalar(
        "magnetic_field_G", magnetic_field_G, allow_zero=True
    )
    radius_cm = _finite_scalar("radius_cm", radius_cm, allow_zero=False)
    particle_mass_g = _finite_scalar(
        "particle_mass_g", particle_mass_g, allow_zero=False
    )
    charge_esu = _finite_scalar("charge_esu", charge_esu, allow_zero=False)
    _gamma_squared(gamma)

    with np.errstate(over="ignore", invalid="ignore"):
        radius_cubed = np.float64(radius_cm) ** 3
    if not np.isfinite(radius_cubed):
        raise FloatingPointError(
            "radius_cm**3 exceeds the finite float64 numerical range"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        volume_cm3 = (4.0 * np.pi / 3.0) * radius_cubed
    if not np.isfinite(volume_cm3):
        raise FloatingPointError(
            "sphere volume exceeds the finite float64 numerical range"
        )

    if magnetic_field_G == 0.0 or not np.any(dn_dgamma_cm3):
        return np.zeros(photon_energy_erg.shape, dtype=float)

    single_particle_power = synchrotron_kernel(
        photon_energy_erg,
        gamma,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=particle_mass_g,
        charge_esu=charge_esu,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        integrand = single_particle_power * (gamma * dn_dgamma_cm3)[None, :]
    if not np.all(np.isfinite(integrand)):
        raise FloatingPointError("particle-distribution integrand is not finite")

    epsilon_E = trapezoid(integrand, x=np.log(gamma), axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        sed = photon_energy_erg * epsilon_E * volume_cm3
    if not np.all(np.isfinite(sed)) or np.any(sed < 0.0):
        raise FloatingPointError("synchrotron SED exceeds the finite float64 range")
    return sed


def electron_synchrotron_sed(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
) -> np.ndarray:
    """Return a comoving electron synchrotron ``E L_E`` in erg s^-1.

    ``photon_energy_erg`` is a one-dimensional array of positive finite
    comoving photon energies in erg. ``gamma`` contains at least two strictly
    increasing finite Lorentz factors greater than 1. It must match the
    one-dimensional, finite, non-negative ``dn_dgamma_cm3 = dn/dgamma`` array;
    the latter is a comoving number density in cm^-3, not ``dn/dE``,
    ``dn/dln(gamma)``, a total particle count, or a distribution already
    multiplied by volume. ``magnetic_field_G`` is the finite comoving field in
    Gauss and must be non-negative; ``radius_cm`` is the finite, positive radius
    in cm of a homogeneous comoving sphere. The returned shape follows
    ``photon_energy_erg`` and contains comoving
    ``E L_E = nu L_nu`` in erg s^-1.

    This thin wrapper fixes ``m=m_e`` and ``|q|=e`` and otherwise uses
    :func:`synchrotron_sed`, including its exact isotropic pitch-angle average,
    classical ``gamma >> 1`` and ``chi << 1`` range, and exclusions of SSA,
    cooling, transport, and observer-frame transformations.
    """
    return synchrotron_sed(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_E_CGS,
        charge_esu=_E_ESU,
    )


def proton_synchrotron_sed(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
) -> np.ndarray:
    """Return a comoving proton synchrotron ``E L_E`` in erg s^-1.

    ``photon_energy_erg`` is a one-dimensional array of positive finite
    comoving photon energies in erg. ``gamma`` contains at least two strictly
    increasing finite Lorentz factors greater than 1. It must match the
    one-dimensional, finite, non-negative ``dn_dgamma_cm3 = dn/dgamma`` array;
    the latter is a comoving number density in cm^-3, not ``dn/dE``,
    ``dn/dln(gamma)``, a total particle count, or a distribution already
    multiplied by volume. ``magnetic_field_G`` is the finite comoving field in
    Gauss and must be non-negative; ``radius_cm`` is the finite, positive radius
    in cm of a homogeneous comoving sphere. The returned shape follows
    ``photon_energy_erg`` and contains comoving
    ``E L_E = nu L_nu`` in erg s^-1.

    This thin wrapper fixes ``m=m_p`` and ``|q|=e`` and otherwise uses
    :func:`synchrotron_sed`, including its exact isotropic pitch-angle average,
    classical ``gamma >> 1`` and ``chi << 1`` range, and exclusions of SSA,
    cooling, transport, and observer-frame transformations.
    """
    return synchrotron_sed(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_P_CGS,
        charge_esu=_E_ESU,
    )


def power_law_distribution(
    gamma,
    normalization_cm3,
    index,
    gamma_min,
    gamma_max,
) -> np.ndarray:
    """Return ``dn/dgamma = K gamma**(-p)`` between inclusive cutoffs.

    Parameters are comoving: ``gamma`` is a non-empty one-dimensional array of
    finite values greater than 1; ``normalization_cm3 = K`` is a finite
    non-negative coefficient in cm^-3; ``index = p`` is finite and positive;
    and ``gamma_min`` and ``gamma_max`` are finite cutoffs satisfying
    ``1 < gamma_min < gamma_max``. The returned cm^-3 array preserves the input
    shape and order and is exactly zero outside the inclusive cutoff interval.

    This convenience function does not normalize a total particle number or
    energy density and applies no geometry. When passed to
    :func:`synchrotron_sed`, that function supplies the homogeneous comoving
    sphere and exact isotropic pitch-angle kernel. The associated synchrotron
    model is classical (``gamma >> 1``, ``chi << 1``) and excludes SSA, cooling,
    transport, and observer-frame transformations.
    """
    gamma = _positive_1d("gamma", gamma, lower_bound=1.0)
    normalization_cm3 = _finite_scalar(
        "normalization_cm3", normalization_cm3, allow_zero=True
    )
    index = _finite_scalar("index", index, allow_zero=False)
    gamma_min = _finite_scalar("gamma_min", gamma_min, allow_zero=False)
    gamma_max = _finite_scalar("gamma_max", gamma_max, allow_zero=False)
    if gamma_min <= 1.0:
        raise ValueError("gamma_min must be greater than 1")
    if gamma_max <= gamma_min:
        raise ValueError("gamma_max must be greater than gamma_min")

    result = np.zeros_like(gamma)
    if normalization_cm3 == 0.0:
        return result
    inside = (gamma >= gamma_min) & (gamma <= gamma_max)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result[inside] = normalization_cm3 * gamma[inside] ** (-index)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("power-law distribution exceeds finite float64 range")
    return result
