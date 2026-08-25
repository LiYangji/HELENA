"""Observer transforms for one discrete isotropically emitting moving blob.

The source-side result is a line-of-sight isotropic-equivalent SED luminosity,
not a flux or the true luminosity integrated over every emission direction.
"""

import numpy as np

__all__ = [
    "transform_comoving_sed",
    "sed_luminosity_to_energy_flux",
]

_LOG_FOUR_PI = np.log(4.0 * np.pi)


def _real_1d(name, values, *, positive, shape=None):
    array = np.asarray(values)
    if (
        array.ndim != 1
        or array.size == 0
        or array.dtype.kind not in "iuf"
        or (shape is not None and array.shape != shape)
    ):
        requirement = "positive" if positive else "non-negative"
        shape_requirement = "" if shape is None else f" with shape {shape}"
        raise ValueError(
            f"{name} must be a non-empty one-dimensional real {requirement} "
            f"array{shape_requirement}"
        )
    array = array.astype(float, copy=False)
    invalid_sign = array <= 0.0 if positive else array < 0.0
    if not np.all(np.isfinite(array)) or np.any(invalid_sign):
        requirement = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must contain only finite {requirement} values")
    return array


def _real_scalar(name, value, *, positive):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(array)
    invalid_sign = value <= 0.0 if positive else value < 0.0
    if not np.isfinite(value) or invalid_sign:
        comparison = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {comparison}")
    return value


def _scale_nonnegative(values, log_scale, *, diagnostic):
    if log_scale == 0.0:
        return values.copy()

    result = np.zeros_like(values)
    positive = values > 0.0
    if not np.any(positive):
        return result

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = np.exp(np.log(values[positive]) + log_scale)
    if not np.all(np.isfinite(scaled)) or np.any(scaled <= 0.0):
        raise FloatingPointError(f"{diagnostic} exceeds the finite float64 range")
    result[positive] = scaled
    return result


def transform_comoving_sed(
    photon_energy_comoving_erg,
    sed_comoving_erg_s,
    *,
    doppler_factor,
    redshift,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a discrete isotropic blob SED to observer energy coordinates.

    Parameters
    ----------
    photon_energy_comoving_erg : array_like, shape (n_energy,)
        Positive finite photon energies ``E'`` in the blob comoving frame, in
        erg. Input order is preserved.
    sed_comoving_erg_s : array_like, shape (n_energy,)
        Finite non-negative comoving ``S'(E') = E' L'_E'`` in erg s^-1. This
        is not ``L'_E'``, a flux, or a photon-number spectrum.
    doppler_factor : float
        Finite positive Doppler factor ``delta``. Values below one represent
        deboosting and are valid.
    redshift : float
        Finite cosmological redshift ``z >= 0`` supplied by the caller.

    Returns
    -------
    observed_photon_energy_erg : numpy.ndarray, shape (n_energy,)
        Observer-frame photon energies ``E_obs = delta E' / (1+z)`` in erg.
    isotropic_equivalent_sed_luminosity_erg_s : numpy.ndarray, shape (n_energy,)
        Source-side line-of-sight isotropic-equivalent SED luminosity
        ``E_em L_E,em,iso = delta^4 E' L'_E'`` in erg s^-1, paired with the
        observer energy grid. It is neither flux nor the true angle-integrated
        host-frame luminosity.

    Notes
    -----
    This pointwise transformation applies only to one discrete moving blob
    whose emergent radiation is isotropic in its comoving frame. It creates no
    grid and performs no sorting, interpolation, regridding, attenuation,
    cosmology calculation, or time transformation. It is not the beaming law
    for a continuous jet, anisotropic external-Compton radiation, ordered-field
    directional emission, or polarization.

    The host/emission frame has ``E_em = delta E'``; cosmological propagation
    then gives ``E_obs = E_em/(1+z)``. At corresponding mapped points and before
    the separate luminosity-distance conversion, redshift moves the horizontal
    coordinate but adds no vertical factor to this source-side SED luminosity.
    """
    photon_energy_comoving_erg = _real_1d(
        "photon_energy_comoving_erg",
        photon_energy_comoving_erg,
        positive=True,
    )
    sed_comoving_erg_s = _real_1d(
        "sed_comoving_erg_s",
        sed_comoving_erg_s,
        positive=False,
        shape=photon_energy_comoving_erg.shape,
    )
    doppler_factor = _real_scalar("doppler_factor", doppler_factor, positive=True)
    redshift = _real_scalar("redshift", redshift, positive=False)

    log_doppler = np.log(doppler_factor)
    observed_energy = _scale_nonnegative(
        photon_energy_comoving_erg,
        log_doppler - np.log1p(redshift),
        diagnostic="observer photon-energy transform",
    )
    isotropic_equivalent_sed = _scale_nonnegative(
        sed_comoving_erg_s,
        4.0 * log_doppler,
        diagnostic="isotropic-equivalent SED transform",
    )
    return observed_energy, isotropic_equivalent_sed


def sed_luminosity_to_energy_flux(
    isotropic_equivalent_sed_erg_s,
    *,
    luminosity_distance_cm,
) -> np.ndarray:
    """Convert isotropic-equivalent SED luminosity to observed energy flux.

    Parameters
    ----------
    isotropic_equivalent_sed_erg_s : array_like, shape (n_energy,)
        Finite non-negative source-side line-of-sight isotropic-equivalent SED
        luminosity ``E_em L_E,em,iso`` in erg s^-1. It is not flux or the true
        luminosity integrated over every source-frame direction.
    luminosity_distance_cm : float
        Finite positive standard luminosity distance ``d_L`` in cm, supplied
        directly by the caller and not inferred from redshift.

    Returns
    -------
    numpy.ndarray, shape (n_energy,)
        Observer-frame energy flux ``E_obs F_E,obs`` in erg cm^-2 s^-1,
        equal to the input divided by ``4 pi d_L^2``.

    Notes
    -----
    The luminosity-distance definition already contains cosmological energy
    and arrival-rate dimming. Therefore ``E_obs F_E,obs`` has no additional
    explicit factor of ``1+z``. This function performs no cosmology,
    interpolation, attenuation, band integration, or time transformation.
    """
    isotropic_equivalent_sed_erg_s = _real_1d(
        "isotropic_equivalent_sed_erg_s",
        isotropic_equivalent_sed_erg_s,
        positive=False,
    )
    luminosity_distance_cm = _real_scalar(
        "luminosity_distance_cm", luminosity_distance_cm, positive=True
    )
    return _scale_nonnegative(
        isotropic_equivalent_sed_erg_s,
        -_LOG_FOUR_PI - 2.0 * np.log(luminosity_distance_cm),
        diagnostic="observed energy flux",
    )
