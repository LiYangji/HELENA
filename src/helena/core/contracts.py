"""Backend-neutral NumPy validation and array-shape contracts."""

import numpy as np


def real_1d(name, values, *, minimum_size, positive, shape=None):
    """Validate and return a float64 one-dimensional real array."""
    array = np.asarray(values)
    if (
        array.ndim != 1
        or array.size < minimum_size
        or array.dtype.kind not in "iuf"
        or (shape is not None and array.shape != shape)
    ):
        qualifier = f" with shape {shape}" if shape is not None else ""
        raise ValueError(f"{name} must be a one-dimensional real array{qualifier}")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    if not positive and np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return array


def nonempty_real_1d(name, values, *, positive, shape=None):
    """Validate and return a non-empty float64 one-dimensional array."""
    array = np.asarray(values)
    if (
        array.ndim != 1
        or array.size == 0
        or array.dtype.kind not in "iuf"
        or (shape is not None and array.shape != shape)
    ):
        qualifier = f" with shape {shape}" if shape is not None else ""
        raise ValueError(f"{name} must be a non-empty real 1D array{qualifier}")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    if not positive and np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return array


def positive_scalar(name, value):
    """Validate and return a finite positive real scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite positive real scalar")
    value = float(array)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return value


def electron_gamma(values):
    """Validate an ordered electron Lorentz-factor grid."""
    gamma = np.asarray(values)
    if gamma.ndim != 1 or gamma.size < 2 or gamma.dtype.kind not in "iuf":
        raise ValueError(
            "electron_gamma must be a one-dimensional real array with at least "
            "two values"
        )
    gamma = gamma.astype(np.float64, copy=False)
    if not np.all(np.isfinite(gamma)) or np.any(gamma <= 1.0):
        raise ValueError("electron_gamma must contain only finite values > 1")
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("electron_gamma must be strictly increasing")
    return gamma


def nonnegative_on_grid(name, values, gamma, *, allow_scalar):
    """Validate a finite non-negative scalar or grid-shaped array."""
    array = np.asarray(values)
    if allow_scalar and array.ndim == 0:
        if array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a real scalar or grid-shaped array")
        array = np.full(gamma.shape, float(array), dtype=np.float64)
    elif array.ndim == 1 and array.shape == gamma.shape and array.dtype.kind in "iuf":
        array = array.astype(np.float64, copy=False)
    else:
        qualifier = "a scalar or " if allow_scalar else ""
        raise ValueError(
            f"{name} must be {qualifier}a real array with shape {gamma.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative values")
    return array


def muon_gamma(values):
    """Validate a non-empty muon Lorentz-factor array."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError("muon_gamma must be a non-empty one-dimensional real array")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)) or np.any(array <= 1.0):
        raise ValueError("muon_gamma must contain only finite values > 1")
    return array


def pion_gamma(values):
    """Validate a non-empty charged-pion Lorentz-factor array."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError("pion_gamma must be a non-empty one-dimensional real array")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)) or np.any(array <= 1.0):
        raise ValueError("pion_gamma must contain only finite values > 1")
    return array


def positive_real_scalar(name, value):
    """Validate and return a finite real scalar greater than zero."""
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real scalar > 0")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar > 0")
    value = float(array)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite real scalar > 0")
    return value


def positive_magnetic_field(value):
    """Validate a magnetic-field magnitude in Gauss."""
    return positive_real_scalar("magnetic_field_G", value)


def proton_gamma(values):
    """Validate a non-empty proton Lorentz-factor array."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError("proton_gamma must be a non-empty one-dimensional real array")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)) or np.any(array <= 1.0):
        raise ValueError("proton_gamma must contain only finite values > 1")
    return array


def nonnegative_scalar(name, value):
    """Validate and return a finite non-negative real scalar."""
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    value = float(array)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return value


def rate_on_grid(name, values, gamma):
    """Validate a non-negative scalar or proton-grid-shaped rate."""
    array = np.asarray(values)
    if array.ndim == 0:
        if array.dtype.kind not in "iuf":
            raise ValueError(
                f"{name} must be finite, non-negative, and scalar or grid-shaped"
            )
        value = float(array)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be finite, non-negative, and scalar or grid-shaped"
            )
        return np.full(gamma.shape, value, dtype=np.float64)
    if array.ndim != 1 or array.shape != gamma.shape or array.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must be a finite non-negative scalar or an array with shape "
            f"{gamma.shape}"
        )
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative values")
    return array


def proton_injection(values, gamma):
    """Validate a proton-grid-shaped injection spectrum."""
    array = np.asarray(values)
    if array.ndim != 1 or array.shape != gamma.shape or array.dtype.kind not in "iuf":
        raise ValueError(
            "q_proton_injection_dgamma_cm3_s must be a real array with shape "
            f"{gamma.shape}"
        )
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(
            "q_proton_injection_dgamma_cm3_s must contain finite non-negative values"
        )
    return array


def transport_array(name, values, *, shape=None, positive=False):
    """Validate a finite transport array with at least two elements."""
    array = np.asarray(values)
    if (
        array.ndim != 1
        or array.size < 2
        or array.dtype.kind not in "iuf"
        or (shape is not None and array.shape != shape)
    ):
        qualifier = f" with shape {shape}" if shape is not None else ""
        raise ValueError(
            f"{name} must be a one-dimensional real array with at least two values"
            f"{qualifier}"
        )
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    if not positive and np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return array


def positive_scalar_or_1d(name, values, *, lower_bound):
    """Validate a scalar or 1D array strictly above a lower bound."""
    array = np.asarray(values)
    if array.ndim not in (0, 1) or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must be a real scalar or non-empty one-dimensional array"
        )
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)) or np.any(array <= lower_bound):
        raise ValueError(f"{name} must contain only finite values > {lower_bound}")
    return array


def positive_1d(name, values, *, lower_bound):
    """Validate a non-empty 1D array strictly above a lower bound."""
    array = positive_scalar_or_1d(name, values, lower_bound=lower_bound)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional real array")
    return array


def finite_scalar(name, value, *, allow_zero):
    """Validate a finite positive or non-negative scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(array)
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        comparison = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {comparison}")
    return value


def nonnegative_1d(name, values, *, shape):
    """Validate a finite non-negative array with an exact shape."""
    array = np.asarray(values)
    if array.ndim != 1 or array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} must be a one-dimensional real array with shape {shape}"
        )
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative values")
    return array


def gamma_squared(gamma):
    """Validate that squaring a Lorentz-factor array remains finite."""
    with np.errstate(over="ignore", invalid="ignore"):
        squared = gamma**2
    if not np.all(np.isfinite(squared)):
        raise FloatingPointError("gamma**2 exceeds the finite float64 numerical range")
    return squared


def absorption_inputs(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
):
    """Validate the shared synchrotron-absorption input contract."""
    photon_energy_erg = positive_1d(
        "photon_energy_erg", photon_energy_erg, lower_bound=0.0
    )
    gamma = positive_1d("gamma", gamma, lower_bound=1.0)
    if gamma.size < 3:
        raise ValueError("gamma must contain at least three points")
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("gamma must be strictly increasing")
    dn_dgamma_cm3 = nonnegative_1d(
        "dn_dgamma_cm3", dn_dgamma_cm3, shape=gamma.shape
    )
    magnetic_field_G = finite_scalar(
        "magnetic_field_G", magnetic_field_G, allow_zero=True
    )
    particle_mass_g = finite_scalar(
        "particle_mass_g", particle_mass_g, allow_zero=False
    )
    charge_esu = finite_scalar("charge_esu", charge_esu, allow_zero=False)
    gamma_squared(gamma)
    return (
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G,
        particle_mass_g,
        charge_esu,
    )


def muon_decay_inputs(
    daughter_name,
    daughter_values,
    *,
    daughter_lower_bound,
    muon_gamma,
    parent_rates,
):
    """Validate the common muon-decay daughter-grid contract."""
    daughter = real_1d(
        daughter_name, daughter_values, minimum_size=1, positive=True
    )
    if np.any(daughter <= daughter_lower_bound):
        raise ValueError(
            f"{daughter_name} must contain only values > {daughter_lower_bound}"
        )
    gamma = real_1d("muon_gamma", muon_gamma, minimum_size=2, positive=True)
    if np.any(gamma <= 1.0):
        raise ValueError("muon_gamma must contain only values > 1")
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("muon_gamma must be strictly increasing")
    names = (
        "q_mu_plus_left_decay_dgamma_cm3_s",
        "q_mu_plus_right_decay_dgamma_cm3_s",
        "q_mu_minus_left_decay_dgamma_cm3_s",
        "q_mu_minus_right_decay_dgamma_cm3_s",
    )
    rates = tuple(
        real_1d(name, rate, minimum_size=2, positive=False, shape=gamma.shape)
        for name, rate in zip(names, parent_rates, strict=True)
    )
    return daughter, gamma, rates


def charged_pion_decay_inputs(
    daughter_name,
    daughter_values,
    *,
    daughter_lower_bound,
    pion_gamma,
    q_pi_plus_decay_dgamma_cm3_s,
    q_pi_minus_decay_dgamma_cm3_s,
):
    """Validate the common charged-pion daughter-grid contract."""
    daughter = real_1d(
        daughter_name, daughter_values, minimum_size=1, positive=True
    )
    if np.any(daughter <= daughter_lower_bound):
        raise ValueError(
            f"{daughter_name} must contain only values > {daughter_lower_bound}"
        )
    gamma = real_1d("pion_gamma", pion_gamma, minimum_size=2, positive=True)
    if np.any(gamma <= 1.0):
        raise ValueError("pion_gamma must contain only values > 1")
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("pion_gamma must be strictly increasing")
    plus_rate = real_1d(
        "q_pi_plus_decay_dgamma_cm3_s",
        q_pi_plus_decay_dgamma_cm3_s,
        minimum_size=2,
        positive=False,
        shape=gamma.shape,
    )
    minus_rate = real_1d(
        "q_pi_minus_decay_dgamma_cm3_s",
        q_pi_minus_decay_dgamma_cm3_s,
        minimum_size=2,
        positive=False,
        shape=gamma.shape,
    )
    return daughter, gamma, plus_rate, minus_rate
