"""Charged-pion decay and synchrotron timescale diagnostics."""

import numpy as np

from helena.core.constants import (
    CHARGED_PION_MASS_CGS as _CHARGED_PION_MASS_G,
    CHARGED_PION_REST_LIFETIME_S as _CHARGED_PION_REST_LIFETIME_S,
    ELEMENTARY_CHARGE_ESU as _E_ESU,
)
from helena.core.contracts import (
    pion_gamma as _pion_gamma,
    positive_real_scalar as _positive_real_scalar,
)
from helena.core.synchrotron import synchrotron_cooling_time

__all__ = [
    "charged_pion_decay_time",
    "charged_pion_synchrotron_cooling_time",
    "charged_pion_synchrotron_to_decay_rate_ratio",
]


def _finite_positive_result(result, *, diagnostic):
    result = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError(
            f"{diagnostic} exceeds the finite float64 numerical range"
        )
    return result


def charged_pion_decay_time(pion_gamma):
    """Return the charged-pion decay time in the blob comoving frame.

    Parameters
    ----------
    pion_gamma : array_like, shape (n_pion,)
        Non-empty one-dimensional array of finite charged-pion Lorentz factors,
        all strictly greater than one. Input order is preserved.

    Returns
    -------
    numpy.ndarray, shape (n_pion,)
        Blob-comoving decay time ``t_dec = pion_gamma * tau_pi`` in seconds.

    Notes
    -----
    The rest-frame lifetime is the PDG 2024 value ``tau_pi = 2.6033e-8 s``.
    This is an independent timescale diagnostic: it does not evolve a pion
    population or calculate decay products, transport, injection, cascade,
    observer-frame quantities, fitting, or a JAX backend.
    """
    gamma = _pion_gamma(pion_gamma)
    result = gamma * _CHARGED_PION_REST_LIFETIME_S
    return _finite_positive_result(result, diagnostic="charged-pion decay time")


def charged_pion_synchrotron_cooling_time(pion_gamma, magnetic_field_G):
    """Return the charged-pion synchrotron cooling time.

    Parameters
    ----------
    pion_gamma : array_like, shape (n_pion,)
        Non-empty one-dimensional array of finite charged-pion Lorentz factors,
        all strictly greater than one. Input order is preserved.
    magnetic_field_G : float
        Finite positive blob-comoving magnetic-field magnitude in Gauss.

    Returns
    -------
    numpy.ndarray, shape (n_pion,)
        Blob-comoving synchrotron cooling time in seconds.

    Notes
    -----
    The calculation fixes ``m_pi c^2 = 139.57039 MeV`` and ``|q_pi| = e`` and
    reuses the existing classical isotropic synchrotron diagnostic. Equivalently,
    ``t_syn = 6 pi m_pi^3 c / (sigma_T m_e^2 B^2 pion_gamma)``. It assumes
    ``pion_gamma >> 1`` and ``chi << 1`` and deliberately has no ``beta^2``
    correction. It includes no cooling evolution, synchrotron spectrum,
    transport, decay products, cascade, observer transformation, fitting, or
    JAX backend.
    """
    gamma = _pion_gamma(pion_gamma)
    magnetic_field_G = _positive_real_scalar("magnetic_field_G", magnetic_field_G)
    result = synchrotron_cooling_time(
        gamma,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=_CHARGED_PION_MASS_G,
        charge_esu=_E_ESU,
    )
    return _finite_positive_result(
        result, diagnostic="charged-pion synchrotron cooling time"
    )


def charged_pion_synchrotron_to_decay_rate_ratio(
    pion_gamma,
    magnetic_field_G,
):
    """Return the charged-pion synchrotron-to-decay rate ratio.

    Parameters
    ----------
    pion_gamma : array_like, shape (n_pion,)
        Non-empty one-dimensional array of finite charged-pion Lorentz factors,
        all strictly greater than one. Input order is preserved.
    magnetic_field_G : float
        Finite positive blob-comoving magnetic-field magnitude in Gauss.

    Returns
    -------
    numpy.ndarray, shape (n_pion,)
        Dimensionless rate ratio
        ``eta = t_syn^-1 / t_dec^-1 = t_dec / t_syn``.

    Notes
    -----
    ``eta = 1`` marks equal local rates, while ``eta > 1`` means synchrotron
    cooling is faster than decay. This continuous diagnostic does not decide
    whether decay is prompt; that choice depends on the caller's energy range
    and accuracy requirements. It shares the classical ultrarelativistic,
    blob-comoving assumptions and exclusions of the two timescale functions.
    """
    decay_time = charged_pion_decay_time(pion_gamma)
    synchrotron_time = charged_pion_synchrotron_cooling_time(
        pion_gamma, magnetic_field_G
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        result = decay_time / synchrotron_time
    return _finite_positive_result(
        result, diagnostic="charged-pion synchrotron-to-decay rate ratio"
    )
