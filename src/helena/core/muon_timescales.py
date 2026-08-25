"""Muon decay and synchrotron timescale diagnostics."""

import numpy as np

from helena.core.constants import (
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    MUON_MASS_CGS as _MUON_MASS_G,
    MUON_REST_LIFETIME_S as _MUON_REST_LIFETIME_S,
)
from helena.core.contracts import (
    muon_gamma as _muon_gamma,
    positive_real_scalar as _positive_real_scalar,
)
from helena.core.synchrotron import synchrotron_cooling_time

__all__ = [
    "muon_decay_time",
    "muon_synchrotron_cooling_time",
    "muon_synchrotron_to_decay_rate_ratio",
]


def _finite_positive_result(result, *, diagnostic):
    result = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError(
            f"{diagnostic} exceeds the finite float64 numerical range"
        )
    return result


def muon_decay_time(muon_gamma):
    """Return the muon decay time in the blob comoving frame.

    Parameters
    ----------
    muon_gamma : array_like, shape (n_muon,)
        Non-empty one-dimensional array of finite muon Lorentz factors, all
        strictly greater than one. Input order is preserved.

    Returns
    -------
    numpy.ndarray, shape (n_muon,)
        Blob-comoving decay time ``t_dec = muon_gamma * tau_mu`` in seconds.

    Notes
    -----
    The rest-frame lifetime is the PDG 2026 value ``tau_mu = 2.1969811e-6 s``.
    Charge and helicity do not change this lifetime. This independent
    diagnostic includes no decay products, injection conversion, cooling
    evolution, transport, observer transformation, fitting, or JAX backend.
    """
    gamma = _muon_gamma(muon_gamma)
    result = gamma * _MUON_REST_LIFETIME_S
    return _finite_positive_result(result, diagnostic="muon decay time")


def muon_synchrotron_cooling_time(muon_gamma, magnetic_field_G):
    """Return the muon synchrotron cooling time in the blob frame.

    Parameters
    ----------
    muon_gamma : array_like, shape (n_muon,)
        Non-empty one-dimensional array of finite muon Lorentz factors, all
        strictly greater than one. Input order is preserved.
    magnetic_field_G : float
        Finite positive blob-comoving magnetic-field magnitude in Gauss.

    Returns
    -------
    numpy.ndarray, shape (n_muon,)
        Blob-comoving synchrotron cooling time in seconds.

    Notes
    -----
    The calculation fixes ``m_mu c^2 = 105.6583755 MeV`` and ``|q_mu| = e``
    and reuses the existing classical isotropic synchrotron diagnostic.
    Equivalently,
    ``t_syn = 6 pi m_mu^3 c / (sigma_T m_e^2 B^2 muon_gamma)``. It assumes
    ``muon_gamma >> 1``, ``chi_mu << 1``, and ``beta_mu^2 ~= 1``. Charge and
    helicity do not change this classical cooling time. It includes no cooling
    evolution, synchrotron spectrum, decay products, transport, observer
    transformation, fitting, or JAX backend.
    """
    gamma = _muon_gamma(muon_gamma)
    magnetic_field_G = _positive_real_scalar("magnetic_field_G", magnetic_field_G)
    result = synchrotron_cooling_time(
        gamma,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=_MUON_MASS_G,
        charge_esu=_E_ESU,
    )
    return _finite_positive_result(result, diagnostic="muon synchrotron cooling time")


def muon_synchrotron_to_decay_rate_ratio(muon_gamma, magnetic_field_G):
    """Return the muon synchrotron-to-decay rate ratio.

    Parameters
    ----------
    muon_gamma : array_like, shape (n_muon,)
        Non-empty one-dimensional array of finite muon Lorentz factors, all
        strictly greater than one. Input order is preserved.
    magnetic_field_G : float
        Finite positive blob-comoving magnetic-field magnitude in Gauss.

    Returns
    -------
    numpy.ndarray, shape (n_muon,)
        Dimensionless rate ratio
        ``eta = t_syn^-1 / t_dec^-1 = t_dec / t_syn``.

    Notes
    -----
    ``eta = 1`` marks equal local rates, while ``eta > 1`` means synchrotron
    cooling is faster than decay. This charge- and helicity-independent
    continuous diagnostic does not select prompt decay; callers must decide
    from their muon energy range and accuracy needs. It is separate from pion
    and future muon decay kinematics and includes no transport or decay products.
    """
    decay_time = muon_decay_time(muon_gamma)
    synchrotron_time = muon_synchrotron_cooling_time(muon_gamma, magnetic_field_G)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        result = decay_time / synchrotron_time
    return _finite_positive_result(
        result, diagnostic="muon synchrotron-to-decay rate ratio"
    )
