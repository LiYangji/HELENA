"""Steady-state transport of helicity-resolved muons."""

import numpy as np

from helena.core.contracts import positive_magnetic_field as _positive_magnetic_field
from helena.core.muon_timescales import (
    muon_decay_time,
    muon_synchrotron_cooling_time,
)
from helena.core.steady_state_transport import (
    steady_state_continuous_loss_solution,
)

__all__ = ["helicity_resolved_muon_steady_state_transport"]


def helicity_resolved_muon_steady_state_transport(
    muon_gamma,
    q_mu_plus_left_injection_dgamma_cm3_s,
    q_mu_plus_right_injection_dgamma_cm3_s,
    q_mu_minus_left_injection_dgamma_cm3_s,
    q_mu_minus_right_injection_dgamma_cm3_s,
    magnetic_field_G,
):
    """Return transported muon densities and actual decay-event rates.

    Parameters
    ----------
    muon_gamma : array_like, shape (n_muon,)
        At least two finite, strictly increasing muon Lorentz factors, all
        strictly greater than one. All quantities are in the blob comoving
        frame.
    q_mu_plus_left_injection_dgamma_cm3_s,
    q_mu_plus_right_injection_dgamma_cm3_s,
    q_mu_minus_left_injection_dgamma_cm3_s,
    q_mu_minus_right_injection_dgamma_cm3_s : array_like, shape (n_muon,)
        Finite non-negative helicity-resolved muon injection rates per unit
        muon Lorentz factor, in cm^-3 s^-1. Left and right mean ``h=-1`` and
        ``h=+1``, respectively.
    magnetic_field_G : float
        Finite positive blob-comoving magnetic-field magnitude in Gauss.

    Returns
    -------
    tuple of numpy.ndarray
        ``(N_mu+_L, N_mu+_R, N_mu-_L, N_mu-_R, Q_mu+_L,dec, Q_mu+_R,dec,
        Q_mu-_L,dec, Q_mu-_R,dec)`` on ``muon_gamma``. Densities are in cm^-3
        and decay-event rates are in cm^-3 s^-1, each per unit muon Lorentz
        factor. This order matches the parent order used by ``muon_decay.py``.

    Notes
    -----
    All four charge/helicity channels independently use continuous muon
    synchrotron loss ``b_syn = muon_gamma / t_syn`` and the local decay sink
    ``lambda_dec = 1 / t_dec``. Charge and helicity do not alter either rate;
    no helicity mixing, spin evolution, or depolarization is included.

    The returned decay-event rates are ``N_mu / t_dec``, not the input
    injection rates. Escape is zero, so this approximation requires decay or
    synchrotron transport to dominate escape over the relevant energy range.
    No prompt-decay switch, decay branching ratio, daughter kernel, finite
    source age, cascade, observer transformation, JAX, or fitting is included.
    """
    magnetic_field_G = _positive_magnetic_field(magnetic_field_G)
    decay_time = muon_decay_time(muon_gamma)
    synchrotron_time = muon_synchrotron_cooling_time(muon_gamma, magnetic_field_G)
    gamma = np.asarray(muon_gamma, dtype=float)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        synchrotron_loss_rate = gamma / synchrotron_time
        decay_rate = 1.0 / decay_time

    injections = (
        q_mu_plus_left_injection_dgamma_cm3_s,
        q_mu_plus_right_injection_dgamma_cm3_s,
        q_mu_minus_left_injection_dgamma_cm3_s,
        q_mu_minus_right_injection_dgamma_cm3_s,
    )
    densities = tuple(
        steady_state_continuous_loss_solution(
            gamma, injection, synchrotron_loss_rate, decay_rate
        )
        for injection in injections
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        outputs = densities + tuple(density / decay_time for density in densities)
    if any(
        not np.all(np.isfinite(output)) or np.any(output < 0.0) for output in outputs
    ):
        raise FloatingPointError(
            "muon transport output exceeds the finite float64 range"
        )
    return outputs
