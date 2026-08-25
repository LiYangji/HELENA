"""Steady-state transport of charged pions."""

import numpy as np

from helena.core.contracts import positive_magnetic_field as _positive_magnetic_field
from helena.core.pion_timescales import (
    charged_pion_decay_time,
    charged_pion_synchrotron_cooling_time,
)
from helena.core.steady_state_transport import (
    steady_state_continuous_loss_solution,
)

__all__ = ["charged_pion_steady_state_transport"]


def charged_pion_steady_state_transport(
    pion_gamma,
    q_pi_plus_injection_dgamma_cm3_s,
    q_pi_minus_injection_dgamma_cm3_s,
    magnetic_field_G,
):
    """Return transported charged-pion densities and decay-event rates.

    Parameters
    ----------
    pion_gamma : array_like, shape (n_pion,)
        At least two finite, strictly increasing charged-pion Lorentz factors,
        all strictly greater than one. All quantities are in the blob comoving
        frame.
    q_pi_plus_injection_dgamma_cm3_s,
    q_pi_minus_injection_dgamma_cm3_s : array_like, shape (n_pion,)
        Finite non-negative ``pi+`` and ``pi-`` injection rates per unit pion
        Lorentz factor, in cm^-3 s^-1.
    magnetic_field_G : float
        Finite positive blob-comoving magnetic-field magnitude in Gauss.

    Returns
    -------
    tuple of numpy.ndarray
        ``(N_pi+, N_pi-, Q_pi+_dec, Q_pi-_dec)`` on ``pion_gamma``. The first
        two arrays are differential densities in cm^-3 per unit pion Lorentz
        factor. The final two are actual decay-event rate spectra in
        cm^-3 s^-1 per unit pion Lorentz factor.

    Notes
    -----
    The transport equation uses continuous charged-pion synchrotron loss
    ``b_syn = pion_gamma / t_syn`` and the local decay sink
    ``lambda_dec = 1 / t_dec``. Escape is zero, so ``lambda_total`` is exactly
    ``lambda_dec``; this approximation requires ``t_dec << t_escape`` over the
    relevant energy range.

    The decay-event rates are ``N_pi / t_dec``, not the input injection rates.
    No prompt-decay switch or decay branching ratio is applied. Neutral pions,
    daughter spectra, escape, finite source age, time dependence, diffusion,
    cascade, observer transformation, JAX, and fitting are not included.
    """
    magnetic_field_G = _positive_magnetic_field(magnetic_field_G)
    decay_time = charged_pion_decay_time(pion_gamma)
    synchrotron_time = charged_pion_synchrotron_cooling_time(
        pion_gamma, magnetic_field_G
    )
    gamma = np.asarray(pion_gamma, dtype=float)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        synchrotron_loss_rate = gamma / synchrotron_time
        decay_rate = 1.0 / decay_time

    plus_density = steady_state_continuous_loss_solution(
        gamma,
        q_pi_plus_injection_dgamma_cm3_s,
        synchrotron_loss_rate,
        decay_rate,
    )
    minus_density = steady_state_continuous_loss_solution(
        gamma,
        q_pi_minus_injection_dgamma_cm3_s,
        synchrotron_loss_rate,
        decay_rate,
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        outputs = (
            plus_density,
            minus_density,
            plus_density / decay_time,
            minus_density / decay_time,
        )
    if any(
        not np.all(np.isfinite(output)) or np.any(output < 0.0) for output in outputs
    ):
        raise FloatingPointError(
            "charged-pion transport output exceeds the finite float64 range"
        )
    return outputs
