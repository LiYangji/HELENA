"""Local daughter injection from neutral- and charged-pion decay."""

import numpy as np
from scipy.integrate import trapezoid

from helena.core.constants import (
    CHARGED_PION_MUONIC_BRANCHING_RATIO as _MUONIC_BRANCHING_RATIO,
    CHARGED_PION_REST_ENERGY_CGS as _CHARGED_PION_REST_ENERGY_CGS,
    MUON_TO_PION_MASS_RATIO as _MUON_TO_PION_MASS_RATIO,
    MUON_TO_PION_MASS_RATIO_SQUARED as _R_PI,
    NEUTRAL_PION_REST_ENERGY_CGS as _PI0_REST_ENERGY_CGS,
    NEUTRAL_PION_TWO_PHOTON_BRANCHING_RATIO as _TWO_PHOTON_BRANCHING_RATIO,
)
from helena.core.contracts import (
    charged_pion_decay_inputs as _charged_pion_decay_inputs,
    real_1d as _real_1d,
)

__all__ = [
    "neutral_pion_decay_photon_injection",
    "charged_pion_decay_muon_injection",
    "charged_pion_decay_neutrino_injection",
]


def _direct_neutrino_scaling(x):
    result = np.zeros(x.shape, dtype=float)
    support = (x > 0.0) & (x < 1.0 - _R_PI)
    result[support] = 1.0 / (1.0 - _R_PI)
    return result


def _positive_pion_muon_scaling(x):
    left = np.zeros(x.shape, dtype=float)
    right = np.zeros(x.shape, dtype=float)
    support = (x > _R_PI) & (x < 1.0)
    supported_x = x[support]
    denominator = (1.0 - _R_PI) ** 2 * supported_x
    left[support] = (supported_x - _R_PI) / denominator
    right[support] = _R_PI * (1.0 - supported_x) / denominator
    return left, right


def _scaling_injection_integral(pion_gamma, decay_rate, scaling):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return trapezoid(
            scaling * (decay_rate / pion_gamma)[None, :],
            x=pion_gamma,
            axis=1,
        )


def _checked_decay_outputs(outputs, *, diagnostic):
    if any(
        not np.all(np.isfinite(output)) or np.any(output < 0.0) for output in outputs
    ):
        raise FloatingPointError(f"{diagnostic} exceeds the finite float64 range")
    return outputs


def neutral_pion_decay_photon_injection(
    photon_energy_erg,
    pion_gamma,
    q_pi0_dgamma_cm3_s,
):
    """Return local photon injection from ``pi0 -> gamma + gamma``.

    Parameters
    ----------
    photon_energy_erg : array_like, shape (n_photon,)
        Positive finite photon energies in the blob comoving frame, in erg.
        Their order is preserved and need not be increasing.
    pion_gamma : array_like, shape (n_pion,)
        At least two strictly increasing finite neutral-pion Lorentz factors,
        all strictly greater than one.
    q_pi0_dgamma_cm3_s : array_like, shape (n_pion,)
        Finite non-negative local neutral-pion injection per unit pion Lorentz
        factor, in cm^-3 s^-1, on ``pion_gamma``.

    Returns
    -------
    numpy.ndarray, shape (n_photon,)
        Blob-comoving local photon injection ``dn_gamma/(dE_gamma dt)`` in
        cm^-3 s^-1 erg^-1, following the photon-energy input order.

    Notes
    -----
    The calculation uses exact relativistic two-body kinematics and the PDG
    ``pi0 -> gamma gamma`` branching ratio. A single cumulative trapezoidal
    tail integral is linearly interpolated on the caller's pion grid, with no
    extrapolation. ``pion_gamma = 1`` is excluded because its delta-function
    photon line cannot be represented by this continuous spectral interface.
    No pion evolution, cooling, charged-pion decay, downstream decay,
    absorption, cascade, frame transformation, fitting, or JAX backend is
    included.
    """
    photon_energy = _real_1d(
        "photon_energy_erg", photon_energy_erg, minimum_size=1, positive=True
    )
    gamma = _real_1d("pion_gamma", pion_gamma, minimum_size=2, positive=True)
    if np.any(gamma <= 1.0):
        raise ValueError("pion_gamma must contain only values > 1")
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("pion_gamma must be strictly increasing")
    injection = _real_1d(
        "q_pi0_dgamma_cm3_s",
        q_pi0_dgamma_cm3_s,
        minimum_size=2,
        positive=False,
        shape=gamma.shape,
    )

    result = np.zeros(photon_energy.shape, dtype=float)
    if not np.any(injection):
        return result

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_gamma = 1.0 / gamma
        denominator = gamma * np.sqrt((1.0 - inverse_gamma) * (1.0 + inverse_gamma))
        integrand = injection / denominator
    if not np.all(np.isfinite(integrand)):
        raise FloatingPointError("neutral-pion decay integrand exceeds float64 range")
    with np.errstate(over="ignore", invalid="ignore"):
        intervals = (0.5 * integrand[:-1] + 0.5 * integrand[1:]) * np.diff(gamma)
        tail = np.empty_like(gamma)
        tail[-1] = 0.0
        tail[:-1] = np.cumsum(intervals[::-1])[::-1]
    if not np.all(np.isfinite(tail)):
        raise FloatingPointError("neutral-pion decay integral exceeds float64 range")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x_gamma = photon_energy / _PI0_REST_ENERGY_CGS
    gamma_minimum = np.full(photon_energy.shape, np.inf, dtype=float)
    safe = np.isfinite(x_gamma) & (x_gamma >= 0.25 / np.finfo(float).max)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gamma_minimum[safe] = x_gamma[safe] + 0.25 / x_gamma[safe]
        result = (
            2.0
            * _TWO_PHOTON_BRANCHING_RATIO
            / _PI0_REST_ENERGY_CGS
            * np.interp(gamma_minimum, gamma, tail, left=tail[0], right=0.0)
        )
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "neutral-pion decay photon injection exceeds float64 range"
        )
    return result


def charged_pion_decay_muon_injection(
    muon_gamma,
    pion_gamma,
    q_pi_plus_decay_dgamma_cm3_s,
    q_pi_minus_decay_dgamma_cm3_s,
):
    """Return helicity-resolved muon injection from charged-pion decay.

    Parameters
    ----------
    muon_gamma : array_like, shape (n_muon,)
        Finite blob-comoving muon Lorentz factors, all strictly greater than
        one. Input order is preserved and need not be increasing.
    pion_gamma : array_like, shape (n_pion,)
        At least two strictly increasing finite charged-pion Lorentz factors,
        all strictly greater than one.
    q_pi_plus_decay_dgamma_cm3_s, q_pi_minus_decay_dgamma_cm3_s : array_like
        Actual blob-comoving pion decay-event rate spectra per unit pion
        Lorentz factor, in cm^-3 s^-1. Each must be finite, non-negative, and
        have the same shape as ``pion_gamma``; neither is a pion density.

    Returns
    -------
    tuple of numpy.ndarray
        ``(Q_mu+_L, Q_mu+_R, Q_mu-_L, Q_mu-_R)`` on ``muon_gamma``, each in
        cm^-3 s^-1 per unit muon Lorentz factor. ``L`` has helicity ``h=-1``
        and ``R`` has ``h=+1``.

    Notes
    -----
    The Hümmer/Lipari ultrarelativistic scaling approximation is used and is
    physically intended for ``pion_gamma >> 1``. The PDG 2024 muonic branching
    ratio is applied once outside the separately normalized scaling functions.
    After an external prompt-decay decision, a caller may pass photomeson pion
    injection as the decay-event rate; a future transport calculation may
    instead pass ``N_pi / t_dec``. This function makes neither choice and does
    not call the independent pion-timescale diagnostics. It includes no pion
    cooling or transport, muon decay or cooling, observer transform, cascade,
    JAX, or fitting.
    """
    muon_gamma, pion_gamma, plus_rate, minus_rate = _charged_pion_decay_inputs(
        "muon_gamma",
        muon_gamma,
        daughter_lower_bound=1.0,
        pion_gamma=pion_gamma,
        q_pi_plus_decay_dgamma_cm3_s=q_pi_plus_decay_dgamma_cm3_s,
        q_pi_minus_decay_dgamma_cm3_s=q_pi_minus_decay_dgamma_cm3_s,
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        x = muon_gamma[:, None] * _MUON_TO_PION_MASS_RATIO / pion_gamma[None, :]
    plus_left_scaling, plus_right_scaling = _positive_pion_muon_scaling(x)
    coefficient = _MUONIC_BRANCHING_RATIO * _MUON_TO_PION_MASS_RATIO
    outputs = (
        coefficient
        * _scaling_injection_integral(pion_gamma, plus_rate, plus_left_scaling),
        coefficient
        * _scaling_injection_integral(pion_gamma, plus_rate, plus_right_scaling),
        coefficient
        * _scaling_injection_integral(pion_gamma, minus_rate, plus_right_scaling),
        coefficient
        * _scaling_injection_integral(pion_gamma, minus_rate, plus_left_scaling),
    )
    return _checked_decay_outputs(outputs, diagnostic="charged-pion muon injection")


def charged_pion_decay_neutrino_injection(
    neutrino_energy_erg,
    pion_gamma,
    q_pi_plus_decay_dgamma_cm3_s,
    q_pi_minus_decay_dgamma_cm3_s,
):
    """Return direct muon-neutrino injection from charged-pion decay.

    Parameters
    ----------
    neutrino_energy_erg : array_like, shape (n_neutrino,)
        Positive finite blob-comoving neutrino energies in erg. Input order is
        preserved and need not be increasing.
    pion_gamma : array_like, shape (n_pion,)
        At least two strictly increasing finite charged-pion Lorentz factors,
        all strictly greater than one.
    q_pi_plus_decay_dgamma_cm3_s, q_pi_minus_decay_dgamma_cm3_s : array_like
        Actual blob-comoving pion decay-event rate spectra per unit pion
        Lorentz factor, in cm^-3 s^-1. Each must be finite, non-negative, and
        have the same shape as ``pion_gamma``; neither is a pion density.

    Returns
    -------
    tuple of numpy.ndarray
        ``(Q_nu_mu, Q_anti_nu_mu)`` on ``neutrino_energy_erg``, each in
        cm^-3 s^-1 erg^-1. The first comes from ``pi+`` and the second from
        ``pi-``.

    Notes
    -----
    The Hümmer/Lipari ultrarelativistic scaling approximation is used and is
    physically intended for ``pion_gamma >> 1``. The PDG 2024 muonic branching
    ratio is applied once outside the normalized direct-neutrino scaling
    function. After an external prompt-decay decision, a caller may pass
    photomeson pion injection as the decay-event rate; a future transport
    calculation may instead pass ``N_pi / t_dec``. This function makes neither
    choice and does not call the independent pion-timescale diagnostics. It
    includes no muon-decay neutrinos, pion cooling or transport, muon cooling,
    oscillation, observer transform, cascade, JAX, or fitting.
    """
    neutrino_energy, pion_gamma, plus_rate, minus_rate = _charged_pion_decay_inputs(
        "neutrino_energy_erg",
        neutrino_energy_erg,
        daughter_lower_bound=0.0,
        pion_gamma=pion_gamma,
        q_pi_plus_decay_dgamma_cm3_s=q_pi_plus_decay_dgamma_cm3_s,
        q_pi_minus_decay_dgamma_cm3_s=q_pi_minus_decay_dgamma_cm3_s,
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        x = neutrino_energy[:, None] / (
            pion_gamma[None, :] * _CHARGED_PION_REST_ENERGY_CGS
        )
    scaling = _direct_neutrino_scaling(x)
    coefficient = _MUONIC_BRANCHING_RATIO / _CHARGED_PION_REST_ENERGY_CGS
    outputs = (
        coefficient * _scaling_injection_integral(pion_gamma, plus_rate, scaling),
        coefficient * _scaling_injection_integral(pion_gamma, minus_rate, scaling),
    )
    return _checked_decay_outputs(outputs, diagnostic="charged-pion neutrino injection")
