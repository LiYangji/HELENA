"""Helicity-dependent daughter injection from muon decay."""

import numpy as np
from scipy.integrate import trapezoid

from helena.core.constants import (
    ELECTRON_TO_MUON_MASS_RATIO as _ELECTRON_TO_MUON_MASS_RATIO,
    MUON_REST_ENERGY_CGS as _MUON_REST_ENERGY_CGS,
)
from helena.core.contracts import muon_decay_inputs as _muon_decay_inputs

__all__ = [
    "muon_decay_electron_positron_injection",
    "muon_decay_neutrino_injection",
]


def _michel_scalings(x):
    ab_minus = np.zeros(x.shape, dtype=float)
    ab_plus = np.zeros(x.shape, dtype=float)
    cd_minus = np.zeros(x.shape, dtype=float)
    cd_plus = np.zeros(x.shape, dtype=float)
    support = (x > 0.0) & (x < 1.0)
    supported_x = x[support]
    x_squared = supported_x**2
    x_cubed = supported_x * x_squared

    a = 5.0 / 3.0 - 3.0 * x_squared + 4.0 / 3.0 * x_cubed
    b = -1.0 / 3.0 + 3.0 * x_squared - 8.0 / 3.0 * x_cubed
    c = 2.0 - 6.0 * x_squared + 4.0 * x_cubed
    d = 2.0 - 12.0 * supported_x + 18.0 * x_squared - 8.0 * x_cubed

    ab_minus[support] = np.maximum(a - b, 0.0)
    ab_plus[support] = np.maximum(a + b, 0.0)
    cd_minus[support] = np.maximum(c - d, 0.0)
    cd_plus[support] = np.maximum(c + d, 0.0)
    return ab_minus, ab_plus, cd_minus, cd_plus


def _scaling_injection_integral(muon_gamma, decay_rate, scaling):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return trapezoid(
            scaling * (decay_rate / muon_gamma)[None, :],
            x=muon_gamma,
            axis=1,
        )


def _checked_decay_outputs(outputs, *, diagnostic):
    if any(
        not np.all(np.isfinite(output)) or np.any(output < 0.0) for output in outputs
    ):
        raise FloatingPointError(f"{diagnostic} exceeds the finite float64 range")
    return outputs


def muon_decay_electron_positron_injection(
    electron_gamma,
    muon_gamma,
    q_mu_plus_left_decay_dgamma_cm3_s,
    q_mu_plus_right_decay_dgamma_cm3_s,
    q_mu_minus_left_decay_dgamma_cm3_s,
    q_mu_minus_right_decay_dgamma_cm3_s,
):
    """Return positron and electron injection from helicity-resolved muons.

    Parameters
    ----------
    electron_gamma : array_like, shape (n_electron,)
        Finite blob-comoving electron/positron Lorentz factors, all strictly
        greater than one. Input order is preserved and need not be increasing.
    muon_gamma : array_like, shape (n_muon,)
        At least two strictly increasing finite muon Lorentz factors, all
        strictly greater than one.
    q_mu_plus_left_decay_dgamma_cm3_s,
    q_mu_plus_right_decay_dgamma_cm3_s,
    q_mu_minus_left_decay_dgamma_cm3_s,
    q_mu_minus_right_decay_dgamma_cm3_s : array_like, shape (n_muon,)
        Actual blob-comoving muon decay-event rate spectra per unit muon
        Lorentz factor, in cm^-3 s^-1. Each is finite and non-negative; these
        inputs are not muon densities. Left and right mean ``h=-1`` and
        ``h=+1``, respectively.

    Returns
    -------
    tuple of numpy.ndarray
        ``(Q_positron, Q_electron)`` on ``electron_gamma``, each in cm^-3 s^-1
        per unit electron Lorentz factor.

    Notes
    -----
    The leading-order ultrarelativistic Michel scaling functions neglect the
    electron mass in their kinematics; the electron mass enters only through
    the ``E_e <-> electron_gamma`` mapping and Jacobian. No branching ratio is
    applied. After an external prompt-decay decision, a caller may pass
    pion-decay muon injection as the decay-event rate; future transport may
    instead pass ``N_mu / t_dec``. This function makes neither choice and does
    not call muon-timescale diagnostics. It includes no radiative correction,
    explicit radiative decay, cooling, transport, secondary radiation,
    observer transformation, JAX, or fitting.
    """
    electron_gamma, muon_gamma, rates = _muon_decay_inputs(
        "electron_gamma",
        electron_gamma,
        daughter_lower_bound=1.0,
        muon_gamma=muon_gamma,
        parent_rates=(
            q_mu_plus_left_decay_dgamma_cm3_s,
            q_mu_plus_right_decay_dgamma_cm3_s,
            q_mu_minus_left_decay_dgamma_cm3_s,
            q_mu_minus_right_decay_dgamma_cm3_s,
        ),
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        x = electron_gamma[:, None] * _ELECTRON_TO_MUON_MASS_RATIO / muon_gamma[None, :]
    ab_minus, ab_plus, _, _ = _michel_scalings(x)
    plus_left, plus_right, minus_left, minus_right = rates
    coefficient = _ELECTRON_TO_MUON_MASS_RATIO
    outputs = (
        coefficient
        * (
            _scaling_injection_integral(muon_gamma, plus_left, ab_minus)
            + _scaling_injection_integral(muon_gamma, plus_right, ab_plus)
        ),
        coefficient
        * (
            _scaling_injection_integral(muon_gamma, minus_left, ab_plus)
            + _scaling_injection_integral(muon_gamma, minus_right, ab_minus)
        ),
    )
    return _checked_decay_outputs(
        outputs, diagnostic="muon-decay electron/positron injection"
    )


def muon_decay_neutrino_injection(
    neutrino_energy_erg,
    muon_gamma,
    q_mu_plus_left_decay_dgamma_cm3_s,
    q_mu_plus_right_decay_dgamma_cm3_s,
    q_mu_minus_left_decay_dgamma_cm3_s,
    q_mu_minus_right_decay_dgamma_cm3_s,
):
    """Return flavor-resolved neutrino injection from muon decay.

    Parameters
    ----------
    neutrino_energy_erg : array_like, shape (n_neutrino,)
        Positive finite blob-comoving neutrino energies in erg. Input order is
        preserved and need not be increasing.
    muon_gamma : array_like, shape (n_muon,)
        At least two strictly increasing finite muon Lorentz factors, all
        strictly greater than one.
    q_mu_plus_left_decay_dgamma_cm3_s,
    q_mu_plus_right_decay_dgamma_cm3_s,
    q_mu_minus_left_decay_dgamma_cm3_s,
    q_mu_minus_right_decay_dgamma_cm3_s : array_like, shape (n_muon,)
        Actual blob-comoving muon decay-event rate spectra per unit muon
        Lorentz factor, in cm^-3 s^-1. Each is finite and non-negative; these
        inputs are not muon densities. Left and right mean ``h=-1`` and
        ``h=+1``, respectively.

    Returns
    -------
    tuple of numpy.ndarray
        ``(Q_nu_e, Q_anti_nu_e, Q_nu_mu, Q_anti_nu_mu)`` on
        ``neutrino_energy_erg``, each in cm^-3 s^-1 erg^-1.

    Notes
    -----
    The leading-order ultrarelativistic Michel approximation treats daughter
    neutrinos as massless and applies no branching ratio. Charge conjugation
    maps ``h -> -h``. After an external prompt-decay decision, a caller may
    pass pion-decay muon injection as the decay-event rate; future transport
    may instead pass ``N_mu / t_dec``. This function makes neither choice and
    does not call muon-timescale diagnostics. It includes no tau neutrinos,
    QED radiative correction, explicit radiative decay, oscillation, cooling,
    transport, cascade, observer transformation, JAX, or fitting.
    """
    neutrino_energy, muon_gamma, rates = _muon_decay_inputs(
        "neutrino_energy_erg",
        neutrino_energy_erg,
        daughter_lower_bound=0.0,
        muon_gamma=muon_gamma,
        parent_rates=(
            q_mu_plus_left_decay_dgamma_cm3_s,
            q_mu_plus_right_decay_dgamma_cm3_s,
            q_mu_minus_left_decay_dgamma_cm3_s,
            q_mu_minus_right_decay_dgamma_cm3_s,
        ),
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        x = neutrino_energy[:, None] / (muon_gamma[None, :] * _MUON_REST_ENERGY_CGS)
    ab_minus, ab_plus, cd_minus, cd_plus = _michel_scalings(x)
    plus_left, plus_right, minus_left, minus_right = rates
    coefficient = 1.0 / _MUON_REST_ENERGY_CGS
    outputs = (
        coefficient
        * (
            _scaling_injection_integral(muon_gamma, plus_left, cd_minus)
            + _scaling_injection_integral(muon_gamma, plus_right, cd_plus)
        ),
        coefficient
        * (
            _scaling_injection_integral(muon_gamma, minus_left, cd_plus)
            + _scaling_injection_integral(muon_gamma, minus_right, cd_minus)
        ),
        coefficient
        * (
            _scaling_injection_integral(muon_gamma, minus_left, ab_plus)
            + _scaling_injection_integral(muon_gamma, minus_right, ab_minus)
        ),
        coefficient
        * (
            _scaling_injection_integral(muon_gamma, plus_left, ab_minus)
            + _scaling_injection_integral(muon_gamma, plus_right, ab_plus)
        ),
    )
    return _checked_decay_outputs(outputs, diagnostic="muon-decay neutrino injection")
