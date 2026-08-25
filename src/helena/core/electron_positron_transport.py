"""Steady electron/positron transport with shared continuous losses and escape."""

import numpy as np

from helena.core.contracts import (
    electron_gamma as _electron_gamma,
    nonnegative_on_grid as _nonnegative_on_grid,
)
from helena.core.steady_state_finite_volume import (
    steady_state_finite_volume_solution,
)
from helena.core.synchrotron import electron_synchrotron_cooling_time

__all__ = ["electron_positron_steady_state_transport"]


def electron_positron_steady_state_transport(
    electron_gamma,
    q_electron_injection_dgamma_cm3_s,
    q_positron_injection_dgamma_cm3_s,
    magnetic_field_G,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
    *,
    additional_continuous_loss_rate_dgamma_s=0.0,
):
    """Return blob-frame steady electron/positron densities and escape spectra.

    Parameters
    ----------
    electron_gamma : array_like, shape (n_electron,)
        At least two finite, strictly increasing blob-comoving Lorentz factors,
        all greater than one.
    q_electron_injection_dgamma_cm3_s,
    q_positron_injection_dgamma_cm3_s : array_like, shape (n_electron,)
        Finite non-negative electron and positron injection rates in cm^-3 s^-1
        per unit Lorentz factor.
    magnetic_field_G : float
        Finite non-negative blob-comoving magnetic field in Gauss.
    adiabatic_rate_s_inv, escape_rate_s_inv : float or array_like
        Finite non-negative scalar or electron-grid-shaped rates in s^-1.
        Escape is a local particle sink, not a continuous energy loss.
    additional_continuous_loss_rate_dgamma_s : float or array_like, optional
        Caller-supplied finite non-negative ``b_additional = -d gamma/dt`` in
        s^-1, either scalar or electron-grid-shaped. It is not a fractional
        loss rate and is therefore used without another factor of ``gamma``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(N_electron, N_positron, Q_electron_escape, Q_positron_escape)`` as
        float64 arrays. Densities are ``dn/dgamma`` in cm^-3 per unit Lorentz
        factor; escape spectra are ``lambda_escape N`` in cm^-3 s^-1 per unit
        Lorentz factor.

    Notes
    -----
    Both charges use the same conservative finite-volume operator and total
    continuous loss
    ``b = gamma/t_syn + gamma*adiabatic_rate + b_additional``; only their
    injection arrays differ. The solver supplies zero high-energy inflow and
    low-energy cooling outflow. No inverse-Compton cooling, annihilation,
    radiation, cascade, observer transformation, or JAX backend is included.
    """
    gamma = _electron_gamma(electron_gamma)
    electron_injection = _nonnegative_on_grid(
        "q_electron_injection_dgamma_cm3_s",
        q_electron_injection_dgamma_cm3_s,
        gamma,
        allow_scalar=False,
    )
    positron_injection = _nonnegative_on_grid(
        "q_positron_injection_dgamma_cm3_s",
        q_positron_injection_dgamma_cm3_s,
        gamma,
        allow_scalar=False,
    )
    adiabatic_rate = _nonnegative_on_grid(
        "adiabatic_rate_s_inv", adiabatic_rate_s_inv, gamma, allow_scalar=True
    )
    escape_rate = _nonnegative_on_grid(
        "escape_rate_s_inv", escape_rate_s_inv, gamma, allow_scalar=True
    )
    additional_loss = _nonnegative_on_grid(
        "additional_continuous_loss_rate_dgamma_s",
        additional_continuous_loss_rate_dgamma_s,
        gamma,
        allow_scalar=True,
    )

    synchrotron_time = electron_synchrotron_cooling_time(
        gamma, magnetic_field_G=magnetic_field_G
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        total_loss = gamma / synchrotron_time + gamma * adiabatic_rate + additional_loss
    if not np.all(np.isfinite(total_loss)) or np.any(total_loss < 0.0):
        raise FloatingPointError("total electron/positron loss exceeds float64 range")

    electron_density = steady_state_finite_volume_solution(
        gamma, electron_injection, total_loss, escape_rate
    )
    positron_density = steady_state_finite_volume_solution(
        gamma, positron_injection, total_loss, escape_rate
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        electron_escape = escape_rate * electron_density
        positron_escape = escape_rate * positron_density
    outputs = (
        electron_density,
        positron_density,
        electron_escape,
        positron_escape,
    )
    if any(
        not np.all(np.isfinite(output)) or np.any(output < 0.0) for output in outputs
    ):
        raise FloatingPointError(
            "electron/positron transport output exceeds the finite float64 range"
        )
    return outputs
