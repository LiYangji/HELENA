"""Generic steady-state transport with continuous losses and local sinks."""

import numpy as np

__all__ = ["steady_state_continuous_loss_solution"]


def _transport_array(name, values, *, shape=None, positive=False):
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
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    if not positive and np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return array


def _reverse_transport_recurrence(gamma, injection, loss_rate, sink_rate):
    flux = np.zeros_like(gamma)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for index in range(gamma.size - 2, -1, -1):
            delta_gamma = gamma[index + 1] - gamma[index]
            injection_mid = 0.5 * injection[index] + 0.5 * injection[index + 1]
            loss_mid = 0.5 * loss_rate[index] + 0.5 * loss_rate[index + 1]
            sink_mid = 0.5 * sink_rate[index] + 0.5 * sink_rate[index + 1]

            if sink_mid == 0.0:
                attenuation = 1.0
                source = injection_mid * delta_gamma
            else:
                injection_over_sink = injection_mid / sink_mid
                equilibrium_flux = (
                    injection_over_sink * loss_mid
                    if 0.0 < injection_over_sink < np.inf
                    else injection_mid * (loss_mid / sink_mid)
                )
                z = (sink_mid / loss_mid) * delta_gamma
                if not np.isfinite(z):
                    attenuation = 0.0
                    source = equilibrium_flux
                elif z == 0.0:
                    attenuation = 1.0
                    source = injection_mid * delta_gamma
                elif z <= 1.0:
                    attenuation = np.exp(-z)
                    source = injection_mid * delta_gamma * (-np.expm1(-z) / z)
                else:
                    attenuation = np.exp(-z)
                    source = equilibrium_flux * (-np.expm1(-z))
            flux[index] = flux[index + 1] * attenuation + source
    return flux


def steady_state_continuous_loss_solution(
    particle_gamma,
    injection_dgamma_cm3_s,
    continuous_loss_rate_dgamma_s,
    sink_rate_s_inv,
):
    """Return a steady-state density under continuous losses and local sinks.

    Parameters
    ----------
    particle_gamma : array_like, shape (n_gamma,)
        At least two finite, positive, strictly increasing Lorentz-factor
        coordinates. All quantities are local to the blob comoving frame.
    injection_dgamma_cm3_s : array_like, shape (n_gamma,)
        Finite non-negative injection ``Q = dn/(d gamma dt)`` in cm^-3 s^-1
        per unit Lorentz factor.
    continuous_loss_rate_dgamma_s : array_like, shape (n_gamma,)
        Finite strictly positive continuous loss rate
        ``b = -d gamma/dt`` in s^-1.
    sink_rate_s_inv : array_like, shape (n_gamma,)
        Finite non-negative total local sink rate ``lambda`` in s^-1.

    Returns
    -------
    numpy.ndarray, shape (n_gamma,)
        Steady-state differential density ``N = dn/d gamma`` in cm^-3, with
        float64 dtype and high-energy boundary ``N(gamma_max) = 0``.

    Notes
    -----
    This solves ``-d(b N)/d gamma + lambda N = Q`` with midpoint coefficients
    and an O(n_gamma) high-to-low recurrence. It assumes deterministic
    continuous losses with ``b > 0`` and local catastrophic sinks; it includes
    no energy diffusion, acceleration, time dependence, geometry, hidden grid,
    species physics, or observer transformation.

    ``sink_rate_s_inv`` may combine decay, escape, and other local sinks, but
    callers must retain those rates separately when constructing physical
    outputs: for example, ``Q_decay = lambda_decay * N`` rather than
    ``lambda_total * N``. The function does not define an escape model, decide
    whether escape is negligible, or select a prompt-decay approximation.
    """
    gamma = _transport_array("particle_gamma", particle_gamma, positive=True)
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("particle_gamma must be strictly increasing")
    injection = _transport_array(
        "injection_dgamma_cm3_s",
        injection_dgamma_cm3_s,
        shape=gamma.shape,
    )
    loss_rate = _transport_array(
        "continuous_loss_rate_dgamma_s",
        continuous_loss_rate_dgamma_s,
        shape=gamma.shape,
        positive=True,
    )
    sink_rate = _transport_array(
        "sink_rate_s_inv",
        sink_rate_s_inv,
        shape=gamma.shape,
    )

    if not np.any(injection):
        return np.zeros_like(gamma)

    flux = _reverse_transport_recurrence(gamma, injection, loss_rate, sink_rate)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        density = flux / loss_rate
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise FloatingPointError(
            "steady-state transport solution exceeds the finite float64 range"
        )
    return density
