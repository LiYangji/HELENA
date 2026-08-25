"""Blob-frame proton synchrotron and adiabatic continuous-loss rates."""

import numpy as np

from helena.core.contracts import (
    nonnegative_scalar as _nonnegative_scalar,
    proton_gamma as _proton_gamma,
    rate_on_grid as _rate_on_grid,
)
from helena.core.synchrotron import proton_synchrotron_cooling_time

__all__ = [
    "proton_synchrotron_continuous_loss_rate",
    "proton_adiabatic_continuous_loss_rate",
]


def proton_synchrotron_continuous_loss_rate(proton_gamma, magnetic_field_G):
    """Return the classical proton synchrotron loss ``-d gamma_p/dt``.

    ``proton_gamma`` is a non-empty one-dimensional array of finite values
    strictly greater than one. ``magnetic_field_G`` is a finite non-negative
    scalar blob-comoving magnetic field in Gauss. The float64 result is in
    s^-1 and preserves the proton-grid order.

    The function reuses the existing exact-isotropic-pitch-angle, classical
    ultrarelativistic proton cooling-time convention, with ``gamma_p >> 1``
    and ``chi_p << 1``. It adds no finite-beta correction. For ``B=0`` it
    returns exact zeros without evaluating ``gamma/inf``. It includes no
    transport, escape, radiation spectrum, observer transform, or JAX backend.
    """
    gamma = _proton_gamma(proton_gamma)
    magnetic_field_G = _nonnegative_scalar("magnetic_field_G", magnetic_field_G)
    if magnetic_field_G == 0.0:
        return np.zeros(gamma.shape, dtype=np.float64)
    cooling_time = proton_synchrotron_cooling_time(
        gamma, magnetic_field_G=magnetic_field_G
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        result = gamma / cooling_time
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "proton synchrotron loss rate exceeds the finite float64 range"
        )
    return result


def proton_adiabatic_continuous_loss_rate(proton_gamma, adiabatic_rate_s_inv):
    """Return the one-zone ultrarelativistic adiabatic loss rate.

    ``proton_gamma`` is a non-empty one-dimensional array of finite values
    strictly greater than one. ``adiabatic_rate_s_inv = 1/t_ad`` is either a
    finite non-negative scalar or a matching array in s^-1. The returned
    float64 ``b_ad = gamma_p/t_ad`` is in s^-1 and follows the proton grid.

    This is energy-space advection, not particle escape. It includes no extra
    volume-dilution sink and assumes the caller has constructed the one-zone
    adiabatic timescale. Zero rate returns exact zeros. No geometry, expansion
    model, transport, observer transform, or JAX backend is included.
    """
    gamma = _proton_gamma(proton_gamma)
    rate = _rate_on_grid("adiabatic_rate_s_inv", adiabatic_rate_s_inv, gamma)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = gamma * rate
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "proton adiabatic loss rate exceeds the finite float64 range"
        )
    return result
