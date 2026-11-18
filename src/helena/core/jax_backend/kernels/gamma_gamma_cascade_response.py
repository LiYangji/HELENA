"""Finite-grid collision response for gamma-gamma pair transport only."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import ELECTRON_REST_ENERGY_CGS, SPEED_OF_LIGHT_CGS
from helena.core.contracts import real_1d
from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
    _isotropic_differential_pair_kernel_jax,
    _trapezoid_weights,
)

CASCADE_PAIR_METHOD = "support-headon-incoming-log-sin2-gl8-hat-mass-v2"
# Integrate in log(min(gamma, x+eta-gamma)). Split a wide last bin even when
# the entire high-energy endpoint is inside that one transport interval.
_MAX_LOG_STEP = 0.7
_BATCH = 2048
_nodes, _weights = np.polynomial.legendre.leggauss(8)
_t = 0.5 * (_nodes + 1)
_FRACTION = np.sin(0.5 * np.pi * _t) ** 2
_WEIGHT = 0.25 * np.pi * _weights * np.sin(np.pi * _t)


def _interval_deposits(x, eta, lo, logspan, left, right, reflected, feature):
    """Integrate the true per-charge kernel against both adjacent hat functions."""
    h = lo * jnp.exp(jnp.asarray(_FRACTION) * logspan)
    incident = jnp.minimum(x, eta)
    q = jnp.where(feature == 0, h, incident + feature * (h - 1))
    # Charge/angle symmetry k(T-q)=k(q) avoids evaluating T-(T-q) at the
    # narrow high-energy endpoint. The hat functions still see physical gamma.
    k = _isotropic_differential_pair_kernel_jax(
        q, x * ELECTRON_REST_ENERGY_CGS, (eta * ELECTRON_REST_ENERGY_CGS)[None]
    )[:, 0]
    mass = jnp.asarray(_WEIGHT) * logspan * h * k
    total = x + eta
    left_hat = jnp.where(reflected, q - (total - right), right - q) / (right - left)
    right_hat = jnp.where(reflected, (total - left) - q, q - left) / (right - left)
    return jnp.stack((jnp.dot(mass, left_hat), jnp.dot(mass, right_hat)))


_batched_deposits = jax.jit(jax.vmap(_interval_deposits))


def _project_half(row, x, eta, daughter, lower, midpoint, head_on, reflected):
    """Stream active intersections for one primary and one side of the support."""
    total = x + eta[:, None]
    left, right = daughter[:-1], daughter[1:]
    if reflected:
        lo = np.maximum(lower[:, None], total - right)
        hi = np.minimum(midpoint[:, None], total - left)
    else:
        lo = np.maximum(lower[:, None], left)
        hi = np.minimum(midpoint[:, None], right)
    # For two hard photons k has an additional narrow structure where the
    # daughter's energy approaches either incoming energy. On the low half
    # resolve that structure in log(1+|q-min(x,eta)|), on both sides.
    incident = np.minimum(x, eta)[:, None]
    crossover = np.where(incident > 1, (incident + 1) / 2, np.inf)
    for feature, a, b in (
        (0, lo, np.minimum(hi, crossover)),
        (-1, np.maximum(lo, crossover), np.minimum(hi, incident)),
        (1, np.maximum(lo, np.maximum(incident, crossover)), hi),
    ):
        target, cell = np.nonzero(b > a)
        if not target.size:
            continue
        lower, upper = a[target, cell], b[target, cell]
        # At the head-on daughter energy the angular upper bound switches
        # between the kinematic root and s=x*eta. Integrate each side of
        # this kink separately, including its reflected location on the high half.
        kink = head_on[target]
        cross = (lower < kink) & (kink < upper)
        if np.any(cross):
            target = np.concatenate((target, target[cross]))
            cell = np.concatenate((cell, cell[cross]))
            lower, upper = (
                np.r_[lower, kink[cross]],
                np.r_[np.where(cross, kink, upper), upper[cross]],
            )
        line = incident[target, 0]
        if feature == -1:
            lower, upper = 1 + (line - upper), 1 + (line - lower)
        elif feature == 1:
            lower, upper = 1 + (lower - line), 1 + (upper - line)
        _accumulate_intervals(
            row, x, eta, daughter, target, cell, lower, upper, reflected, feature
        )


def _accumulate_intervals(
    row, x, eta, daughter, target, cell, lo, hi, reflected, feature
):
    left, right = daughter[:-1], daughter[1:]
    logspan = np.log1p((hi - lo) / lo)
    pieces = np.maximum(1, np.ceil(logspan / _MAX_LOG_STEP).astype(int))
    index = np.repeat(np.arange(lo.size), pieces)
    part = np.arange(index.size) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    step = (logspan / pieces)[index]
    lo = lo[index] * np.exp(part * step)
    target, cell = target[index], cell[index]
    # Pad only the last batch, reusing the same compiled shape across grids.
    for start in range(0, target.size, _BATCH):
        stop = min(start + _BATCH, target.size)
        ids = np.minimum(np.arange(start, start + _BATCH), stop - 1)
        j, i = target[ids], cell[ids]
        result = np.asarray(
            _batched_deposits(
                np.full(_BATCH, x),
                eta[j],
                lo[ids],
                step[ids],
                left[i],
                right[i],
                np.full(_BATCH, reflected),
                np.full(_BATCH, feature),
            )
        )[: stop - start]
        j, i = target[start:stop], cell[start:stop]
        np.add.at(row, (j, i), result[:, 0])
        np.add.at(row, (j, i + 1), result[:, 1])


def prepare_cascade_pair_response_jax(daughter, primary, target):
    """Construct the host-prepared x64 transport response on a fixed grid.

    Each per-charge nodal mass is divided by its linear-gamma trapezoid
    weight. Contracting photon densities therefore returns the diagnostic
    differential-density representation; finite-volume transport consumes the
    same masses divided by its own widths. The response includes c and both
    photon dE weights, without an identical-population event factor.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("gamma-gamma cascade response requires JAX float64")
    daughter = real_1d("daughter", daughter, minimum_size=2, positive=True)
    primary = real_1d("primary", primary, minimum_size=2, positive=True)
    target = real_1d("target", target, minimum_size=2, positive=True)
    if daughter[0] <= 1 or any(
        np.any(np.diff(a) <= 0) for a in (daughter, primary, target)
    ):
        raise ValueError("response grids must be increasing and daughter gamma > 1")
    eta = target / ELECTRON_REST_ENERGY_CGS
    result = np.zeros((primary.size, target.size, daughter.size), dtype=np.float64)
    for index, x in enumerate(primary / ELECTRON_REST_ENERGY_CGS):
        active = np.flatnonzero(x * eta > 1.0)
        if not active.size:
            continue
        y = eta[active]
        total, product = x + y, x * y
        s = np.minimum(np.maximum(1.0, 0.5 * total), product)
        width = np.sqrt(1.0 - 1.0 / s) * np.sqrt((x - y) ** 2 + 4 * (product - s))
        lower = np.maximum(1.0, (4 * s + total**2 / s - 4) / (2 * (total + width)))
        # Rationalized lower daughter endpoint at s=x*y (head-on).
        diff = x - y
        head_on = (4 * product + diff**2 / product) / (
            2 * (total + np.sqrt(1 - 1 / product) * np.abs(diff))
        )
        row = np.zeros((active.size, daughter.size), dtype=np.float64)
        for reflected in (False, True):
            _project_half(row, x, y, daughter, lower, total / 2, head_on, reflected)
        result[index, active, :] = row
    primary_weight = (
        SPEED_OF_LIGHT_CGS * primary * np.asarray(_trapezoid_weights(np.log(primary)))
    )
    target_weight = target * np.asarray(_trapezoid_weights(np.log(target)))
    result *= primary_weight[:, None, None]
    result *= target_weight[None, :, None]
    result /= np.asarray(_trapezoid_weights(daughter))[None, None, :]
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise FloatingPointError(
            "nonfinite or negative bin-integrated gamma-gamma response"
        )
    return jnp.asarray(np.transpose(result, (2, 0, 1)))
