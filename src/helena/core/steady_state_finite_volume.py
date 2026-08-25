"""Conservative finite-volume steady-state transport on a gamma grid.

The discretization is an upper-bidiagonal linear system solved by a fixed-shape
O(n_gamma) reverse recurrence.  A future JAX implementation can reproduce the
recurrence with ``lax.scan``; gradients should use linear implicit
differentiation, not an artificial cooling or sink floor.
"""

import numpy as np

from helena.core.contracts import transport_array as _transport_array

__all__ = ["steady_state_finite_volume_solution"]


def _finite_volume_coefficients(gamma, loss_rate, sink_rate):
    internal_faces = np.sqrt(gamma[:-1]) * np.sqrt(gamma[1:])
    cell_faces = np.empty(gamma.size + 1, dtype=np.float64)
    cell_faces[1:-1] = internal_faces
    cell_faces[0] = gamma[0] * (gamma[0] / internal_faces[0])
    cell_faces[-1] = gamma[-1] * (gamma[-1] / internal_faces[-1])
    cell_widths = np.diff(cell_faces)
    if not np.all(np.isfinite(cell_widths)) or np.any(cell_widths <= 0.0):
        raise ValueError("particle_gamma must define finite positive cell widths")

    # Index i is the lower-energy face of cell i.  Entries 1: are internal
    # faces; the separate high-energy boundary flux is fixed to zero.
    outflow_face_loss = np.empty_like(gamma)
    outflow_face_loss[0] = loss_rate[0]
    weights = (internal_faces - gamma[:-1]) / np.diff(gamma)
    outflow_face_loss[1:] = (1.0 - weights) * loss_rate[:-1] + weights * loss_rate[1:]

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        diagonal = outflow_face_loss + sink_rate * cell_widths
    if np.any(diagonal == 0.0):
        raise ValueError(
            "every finite-volume cell needs positive cooling outflow or sink removal"
        )
    if not np.all(np.isfinite(diagonal)):
        raise FloatingPointError(
            "finite-volume transport coefficients exceed the float64 range"
        )
    return cell_widths, outflow_face_loss, diagonal


def steady_state_finite_volume_solution(
    particle_gamma,
    injection_dgamma_cm3_s,
    continuous_loss_rate_dgamma_s,
    sink_rate_s_inv,
):
    """Return a conservative steady-state density with cooling and local sinks.

    Parameters
    ----------
    particle_gamma : array_like, shape (n_gamma,)
        At least two finite, positive, strictly increasing dimensionless cell
        centers. All quantities are local to the blob comoving frame.
    injection_dgamma_cm3_s : array_like, shape (n_gamma,)
        Finite non-negative injection ``Q = dn/(d gamma dt)`` in cm^-3 s^-1
        per unit Lorentz factor.
    continuous_loss_rate_dgamma_s : array_like, shape (n_gamma,)
        Finite non-negative continuous loss ``b = -d gamma/dt`` in s^-1.
        Zero-loss cells are allowed when another removal term makes their
        discrete steady state unique.
    sink_rate_s_inv : array_like, shape (n_gamma,)
        Finite non-negative local sink rate ``lambda`` in s^-1.

    Returns
    -------
    numpy.ndarray, shape (n_gamma,)
        Float64 differential density ``N = dn/d gamma`` in cm^-3.

    Notes
    -----
    The function solves ``-d(b N)/d gamma + lambda N = Q``. Input coordinates
    are cell centers, internal faces are geometric midpoints, and face loss
    rates are linearly interpolated in ``gamma``. Cooling uses the upwind
    higher-energy cell. The high-energy boundary has zero inflow, while the
    low-energy boundary permits ``b_face N[0]`` to leave the grid.

    Cooling and sinks remain distinct: a sink is not folded into ``b``, and
    cooling is not treated as local deletion. If a cell has neither cooling
    outflow nor a sink, no unique finite discrete steady state exists and a
    ``ValueError`` is raised. No numerical floor is added.
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
    )
    sink_rate = _transport_array("sink_rate_s_inv", sink_rate_s_inv, shape=gamma.shape)
    cell_widths, outflow_face_loss, diagonal = _finite_volume_coefficients(
        gamma, loss_rate, sink_rate
    )

    density = np.empty_like(gamma)
    incoming_flux = 0.0
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for index in range(gamma.size - 1, -1, -1):
            density[index] = (
                injection[index] * cell_widths[index] + incoming_flux
            ) / diagonal[index]
            incoming_flux = outflow_face_loss[index] * density[index]
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise FloatingPointError(
            "finite-volume transport solution exceeds the finite float64 range"
        )
    return density
