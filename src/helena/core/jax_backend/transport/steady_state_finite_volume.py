"""Implicitly differentiable JAX conservative steady-state transport."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.contracts import transport_array as _transport_array
from helena.diagnostics import (
    validate_finite_volume_coefficients as _validate_finite_volume_coefficients,
)

__all__ = ["steady_state_finite_volume_solution_jax"]


def _steady_state_finite_volume_kernel(gamma, injection, loss_rate, sink_rate):
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    injection = jnp.asarray(injection, dtype=jnp.float64)
    loss_rate = jnp.asarray(loss_rate, dtype=jnp.float64)
    sink_rate = jnp.asarray(sink_rate, dtype=jnp.float64)

    internal_faces = jnp.sqrt(gamma[:-1]) * jnp.sqrt(gamma[1:])
    cell_faces = jnp.concatenate(
        (
            gamma[:1] * (gamma[:1] / internal_faces[:1]),
            internal_faces,
            gamma[-1:] * (gamma[-1:] / internal_faces[-1:]),
        )
    )
    cell_widths = jnp.diff(cell_faces)
    weights = (internal_faces - gamma[:-1]) / jnp.diff(gamma)
    outflow_face_loss = jnp.concatenate(
        (
            loss_rate[:1],
            (1.0 - weights) * loss_rate[:-1] + weights * loss_rate[1:],
        )
    )
    diagonal = outflow_face_loss + sink_rate * cell_widths

    right_hand_side = injection * cell_widths

    def matvec(vector):
        incoming = jnp.concatenate(
            (
                outflow_face_loss[1:] * vector[1:],
                jnp.zeros_like(vector[-1:]),
            )
        )
        return diagonal * vector - incoming

    def solve(_, rhs):
        def recurrence(incoming_flux, values):
            rhs_i, outflow_i, diagonal_i = values
            density_i = (rhs_i + incoming_flux) / diagonal_i
            return outflow_i * density_i, density_i

        _, reverse_density = jax.lax.scan(
            recurrence,
            jnp.float64(0.0),
            (
                rhs[::-1],
                outflow_face_loss[::-1],
                diagonal[::-1],
            ),
        )
        return reverse_density[::-1]

    def transpose_solve(_, cotangent):
        def recurrence(previous, values):
            cotangent_i, outflow_i, diagonal_i = values
            solution_i = (cotangent_i + outflow_i * previous) / diagonal_i
            return solution_i, solution_i

        _, solution = jax.lax.scan(
            recurrence,
            jnp.float64(0.0),
            (cotangent, outflow_face_loss, diagonal),
        )
        return solution

    return jax.lax.custom_linear_solve(
        matvec,
        right_hand_side,
        solve,
        transpose_solve,
        symmetric=False,
    )


def steady_state_finite_volume_solution_jax(
    particle_gamma,
    injection_dgamma_cm3_s,
    continuous_loss_rate_dgamma_s,
    sink_rate_s_inv,
):
    """Return the JAX float64 conservative steady-state density.

    Parameters are one-dimensional arrays on the blob-comoving Lorentz-factor
    cell centers. ``particle_gamma`` has at least two finite, positive,
    strictly increasing values. Injection is in cm^-3 s^-1 per unit Lorentz
    factor, continuous loss ``b = -d gamma/dt`` and local sink rate are in
    s^-1, and all three arrays are finite, non-negative, and grid-shaped.

    The returned JAX array is ``dn/d gamma`` in cm^-3. The discretization and
    high-energy no-inflow/low-energy outflow boundaries exactly match the
    NumPy reference. Forward and transpose systems use O(n_gamma) ``lax.scan``
    solves, and gradients use implicit differentiation through
    ``custom_linear_solve`` rather than the forward recurrence history.
    Callers must enable ``jax_enable_x64`` before tracing. Eager calls validate
    values; JIT-traced calls require already-valid inputs. No cooling or sink
    floor is added.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("steady-state JAX transport requires jax_enable_x64")

    arguments = (
        particle_gamma,
        injection_dgamma_cm3_s,
        continuous_loss_rate_dgamma_s,
        sink_rate_s_inv,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _steady_state_finite_volume_kernel(*arguments)

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
    _validate_finite_volume_coefficients(gamma, loss_rate, sink_rate)

    density = _steady_state_finite_volume_kernel(gamma, injection, loss_rate, sink_rate)
    density_array = np.asarray(density)
    if not np.all(np.isfinite(density_array)) or np.any(density_array < 0.0):
        raise FloatingPointError(
            "finite-volume transport solution exceeds the finite float64 range"
        )
    return density
