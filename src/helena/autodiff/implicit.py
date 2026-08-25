"""Matrix-free implicit differentiation for an already-solved fixed point."""

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


def fixed_point_residual(transition_and_loss, state, parameters):
    """Return ``F(state, parameters) - state`` in the state's inherited units.

    ``transition_and_loss`` must return ``(next_state, scalar_loss)``. This
    helper assigns no physics, units, frame, or validity range beyond those of
    that caller-supplied map.
    """

    next_state, _ = transition_and_loss(state, parameters)
    return next_state - state


def implicit_adjoint_value_and_grad(
    transition_and_loss,
    converged_state,
    parameters,
    *,
    state_scale=None,
    tolerance=1.0e-10,
    absolute_tolerance=0.0,
    restart=20,
    maxiter=10,
):
    """Return a scalar loss and implicit gradient at a solved fixed point.

    ``transition_and_loss(x, theta)`` returns ``(F(x, theta), L(x, theta))``.
    ``converged_state`` is a one-dimensional vector in caller-defined units and
    is stopped at this boundary, so no primal-solver history enters reverse
    mode. ``parameters`` are dimensionless transformed fit coordinates.

    GMRES solves ``(I - F_x)^T lambda = L_x^T`` matrix-free, after the optional
    diagonal change of coordinates ``x = state_scale * y``. The returned
    gradient is ``L_theta + lambda^T F_theta`` and therefore retains direct
    parameter dependence of the loss. Units, frame, assumptions, and valid
    ranges are inherited unchanged from the supplied map and loss.
    """

    state = jnp.asarray(converged_state)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("converged_state must be a non-empty vector")
    if tolerance <= 0.0 or absolute_tolerance < 0.0:
        raise ValueError("linear tolerances must be non-negative with tolerance > 0")
    if restart < 1 or maxiter < 1:
        raise ValueError("restart and maxiter must be positive integers")
    scale = (
        jnp.ones_like(state)
        if state_scale is None
        else jnp.asarray(state_scale, dtype=state.dtype)
    )
    if scale.shape != state.shape:
        raise ValueError("state_scale must have the converged-state shape")

    stopped_state = jax.lax.stop_gradient(state)
    scaled_state = stopped_state / scale

    def scaled_transition_and_loss(scaled, theta):
        next_state, value = transition_and_loss(scale * scaled, theta)
        return next_state / scale, value

    (next_scaled, value), pullback = jax.vjp(
        scaled_transition_and_loss, scaled_state, parameters
    )
    if next_scaled.shape != scaled_state.shape or value.shape != ():
        raise ValueError("transition_and_loss must return (state-shaped vector, scalar)")
    zero_state = jnp.zeros_like(next_scaled)
    unit_loss = jnp.ones_like(value)
    loss_state_gradient, direct_parameter_gradient = pullback(
        (zero_state, unit_loss)
    )

    def transpose_operator(vector):
        map_state_gradient, _ = pullback((vector, jnp.zeros_like(value)))
        return vector - map_state_gradient

    adjoint, _ = gmres(
        transpose_operator,
        loss_state_gradient,
        tol=tolerance,
        atol=absolute_tolerance,
        restart=restart,
        maxiter=maxiter,
        solve_method="incremental",
    )
    linear_residual = transpose_operator(adjoint) - loss_state_gradient
    _, fixed_point_parameter_gradient = pullback((adjoint, jnp.zeros_like(value)))
    gradient = direct_parameter_gradient + fixed_point_parameter_gradient
    rhs_norm = jnp.linalg.norm(loss_state_gradient)
    residual_norm = jnp.linalg.norm(linear_residual)
    denominator = jnp.maximum(rhs_norm, jnp.finfo(state.dtype).eps)
    maximum_rhs = jnp.max(jnp.abs(loss_state_gradient))
    maximum_residual = jnp.max(jnp.abs(linear_residual))
    return value, gradient, {
        "adjoint": adjoint,
        "converged": residual_norm
        <= jnp.maximum(tolerance * rhs_norm, absolute_tolerance),
        "direct_parameter_gradient": direct_parameter_gradient,
        "fixed_point_parameter_gradient": fixed_point_parameter_gradient,
        "initial_residual_norm": rhs_norm,
        "final_residual_norm": residual_norm,
        "relative_residual_norm": residual_norm / denominator,
        "maximum_relative_residual": maximum_residual
        / jnp.maximum(maximum_rhs, jnp.finfo(state.dtype).eps),
    }
