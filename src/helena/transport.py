"""Backend-neutral public facade for steady particle transport."""

from helena.core.jax_backend.transport.steady_state_finite_volume import (
    steady_state_finite_volume_solution_jax as _steady_state_solution,
)

__all__ = ["solve_steady_state"]


def solve_steady_state(
    particle_gamma,
    injection_dgamma_cm3_s,
    continuous_loss_rate_dgamma_s,
    sink_rate_s_inv,
):
    """Return steady blob-frame ``N(gamma)`` in cm^-3 per gamma.

    ``particle_gamma`` is a strictly increasing total/rest-energy grid.
    Injection is ``Q(gamma)`` in cm^-3 s^-1 per gamma, continuous loss is
    ``-d gamma/dt`` in s^-1, and the local escape/decay sink is in s^-1.
    The high-energy boundary has no inflow and the low-energy face is outflow;
    finite-volume coefficients and work arrays remain internal.
    """
    return _steady_state_solution(
        particle_gamma,
        injection_dgamma_cm3_s,
        continuous_loss_rate_dgamma_s,
        sink_rate_s_inv,
    )
