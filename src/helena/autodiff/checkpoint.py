"""Checkpointed execution helpers for already-defined discrete physics maps."""

import jax


def checkpointed_fixed_unroll(step, initial_state, rounds):
    """Apply a pure JAX transition exactly ``rounds`` times with rematerialization.

    ``step(state)`` must return ``(next_state, output)`` pytrees. Input and output
    units, valid physical ranges, and frame conventions are inherited unchanged
    from ``step``; this helper assigns no physics meaning. ``rounds`` must be a
    positive Python integer. Static grids and prepared responses that cannot be
    traced should remain in the ``step`` closure rather than ``initial_state``.
    Only reverse-mode residual storage changes: the primal transition and
    fixed-unroll result are otherwise identical.
    """

    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("rounds must be a positive Python integer")
    checkpointed_step = jax.checkpoint(step)
    state = initial_state
    for _ in range(rounds):
        state, output = checkpointed_step(state)
    return state, output
