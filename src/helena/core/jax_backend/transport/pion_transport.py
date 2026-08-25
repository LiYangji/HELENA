"""JAX steady-state transport of charged pions."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.contracts import (
    pion_gamma as _pion_gamma,
    positive_magnetic_field as _positive_magnetic_field,
    transport_array as _transport_array,
)
from helena.core.jax_backend.kernels.pion_timescales import (
    _charged_pion_decay_time_kernel,
    _charged_pion_synchrotron_cooling_time_kernel,
)
from helena.core.jax_backend.transport.steady_state_finite_volume import (
    steady_state_finite_volume_solution_jax,
)

__all__ = ["charged_pion_steady_state_transport_jax"]


def _charged_pion_transport_kernel(
    pion_gamma,
    q_pi_plus_injection_dgamma_cm3_s,
    q_pi_minus_injection_dgamma_cm3_s,
    magnetic_field_G,
):
    gamma = jnp.asarray(pion_gamma, dtype=jnp.float64)
    plus_injection = jnp.asarray(q_pi_plus_injection_dgamma_cm3_s, dtype=jnp.float64)
    minus_injection = jnp.asarray(q_pi_minus_injection_dgamma_cm3_s, dtype=jnp.float64)
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)

    decay_time = _charged_pion_decay_time_kernel(gamma)
    synchrotron_time = _charged_pion_synchrotron_cooling_time_kernel(
        gamma, magnetic_field
    )
    synchrotron_loss = gamma / synchrotron_time
    decay_rate = 1.0 / decay_time

    plus_density = steady_state_finite_volume_solution_jax(
        gamma, plus_injection, synchrotron_loss, decay_rate
    )
    minus_density = steady_state_finite_volume_solution_jax(
        gamma, minus_injection, synchrotron_loss, decay_rate
    )
    return (
        plus_density,
        minus_density,
        decay_rate * plus_density,
        decay_rate * minus_density,
    )


def charged_pion_steady_state_transport_jax(
    pion_gamma,
    q_pi_plus_injection_dgamma_cm3_s,
    q_pi_minus_injection_dgamma_cm3_s,
    magnetic_field_G,
):
    """Return JAX charged-pion densities and actual decay-event rates.

    Parameters
    ----------
    pion_gamma : array_like, shape (n_pion,)
        At least two finite, strictly increasing blob-frame charged-pion
        Lorentz factors, all greater than one.
    q_pi_plus_injection_dgamma_cm3_s,
    q_pi_minus_injection_dgamma_cm3_s : array_like, shape (n_pion,)
        Finite non-negative pion injection rates in cm^-3 s^-1 per unit
        Lorentz factor.
    magnetic_field_G : float
        Finite positive blob-frame magnetic-field magnitude in Gauss.

    Returns
    -------
    tuple of jax.Array
        ``(N_pi+, N_pi-, Q_pi+_dec, Q_pi-_dec)`` as float64 arrays. Densities
        are in cm^-3 per unit Lorentz factor and decay-event rates are in
        cm^-3 s^-1 per unit Lorentz factor.

    Notes
    -----
    The conservative finite-volume equation contains only continuous
    classical synchrotron loss ``b_syn = gamma_pi / t_syn`` and the local
    decay sink ``lambda_dec = 1 / t_dec``. The returned decay spectra are
    exactly ``lambda_dec N_pi``; no prompt-decay switch or branching ratio is
    applied. All quantities are blob-comoving. The synchrotron approximation
    assumes ``gamma_pi >> 1`` and ``chi_pi << 1``. Eager calls validate inputs;
    JIT-traced calls require already-valid inputs. The function includes no
    escape, daughters, cascade, observer transformation, or fitting wrapper.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX charged-pion transport requires jax_enable_x64")

    arguments = (
        pion_gamma,
        q_pi_plus_injection_dgamma_cm3_s,
        q_pi_minus_injection_dgamma_cm3_s,
        magnetic_field_G,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _charged_pion_transport_kernel(*arguments)

    gamma = _pion_gamma(pion_gamma)
    if gamma.size < 2 or np.any(np.diff(gamma) <= 0.0):
        raise ValueError(
            "pion_gamma must contain at least two strictly increasing values"
        )
    plus_injection = _transport_array(
        "q_pi_plus_injection_dgamma_cm3_s",
        q_pi_plus_injection_dgamma_cm3_s,
        shape=gamma.shape,
    )
    minus_injection = _transport_array(
        "q_pi_minus_injection_dgamma_cm3_s",
        q_pi_minus_injection_dgamma_cm3_s,
        shape=gamma.shape,
    )
    magnetic_field_input = magnetic_field_G
    if isinstance(magnetic_field_G, jax.Array) and magnetic_field_G.ndim == 0:
        magnetic_field_input = float(magnetic_field_G)
    magnetic_field = _positive_magnetic_field(magnetic_field_input)

    outputs = _charged_pion_transport_kernel(
        gamma,
        plus_injection,
        minus_injection,
        magnetic_field,
    )
    if any(
        not np.all(np.isfinite(np.asarray(output))) or np.any(np.asarray(output) < 0.0)
        for output in outputs
    ):
        raise FloatingPointError(
            "JAX charged-pion transport output exceeds the finite float64 range"
        )
    return outputs
