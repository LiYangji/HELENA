"""JAX steady-state transport of helicity-resolved muons."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.contracts import (
    muon_gamma as _muon_gamma,
    positive_magnetic_field as _positive_magnetic_field,
    transport_array as _transport_array,
)
from helena.core.jax_backend.kernels.muon_timescales import (
    _muon_decay_time_kernel,
    _muon_synchrotron_cooling_time_kernel,
)
from helena.core.jax_backend.transport.steady_state_finite_volume import (
    steady_state_finite_volume_solution_jax,
)

__all__ = ["helicity_resolved_muon_steady_state_transport_jax"]


def _muon_transport_kernel(
    muon_gamma,
    q_mu_plus_left_injection_dgamma_cm3_s,
    q_mu_plus_right_injection_dgamma_cm3_s,
    q_mu_minus_left_injection_dgamma_cm3_s,
    q_mu_minus_right_injection_dgamma_cm3_s,
    magnetic_field_G,
):
    gamma = jnp.asarray(muon_gamma, dtype=jnp.float64)
    injections = tuple(
        jnp.asarray(injection, dtype=jnp.float64)
        for injection in (
            q_mu_plus_left_injection_dgamma_cm3_s,
            q_mu_plus_right_injection_dgamma_cm3_s,
            q_mu_minus_left_injection_dgamma_cm3_s,
            q_mu_minus_right_injection_dgamma_cm3_s,
        )
    )
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)
    decay_time = _muon_decay_time_kernel(gamma)
    synchrotron_time = _muon_synchrotron_cooling_time_kernel(gamma, magnetic_field)
    synchrotron_loss = gamma / synchrotron_time
    decay_rate = 1.0 / decay_time
    densities = tuple(
        steady_state_finite_volume_solution_jax(
            gamma, injection, synchrotron_loss, decay_rate
        )
        for injection in injections
    )
    return densities + tuple(decay_rate * density for density in densities)


def helicity_resolved_muon_steady_state_transport_jax(
    muon_gamma,
    q_mu_plus_left_injection_dgamma_cm3_s,
    q_mu_plus_right_injection_dgamma_cm3_s,
    q_mu_minus_left_injection_dgamma_cm3_s,
    q_mu_minus_right_injection_dgamma_cm3_s,
    magnetic_field_G,
):
    """Return JAX muon densities and actual decay-event rates.

    Parameters
    ----------
    muon_gamma : array_like, shape (n_muon,)
        At least two finite, strictly increasing blob-frame muon Lorentz
        factors, all greater than one.
    q_mu_plus_left_injection_dgamma_cm3_s,
    q_mu_plus_right_injection_dgamma_cm3_s,
    q_mu_minus_left_injection_dgamma_cm3_s,
    q_mu_minus_right_injection_dgamma_cm3_s : array_like, shape (n_muon,)
        Finite non-negative injection rates in cm^-3 s^-1 per unit muon
        Lorentz factor. Left and right mean ``h=-1`` and ``h=+1``.
    magnetic_field_G : float
        Finite positive blob-frame magnetic-field magnitude in Gauss.

    Returns
    -------
    tuple of jax.Array
        ``(N_mu+_L, N_mu+_R, N_mu-_L, N_mu-_R, Q_mu+_L,dec,
        Q_mu+_R,dec, Q_mu-_L,dec, Q_mu-_R,dec)`` as float64 arrays. Densities
        are in cm^-3 and decay-event rates are in cm^-3 s^-1, each per unit
        muon Lorentz factor.

    Notes
    -----
    Every channel independently uses the same conservative finite-volume
    continuous synchrotron loss ``b_syn = gamma_mu / t_syn`` and local decay
    sink ``lambda_dec = 1 / t_dec``. Returned decay rates are exactly
    ``lambda_dec N_mu``. No channel mixing, helicity evolution, depolarization,
    branching ratio, prompt switch, escape, adiabatic loss, floor, or daughter
    kernel is included. All quantities are blob-comoving. The synchrotron
    convention assumes ``gamma_mu >> 1`` and ``chi_mu << 1``. Eager calls
    validate inputs; JIT-traced calls require already-valid fixed-shape inputs.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX muon transport requires jax_enable_x64")
    arguments = (
        muon_gamma,
        q_mu_plus_left_injection_dgamma_cm3_s,
        q_mu_plus_right_injection_dgamma_cm3_s,
        q_mu_minus_left_injection_dgamma_cm3_s,
        q_mu_minus_right_injection_dgamma_cm3_s,
        magnetic_field_G,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _muon_transport_kernel(*arguments)

    gamma = _muon_gamma(muon_gamma)
    if gamma.size < 2 or np.any(np.diff(gamma) <= 0.0):
        raise ValueError(
            "muon_gamma must contain at least two strictly increasing values"
        )
    names = (
        "q_mu_plus_left_injection_dgamma_cm3_s",
        "q_mu_plus_right_injection_dgamma_cm3_s",
        "q_mu_minus_left_injection_dgamma_cm3_s",
        "q_mu_minus_right_injection_dgamma_cm3_s",
    )
    injections = tuple(
        _transport_array(name, injection, shape=gamma.shape)
        for name, injection in zip(names, arguments[1:5], strict=True)
    )
    magnetic_field_input = magnetic_field_G
    if isinstance(magnetic_field_G, jax.Array) and magnetic_field_G.ndim == 0:
        magnetic_field_input = float(magnetic_field_G)
    magnetic_field = _positive_magnetic_field(magnetic_field_input)
    outputs = _muon_transport_kernel(gamma, *injections, magnetic_field)
    if any(
        not np.all(np.isfinite(np.asarray(output))) or np.any(np.asarray(output) < 0.0)
        for output in outputs
    ):
        raise FloatingPointError(
            "JAX muon transport output exceeds the finite float64 range"
        )
    return outputs
