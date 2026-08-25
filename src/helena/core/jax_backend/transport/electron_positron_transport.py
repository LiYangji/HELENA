"""JAX electron/positron transport matching the NumPy reference."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_MASS_CGS as _M_E_CGS,
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    SYNCHROTRON_LOG_COOLING_COEFFICIENT as _LOG_COOLING_COEFFICIENT,
)
from helena.core.contracts import (
    electron_gamma as _electron_gamma,
    finite_scalar as _finite_scalar,
    nonnegative_on_grid as _nonnegative_on_grid,
)
from helena.core.jax_backend.transport.steady_state_finite_volume import (
    steady_state_finite_volume_solution_jax,
)

__all__ = ["electron_positron_steady_state_transport_jax"]


_ELECTRON_SYNCHROTRON_LOSS_COEFFICIENT = float(
    np.exp(-_LOG_COOLING_COEFFICIENT - 3.0 * np.log(_M_E_CGS) + 4.0 * np.log(_E_ESU))
)


def _electron_positron_transport_kernel(
    electron_gamma,
    electron_injection,
    positron_injection,
    magnetic_field_G,
    adiabatic_rate,
    escape_rate,
    additional_loss,
):
    gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    electron_injection = jnp.asarray(electron_injection, dtype=jnp.float64)
    positron_injection = jnp.asarray(positron_injection, dtype=jnp.float64)
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)
    adiabatic_rate = jnp.broadcast_to(
        jnp.asarray(adiabatic_rate, dtype=jnp.float64), gamma.shape
    )
    escape_rate = jnp.broadcast_to(
        jnp.asarray(escape_rate, dtype=jnp.float64), gamma.shape
    )
    additional_loss = jnp.broadcast_to(
        jnp.asarray(additional_loss, dtype=jnp.float64), gamma.shape
    )

    synchrotron_loss = (
        _ELECTRON_SYNCHROTRON_LOSS_COEFFICIENT * magnetic_field**2 * gamma**2
    )
    total_loss = synchrotron_loss + gamma * adiabatic_rate + additional_loss
    electron_density = steady_state_finite_volume_solution_jax(
        gamma, electron_injection, total_loss, escape_rate
    )
    positron_density = steady_state_finite_volume_solution_jax(
        gamma, positron_injection, total_loss, escape_rate
    )
    return (
        electron_density,
        positron_density,
        escape_rate * electron_density,
        escape_rate * positron_density,
    )


def electron_positron_steady_state_transport_jax(
    electron_gamma,
    q_electron_injection_dgamma_cm3_s,
    q_positron_injection_dgamma_cm3_s,
    magnetic_field_G,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
    *,
    additional_continuous_loss_rate_dgamma_s=0.0,
):
    """Return JAX steady electron/positron densities and escape spectra.

    Parameters
    ----------
    electron_gamma : array_like, shape (n_electron,)
        At least two finite, strictly increasing blob-comoving Lorentz factors,
        all greater than one.
    q_electron_injection_dgamma_cm3_s,
    q_positron_injection_dgamma_cm3_s : array_like, shape (n_electron,)
        Finite non-negative electron and positron injection rates in cm^-3 s^-1
        per unit Lorentz factor.
    magnetic_field_G : float
        Finite non-negative blob-comoving magnetic field in Gauss.
    adiabatic_rate_s_inv, escape_rate_s_inv : float or array_like
        Finite non-negative scalar or electron-grid-shaped rates in s^-1.
        Escape is a local particle sink, not a continuous energy loss.
    additional_continuous_loss_rate_dgamma_s : float or array_like, optional
        Caller-supplied finite non-negative ``b_additional = -d gamma/dt`` in
        s^-1, either scalar or electron-grid-shaped. It is used directly, not
        multiplied by ``gamma`` or treated as a sink.

    Returns
    -------
    tuple of jax.Array
        ``(N_electron, N_positron, Q_electron_escape, Q_positron_escape)`` as
        float64 arrays. Densities are ``dn/dgamma`` in cm^-3 per unit Lorentz
        factor; escape spectra are ``lambda_escape N`` in cm^-3 s^-1 per unit
        Lorentz factor.

    Notes
    -----
    Both charges use the same conservative finite-volume operator and total
    loss ``b = gamma/t_syn + gamma*adiabatic_rate + b_additional``; only their
    injections differ. Eager calls validate values, while JIT-traced calls
    require already-valid fixed-shape inputs. JAX float64 mode is required.
    No inverse-Compton cooling, annihilation, radiation, cascade, observer
    transformation, or high-level proton-secondary connection is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX electron/positron transport requires jax_enable_x64")

    arguments = (
        electron_gamma,
        q_electron_injection_dgamma_cm3_s,
        q_positron_injection_dgamma_cm3_s,
        magnetic_field_G,
        adiabatic_rate_s_inv,
        escape_rate_s_inv,
        additional_continuous_loss_rate_dgamma_s,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _electron_positron_transport_kernel(*arguments)

    gamma = _electron_gamma(electron_gamma)
    electron_injection = _nonnegative_on_grid(
        "q_electron_injection_dgamma_cm3_s",
        q_electron_injection_dgamma_cm3_s,
        gamma,
        allow_scalar=False,
    )
    positron_injection = _nonnegative_on_grid(
        "q_positron_injection_dgamma_cm3_s",
        q_positron_injection_dgamma_cm3_s,
        gamma,
        allow_scalar=False,
    )
    magnetic_field_input = magnetic_field_G
    if isinstance(magnetic_field_G, jax.Array) and magnetic_field_G.ndim == 0:
        magnetic_field_input = float(magnetic_field_G)
    magnetic_field = _finite_scalar(
        "magnetic_field_G", magnetic_field_input, allow_zero=True
    )
    adiabatic_rate = _nonnegative_on_grid(
        "adiabatic_rate_s_inv", adiabatic_rate_s_inv, gamma, allow_scalar=True
    )
    escape_rate = _nonnegative_on_grid(
        "escape_rate_s_inv", escape_rate_s_inv, gamma, allow_scalar=True
    )
    additional_loss = _nonnegative_on_grid(
        "additional_continuous_loss_rate_dgamma_s",
        additional_continuous_loss_rate_dgamma_s,
        gamma,
        allow_scalar=True,
    )

    outputs = _electron_positron_transport_kernel(
        gamma,
        electron_injection,
        positron_injection,
        magnetic_field,
        adiabatic_rate,
        escape_rate,
        additional_loss,
    )
    if any(isinstance(output, jax.core.Tracer) for output in outputs):
        return outputs
    if any(
        not np.all(np.isfinite(np.asarray(output))) or np.any(np.asarray(output) < 0.0)
        for output in outputs
    ):
        raise FloatingPointError(
            "JAX electron/positron transport output exceeds float64 range"
        )
    return outputs
