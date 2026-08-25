"""JAX charged-pion decay and synchrotron timescales."""

import jax
import jax.numpy as jnp
import numpy as np


from helena.core.constants import (
    CHARGED_PION_MASS_CGS as _CHARGED_PION_MASS_G,
    CHARGED_PION_REST_LIFETIME_S as _CHARGED_PION_REST_LIFETIME_S,
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    SYNCHROTRON_LOG_COOLING_COEFFICIENT as _LOG_COOLING_COEFFICIENT,
)
from helena.core.contracts import (
    pion_gamma as _pion_gamma,
    positive_real_scalar as _positive_real_scalar,
)

__all__ = [
    "charged_pion_decay_time_jax",
    "charged_pion_synchrotron_cooling_time_jax",
]


def _charged_pion_decay_time_kernel(pion_gamma):
    gamma = jnp.asarray(pion_gamma, dtype=jnp.float64)
    return gamma * _CHARGED_PION_REST_LIFETIME_S


def _charged_pion_synchrotron_cooling_time_kernel(
    pion_gamma,
    magnetic_field_G,
):
    gamma = jnp.asarray(pion_gamma, dtype=jnp.float64)
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)
    log_time = (
        _LOG_COOLING_COEFFICIENT
        + 3.0 * np.log(_CHARGED_PION_MASS_G)
        - 4.0 * np.log(_E_ESU)
        - 2.0 * jnp.log(magnetic_field)
        - jnp.log(gamma)
    )
    return jnp.exp(log_time)


def charged_pion_decay_time_jax(pion_gamma):
    """Return the blob-frame charged-pion decay time as a JAX array.

    ``pion_gamma`` is a non-empty one-dimensional array of finite Lorentz
    factors greater than one. The float64 result is ``gamma_pi tau_pi`` in
    seconds and preserves input order, using the same PDG 2024 rest lifetime
    as the NumPy reference. This diagnostic includes no transport, daughter
    injection, or observer-frame transformation. Eager calls validate inputs;
    JIT-traced calls require already-valid inputs.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX charged-pion timescales require jax_enable_x64")

    if isinstance(pion_gamma, jax.core.Tracer):
        return _charged_pion_decay_time_kernel(pion_gamma)

    gamma = _pion_gamma(pion_gamma)
    result = _charged_pion_decay_time_kernel(gamma)
    result_array = np.asarray(result)
    if not np.all(np.isfinite(result_array)) or np.any(result_array <= 0.0):
        raise FloatingPointError(
            "JAX charged-pion decay time exceeds the finite float64 range"
        )
    return result


def charged_pion_synchrotron_cooling_time_jax(
    pion_gamma,
    magnetic_field_G,
):
    """Return the blob-frame charged-pion synchrotron cooling time.

    ``pion_gamma`` is a non-empty one-dimensional array of finite Lorentz
    factors greater than one, and ``magnetic_field_G`` is a finite positive
    scalar in Gauss. The float64 result is in seconds and follows the existing
    classical isotropic convention
    ``t_syn = 9 m_pi^3 c^5 / (4 e^4 B^2 gamma_pi)``. It assumes
    ``gamma_pi >> 1`` and ``chi_pi << 1`` and has no ``beta^2`` correction.
    Eager calls validate inputs; JIT-traced calls require already-valid inputs.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX charged-pion timescales require jax_enable_x64")

    arguments = (pion_gamma, magnetic_field_G)
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _charged_pion_synchrotron_cooling_time_kernel(*arguments)

    gamma = _pion_gamma(pion_gamma)
    magnetic_field_input = magnetic_field_G
    if isinstance(magnetic_field_G, jax.Array) and magnetic_field_G.ndim == 0:
        magnetic_field_input = float(magnetic_field_G)
    magnetic_field = _positive_real_scalar("magnetic_field_G", magnetic_field_input)
    result = _charged_pion_synchrotron_cooling_time_kernel(gamma, magnetic_field)
    result_array = np.asarray(result)
    if not np.all(np.isfinite(result_array)) or np.any(result_array <= 0.0):
        raise FloatingPointError(
            "JAX charged-pion synchrotron cooling time exceeds the finite "
            "float64 range"
        )
    return result
