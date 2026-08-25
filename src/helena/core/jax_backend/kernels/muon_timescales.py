"""JAX muon decay and synchrotron timescales."""

import jax
import jax.numpy as jnp
import numpy as np


from helena.core.constants import (
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    MUON_MASS_CGS as _MUON_MASS_G,
    MUON_REST_LIFETIME_S as _MUON_REST_LIFETIME_S,
    SYNCHROTRON_LOG_COOLING_COEFFICIENT as _LOG_COOLING_COEFFICIENT,
)
from helena.core.contracts import (
    muon_gamma as _muon_gamma,
    positive_real_scalar as _positive_real_scalar,
)

__all__ = [
    "muon_decay_time_jax",
    "muon_synchrotron_cooling_time_jax",
]


def _muon_decay_time_kernel(muon_gamma):
    gamma = jnp.asarray(muon_gamma, dtype=jnp.float64)
    return gamma * _MUON_REST_LIFETIME_S


def _muon_synchrotron_cooling_time_kernel(muon_gamma, magnetic_field_G):
    gamma = jnp.asarray(muon_gamma, dtype=jnp.float64)
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)
    log_time = (
        _LOG_COOLING_COEFFICIENT
        + 3.0 * np.log(_MUON_MASS_G)
        - 4.0 * np.log(_E_ESU)
        - 2.0 * jnp.log(magnetic_field)
        - jnp.log(gamma)
    )
    return jnp.exp(log_time)


def muon_decay_time_jax(muon_gamma):
    """Return the blob-frame muon decay time as a JAX float64 array.

    ``muon_gamma`` is a non-empty one-dimensional array of finite Lorentz
    factors greater than one. The result is ``gamma_mu tau_mu`` in seconds,
    preserves input order, and uses the same PDG 2026 lifetime as the NumPy
    reference. Charge and helicity do not alter the lifetime. Eager calls
    validate inputs; JIT-traced calls require already-valid inputs. No
    transport, daughter injection, or observer transformation is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX muon timescales require jax_enable_x64")
    if isinstance(muon_gamma, jax.core.Tracer):
        return _muon_decay_time_kernel(muon_gamma)

    gamma = _muon_gamma(muon_gamma)
    result = _muon_decay_time_kernel(gamma)
    values = np.asarray(result)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise FloatingPointError("JAX muon decay time exceeds the finite float64 range")
    return result


def muon_synchrotron_cooling_time_jax(muon_gamma, magnetic_field_G):
    """Return the blob-frame muon synchrotron cooling time.

    ``muon_gamma`` is a non-empty one-dimensional array of finite Lorentz
    factors greater than one; ``magnetic_field_G`` is a finite positive scalar
    in Gauss. The float64 result is in seconds and follows the existing
    classical isotropic convention
    ``t_syn = 9 m_mu^3 c^5 / (4 e^4 B^2 gamma_mu)``. It assumes
    ``gamma_mu >> 1``, ``chi_mu << 1``, and ``beta_mu^2 ~= 1``. Charge and
    helicity do not alter this time. Eager calls validate inputs; JIT-traced
    calls require already-valid inputs. No cooling evolution or radiation
    spectrum is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX muon timescales require jax_enable_x64")
    arguments = (muon_gamma, magnetic_field_G)
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _muon_synchrotron_cooling_time_kernel(*arguments)

    gamma = _muon_gamma(muon_gamma)
    magnetic_field_input = magnetic_field_G
    if isinstance(magnetic_field_G, jax.Array) and magnetic_field_G.ndim == 0:
        magnetic_field_input = float(magnetic_field_G)
    magnetic_field = _positive_real_scalar("magnetic_field_G", magnetic_field_input)
    result = _muon_synchrotron_cooling_time_kernel(gamma, magnetic_field)
    values = np.asarray(result)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise FloatingPointError(
            "JAX muon synchrotron cooling time exceeds the finite float64 range"
        )
    return result
