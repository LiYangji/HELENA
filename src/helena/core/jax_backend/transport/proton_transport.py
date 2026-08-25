"""JAX steady-state proton transport with prescribed fractional loss rates."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    PROTON_MASS_CGS as _M_P_CGS,
    SYNCHROTRON_LOG_COOLING_COEFFICIENT as _LOG_COOLING_COEFFICIENT,
)
from helena.core.contracts import (
    nonnegative_scalar as _nonnegative_scalar,
    proton_gamma as _proton_gamma,
    proton_injection as _proton_injection,
    rate_on_grid as _rate_on_grid,
    transport_array as _transport_array,
)
from helena.core.jax_backend.transport.steady_state_finite_volume import (
    steady_state_finite_volume_solution_jax,
)

__all__ = ["proton_steady_state_transport_jax"]


_PROTON_SYNCHROTRON_LOSS_COEFFICIENT = float(
    np.exp(-_LOG_COOLING_COEFFICIENT - 3.0 * np.log(_M_P_CGS) + 4.0 * np.log(_E_ESU))
)


def _proton_transport_kernel(
    proton_gamma,
    injection,
    magnetic_field_G,
    fractional_photomeson_loss,
    adiabatic_rate,
    escape_rate,
    fractional_bethe_heitler_loss,
    fractional_proton_proton_loss,
):
    gamma = jnp.asarray(proton_gamma, dtype=jnp.float64)
    injection = jnp.asarray(injection, dtype=jnp.float64)
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)
    fractional_photomeson_loss = jnp.asarray(
        fractional_photomeson_loss, dtype=jnp.float64
    )
    adiabatic_rate = jnp.broadcast_to(
        jnp.asarray(adiabatic_rate, dtype=jnp.float64), gamma.shape
    )
    escape_rate = jnp.broadcast_to(
        jnp.asarray(escape_rate, dtype=jnp.float64), gamma.shape
    )
    fractional_bethe_heitler_loss = jnp.broadcast_to(
        jnp.asarray(fractional_bethe_heitler_loss, dtype=jnp.float64), gamma.shape
    )
    fractional_proton_proton_loss = jnp.broadcast_to(
        jnp.asarray(fractional_proton_proton_loss, dtype=jnp.float64), gamma.shape
    )

    synchrotron_loss = (
        _PROTON_SYNCHROTRON_LOSS_COEFFICIENT * magnetic_field**2 * gamma**2
    )
    adiabatic_loss = gamma * adiabatic_rate
    photomeson_loss = gamma * fractional_photomeson_loss
    bethe_heitler_loss = gamma * fractional_bethe_heitler_loss
    proton_proton_loss = gamma * fractional_proton_proton_loss
    total_loss = (
        synchrotron_loss
        + adiabatic_loss
        + photomeson_loss
        + bethe_heitler_loss
        + proton_proton_loss
    )
    density = steady_state_finite_volume_solution_jax(
        gamma, injection, total_loss, escape_rate
    )
    return density, escape_rate * density


def proton_steady_state_transport_jax(
    proton_gamma,
    q_proton_injection_dgamma_cm3_s,
    magnetic_field_G,
    fractional_photomeson_energy_loss_rate_s_inv,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
    *,
    fractional_bethe_heitler_energy_loss_rate_s_inv=0.0,
    fractional_proton_proton_energy_loss_rate_s_inv=0.0,
):
    """Return a JAX steady proton density and escape-rate spectrum.

    Parameters
    ----------
    proton_gamma : array_like, shape (n_proton,)
        At least two finite, strictly increasing blob-frame proton Lorentz
        factors greater than one.
    q_proton_injection_dgamma_cm3_s : array_like, shape (n_proton,)
        Finite non-negative injection ``dn_p/(d gamma_p dt)`` in
        cm^-3 s^-1 per unit Lorentz factor.
    magnetic_field_G : float
        Finite non-negative blob-comoving magnetic-field magnitude in Gauss.
    fractional_photomeson_energy_loss_rate_s_inv : array_like, shape (n_proton,)
        Fixed finite non-negative NumPy photomeson fractional energy-loss rate
        in s^-1. This is not the photomeson interaction rate.
    adiabatic_rate_s_inv, escape_rate_s_inv : float or array_like
        Finite non-negative scalar or proton-grid-shaped rates in s^-1.
    fractional_bethe_heitler_energy_loss_rate_s_inv : float or array_like, optional
        Finite non-negative scalar or proton-grid-shaped Bethe--Heitler
        fractional energy-loss rate in s^-1. The default is exact zero.
    fractional_proton_proton_energy_loss_rate_s_inv : float or array_like, optional
        Finite non-negative scalar or proton-grid-shaped cold-target pp
        fractional energy-loss rate in s^-1. The default is exact zero.

    Returns
    -------
    tuple of jax.Array
        ``(N_p, Q_p,esc)`` as float64 arrays. ``N_p = dn_p/d gamma_p`` is in
        cm^-3 and ``Q_p,esc = lambda_p,esc N_p`` is in cm^-3 s^-1, both per
        unit proton Lorentz factor.

    Notes
    -----
    The continuous loss is the sum of the existing classical isotropic proton
    synchrotron convention, ``gamma_p * adiabatic_rate_s_inv``,
    ``gamma_p * fractional_photomeson_energy_loss_rate_s_inv``, and
    ``gamma_p`` times each Bethe--Heitler and proton--proton fractional loss.
    Each fractional input is multiplied by ``gamma_p`` exactly once. These are
    continuous cooling terms, not proton sinks; escape remains separate.

    Synchrotron assumes ``gamma_p >> 1`` and ``chi_p << 1``. All quantities are
    blob-comoving. The function requires JAX float64 mode; eager calls validate
    inputs, while JIT-traced calls require already-valid inputs. It does not
    differentiate through the NumPy photomeson kernel or include proton-neutron
    conversion, Bethe--Heitler pair injection, pp pion injection, acceleration,
    or observer-frame transformations.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX proton transport requires jax_enable_x64")

    arguments = (
        proton_gamma,
        q_proton_injection_dgamma_cm3_s,
        magnetic_field_G,
        fractional_photomeson_energy_loss_rate_s_inv,
        adiabatic_rate_s_inv,
        escape_rate_s_inv,
        fractional_bethe_heitler_energy_loss_rate_s_inv,
        fractional_proton_proton_energy_loss_rate_s_inv,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _proton_transport_kernel(*arguments)

    gamma = _proton_gamma(proton_gamma)
    if gamma.size < 2 or np.any(np.diff(gamma) <= 0.0):
        raise ValueError(
            "proton_gamma must contain at least two strictly increasing values"
        )
    injection = _proton_injection(q_proton_injection_dgamma_cm3_s, gamma)
    magnetic_field_input = magnetic_field_G
    if isinstance(magnetic_field_G, jax.Array) and magnetic_field_G.ndim == 0:
        magnetic_field_input = float(magnetic_field_G)
    magnetic_field = _nonnegative_scalar("magnetic_field_G", magnetic_field_input)
    fractional_photomeson_loss = _transport_array(
        "fractional_photomeson_energy_loss_rate_s_inv",
        fractional_photomeson_energy_loss_rate_s_inv,
        shape=gamma.shape,
    )
    adiabatic_rate = _rate_on_grid("adiabatic_rate_s_inv", adiabatic_rate_s_inv, gamma)
    escape_rate = _rate_on_grid("escape_rate_s_inv", escape_rate_s_inv, gamma)
    fractional_bethe_heitler_loss = _rate_on_grid(
        "fractional_bethe_heitler_energy_loss_rate_s_inv",
        fractional_bethe_heitler_energy_loss_rate_s_inv,
        gamma,
    )
    fractional_proton_proton_loss = _rate_on_grid(
        "fractional_proton_proton_energy_loss_rate_s_inv",
        fractional_proton_proton_energy_loss_rate_s_inv,
        gamma,
    )

    density, escape = _proton_transport_kernel(
        gamma,
        injection,
        magnetic_field,
        fractional_photomeson_loss,
        adiabatic_rate,
        escape_rate,
        fractional_bethe_heitler_loss,
        fractional_proton_proton_loss,
    )
    if any(isinstance(value, jax.core.Tracer) for value in (density, escape)):
        return density, escape
    density_array = np.asarray(density)
    escape_array = np.asarray(escape)
    if (
        not np.all(np.isfinite(density_array))
        or np.any(density_array < 0.0)
        or not np.all(np.isfinite(escape_array))
        or np.any(escape_array < 0.0)
    ):
        raise FloatingPointError("JAX proton transport output exceeds float64 range")
    return density, escape
