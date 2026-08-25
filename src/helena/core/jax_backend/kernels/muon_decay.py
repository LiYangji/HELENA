"""JAX helicity-dependent daughter injection from muon decay."""

import jax
import jax.numpy as jnp
import numpy as np


from helena.core.constants import (
    ELECTRON_TO_MUON_MASS_RATIO as _ELECTRON_TO_MUON_MASS_RATIO,
    MUON_REST_ENERGY_CGS as _MUON_REST_ENERGY_CGS,
)
from helena.core.contracts import muon_decay_inputs as _muon_decay_inputs

__all__ = [
    "muon_decay_electron_positron_injection_jax",
    "muon_decay_neutrino_injection_jax",
]


def _trapezoid_last_axis(values, grid):
    return jnp.sum(jnp.diff(grid) * (values[..., 1:] + values[..., :-1]) / 2.0, axis=-1)


def _michel_scalings(x):
    support = (x > 0.0) & (x < 1.0)
    x_safe = jnp.where(support, x, 0.0)
    x_squared = x_safe**2
    x_cubed = x_safe * x_squared
    a = 5.0 / 3.0 - 3.0 * x_squared + 4.0 / 3.0 * x_cubed
    b = -1.0 / 3.0 + 3.0 * x_squared - 8.0 / 3.0 * x_cubed
    c = 2.0 - 6.0 * x_squared + 4.0 * x_cubed
    d = 2.0 - 12.0 * x_safe + 18.0 * x_squared - 8.0 * x_cubed
    zero = jnp.zeros_like(x)
    return (
        jnp.where(support, jnp.maximum(a - b, 0.0), zero),
        jnp.where(support, jnp.maximum(a + b, 0.0), zero),
        jnp.where(support, jnp.maximum(c - d, 0.0), zero),
        jnp.where(support, jnp.maximum(c + d, 0.0), zero),
    )


def _scaling_injection_integral(muon_gamma, decay_rate, scaling):
    integrand = scaling * (decay_rate / muon_gamma)[None, :]
    return _trapezoid_last_axis(integrand, muon_gamma)


def _electron_positron_kernel(
    electron_gamma,
    muon_gamma,
    q_mu_plus_left_decay_dgamma_cm3_s,
    q_mu_plus_right_decay_dgamma_cm3_s,
    q_mu_minus_left_decay_dgamma_cm3_s,
    q_mu_minus_right_decay_dgamma_cm3_s,
):
    daughter_gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    parent_gamma = jnp.asarray(muon_gamma, dtype=jnp.float64)
    rates = tuple(
        jnp.asarray(rate, dtype=jnp.float64)
        for rate in (
            q_mu_plus_left_decay_dgamma_cm3_s,
            q_mu_plus_right_decay_dgamma_cm3_s,
            q_mu_minus_left_decay_dgamma_cm3_s,
            q_mu_minus_right_decay_dgamma_cm3_s,
        )
    )
    x = daughter_gamma[:, None] * _ELECTRON_TO_MUON_MASS_RATIO / parent_gamma[None, :]
    ab_minus, ab_plus, _, _ = _michel_scalings(x)
    plus_left, plus_right, minus_left, minus_right = rates
    coefficient = _ELECTRON_TO_MUON_MASS_RATIO
    return (
        coefficient
        * (
            _scaling_injection_integral(parent_gamma, plus_left, ab_minus)
            + _scaling_injection_integral(parent_gamma, plus_right, ab_plus)
        ),
        coefficient
        * (
            _scaling_injection_integral(parent_gamma, minus_left, ab_plus)
            + _scaling_injection_integral(parent_gamma, minus_right, ab_minus)
        ),
    )


def _neutrino_kernel(
    neutrino_energy_erg,
    muon_gamma,
    q_mu_plus_left_decay_dgamma_cm3_s,
    q_mu_plus_right_decay_dgamma_cm3_s,
    q_mu_minus_left_decay_dgamma_cm3_s,
    q_mu_minus_right_decay_dgamma_cm3_s,
):
    neutrino_energy = jnp.asarray(neutrino_energy_erg, dtype=jnp.float64)
    parent_gamma = jnp.asarray(muon_gamma, dtype=jnp.float64)
    rates = tuple(
        jnp.asarray(rate, dtype=jnp.float64)
        for rate in (
            q_mu_plus_left_decay_dgamma_cm3_s,
            q_mu_plus_right_decay_dgamma_cm3_s,
            q_mu_minus_left_decay_dgamma_cm3_s,
            q_mu_minus_right_decay_dgamma_cm3_s,
        )
    )
    x = neutrino_energy[:, None] / (parent_gamma[None, :] * _MUON_REST_ENERGY_CGS)
    ab_minus, ab_plus, cd_minus, cd_plus = _michel_scalings(x)
    plus_left, plus_right, minus_left, minus_right = rates
    coefficient = 1.0 / _MUON_REST_ENERGY_CGS
    return (
        coefficient
        * (
            _scaling_injection_integral(parent_gamma, plus_left, cd_minus)
            + _scaling_injection_integral(parent_gamma, plus_right, cd_plus)
        ),
        coefficient
        * (
            _scaling_injection_integral(parent_gamma, minus_left, cd_plus)
            + _scaling_injection_integral(parent_gamma, minus_right, cd_minus)
        ),
        coefficient
        * (
            _scaling_injection_integral(parent_gamma, minus_left, ab_plus)
            + _scaling_injection_integral(parent_gamma, minus_right, ab_minus)
        ),
        coefficient
        * (
            _scaling_injection_integral(parent_gamma, plus_left, ab_minus)
            + _scaling_injection_integral(parent_gamma, plus_right, ab_plus)
        ),
    )


def _checked_outputs(outputs, diagnostic):
    values = np.asarray(outputs)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(f"{diagnostic} exceeds the finite float64 range")
    return outputs


def muon_decay_electron_positron_injection_jax(
    electron_gamma,
    muon_gamma,
    q_mu_plus_left_decay_dgamma_cm3_s,
    q_mu_plus_right_decay_dgamma_cm3_s,
    q_mu_minus_left_decay_dgamma_cm3_s,
    q_mu_minus_right_decay_dgamma_cm3_s,
):
    """Return JAX positron and electron injection from muon decay.

    ``electron_gamma`` is a finite one-dimensional blob-frame grid greater
    than one in any order. ``muon_gamma`` has at least two finite, strictly
    increasing values greater than one. The four matching inputs are actual
    non-negative decay-event rates in cm^-3 s^-1 per unit muon Lorentz factor,
    ordered as ``mu+_L, mu+_R, mu-_L, mu-_R`` with ``L`` meaning ``h=-1``.

    The fixed float64 return order is ``(Q_positron, Q_electron)`` in cm^-3
    s^-1 per unit electron Lorentz factor. The leading-order ultrarelativistic
    Michel scaling, helicity/charge mapping, electron-mass Jacobian, support,
    and no-branching-factor convention match the NumPy reference. Eager calls
    validate inputs; JIT-traced calls require valid fixed-shape inputs. No
    timescale decision, transport, cooling, radiation, or frame transform is
    included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX muon decay requires jax_enable_x64")
    arguments = (
        electron_gamma,
        muon_gamma,
        q_mu_plus_left_decay_dgamma_cm3_s,
        q_mu_plus_right_decay_dgamma_cm3_s,
        q_mu_minus_left_decay_dgamma_cm3_s,
        q_mu_minus_right_decay_dgamma_cm3_s,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _electron_positron_kernel(*arguments)

    daughter, gamma, rates = _muon_decay_inputs(
        "electron_gamma",
        electron_gamma,
        daughter_lower_bound=1.0,
        muon_gamma=muon_gamma,
        parent_rates=arguments[2:],
    )
    return _checked_outputs(
        _electron_positron_kernel(daughter, gamma, *rates),
        "muon-decay electron/positron injection",
    )


def muon_decay_neutrino_injection_jax(
    neutrino_energy_erg,
    muon_gamma,
    q_mu_plus_left_decay_dgamma_cm3_s,
    q_mu_plus_right_decay_dgamma_cm3_s,
    q_mu_minus_left_decay_dgamma_cm3_s,
    q_mu_minus_right_decay_dgamma_cm3_s,
):
    """Return JAX flavor-resolved neutrino injection from muon decay.

    ``neutrino_energy_erg`` is a finite positive one-dimensional blob-frame
    grid in erg in any order. ``muon_gamma`` has at least two finite, strictly
    increasing values greater than one. The four matching inputs are actual
    non-negative decay-event rates in cm^-3 s^-1 per unit muon Lorentz factor,
    ordered as ``mu+_L, mu+_R, mu-_L, mu-_R`` with ``L`` meaning ``h=-1``.

    The fixed float64 return order is ``(Q_nu_e, Q_anti_nu_e, Q_nu_mu,
    Q_anti_nu_mu)`` in cm^-3 s^-1 erg^-1. The leading-order
    ultrarelativistic Michel scaling, helicity/charge/flavor mapping, massless
    neutrino Jacobian, exact support, and no-branching-factor convention match
    the NumPy reference. Eager calls validate inputs; JIT-traced calls require
    valid fixed-shape inputs. No timescale decision, transport, cooling,
    oscillation, cascade, or frame transform is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX muon decay requires jax_enable_x64")
    arguments = (
        neutrino_energy_erg,
        muon_gamma,
        q_mu_plus_left_decay_dgamma_cm3_s,
        q_mu_plus_right_decay_dgamma_cm3_s,
        q_mu_minus_left_decay_dgamma_cm3_s,
        q_mu_minus_right_decay_dgamma_cm3_s,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _neutrino_kernel(*arguments)

    daughter, gamma, rates = _muon_decay_inputs(
        "neutrino_energy_erg",
        neutrino_energy_erg,
        daughter_lower_bound=0.0,
        muon_gamma=muon_gamma,
        parent_rates=arguments[2:],
    )
    return _checked_outputs(
        _neutrino_kernel(daughter, gamma, *rates),
        "muon-decay neutrino injection",
    )
