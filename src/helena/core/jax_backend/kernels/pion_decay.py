"""JAX daughter injection from neutral- and charged-pion decay."""

import jax
import jax.numpy as jnp
import numpy as np


from helena.core.constants import (
    CHARGED_PION_MUONIC_BRANCHING_RATIO as _MUONIC_BRANCHING_RATIO,
    CHARGED_PION_REST_ENERGY_CGS as _CHARGED_PION_REST_ENERGY_CGS,
    MUON_TO_PION_MASS_RATIO as _MUON_TO_PION_MASS_RATIO,
    MUON_TO_PION_MASS_RATIO_SQUARED as _R_PI,
    NEUTRAL_PION_REST_ENERGY_CGS as _PI0_REST_ENERGY_CGS,
    NEUTRAL_PION_TWO_PHOTON_BRANCHING_RATIO as _TWO_PHOTON_BRANCHING_RATIO,
)
from helena.core.contracts import (
    charged_pion_decay_inputs as _charged_pion_decay_inputs,
    real_1d as _real_1d,
)

__all__ = [
    "neutral_pion_decay_photon_injection_jax",
    "charged_pion_decay_muon_injection_jax",
    "charged_pion_decay_neutrino_injection_jax",
]


def _trapezoid_last_axis(values, grid):
    intervals = jnp.diff(grid) * (values[..., 1:] + values[..., :-1]) / 2.0
    return jnp.sum(intervals, axis=-1)


def _neutral_pion_decay_photon_kernel(
    photon_energy_erg,
    pion_gamma,
    q_pi0_dgamma_cm3_s,
):
    photon_energy = jnp.asarray(photon_energy_erg, dtype=jnp.float64)
    gamma = jnp.asarray(pion_gamma, dtype=jnp.float64)
    injection = jnp.asarray(q_pi0_dgamma_cm3_s, dtype=jnp.float64)

    inverse_gamma = 1.0 / gamma
    denominator = gamma * jnp.sqrt((1.0 - inverse_gamma) * (1.0 + inverse_gamma))
    integrand = injection / denominator
    intervals = (integrand[:-1] + integrand[1:]) * jnp.diff(gamma) / 2.0
    tail = jnp.concatenate(
        (jnp.cumsum(intervals[::-1])[::-1], jnp.zeros(1, dtype=jnp.float64))
    )

    x_gamma = photon_energy / _PI0_REST_ENERGY_CGS
    safe = jnp.isfinite(x_gamma) & (x_gamma >= 0.25 / jnp.finfo(jnp.float64).max)
    x_safe = jnp.where(safe, x_gamma, 1.0)
    gamma_minimum = jnp.where(safe, x_safe + 0.25 / x_safe, jnp.inf)
    return (
        2.0
        * _TWO_PHOTON_BRANCHING_RATIO
        / _PI0_REST_ENERGY_CGS
        * jnp.interp(gamma_minimum, gamma, tail, left=tail[0], right=0.0)
    )


def _direct_neutrino_scaling(x):
    support = (x > 0.0) & (x < 1.0 - _R_PI)
    return jnp.where(support, 1.0 / (1.0 - _R_PI), 0.0)


def _positive_pion_muon_scaling(x):
    support = (x > _R_PI) & (x < 1.0)
    x_safe = jnp.where(support, x, 1.0)
    denominator = (1.0 - _R_PI) ** 2 * x_safe
    left = jnp.where(support, (x_safe - _R_PI) / denominator, 0.0)
    right = jnp.where(support, _R_PI * (1.0 - x_safe) / denominator, 0.0)
    return left, right


def _scaling_injection_integral(pion_gamma, decay_rate, scaling):
    integrand = scaling * (decay_rate / pion_gamma)[None, :]
    return _trapezoid_last_axis(integrand, pion_gamma)


def _charged_pion_decay_muon_kernel(
    muon_gamma,
    pion_gamma,
    q_pi_plus_decay_dgamma_cm3_s,
    q_pi_minus_decay_dgamma_cm3_s,
):
    daughter_gamma = jnp.asarray(muon_gamma, dtype=jnp.float64)
    parent_gamma = jnp.asarray(pion_gamma, dtype=jnp.float64)
    plus_rate = jnp.asarray(q_pi_plus_decay_dgamma_cm3_s, dtype=jnp.float64)
    minus_rate = jnp.asarray(q_pi_minus_decay_dgamma_cm3_s, dtype=jnp.float64)
    x = daughter_gamma[:, None] * _MUON_TO_PION_MASS_RATIO / parent_gamma[None, :]
    plus_left_scaling, plus_right_scaling = _positive_pion_muon_scaling(x)
    coefficient = _MUONIC_BRANCHING_RATIO * _MUON_TO_PION_MASS_RATIO
    return (
        coefficient
        * _scaling_injection_integral(parent_gamma, plus_rate, plus_left_scaling),
        coefficient
        * _scaling_injection_integral(parent_gamma, plus_rate, plus_right_scaling),
        coefficient
        * _scaling_injection_integral(parent_gamma, minus_rate, plus_right_scaling),
        coefficient
        * _scaling_injection_integral(parent_gamma, minus_rate, plus_left_scaling),
    )


def _charged_pion_decay_neutrino_kernel(
    neutrino_energy_erg,
    pion_gamma,
    q_pi_plus_decay_dgamma_cm3_s,
    q_pi_minus_decay_dgamma_cm3_s,
):
    neutrino_energy = jnp.asarray(neutrino_energy_erg, dtype=jnp.float64)
    parent_gamma = jnp.asarray(pion_gamma, dtype=jnp.float64)
    plus_rate = jnp.asarray(q_pi_plus_decay_dgamma_cm3_s, dtype=jnp.float64)
    minus_rate = jnp.asarray(q_pi_minus_decay_dgamma_cm3_s, dtype=jnp.float64)
    x = neutrino_energy[:, None] / (
        parent_gamma[None, :] * _CHARGED_PION_REST_ENERGY_CGS
    )
    scaling = _direct_neutrino_scaling(x)
    coefficient = _MUONIC_BRANCHING_RATIO / _CHARGED_PION_REST_ENERGY_CGS
    return (
        coefficient * _scaling_injection_integral(parent_gamma, plus_rate, scaling),
        coefficient * _scaling_injection_integral(parent_gamma, minus_rate, scaling),
    )


def _checked_outputs(outputs, diagnostic):
    values = np.asarray(outputs)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(f"{diagnostic} exceeds the finite float64 range")
    return outputs


def neutral_pion_decay_photon_injection_jax(
    photon_energy_erg,
    pion_gamma,
    q_pi0_dgamma_cm3_s,
):
    """Return JAX photon injection from blob-frame ``pi0 -> gamma gamma``.

    ``photon_energy_erg`` is a finite positive one-dimensional blob-comoving
    grid in erg in any order. ``pion_gamma`` contains at least two finite,
    strictly increasing values greater than one, and matching
    ``q_pi0_dgamma_cm3_s`` is a finite non-negative decay-event rate in
    cm^-3 s^-1 per unit pion Lorentz factor.

    The float64 result follows the photon grid and is
    ``dn_gamma/(dE_gamma dt)`` in cm^-3 s^-1 erg^-1. It uses the NumPy
    reference's exact finite-beta two-body kinematics, cumulative trapezoidal
    tail, and ``pi0 -> gamma gamma`` branching ratio. Eager calls validate
    inputs; JIT-traced calls require valid fixed-shape inputs. No pion
    transport, absorption, cascade, frame transform, or observer physics is
    included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX pion decay requires jax_enable_x64")
    arguments = (photon_energy_erg, pion_gamma, q_pi0_dgamma_cm3_s)
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _neutral_pion_decay_photon_kernel(*arguments)

    photon_energy = _real_1d(
        "photon_energy_erg", photon_energy_erg, minimum_size=1, positive=True
    )
    gamma = _real_1d("pion_gamma", pion_gamma, minimum_size=2, positive=True)
    if np.any(gamma <= 1.0):
        raise ValueError("pion_gamma must contain only values > 1")
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("pion_gamma must be strictly increasing")
    injection = _real_1d(
        "q_pi0_dgamma_cm3_s",
        q_pi0_dgamma_cm3_s,
        minimum_size=2,
        positive=False,
        shape=gamma.shape,
    )
    if not np.any(injection):
        return jnp.zeros(photon_energy.shape, dtype=jnp.float64)

    result = _neutral_pion_decay_photon_kernel(photon_energy, gamma, injection)
    return _checked_outputs(result, "neutral-pion decay photon injection")


def charged_pion_decay_muon_injection_jax(
    muon_gamma,
    pion_gamma,
    q_pi_plus_decay_dgamma_cm3_s,
    q_pi_minus_decay_dgamma_cm3_s,
):
    """Return JAX helicity-resolved muons from charged-pion decay.

    ``muon_gamma`` is a finite one-dimensional blob-frame grid greater than
    one in any order. ``pion_gamma`` has at least two finite, strictly
    increasing values greater than one. The matching ``pi+`` and ``pi-``
    inputs are actual finite non-negative decay-event rates in cm^-3 s^-1 per
    unit pion Lorentz factor, not pion densities.

    The fixed float64 return order is ``(Q_mu+_L, Q_mu+_R, Q_mu-_L,
    Q_mu-_R)`` in cm^-3 s^-1 per unit muon Lorentz factor, with ``L`` having
    ``h=-1`` and ``R`` having ``h=+1``. The Hümmer/Lipari ultrarelativistic
    scaling, CP mapping, mass ratio, support, and muonic branching convention
    exactly match the NumPy reference and are intended for
    ``pion_gamma >> 1``. Eager calls validate inputs; traced calls require
    valid fixed-shape inputs. No timescale decision, pion/muon transport, or
    muon decay is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX pion decay requires jax_enable_x64")
    arguments = (
        muon_gamma,
        pion_gamma,
        q_pi_plus_decay_dgamma_cm3_s,
        q_pi_minus_decay_dgamma_cm3_s,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _charged_pion_decay_muon_kernel(*arguments)

    validated = _charged_pion_decay_inputs(
        "muon_gamma",
        muon_gamma,
        daughter_lower_bound=1.0,
        pion_gamma=pion_gamma,
        q_pi_plus_decay_dgamma_cm3_s=q_pi_plus_decay_dgamma_cm3_s,
        q_pi_minus_decay_dgamma_cm3_s=q_pi_minus_decay_dgamma_cm3_s,
    )
    outputs = _charged_pion_decay_muon_kernel(*validated)
    return _checked_outputs(outputs, "charged-pion muon injection")


def charged_pion_decay_neutrino_injection_jax(
    neutrino_energy_erg,
    pion_gamma,
    q_pi_plus_decay_dgamma_cm3_s,
    q_pi_minus_decay_dgamma_cm3_s,
):
    """Return JAX direct muon-neutrinos from charged-pion decay.

    ``neutrino_energy_erg`` is a finite positive one-dimensional blob-frame
    grid in erg in any order. ``pion_gamma`` has at least two finite, strictly
    increasing values greater than one. The matching ``pi+`` and ``pi-``
    inputs are actual finite non-negative decay-event rates in cm^-3 s^-1 per
    unit pion Lorentz factor, not pion densities.

    The fixed float64 return order is ``(Q_nu_mu, Q_anti_nu_mu)`` in
    cm^-3 s^-1 erg^-1. The Hümmer/Lipari ultrarelativistic direct-neutrino
    scaling, support, and once-applied muonic branching ratio exactly match
    the NumPy reference and are intended for ``pion_gamma >> 1``. Eager calls
    validate inputs; traced calls require valid fixed-shape inputs. No muon
    decay, oscillation, pion/muon transport, or timescale choice is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX pion decay requires jax_enable_x64")
    arguments = (
        neutrino_energy_erg,
        pion_gamma,
        q_pi_plus_decay_dgamma_cm3_s,
        q_pi_minus_decay_dgamma_cm3_s,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _charged_pion_decay_neutrino_kernel(*arguments)

    validated = _charged_pion_decay_inputs(
        "neutrino_energy_erg",
        neutrino_energy_erg,
        daughter_lower_bound=0.0,
        pion_gamma=pion_gamma,
        q_pi_plus_decay_dgamma_cm3_s=q_pi_plus_decay_dgamma_cm3_s,
        q_pi_minus_decay_dgamma_cm3_s=q_pi_minus_decay_dgamma_cm3_s,
    )
    outputs = _charged_pion_decay_neutrino_kernel(*validated)
    return _checked_outputs(outputs, "charged-pion neutrino injection")
