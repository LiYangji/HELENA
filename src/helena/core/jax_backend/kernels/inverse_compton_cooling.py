"""JAX isotropic inverse-Compton continuous cooling."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    INVERSE_COMPTON_SMALL_X_THRESHOLD as _SMALL_X_THRESHOLD,
    SPEED_OF_LIGHT_CGS as _C_CGS,
    THOMSON_CROSS_SECTION_CGS as _SIGMA_T_CGS,
)
from helena.core.contracts import real_1d as _real_1d
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    _checked_result,
    _contains_tracer,
)

__all__ = ["electron_inverse_compton_loss_rate_dgamma_s_jax"]

_DILOGARITHM_TERMS = 40


def _dilogarithm_unit_interval_series_jax(z):
    z = jnp.asarray(z, dtype=jnp.float64)

    def step(index, value):
        order = jnp.asarray(_DILOGARITHM_TERMS - index, dtype=jnp.float64)
        return z * (1.0 / order**2 + value)

    return jax.lax.fori_loop(
        0,
        _DILOGARITHM_TERMS,
        step,
        jnp.zeros_like(z),
    )


def _dilogarithm_negative_jax(x):
    """Return ``Li_2(-x)`` for finite ``x >= 0`` using a fixed series."""
    x = jnp.asarray(x, dtype=jnp.float64)
    large = x > 1.0
    large_x = jnp.where(large, x, 1.0)
    z = jnp.where(large, 1.0 / (1.0 + large_x), x / (1.0 + x))
    series = _dilogarithm_unit_interval_series_jax(z)
    log_y = -jnp.log1p(1.0 / large_x)
    log_one_minus_y = -jnp.log1p(large_x)
    li2_mapped = jnp.where(
        large,
        jnp.pi**2 / 6.0 - log_y * log_one_minus_y - series,
        series,
    )
    log1p_x = jnp.log1p(x)
    return -li2_mapped - 0.5 * log1p_x**2


def _klein_nishina_loss_suppression_jax(x):
    """Return the NumPy reference's isotropic KN loss suppression."""
    x = jnp.asarray(x, dtype=jnp.float64)
    small = x <= _SMALL_X_THRESHOLD
    small_result = 1.0 - 63.0 * x / 40.0 + 441.0 * x**2 / 200.0 - 101.0 * x**3 / 35.0
    full_x = jnp.where(small, 1.0, x)
    g = (
        (full_x / 2.0 + 6.0 + 6.0 / full_x) * jnp.log1p(full_x)
        - (11.0 * full_x**3 / 12.0 + 6.0 * full_x**2 + 9.0 * full_x + 4.0)
        / (1.0 + full_x) ** 2
        - 2.0
        + 2.0 * _dilogarithm_negative_jax(full_x)
    )
    full_result = 9.0 * g / full_x**3
    return jnp.where(small, small_result, full_result)


def _inverse_compton_loss_kernel(
    electron_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    photon_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    photon_density = jnp.asarray(dn_target_photon_denergy_cm3_erg, dtype=jnp.float64)
    x = 4.0 * gamma[:, None] * photon_energy[None, :] / _MEC2_CGS
    suppression = _klein_nishina_loss_suppression_jax(x)
    energy_density = jnp.trapezoid(
        photon_energy[None, :] * photon_density[None, :] * suppression,
        x=photon_energy,
        axis=1,
    )
    return 4.0 * _SIGMA_T_CGS * _C_CGS / (3.0 * _MEC2_CGS) * gamma**2 * energy_density


def electron_inverse_compton_loss_rate_dgamma_s_jax(
    electron_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return JAX isotropic electron inverse-Compton cooling ``-dγ/dt``.

    ``electron_gamma`` is a non-empty one-dimensional array of finite
    dimensionless Lorentz factors greater than one. Target photon energies are
    at least two finite, positive, strictly increasing blob-comoving values in
    erg, with matching finite non-negative isotropic ``dn_ph/dE`` in
    cm^-3 erg^-1. The float64 output follows the electron order and is the
    continuous loss ``b_IC`` in s^-1.

    The implementation matches the NumPy reference's full analytic isotropic
    Klein--Nishina suppression, small-argument branch, constants, and direct
    photon-energy trapezoid. Eager calls validate values; JIT-traced calls
    require valid fixed-shape inputs. It performs no interpolation, transport,
    radiation calculation, feedback, or cascade.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX inverse-Compton cooling requires jax_enable_x64")

    arguments = (
        electron_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if _contains_tracer(arguments):
        return _inverse_compton_loss_kernel(*arguments)

    gamma = _real_1d("electron_gamma", electron_gamma, minimum_size=1, positive=True)
    if np.any(gamma <= 1.0):
        raise ValueError("electron_gamma must contain only values > 1")
    photon_energy = _real_1d(
        "target_photon_energy_erg",
        target_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(photon_energy) <= 0.0):
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    photon_density = _real_1d(
        "dn_target_photon_denergy_cm3_erg",
        dn_target_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=photon_energy.shape,
    )
    if not np.any(photon_density):
        return jnp.zeros(gamma.shape, dtype=jnp.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        maximum_x = 4.0 * np.max(gamma) * np.max(photon_energy) / _MEC2_CGS
        maximum_gamma_squared = np.max(gamma) ** 2
    if not np.isfinite(maximum_x) or not np.isfinite(maximum_gamma_squared):
        raise FloatingPointError("inverse-Compton inputs exceed float64 range")

    result = _inverse_compton_loss_kernel(gamma, photon_energy, photon_density)
    return _checked_result(result, "JAX inverse-Compton cooling")
