"""JAX synchrotron radiation in the comoving frame."""

import jax
import jax.numpy as jnp
import numpy as np


from helena.core.constants import (
    ELECTRON_MASS_CGS as _M_E_CGS,
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    PLANCK_CONSTANT_CGS as _H_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
    SYNCHROTRON_LARGE_X as _LARGE_X,
    SYNCHROTRON_SMALL_X as _SMALL_X,
    SYNCHROTRON_SMALL_X_COEFFICIENT as _SMALL_X_COEFFICIENT,
)
from helena.core.contracts import (
    finite_scalar as _finite_scalar,
    gamma_squared as _gamma_squared,
    nonnegative_1d as _nonnegative_1d,
    positive_1d as _positive_1d,
)

__all__ = ["synchrotron_sed_jax", "electron_synchrotron_sed_jax"]

_BESSEL_INTERVALS = 160
_BESSEL_T_MAX = 14.0


def _contains_tracer(value):
    return any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )


def _scaled_modified_bessel_k_jax(order, z):
    """Return ``exp(z) K_order(z)`` by fixed-shape quadrature."""
    step_size = _BESSEL_T_MAX / _BESSEL_INTERVALS

    def integrand(t):
        return jnp.exp(-z * (jnp.cosh(t) - 1.0)) * jnp.cosh(order * t)

    initial = 0.5 * (integrand(0.0) + integrand(_BESSEL_T_MAX))

    def add_point(index, total):
        return total + integrand(index * step_size)

    integral = jax.lax.fori_loop(1, _BESSEL_INTERVALS, add_point, initial)
    return step_size * integral


def _isotropic_synchrotron_function_jax(x):
    """Return the NumPy reference's isotropic pitch-angle function ``R(x)``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    small = x <= _SMALL_X
    small_x = jnp.where(small, x, 0.0)
    small_result = (
        _SMALL_X_COEFFICIENT * jnp.cbrt(small_x) - jnp.pi / jnp.sqrt(3.0) * small_x
    )

    middle = (x > _SMALL_X) & (x < _LARGE_X)
    middle_x = jnp.where(middle, x, 1.0)
    z = middle_x / 2.0
    k_4_3 = _scaled_modified_bessel_k_jax(4.0 / 3.0, z)
    k_1_3 = _scaled_modified_bessel_k_jax(1.0 / 3.0, z)
    difference_of_squares = (k_4_3 - k_1_3) * (k_4_3 + k_1_3)
    scaled = k_4_3 * k_1_3 - 3.0 * middle_x / 10.0 * difference_of_squares
    middle_result = middle_x**2 / 2.0 * jnp.exp(-middle_x) * scaled

    large_x = jnp.where(x >= _LARGE_X, x, _LARGE_X)
    large_result = jnp.exp(
        jnp.log(jnp.pi / 2.0) + jnp.log1p(-11.0 / (18.0 * large_x)) - large_x
    )
    return jnp.where(
        small, small_result, jnp.where(middle, middle_result, large_result)
    )


def _synchrotron_sed_kernel(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    magnetic_field_G,
    radius_cm,
    particle_mass_g,
    charge_esu,
):
    photon_energy = jnp.asarray(photon_energy_erg, dtype=jnp.float64)
    particle_gamma = jnp.asarray(gamma, dtype=jnp.float64)
    particle_density = jnp.asarray(dn_dgamma_cm3, dtype=jnp.float64)
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)
    radius = jnp.asarray(radius_cm, dtype=jnp.float64)
    particle_mass = jnp.asarray(particle_mass_g, dtype=jnp.float64)
    charge = jnp.asarray(charge_esu, dtype=jnp.float64)

    frequency_hz = photon_energy[:, None] / _H_CGS
    critical_frequency_hz = (
        3.0
        * charge
        * magnetic_field
        * particle_gamma[None, :] ** 2
        / (4.0 * jnp.pi * particle_mass * _C_CGS)
    )
    x = frequency_hz / critical_frequency_hz
    prefactor = (
        jnp.sqrt(3.0)
        * charge**3
        * magnetic_field
        / (particle_mass * _C_CGS**2 * _H_CGS)
    )
    single_particle_power = prefactor * _isotropic_synchrotron_function_jax(x)
    emissivity = jnp.trapezoid(
        single_particle_power * (particle_gamma * particle_density)[None, :],
        x=jnp.log(particle_gamma),
        axis=1,
    )
    volume_cm3 = 4.0 * jnp.pi * radius**3 / 3.0
    return photon_energy * emissivity * volume_cm3


def synchrotron_sed_jax(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
    particle_mass_g,
    charge_esu,
):
    """Return blob-frame synchrotron ``E L_E`` in erg s^-1.

    Parameters match :func:`helena.core.synchrotron.synchrotron_sed`.
    ``photon_energy_erg`` is the one-dimensional blob-comoving photon-energy
    grid in erg. ``gamma`` is a strictly increasing particle Lorentz-factor
    grid with at least two values greater than one. ``dn_dgamma_cm3`` is the
    source-independent particle density in cm^-3 per Lorentz factor.
    ``magnetic_field_G`` is the non-negative blob-comoving field in Gauss,
    ``radius_cm`` is the positive homogeneous-sphere radius in cm,
    ``particle_mass_g`` is the positive particle mass in grams, and
    ``charge_esu`` is the positive charge magnitude in esu/statC.

    The float64 output follows the photon grid and uses the NumPy reference's
    classical optically thin, exact isotropic pitch-angle kernel and
    deterministic trapezoid in ``ln(gamma)``. Eager calls validate inputs;
    JIT-traced calls require valid fixed-shape inputs. No cooling, SSA,
    transport, observer transform, inverse-Compton radiation, or cascade is
    included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX synchrotron radiation requires jax_enable_x64")

    arguments = (
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G,
        radius_cm,
        particle_mass_g,
        charge_esu,
    )
    if _contains_tracer(arguments):
        return _synchrotron_sed_kernel(*arguments)

    photon_energy = _positive_1d(
        "photon_energy_erg", photon_energy_erg, lower_bound=0.0
    )
    particle_gamma = _positive_1d("gamma", gamma, lower_bound=1.0)
    if particle_gamma.size < 2:
        raise ValueError("gamma must contain at least two points")
    if np.any(np.diff(particle_gamma) <= 0.0):
        raise ValueError("gamma must be strictly increasing")
    particle_density = _nonnegative_1d(
        "dn_dgamma_cm3", dn_dgamma_cm3, shape=particle_gamma.shape
    )
    magnetic_field = _finite_scalar(
        "magnetic_field_G", magnetic_field_G, allow_zero=True
    )
    radius = _finite_scalar("radius_cm", radius_cm, allow_zero=False)
    particle_mass = _finite_scalar("particle_mass_g", particle_mass_g, allow_zero=False)
    charge = _finite_scalar("charge_esu", charge_esu, allow_zero=False)
    _gamma_squared(particle_gamma)
    with np.errstate(over="ignore", invalid="ignore"):
        radius_cubed = np.float64(radius) ** 3
    if not np.isfinite(radius_cubed):
        raise FloatingPointError(
            "radius_cm**3 exceeds the finite float64 numerical range"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        volume_cm3 = (4.0 * np.pi / 3.0) * radius_cubed
    if not np.isfinite(volume_cm3):
        raise FloatingPointError("sphere volume exceeds the finite float64 range")
    if magnetic_field == 0.0 or not np.any(particle_density):
        return jnp.zeros(photon_energy.shape, dtype=jnp.float64)

    result = _synchrotron_sed_kernel(
        photon_energy,
        particle_gamma,
        particle_density,
        magnetic_field,
        radius,
        particle_mass,
        charge,
    )
    if _contains_tracer(result):
        return result
    values = np.asarray(result)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError("JAX synchrotron SED exceeds the finite float64 range")
    return result


def electron_synchrotron_sed_jax(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return blob-frame electron/positron synchrotron ``E L_E`` in erg s^-1.

    Inputs, output, assumptions, and JAX conventions match
    :func:`synchrotron_sed_jax`; this wrapper fixes ``m=m_e`` and ``|q|=e``.
    The supplied density may be the sum of any electron and positron sources.
    """
    return synchrotron_sed_jax(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_E_CGS,
        charge_esu=_E_ESU,
    )
