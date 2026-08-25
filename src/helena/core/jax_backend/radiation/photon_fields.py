"""JAX internal photon fields in a homogeneous comoving sphere."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import SPEED_OF_LIGHT_CGS as _C_CGS
from helena.core.contracts import nonempty_real_1d as _real_1d
from helena.core.jax_backend.radiation.ssa import (
    _contains_tracer,
    _sphere_attenuation_factor_jax,
)

__all__ = [
    "produced_sed_to_internal_photon_density_jax",
    "synchrotron_sed_to_internal_photon_density_jax",
]


def _homogeneous_sphere_effective_path_length_jax(
    absorption_coefficient_cm1,
    radius_cm,
):
    alpha = jnp.asarray(absorption_coefficient_cm1, dtype=jnp.float64)
    radius = jnp.asarray(radius_cm, dtype=jnp.float64)
    tau = 2.0 * alpha * radius
    small = tau <= 0.1
    small_tau = jnp.where(small, tau, 0.0)
    small_ratio = (
        2.0
        * radius
        * (
            3.0 / 8.0
            + small_tau
            * (
                -1.0 / 10.0
                + small_tau
                * (
                    1.0 / 48.0
                    + small_tau
                    * (-1.0 / 280.0 + small_tau * (1.0 / 1920.0 - small_tau / 15120.0))
                )
            )
        )
    )
    safe_alpha = jnp.where(small, 1.0, alpha)
    regular_ratio = (1.0 - _sphere_attenuation_factor_jax(tau)) / safe_alpha
    return jnp.where(small, small_ratio, regular_ratio)


def _internal_photon_density_kernel(
    photon_energy_erg,
    produced_sed_erg_s,
    absorption_coefficient_cm1,
    radius_cm,
):
    energy = jnp.asarray(photon_energy_erg, dtype=jnp.float64)
    produced_sed = jnp.asarray(produced_sed_erg_s, dtype=jnp.float64)
    radius = jnp.asarray(radius_cm, dtype=jnp.float64)
    ratio = _homogeneous_sphere_effective_path_length_jax(
        absorption_coefficient_cm1,
        radius,
    )

    volume = 4.0 * jnp.pi * radius**3 / 3.0
    return produced_sed * ratio / (volume * _C_CGS * energy**2)


def produced_sed_to_internal_photon_density_jax(
    photon_energy_erg,
    produced_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return a JAX homogeneous-sphere internal ``dn_ph/dE``.

    Inputs and assumptions match
    :func:`helena.core.photon_fields.produced_sed_to_internal_photon_density`.
    Photon energy is blob-comoving in erg, ``produced_sed_erg_s`` is intrinsic
    comoving ``E L_E`` in erg s^-1, ``absorption_coefficient_cm1`` is finite
    non-negative local absorption in cm^-1, and ``radius_cm`` is the positive
    homogeneous-sphere radius. The return is float64 in cm^-3 erg^-1. The
    canonical Dermer volume-and-direction average is used; this is not an
    emergent-SED escape-time approximation.

    Eager calls validate inputs. JIT-traced calls require valid fixed-shape
    inputs and JAX float64 mode. No source-specific radiation, observer
    transform, absorption-coefficient calculation, feedback, or cascade is
    included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX internal photon fields require jax_enable_x64")

    arguments = (
        photon_energy_erg,
        produced_sed_erg_s,
        absorption_coefficient_cm1,
        radius_cm,
    )
    if _contains_tracer(arguments):
        return _internal_photon_density_kernel(*arguments)

    energy = _real_1d("photon_energy_erg", photon_energy_erg, positive=True)
    produced_sed = _real_1d(
        "produced_sed_erg_s",
        produced_sed_erg_s,
        positive=False,
        shape=energy.shape,
    )
    alpha = _real_1d(
        "absorption_coefficient_cm1",
        absorption_coefficient_cm1,
        positive=False,
        shape=energy.shape,
    )
    radius_array = np.asarray(radius_cm)
    if radius_array.ndim != 0 or radius_array.dtype.kind not in "iuf":
        raise ValueError("radius_cm must be a finite positive real scalar")
    radius = float(radius_array)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_cm must be a finite positive real scalar")

    if not np.any(produced_sed):
        return jnp.zeros(energy.shape, dtype=jnp.float64)

    result = _internal_photon_density_kernel(energy, produced_sed, alpha, radius)
    if _contains_tracer(result):
        return result
    result_values = np.asarray(result)
    if not np.all(np.isfinite(result_values)) or np.any(result_values < 0.0):
        raise FloatingPointError("internal photon density exceeds finite float64 range")
    return result


def synchrotron_sed_to_internal_photon_density_jax(
    photon_energy_erg,
    thin_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return the JAX SSA-aware internal ``dn_syn/dE`` in cm^-3 erg^-1.

    Inputs, units, and homogeneous-sphere behavior match
    :func:`produced_sed_to_internal_photon_density_jax`; this compatibility
    wrapper keeps the existing synchrotron API and physical output unchanged.
    """
    return produced_sed_to_internal_photon_density_jax(
        photon_energy_erg,
        thin_sed_erg_s,
        absorption_coefficient_cm1,
        radius_cm=radius_cm,
    )
