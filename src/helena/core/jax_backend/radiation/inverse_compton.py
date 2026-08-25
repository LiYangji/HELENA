"""JAX isotropic inverse-Compton radiation in the blob comoving frame."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
    THOMSON_CROSS_SECTION_CGS as _SIGMA_T_CGS,
)
from helena.core.contracts import (
    positive_scalar as _positive_scalar,
    real_1d as _real_1d,
)
from helena.core.jax_backend.radiation.synchrotron import _contains_tracer

__all__ = ["inverse_compton_sed_jax"]


def _inverse_compton_sed_kernel(
    scattered_photon_energy_erg,
    electron_gamma,
    dn_electron_dgamma_cm3,
    seed_photon_energy_erg,
    dn_seed_photon_denergy_cm3_erg,
    radius_cm,
):
    scattered = jnp.asarray(scattered_photon_energy_erg, dtype=jnp.float64)
    gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    lepton_density = jnp.asarray(dn_electron_dgamma_cm3, dtype=jnp.float64)
    seed = jnp.asarray(seed_photon_energy_erg, dtype=jnp.float64)
    seed_density = jnp.asarray(dn_seed_photon_denergy_cm3_erg, dtype=jnp.float64)
    radius = jnp.asarray(radius_cm, dtype=jnp.float64)

    epsilon = seed / _MEC2_CGS
    epsilon_s = scattered[:, None, None] / _MEC2_CGS
    gamma_3d = gamma[None, :, None]
    gamma_e = 4.0 * gamma_3d * epsilon[None, None, :]
    allowed_gamma = gamma_3d > epsilon_s
    denominator = gamma_e * (gamma_3d - epsilon_s)
    q = epsilon_s / jnp.where(allowed_gamma, denominator, 1.0)
    allowed = allowed_gamma & (q >= 1.0 / (4.0 * gamma_3d**2)) & (q <= 1.0)
    safe_q = jnp.where(allowed, q, 1.0)
    gamma_q = gamma_e * safe_q
    kernel = jnp.where(
        allowed,
        2.0 * safe_q * jnp.log(safe_q)
        + (1.0 + 2.0 * safe_q) * (1.0 - safe_q)
        + gamma_q**2 * (1.0 - safe_q) / (2.0 * (1.0 + gamma_q)),
        0.0,
    )
    weights = (
        lepton_density[None, :, None]
        / gamma_3d
        * (_MEC2_CGS * seed_density)[None, None, :]
    )
    seed_integral = jnp.trapezoid(
        weights * kernel,
        x=jnp.log(epsilon),
        axis=2,
    )
    rate = (
        3.0
        * _SIGMA_T_CGS
        * _C_CGS
        / 4.0
        * jnp.trapezoid(seed_integral, x=jnp.log(gamma), axis=1)
    )
    volume_cm3 = 4.0 * jnp.pi * radius**3 / 3.0
    return volume_cm3 * _MEC2_CGS * (scattered / _MEC2_CGS) ** 2 * rate


def inverse_compton_sed_jax(
    scattered_photon_energy_erg,
    electron_gamma,
    dn_electron_dgamma_cm3,
    seed_photon_energy_erg,
    dn_seed_photon_denergy_cm3_erg,
    *,
    radius_cm,
):
    """Return isotropic blob-frame inverse-Compton ``E_s L_Es`` in erg s^-1.

    Parameters and units match
    :func:`helena.core.compton.inverse_compton_sed`. Scattered and seed
    photon energies are blob-comoving values in erg. ``electron_gamma`` is a
    strictly increasing Lorentz-factor grid. Despite the canonical parameter
    name, ``dn_electron_dgamma_cm3`` is the source-independent total lepton
    density ``N_electron + N_positron`` in cm^-3 per Lorentz factor and may
    contain any primary or secondary components. The seed density is
    ``dn_ph/dE`` in cm^-3 erg^-1, and ``radius_cm`` is the positive
    homogeneous-sphere radius in cm.

    The float64 output follows the scattered-energy order. It preserves the
    NumPy reference's full angle-averaged differential Klein--Nishina kernel,
    kinematic mask, sphere volume, and trapezoidal integration first in
    ``ln(E_seed / m_e c^2)`` and then in ``ln(gamma)``. Eager calls validate
    inputs; JIT-traced calls require valid fixed-shape inputs. No cooling,
    transport, SSC feedback, anisotropic emission, observer transform,
    gamma-gamma interaction, or cascade is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX inverse-Compton radiation requires jax_enable_x64")

    arguments = (
        scattered_photon_energy_erg,
        electron_gamma,
        dn_electron_dgamma_cm3,
        seed_photon_energy_erg,
        dn_seed_photon_denergy_cm3_erg,
        radius_cm,
    )
    if _contains_tracer(arguments):
        return _inverse_compton_sed_kernel(*arguments)

    scattered = _real_1d(
        "scattered_photon_energy_erg",
        scattered_photon_energy_erg,
        minimum_size=1,
        positive=True,
    )
    gamma = _real_1d("electron_gamma", electron_gamma, minimum_size=2, positive=True)
    if np.any(gamma <= 1.0) or np.any(np.diff(gamma) <= 0.0):
        raise ValueError("electron_gamma must be strictly increasing and > 1")
    lepton_density = _real_1d(
        "dn_electron_dgamma_cm3",
        dn_electron_dgamma_cm3,
        minimum_size=2,
        positive=False,
        shape=gamma.shape,
    )
    seed = _real_1d(
        "seed_photon_energy_erg",
        seed_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(seed) <= 0.0):
        raise ValueError("seed_photon_energy_erg must be strictly increasing")
    seed_density = _real_1d(
        "dn_seed_photon_denergy_cm3_erg",
        dn_seed_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=seed.shape,
    )
    radius = _positive_scalar("radius_cm", radius_cm)
    if not np.any(lepton_density) or not np.any(seed_density):
        return jnp.zeros(scattered.shape, dtype=jnp.float64)

    result = _inverse_compton_sed_kernel(
        scattered,
        gamma,
        lepton_density,
        seed,
        seed_density,
        radius,
    )
    if _contains_tracer(result):
        return result
    values = np.asarray(result)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(
            "JAX inverse-Compton SED exceeds the finite float64 range"
        )
    return result
