"""JAX synchrotron self-absorption in a homogeneous comoving sphere."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_MASS_CGS as _M_E_CGS,
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    PLANCK_CONSTANT_CGS as _H_CGS,
    PROTON_MASS_CGS as _M_P_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
    SSA_NEGATIVE_ALPHA_ROUNDOFF_FACTOR as _NEGATIVE_ALPHA_ROUNDOFF_FACTOR,
)
from helena.core.contracts import (
    absorption_inputs as _absorption_inputs,
    finite_scalar as _finite_scalar,
)
from helena.core.jax_backend.radiation.synchrotron import (
    _contains_tracer,
    _isotropic_synchrotron_function_jax,
)

__all__ = [
    "synchrotron_absorption_coefficient_jax",
    "synchrotron_self_absorbed_sed_jax",
    "electron_synchrotron_self_absorbed_sed_jax",
    "proton_synchrotron_self_absorbed_sed_jax",
]


def _gradient_interior(values, coordinates):
    """Return the centered second-order derivative at interior grid points."""
    lower = coordinates[1:-1] - coordinates[:-2]
    upper = coordinates[2:] - coordinates[1:-1]
    lower_slope = (values[1:-1] - values[:-2]) / lower
    upper_slope = (values[2:] - values[1:-1]) / upper
    return (upper * lower_slope + lower * upper_slope) / (lower + upper)


def _synchrotron_power_per_energy_kernel(
    photon_energy_erg,
    gamma,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
):
    photon_energy = jnp.asarray(photon_energy_erg, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)
    particle_mass = jnp.asarray(particle_mass_g, dtype=jnp.float64)
    charge = jnp.asarray(charge_esu, dtype=jnp.float64)

    evaluation_field = jnp.where(magnetic_field > 0.0, magnetic_field, 1.0)
    frequency_hz = photon_energy[:, None] / _H_CGS
    critical_frequency_hz = (
        3.0
        * charge
        * evaluation_field
        * gamma[None, :] ** 2
        / (4.0 * jnp.pi * particle_mass * _C_CGS)
    )
    x = frequency_hz / critical_frequency_hz
    prefactor = (
        jnp.sqrt(3.0)
        * charge**3
        * evaluation_field
        / (particle_mass * _C_CGS**2 * _H_CGS)
    )
    power = prefactor * _isotropic_synchrotron_function_jax(x)
    return jnp.where(magnetic_field > 0.0, power, 0.0)


def _coefficient_from_power(power_per_energy, photon_energy, gamma, density, mass):
    log_gamma = jnp.log(gamma)
    derivative = _gradient_interior(density, log_gamma)
    absorption_weight = 2.0 * density[1:-1] - derivative
    integrand = _H_CGS * power_per_energy[:, 1:-1] * absorption_weight[None, :]
    integral = jnp.trapezoid(integrand, x=log_gamma[1:-1], axis=1)
    absolute_integral = jnp.trapezoid(jnp.abs(integrand), x=log_gamma[1:-1], axis=1)
    denominator = 8.0 * jnp.pi * mass * (photon_energy / _H_CGS) ** 2
    return integral / denominator, absolute_integral / denominator


def _absorption_coefficient_and_scale_kernel(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
):
    photon_energy = jnp.asarray(photon_energy_erg, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    density = jnp.asarray(dn_dgamma_cm3, dtype=jnp.float64)
    particle_mass = jnp.asarray(particle_mass_g, dtype=jnp.float64)
    power = _synchrotron_power_per_energy_kernel(
        photon_energy,
        gamma,
        magnetic_field_G,
        particle_mass,
        charge_esu,
    )
    return _coefficient_from_power(
        power,
        photon_energy,
        gamma,
        density,
        particle_mass,
    )


def _sphere_attenuation_factor_jax(tau_diameter):
    tau_diameter = jnp.asarray(tau_diameter, dtype=jnp.float64)
    small = (tau_diameter > 0.0) & (tau_diameter <= 0.1)
    small_tau = jnp.where(small, tau_diameter, 0.0)
    small_result = 1.0 + small_tau * (
        -3.0 / 8.0
        + small_tau
        * (
            1.0 / 10.0
            + small_tau
            * (
                -1.0 / 48.0
                + small_tau
                * (1.0 / 280.0 + small_tau * (-1.0 / 1920.0 + small_tau / 15120.0))
            )
        )
    )

    regular = tau_diameter > 0.1
    regular_tau = jnp.where(regular, tau_diameter, 1.0)
    inverse_tau = 1.0 / regular_tau
    regular_result = (
        3.0
        * inverse_tau
        * (0.5 - (1.0 - (1.0 + regular_tau) * jnp.exp(-regular_tau)) * inverse_tau**2)
    )
    return jnp.where(
        tau_diameter == 0.0,
        1.0,
        jnp.where(small, small_result, regular_result),
    )


def _ssa_quantities_kernel(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    magnetic_field_G,
    radius_cm,
    particle_mass_g,
    charge_esu,
):
    photon_energy = jnp.asarray(photon_energy_erg, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    density = jnp.asarray(dn_dgamma_cm3, dtype=jnp.float64)
    radius = jnp.asarray(radius_cm, dtype=jnp.float64)
    particle_mass = jnp.asarray(particle_mass_g, dtype=jnp.float64)
    power = _synchrotron_power_per_energy_kernel(
        photon_energy,
        gamma,
        magnetic_field_G,
        particle_mass,
        charge_esu,
    )
    coefficient, coefficient_scale = _coefficient_from_power(
        power,
        photon_energy,
        gamma,
        density,
        particle_mass,
    )
    emissivity = jnp.trapezoid(
        power * (gamma * density)[None, :],
        x=jnp.log(gamma),
        axis=1,
    )
    thin_sed = photon_energy * emissivity * (4.0 * jnp.pi * radius**3 / 3.0)
    return coefficient, coefficient_scale, thin_sed


def _transfer_from_quantities(coefficient, thin_sed, radius_cm):
    transfer_coefficient = jnp.where(coefficient < 0.0, 0.0, coefficient)
    tau_diameter = (
        2.0 * jnp.asarray(radius_cm, dtype=jnp.float64) * transfer_coefficient
    )
    return thin_sed * _sphere_attenuation_factor_jax(tau_diameter)


def _ssa_kernel(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    magnetic_field_G,
    radius_cm,
    particle_mass_g,
    charge_esu,
):
    coefficient, _, thin_sed = _ssa_quantities_kernel(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G,
        radius_cm,
        particle_mass_g,
        charge_esu,
    )
    return _transfer_from_quantities(coefficient, thin_sed, radius_cm)


def synchrotron_absorption_coefficient_jax(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    particle_mass_g,
    charge_esu,
):
    """Return signed JAX comoving synchrotron ``alpha_nu`` in cm^-1.

    Parameters, units, and physical assumptions match
    :func:`helena.core.ssa.synchrotron_absorption_coefficient`. The generic
    interface accepts any positive particle mass in grams and charge magnitude
    in esu/statC. It uses the canonical isotropic synchrotron kernel, the same
    second-order derivative of ``dn/dgamma`` on ``ln(gamma)``, and the same
    interior caller-grid trapezoid. Electron-grid endpoints do not participate
    in the derivative integral. Population inversion remains visible as
    negative output. Eager calls validate values; JIT-traced calls require
    valid fixed-shape inputs and JAX float64 mode.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX synchrotron self-absorption requires jax_enable_x64")

    arguments = (
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G,
        particle_mass_g,
        charge_esu,
    )
    if _contains_tracer(arguments):
        return _absorption_coefficient_and_scale_kernel(*arguments)[0]

    validated = _absorption_inputs(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=particle_mass_g,
        charge_esu=charge_esu,
    )
    if validated[3] == 0.0 or not np.any(validated[2]):
        return jnp.zeros(validated[0].shape, dtype=jnp.float64)

    coefficient, coefficient_scale = _absorption_coefficient_and_scale_kernel(
        *validated
    )
    if _contains_tracer((coefficient, coefficient_scale)):
        return coefficient
    coefficient_values = np.asarray(coefficient)
    scale_values = np.asarray(coefficient_scale)
    if (
        not np.all(np.isfinite(coefficient_values))
        or not np.all(np.isfinite(scale_values))
        or np.any(scale_values < 0.0)
    ):
        raise FloatingPointError(
            "synchrotron absorption coefficient exceeds the finite float64 range"
        )
    return coefficient


def synchrotron_self_absorbed_sed_jax(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
    particle_mass_g,
    charge_esu,
):
    """Return JAX homogeneous-sphere SSA ``E L_E`` in erg s^-1.

    Inputs match :func:`helena.core.ssa.synchrotron_self_absorbed_sed` and
    are blob-comoving: photon energy is in erg, ``dn/dgamma`` in cm^-3,
    magnetic field in Gauss, radius in cm, mass in grams, and charge magnitude
    in esu/statC. The transfer uses the canonical Dermer sphere factor at
    diameter depth ``tau_d = 2 R alpha_nu``. Eager calls reject clear
    population inversion and treat only canonical roundoff-sized negative
    absorption as zero; traced calls require valid non-inverted fixed-shape
    inputs. No transport, internal photon field, SSC, gamma-gamma interaction,
    observer transform, or cascade is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX synchrotron self-absorption requires jax_enable_x64")

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
        return _ssa_kernel(*arguments)

    radius = _finite_scalar("radius_cm", radius_cm, allow_zero=False)
    validated = _absorption_inputs(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=particle_mass_g,
        charge_esu=charge_esu,
    )
    if validated[3] == 0.0 or not np.any(validated[2]):
        return jnp.zeros(validated[0].shape, dtype=jnp.float64)

    coefficient, coefficient_scale, thin_sed = _ssa_quantities_kernel(
        *validated[:3],
        validated[3],
        radius,
        *validated[4:],
    )
    if _contains_tracer((coefficient, coefficient_scale, thin_sed)):
        return _transfer_from_quantities(coefficient, thin_sed, radius)

    coefficient_values = np.asarray(coefficient)
    scale_values = np.asarray(coefficient_scale)
    if (
        not np.all(np.isfinite(coefficient_values))
        or not np.all(np.isfinite(scale_values))
        or np.any(scale_values < 0.0)
    ):
        raise FloatingPointError(
            "synchrotron absorption coefficient exceeds the finite float64 range"
        )
    if np.any(coefficient_values < -_NEGATIVE_ALPHA_ROUNDOFF_FACTOR * scale_values):
        raise ValueError("negative synchrotron absorption requires maser transfer")

    result = _transfer_from_quantities(coefficient, thin_sed, radius)
    result_values = np.asarray(result)
    if not np.all(np.isfinite(result_values)) or np.any(result_values < 0.0):
        raise FloatingPointError("self-absorbed SED exceeds the finite float64 range")
    return result


def electron_synchrotron_self_absorbed_sed_jax(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return electron/positron SSA ``E L_E`` in comoving erg s^-1.

    The wrapper fixes the electron mass and elementary-charge magnitude. The
    density may be any caller-supplied total ``N_electron + N_positron`` in
    cm^-3 per Lorentz factor; component assembly is outside this function.
    """
    return synchrotron_self_absorbed_sed_jax(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_E_CGS,
        charge_esu=_E_ESU,
    )


def proton_synchrotron_self_absorbed_sed_jax(
    photon_energy_erg,
    gamma,
    dn_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return proton SSA ``E L_E`` in comoving erg s^-1."""
    return synchrotron_self_absorbed_sed_jax(
        photon_energy_erg,
        gamma,
        dn_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_P_CGS,
        charge_esu=_E_ESU,
    )
