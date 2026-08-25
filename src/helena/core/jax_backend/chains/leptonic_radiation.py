"""One-way JAX leptonic synchrotron, SSA, SSC, and EC radiation chain."""

import jax.numpy as jnp

from helena.core.constants import SPEED_OF_LIGHT_CGS as _C_CGS
from helena.core.jax_backend.radiation.inverse_compton import (
    inverse_compton_sed_jax,
)
from helena.core.jax_backend.radiation.photon_fields import (
    synchrotron_sed_to_internal_photon_density_jax,
)
from helena.core.jax_backend.radiation.ssa import (
    _E_ESU,
    _M_E_CGS,
    electron_synchrotron_self_absorbed_sed_jax,
    synchrotron_absorption_coefficient_jax,
)
from helena.core.jax_backend.radiation.synchrotron import (
    electron_synchrotron_sed_jax,
)
from helena.core.jax_backend.transport.electron_positron_transport import (
    _ELECTRON_SYNCHROTRON_LOSS_COEFFICIENT,
)

__all__ = ["leptonic_radiation_chain_jax"]


def leptonic_radiation_chain_jax(
    synchrotron_photon_energy_erg,
    inverse_compton_photon_energy_erg,
    electron_gamma,
    electron_density_dgamma_cm3,
    positron_density_dgamma_cm3,
    external_seed_photon_energy_erg,
    dn_external_seed_photon_denergy_cm3_erg,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return one-way blob-comoving synchrotron, SSC, and EC radiation.

    Photon-energy grids are in erg, ``electron_gamma`` is dimensionless, both
    lepton densities are ``dn/dgamma`` in cm^-3, the prescribed external seed
    density is ``dn_ph/dE`` in cm^-3 erg^-1, ``magnetic_field_G`` is in Gauss,
    and ``radius_cm`` is the homogeneous-sphere radius in cm. Electron and
    positron densities are added before every radiation calculation.
    External seed energies and densities must already be blob-comoving; this
    chain does not transform a host/AGN-frame radiation field into the blob.

    Returned synchrotron and inverse-Compton spectra are blob-comoving
    ``E L_E`` in erg s^-1, the signed SSA coefficient is in cm^-1, and the
    internal synchrotron photon density is in cm^-3 erg^-1. Valid fixed-shape
    float64 inputs, eager validation, and JIT conventions are inherited from
    the called public radiation APIs. The dictionary is a JAX pytree.

    This performs one forward SSC calculation from the fixed input lepton
    distributions. It includes no particle transport, observer transform,
    gamma-gamma interaction, feedback iteration, or cascade.
    """
    return _leptonic_radiation_chain_jax(
        synchrotron_photon_energy_erg,
        inverse_compton_photon_energy_erg,
        electron_gamma,
        electron_density_dgamma_cm3,
        positron_density_dgamma_cm3,
        external_seed_photon_energy_erg,
        dn_external_seed_photon_denergy_cm3_erg,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        include_high_energy_synchrotron=False,
    )


def _leptonic_radiation_with_high_synchrotron_jax(
    synchrotron_photon_energy_erg,
    inverse_compton_photon_energy_erg,
    electron_gamma,
    electron_density_dgamma_cm3,
    positron_density_dgamma_cm3,
    external_seed_photon_energy_erg,
    dn_external_seed_photon_denergy_cm3_erg,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return leptonic radiation with non-overlapping low/high synchrotron.

    The low-energy synchrotron band, through its final grid point, remains the
    internal target. The same transported leptons are evaluated directly on
    the inverse-Compton/gamma grid, where synchrotron is exactly zero at and
    below that split and is treated as a gamma primary above it. This excludes
    low-low, high-high, and symmetric identical-photon interactions.
    """
    return _leptonic_radiation_chain_jax(
        synchrotron_photon_energy_erg,
        inverse_compton_photon_energy_erg,
        electron_gamma,
        electron_density_dgamma_cm3,
        positron_density_dgamma_cm3,
        external_seed_photon_energy_erg,
        dn_external_seed_photon_denergy_cm3_erg,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        include_high_energy_synchrotron=True,
    )


def _leptonic_radiation_chain_jax(
    synchrotron_photon_energy_erg,
    inverse_compton_photon_energy_erg,
    electron_gamma,
    electron_density_dgamma_cm3,
    positron_density_dgamma_cm3,
    external_seed_photon_energy_erg,
    dn_external_seed_photon_denergy_cm3_erg,
    *,
    magnetic_field_G,
    radius_cm,
    include_high_energy_synchrotron,
):
    total_lepton_density = jnp.asarray(
        electron_density_dgamma_cm3, dtype=jnp.float64
    ) + jnp.asarray(positron_density_dgamma_cm3, dtype=jnp.float64)
    synchrotron_intrinsic_sed = electron_synchrotron_sed_jax(
        synchrotron_photon_energy_erg,
        electron_gamma,
        total_lepton_density,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
    )
    synchrotron_absorption_coefficient = synchrotron_absorption_coefficient_jax(
        synchrotron_photon_energy_erg,
        electron_gamma,
        total_lepton_density,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=_M_E_CGS,
        charge_esu=_E_ESU,
    )
    synchrotron_emergent_sed = electron_synchrotron_self_absorbed_sed_jax(
        synchrotron_photon_energy_erg,
        electron_gamma,
        total_lepton_density,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
    )
    internal_synchrotron_photon_density = (
        synchrotron_sed_to_internal_photon_density_jax(
            synchrotron_photon_energy_erg,
            synchrotron_intrinsic_sed,
            synchrotron_absorption_coefficient,
            radius_cm=radius_cm,
        )
    )
    ssc_sed = inverse_compton_sed_jax(
        inverse_compton_photon_energy_erg,
        electron_gamma,
        total_lepton_density,
        synchrotron_photon_energy_erg,
        internal_synchrotron_photon_density,
        radius_cm=radius_cm,
    )
    external_inverse_compton_sed = inverse_compton_sed_jax(
        inverse_compton_photon_energy_erg,
        electron_gamma,
        total_lepton_density,
        external_seed_photon_energy_erg,
        dn_external_seed_photon_denergy_cm3_erg,
        radius_cm=radius_cm,
    )
    total_inverse_compton_sed = ssc_sed + external_inverse_compton_sed

    result = {
        "total_lepton_density_dgamma_cm3": total_lepton_density,
        "synchrotron_intrinsic_sed_erg_s": synchrotron_intrinsic_sed,
        "synchrotron_absorption_coefficient_cm1": (synchrotron_absorption_coefficient),
        "synchrotron_emergent_sed_erg_s": synchrotron_emergent_sed,
        "internal_synchrotron_photon_density_cm3_erg": (
            internal_synchrotron_photon_density
        ),
        "ssc_sed_erg_s": ssc_sed,
        "external_inverse_compton_sed_erg_s": external_inverse_compton_sed,
        "total_inverse_compton_sed_erg_s": total_inverse_compton_sed,
    }
    if not include_high_energy_synchrotron:
        return result

    synchrotron_energy = jnp.asarray(
        synchrotron_photon_energy_erg,
        dtype=jnp.float64,
    )
    gamma_energy = jnp.asarray(
        inverse_compton_photon_energy_erg,
        dtype=jnp.float64,
    )
    electron_gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    magnetic_field = jnp.asarray(magnetic_field_G, dtype=jnp.float64)
    radius = jnp.asarray(radius_cm, dtype=jnp.float64)
    split_energy = synchrotron_energy[-1]
    target_band_sed = jnp.where(
        synchrotron_energy <= split_energy,
        synchrotron_intrinsic_sed,
        0.0,
    )
    high_energy_sed = jnp.where(
        gamma_energy > split_energy,
        electron_synchrotron_sed_jax(
            gamma_energy,
            electron_gamma,
            total_lepton_density,
            magnetic_field_G=magnetic_field,
            radius_cm=radius,
        ),
        0.0,
    )
    target_band_luminosity = jnp.trapezoid(
        target_band_sed,
        x=jnp.log(synchrotron_energy),
    )
    high_energy_luminosity = jnp.trapezoid(
        high_energy_sed,
        x=jnp.log(gamma_energy),
    )
    total_synchrotron_luminosity = target_band_luminosity + high_energy_luminosity
    volume = 4.0 * jnp.pi * radius**3 / 3.0
    synchrotron_loss_rate = (
        _ELECTRON_SYNCHROTRON_LOSS_COEFFICIENT * magnetic_field**2 * electron_gamma**2
    )
    cooling_luminosity = (
        volume
        * _M_E_CGS
        * _C_CGS**2
        * jnp.trapezoid(
            synchrotron_loss_rate * total_lepton_density,
            x=electron_gamma,
        )
    )
    zero_power = (total_synchrotron_luminosity == 0.0) & (cooling_luminosity == 0.0)
    safe_cooling = jnp.where(cooling_luminosity == 0.0, 1.0, cooling_luminosity)
    capture_fraction = jnp.where(
        zero_power,
        1.0,
        total_synchrotron_luminosity / safe_cooling,
    )
    deficit_fraction = jnp.where(
        zero_power,
        0.0,
        jnp.abs(1.0 - capture_fraction),
    )
    symmetric_residual = jnp.where(
        zero_power,
        0.0,
        2.0
        * jnp.abs(total_synchrotron_luminosity - cooling_luminosity)
        / (total_synchrotron_luminosity + cooling_luminosity),
    )
    return {
        **result,
        "synchrotron_split_energy_erg": split_energy,
        "electron_synchrotron_target_band_intrinsic_sed_erg_s": target_band_sed,
        "electron_synchrotron_high_energy_intrinsic_sed_erg_s": high_energy_sed,
        "total_electron_synchrotron_intrinsic_luminosity_erg_s": (
            total_synchrotron_luminosity
        ),
        "electron_synchrotron_target_band_luminosity_erg_s": (target_band_luminosity),
        "electron_synchrotron_high_energy_luminosity_erg_s": (high_energy_luminosity),
        "electron_synchrotron_cooling_luminosity_erg_s": cooling_luminosity,
        "electron_synchrotron_energy_capture_fraction": capture_fraction,
        "electron_synchrotron_energy_deficit_fraction": deficit_fraction,
        "electron_synchrotron_energy_symmetric_residual": symmetric_residual,
    }
