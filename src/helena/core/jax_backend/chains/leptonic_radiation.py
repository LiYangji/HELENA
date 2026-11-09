"""One-way JAX leptonic synchrotron, SSA, SSC, and EC radiation chain."""

import jax.numpy as jnp

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
    return result
