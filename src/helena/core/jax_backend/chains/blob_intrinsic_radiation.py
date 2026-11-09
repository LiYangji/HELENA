"""One-way intrinsic radiation assembly for a homogeneous blob."""

import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELEMENTARY_CHARGE_ESU as _E_ESU,
)
from helena.core.constants import (
    PROTON_MASS_CGS as _M_P_CGS,
)
from helena.core.contracts import nonnegative_1d as _nonnegative_1d
from helena.core.jax_backend.chains.leptonic_radiation import (
    leptonic_radiation_chain_jax,
)
from helena.core.jax_backend.radiation.synchrotron import (
    _contains_tracer,
    synchrotron_sed_jax,
)

__all__ = ["blob_intrinsic_radiation_chain_jax"]


def blob_intrinsic_radiation_chain_jax(
    synchrotron_photon_energy_erg,
    gamma_ray_energy_erg,
    electron_gamma,
    electron_density_dgamma_cm3,
    positron_density_dgamma_cm3,
    proton_gamma,
    proton_density_dgamma_cm3,
    pi0_decay_photon_injection_cm3_s_erg,
    external_seed_photon_energy_erg,
    dn_external_seed_photon_denergy_cm3_erg,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Assemble blob-comoving intrinsic lepton, proton, and pi0 radiation.

    Photon energies are blob-comoving in erg, particle distributions are ``dn/dgamma`` in
    cm^-3, the external seed field is ``dn_ph/dE`` in cm^-3 erg^-1, and
    the external seed field must already be blob-comoving rather than
    host/AGN-frame. ``pi0_decay_photon_injection_cm3_s_erg`` is
    ``dn_gamma/(dE dt)`` in
    cm^-3 s^-1 erg^-1 on ``gamma_ray_energy_erg``. ``magnetic_field_G`` and
    ``radius_cm`` describe one shared uniform spherical blob.

    All SED fields are intrinsic comoving ``E L_E`` in erg s^-1. The pi0
    adapter is exactly ``V E^2 Q_gamma`` with ``V = 4 pi R^3 / 3``. This
    fixed-shape JAX composition includes no particle transport, gamma-gamma
    absorption, observer transform, feedback, or cascade.
    """
    gamma_ray_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
    if _contains_tracer((gamma_ray_energy_erg, pi0_decay_photon_injection_cm3_s_erg)):
        pi0_injection = jnp.asarray(
            pi0_decay_photon_injection_cm3_s_erg, dtype=jnp.float64
        )
    else:
        pi0_injection = jnp.asarray(
            _nonnegative_1d(
                "pi0_decay_photon_injection_cm3_s_erg",
                pi0_decay_photon_injection_cm3_s_erg,
                shape=np.shape(gamma_ray_energy_erg),
            ),
            dtype=jnp.float64,
        )
    leptonic = leptonic_radiation_chain_jax(
        synchrotron_photon_energy_erg,
        gamma_ray_energy_erg,
        electron_gamma,
        electron_density_dgamma_cm3,
        positron_density_dgamma_cm3,
        external_seed_photon_energy_erg,
        dn_external_seed_photon_denergy_cm3_erg,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
    )
    proton_synchrotron = synchrotron_sed_jax(
        gamma_ray_energy_erg,
        proton_gamma,
        proton_density_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_P_CGS,
        charge_esu=_E_ESU,
    )
    volume_cm3 = 4.0 * jnp.pi * jnp.asarray(radius_cm, dtype=jnp.float64) ** 3 / 3.0
    pi0_sed = volume_cm3 * gamma_ray_energy**2 * pi0_injection
    if not _contains_tracer(pi0_sed):
        pi0_values = np.asarray(pi0_sed)
        if not np.all(np.isfinite(pi0_values)):
            raise FloatingPointError("pi0 intrinsic SED exceeds finite float64 range")

    ssc = leptonic["ssc_sed_erg_s"]
    external_compton = leptonic["external_inverse_compton_sed_erg_s"]
    total_leptonic_gamma = ssc + external_compton
    total_hadronic_gamma = proton_synchrotron + pi0_sed
    total_gamma = total_leptonic_gamma + total_hadronic_gamma

    result = {
        "total_lepton_density_dgamma_cm3": (
            leptonic["total_lepton_density_dgamma_cm3"]
        ),
        "electron_synchrotron_intrinsic_sed_erg_s": (
            leptonic["synchrotron_intrinsic_sed_erg_s"]
        ),
        "electron_synchrotron_emergent_sed_erg_s": (
            leptonic["synchrotron_emergent_sed_erg_s"]
        ),
        "electron_synchrotron_absorption_coefficient_cm1": (
            leptonic["synchrotron_absorption_coefficient_cm1"]
        ),
        "internal_synchrotron_photon_density_cm3_erg": (
            leptonic["internal_synchrotron_photon_density_cm3_erg"]
        ),
        "ssc_intrinsic_sed_erg_s": ssc,
        "external_compton_intrinsic_sed_erg_s": external_compton,
        "proton_synchrotron_intrinsic_sed_erg_s": proton_synchrotron,
        "pi0_intrinsic_sed_erg_s": pi0_sed,
        "total_leptonic_gamma_intrinsic_sed_erg_s": total_leptonic_gamma,
        "total_hadronic_gamma_intrinsic_sed_erg_s": total_hadronic_gamma,
        "total_gamma_intrinsic_sed_erg_s": total_gamma,
    }
    return result
