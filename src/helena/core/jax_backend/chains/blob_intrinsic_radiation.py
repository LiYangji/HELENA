"""One-way intrinsic radiation assembly for a homogeneous blob."""

import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELEMENTARY_CHARGE_ESU as _E_ESU,
    PROTON_MASS_CGS as _M_P_CGS,
)
from helena.core.contracts import nonnegative_1d as _nonnegative_1d
from helena.core.jax_backend.chains.leptonic_radiation import (
    _leptonic_radiation_with_high_synchrotron_jax,
    leptonic_radiation_chain_jax,
)
from helena.core.jax_backend.radiation.synchrotron import (
    _contains_tracer,
    electron_synchrotron_sed_jax,
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
    return _blob_intrinsic_radiation_chain_jax(
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
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        include_high_energy_synchrotron=False,
        compose_proton_synchrotron_low_energy=False,
        synchrotron_primary_split_energy_erg=None,
    )


def _blob_intrinsic_radiation_with_high_synchrotron_jax(
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
    synchrotron_primary_split_energy_erg=None,
    compose_proton_synchrotron_low_energy=False,
):
    """Assemble intrinsic radiation with high-energy lepton synchrotron.

    ``synchrotron_primary_split_energy_erg`` independently sets the boundary
    above which gamma-grid electron synchrotron is a primary. ``None`` uses the
    final target-grid energy. The complete supplied synchrotron grid remains
    the target field, so extending it above an explicit split does not move the
    primary boundary. Such overlapping primary/target populations remain an
    approximation: no symmetric high-high photon collision operator is added.
    ``compose_proton_synchrotron_low_energy`` is an internal full-forward
    composition mode: the low grid owns the split node and the high grid owns
    only energies above it. The default preserves the standalone one-way
    assembly contract.
    """
    return _blob_intrinsic_radiation_chain_jax(
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
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        include_high_energy_synchrotron=True,
        compose_proton_synchrotron_low_energy=(compose_proton_synchrotron_low_energy),
        synchrotron_primary_split_energy_erg=(synchrotron_primary_split_energy_erg),
    )


def _blob_intrinsic_radiation_chain_jax(
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
    include_high_energy_synchrotron,
    compose_proton_synchrotron_low_energy,
    synchrotron_primary_split_energy_erg,
):
    synchrotron_energy = jnp.asarray(
        synchrotron_photon_energy_erg,
        dtype=jnp.float64,
    )
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
    leptonic_chain = (
        _leptonic_radiation_with_high_synchrotron_jax
        if include_high_energy_synchrotron
        else leptonic_radiation_chain_jax
    )
    leptonic = leptonic_chain(
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
    if not include_high_energy_synchrotron:
        high_energy_synchrotron = jnp.zeros_like(ssc)
    elif synchrotron_primary_split_energy_erg is None:
        split_energy = leptonic["synchrotron_split_energy_erg"]
        target_band_synchrotron = leptonic[
            "electron_synchrotron_target_band_intrinsic_sed_erg_s"
        ]
        high_energy_synchrotron = leptonic[
            "electron_synchrotron_high_energy_intrinsic_sed_erg_s"
        ]
        target_band_luminosity = leptonic[
            "electron_synchrotron_target_band_luminosity_erg_s"
        ]
        high_energy_luminosity = leptonic[
            "electron_synchrotron_high_energy_luminosity_erg_s"
        ]
        total_synchrotron_luminosity = leptonic[
            "total_electron_synchrotron_intrinsic_luminosity_erg_s"
        ]
        capture_fraction = leptonic["electron_synchrotron_energy_capture_fraction"]
        deficit_fraction = leptonic["electron_synchrotron_energy_deficit_fraction"]
        symmetric_residual = leptonic["electron_synchrotron_energy_symmetric_residual"]
    else:
        gamma_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
        split_energy = jnp.asarray(
            synchrotron_primary_split_energy_erg,
            dtype=jnp.float64,
        )
        target_band_synchrotron = jnp.where(
            synchrotron_energy <= split_energy,
            leptonic["synchrotron_intrinsic_sed_erg_s"],
            0.0,
        )
        high_energy_synchrotron = jnp.where(
            gamma_energy > split_energy,
            electron_synchrotron_sed_jax(
                gamma_energy,
                electron_gamma,
                leptonic["total_lepton_density_dgamma_cm3"],
                magnetic_field_G=magnetic_field_G,
                radius_cm=radius_cm,
            ),
            0.0,
        )
        target_band_luminosity = jnp.trapezoid(
            target_band_synchrotron,
            x=jnp.log(synchrotron_energy),
        )
        high_energy_luminosity = jnp.trapezoid(
            high_energy_synchrotron,
            x=jnp.log(gamma_energy),
        )
        total_synchrotron_luminosity = target_band_luminosity + high_energy_luminosity
        cooling_luminosity = leptonic["electron_synchrotron_cooling_luminosity_erg_s"]
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
    if include_high_energy_synchrotron:
        proton_synchrotron_low_energy = jnp.where(
            synchrotron_energy <= split_energy,
            synchrotron_sed_jax(
                synchrotron_energy,
                proton_gamma,
                proton_density_dgamma_cm3,
                magnetic_field_G=magnetic_field_G,
                radius_cm=radius_cm,
                particle_mass_g=_M_P_CGS,
                charge_esu=_E_ESU,
            ),
            0.0,
        )
        proton_synchrotron_high_energy = jnp.where(
            gamma_ray_energy > split_energy,
            proton_synchrotron,
            0.0,
        )
    else:
        proton_synchrotron_high_energy = proton_synchrotron

    total_leptonic_gamma = high_energy_synchrotron + ssc + external_compton
    composed_proton_synchrotron = (
        proton_synchrotron_high_energy
        if compose_proton_synchrotron_low_energy
        else proton_synchrotron
    )
    total_hadronic_gamma = composed_proton_synchrotron + pi0_sed
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
    if not include_high_energy_synchrotron:
        return result
    return {
        **result,
        "synchrotron_split_energy_erg": split_energy,
        "electron_synchrotron_target_band_intrinsic_sed_erg_s": (
            target_band_synchrotron
        ),
        "electron_synchrotron_high_energy_intrinsic_sed_erg_s": (
            high_energy_synchrotron
        ),
        "proton_synchrotron_low_energy_intrinsic_sed_erg_s": (
            proton_synchrotron_low_energy
        ),
        "proton_synchrotron_high_energy_intrinsic_sed_erg_s": (
            proton_synchrotron_high_energy
        ),
        "total_electron_synchrotron_intrinsic_luminosity_erg_s": (
            total_synchrotron_luminosity
        ),
        "electron_synchrotron_target_band_luminosity_erg_s": (target_band_luminosity),
        "electron_synchrotron_high_energy_luminosity_erg_s": (high_energy_luminosity),
        "electron_synchrotron_cooling_luminosity_erg_s": leptonic[
            "electron_synchrotron_cooling_luminosity_erg_s"
        ],
        "electron_synchrotron_energy_capture_fraction": capture_fraction,
        "electron_synchrotron_energy_deficit_fraction": deficit_fraction,
        "electron_synchrotron_energy_symmetric_residual": symmetric_residual,
    }
