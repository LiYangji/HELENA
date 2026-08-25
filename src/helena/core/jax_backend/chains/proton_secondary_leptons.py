"""JAX orchestration of proton-derived secondary lepton injection."""

from helena.core.jax_backend.kernels.bethe_heitler import (
    bethe_heitler_pair_injection_jax,
    bethe_heitler_proton_energy_loss_rate_jax,
)
from helena.core.jax_backend.kernels.photomeson import (
    photomeson_energy_loss_rate_jax,
)
from helena.core.jax_backend.chains.photomeson_chain import (
    photomeson_secondary_chain_jax,
)
from helena.core.jax_backend.transport.proton_transport import (
    proton_steady_state_transport_jax,
)

__all__ = ["proton_secondary_lepton_injection_jax"]


def proton_secondary_lepton_injection_jax(
    proton_gamma,
    q_proton_injection_dgamma_cm3_s,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    pion_gamma,
    muon_gamma,
    photon_energy_erg,
    electron_gamma,
    neutrino_energy_erg,
    magnetic_field_G,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
    *,
    bethe_heitler_pair_response=None,
):
    """Return proton-derived secondary lepton injection from one steady ``N_p``.

    All inputs are blob-comoving. ``proton_gamma``, ``pion_gamma``,
    ``muon_gamma``, and ``electron_gamma`` are Lorentz-factor grids;
    ``photon_energy_erg`` and ``neutrino_energy_erg`` are daughter-energy grids
    in erg. ``q_proton_injection_dgamma_cm3_s`` is in cm^-3 s^-1 per proton
    Lorentz factor. The positive target-energy grid is in erg and its matching
    finite non-negative isotropic or angle-averaged ``dn_ph/dE`` is in
    cm^-3 erg^-1. ``magnetic_field_G`` is in Gauss, while adiabatic and escape
    rates are non-negative scalar or proton-grid values in s^-1. Valid ranges
    and eager/JIT fixed-shape conventions are inherited from the called public
    JAX APIs. ``bethe_heitler_pair_response`` is the optional fixed-grid output
    of ``prepare_bethe_heitler_pair_response_jax``; traced execution requires
    it so only the dynamic pair contraction is differentiated.

    The same target field supplies the photomeson and Bethe--Heitler fractional
    proton losses. Proton transport is solved once with synchrotron,
    adiabatic, photomeson, Bethe--Heitler, and escape effects. Its unique
    ``dn_p/dgamma_p`` in cm^-3 is passed to both photomeson pion production and
    Bethe--Heitler pair injection. The photomeson ``positron`` and ``electron``
    fields remain muon-decay injections; ``bh_positron`` and ``bh_electron``
    are the separate BH injections. Their direct elementwise sums are returned
    as ``total_secondary_positron`` and ``total_secondary_electron``, all in
    cm^-3 s^-1 per electron Lorentz factor on ``electron_gamma``.

    The returned dictionary retains every photomeson-chain field and adds the
    proton escape spectrum, both fractional proton loss rates, BH pair sources,
    and total proton-derived lepton sources. No primary or gamma-gamma pair
    source, electron/positron transport, radiation, cascade, target depletion,
    or observer transformation is included. The inherited Hümmer Sim-B,
    ultrarelativistic pion/muon-decay, and KA/Chodorowski BH assumptions apply.
    """
    fractional_photomeson_loss = photomeson_energy_loss_rate_jax(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    fractional_bethe_heitler_loss = bethe_heitler_proton_energy_loss_rate_jax(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    proton_density, proton_escape = proton_steady_state_transport_jax(
        proton_gamma,
        q_proton_injection_dgamma_cm3_s,
        magnetic_field_G,
        fractional_photomeson_loss,
        adiabatic_rate_s_inv,
        escape_rate_s_inv,
        fractional_bethe_heitler_energy_loss_rate_s_inv=(fractional_bethe_heitler_loss),
    )
    photomeson = photomeson_secondary_chain_jax(
        proton_gamma,
        q_proton_injection_dgamma_cm3_s,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        pion_gamma,
        muon_gamma,
        photon_energy_erg,
        electron_gamma,
        neutrino_energy_erg,
        magnetic_field_G,
        adiabatic_rate_s_inv,
        escape_rate_s_inv,
        proton_density_dgamma_cm3=proton_density,
    )
    bh_positron, bh_electron = bethe_heitler_pair_injection_jax(
        electron_gamma,
        proton_gamma,
        proton_density,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        pair_response=bethe_heitler_pair_response,
    )
    return {
        **photomeson,
        "proton_escape": proton_escape,
        "fractional_photomeson_loss": fractional_photomeson_loss,
        "fractional_bethe_heitler_loss": fractional_bethe_heitler_loss,
        "bh_positron": bh_positron,
        "bh_electron": bh_electron,
        "total_secondary_positron": photomeson["positron"] + bh_positron,
        "total_secondary_electron": photomeson["electron"] + bh_electron,
    }
