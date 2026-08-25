"""JAX transport of proton-derived secondary electrons and positrons."""

from helena.core.jax_backend.chains.proton_secondary_leptons import (
    proton_secondary_lepton_injection_jax,
)
from helena.core.jax_backend.kernels.inverse_compton_cooling import (
    electron_inverse_compton_loss_rate_dgamma_s_jax,
)
from helena.core.jax_backend.transport.electron_positron_transport import (
    electron_positron_steady_state_transport_jax,
)

__all__ = ["proton_secondary_lepton_transport_chain_jax"]


def proton_secondary_lepton_transport_chain_jax(
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
    proton_adiabatic_rate_s_inv,
    proton_escape_rate_s_inv,
    electron_adiabatic_rate_s_inv,
    electron_escape_rate_s_inv,
    *,
    additional_electron_continuous_loss_rate_dgamma_s=0.0,
):
    """Return proton-derived secondary sources, densities, and escape spectra.

    All inputs and outputs are blob-comoving and use cgs units. Lorentz-factor
    grids are dimensionless; photon and neutrino energy grids are in erg.
    ``q_proton_injection_dgamma_cm3_s`` is in cm^-3 s^-1 per proton Lorentz
    factor, while ``dn_target_photon_denergy_cm3_erg`` is in cm^-3 erg^-1.
    ``magnetic_field_G`` is in Gauss. Proton and electron adiabatic and escape
    rates are independent inputs in s^-1. The optional caller-supplied electron
    loss is ``b_user = -d gamma_e/dt`` in s^-1.

    The existing proton-secondary chain computes one steady proton density and
    the muon-decay plus Bethe--Heitler electron/positron injections. Those total
    injections are passed directly to the existing conservative lepton
    transport on ``electron_gamma``. Full Klein--Nishina inverse-Compton cooling
    is computed once from the supplied target field and added to ``b_user``.
    Returned electron/positron densities are in cm^-3 per Lorentz factor, their
    escape spectra are in cm^-3 s^-1 per Lorentz factor, and the appended IC
    loss is in s^-1. All source-chain fields and their existing units are
    retained unchanged. The added mapping keys explicitly end in
    ``_density_dgamma_cm3`` or ``_escape_dgamma_cm3_s``.

    No primary or gamma-gamma pair source, radiation, feedback, or cascade is
    included.
    """
    secondary = proton_secondary_lepton_injection_jax(
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
        proton_adiabatic_rate_s_inv,
        proton_escape_rate_s_inv,
    )
    inverse_compton_loss = electron_inverse_compton_loss_rate_dgamma_s_jax(
        electron_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    total_additional_electron_loss = (
        inverse_compton_loss + additional_electron_continuous_loss_rate_dgamma_s
    )
    (
        electron_density,
        positron_density,
        electron_escape,
        positron_escape,
    ) = electron_positron_steady_state_transport_jax(
        electron_gamma,
        secondary["total_secondary_electron"],
        secondary["total_secondary_positron"],
        magnetic_field_G,
        electron_adiabatic_rate_s_inv,
        electron_escape_rate_s_inv,
        additional_continuous_loss_rate_dgamma_s=(total_additional_electron_loss),
    )
    return {
        **secondary,
        "electron_density_dgamma_cm3": electron_density,
        "positron_density_dgamma_cm3": positron_density,
        "electron_escape_dgamma_cm3_s": electron_escape,
        "positron_escape_dgamma_cm3_s": positron_escape,
        "electron_inverse_compton_loss_rate_dgamma_s": inverse_compton_loss,
    }
