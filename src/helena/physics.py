"""Standalone blob-frame physics operations backed by production JAX kernels."""

from typing import Any, NamedTuple

from helena.core.constants import (
    CHARGED_PION_MASS_CGS as _M_PI_CGS,
)
from helena.core.constants import (
    ELECTRON_MASS_CGS as _M_E_CGS,
)
from helena.core.constants import (
    ELEMENTARY_CHARGE_ESU as _E_ESU,
)
from helena.core.constants import (
    MUON_MASS_CGS as _M_MU_CGS,
)
from helena.core.constants import (
    PROTON_MASS_CGS as _M_P_CGS,
)
from helena.core.jax_backend.kernels.bethe_heitler import (
    bethe_heitler_proton_energy_loss_rate_jax as _bethe_heitler_loss,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    gamma_gamma_absorbed_sed_jax as _gamma_gamma_absorbed_sed,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    gamma_gamma_absorption_coefficient_jax as _gamma_gamma_absorption_coefficient,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    gamma_gamma_optical_depth_jax as _gamma_gamma_optical_depth,
)
from helena.core.jax_backend.kernels.muon_decay import (
    muon_decay_electron_positron_injection_jax as _muon_decay_leptons,
)
from helena.core.jax_backend.kernels.muon_decay import (
    muon_decay_neutrino_injection_jax as _muon_decay_neutrinos,
)
from helena.core.jax_backend.kernels.photomeson import (
    photomeson_energy_loss_rate_jax as _photomeson_energy_loss_rate,
)
from helena.core.jax_backend.kernels.photomeson import (
    photomeson_interaction_rate_jax as _photomeson_interaction_rate,
)
from helena.core.jax_backend.kernels.photomeson import (
    photomeson_pion_injection_jax as _photomeson_pion_injection,
)
from helena.core.jax_backend.kernels.pion_decay import (
    charged_pion_decay_muon_injection_jax as _charged_pion_decay_muons,
)
from helena.core.jax_backend.kernels.pion_decay import (
    charged_pion_decay_neutrino_injection_jax as _charged_pion_decay_neutrinos,
)
from helena.core.jax_backend.kernels.pion_decay import (
    neutral_pion_decay_photon_injection_jax as _neutral_pion_decay_photons,
)
from helena.core.jax_backend.kernels.proton_proton import (
    proton_proton_pion_injection_and_loss_jax as _proton_proton_pions_and_loss,
)
from helena.core.jax_backend.radiation.inverse_compton import (
    inverse_compton_sed_jax as _inverse_compton,
)
from helena.core.jax_backend.radiation.photon_fields import (
    produced_sed_to_internal_photon_density_jax as _internal_photon_density,
)
from helena.core.jax_backend.radiation.ssa import (
    electron_synchrotron_self_absorbed_sed_jax as _electron_synchrotron_self_absorbed_sed,
)
from helena.core.jax_backend.radiation.ssa import (
    synchrotron_absorption_coefficient_jax as _synchrotron_absorption_coefficient,
)
from helena.core.jax_backend.radiation.synchrotron import (
    electron_synchrotron_sed_jax as _electron_synchrotron,
)
from helena.core.jax_backend.radiation.synchrotron import (
    synchrotron_sed_jax as _synchrotron,
)

__all__ = [
    "ChargedPionMuonInjection",
    "MuonDecayElectronPositronInjection",
    "MuonDecayNeutrinoInjection",
    "PionDecayNeutrinoInjection",
    "PionInjection",
    "ProtonProtonResult",
    "bethe_heitler_proton_energy_loss_rate",
    "charged_pion_decay_muon_injection",
    "charged_pion_decay_neutrino_injection",
    "charged_pion_synchrotron",
    "electron_synchrotron",
    "electron_synchrotron_absorption_coefficient",
    "electron_synchrotron_self_absorbed_sed",
    "gamma_gamma_absorbed_sed",
    "gamma_gamma_absorption_coefficient",
    "gamma_gamma_optical_depth",
    "internal_photon_density",
    "inverse_compton",
    "muon_decay_electron_positron_injection",
    "muon_decay_neutrino_injection",
    "muon_synchrotron",
    "neutral_pion_decay_photon_injection",
    "photomeson_energy_loss_rate",
    "photomeson_interaction_rate",
    "photomeson_pion_injection",
    "proton_proton_pion_injection_and_loss",
    "proton_synchrotron",
]


class PionInjection(NamedTuple):
    """Charge-resolved pion ``Q(gamma)`` arrays in cm^-3 s^-1 per gamma."""

    neutral_pion_dgamma_cm3_s: Any
    positive_pion_dgamma_cm3_s: Any
    negative_pion_dgamma_cm3_s: Any


class ProtonProtonResult(NamedTuple):
    """Cold-target pp loss and charge-resolved pion source arrays."""

    fractional_proton_energy_loss_rate_s_inv: Any
    neutral_pion_injection_dgamma_cm3_s: Any
    positive_pion_injection_dgamma_cm3_s: Any
    negative_pion_injection_dgamma_cm3_s: Any


class ChargedPionMuonInjection(NamedTuple):
    """Helicity-resolved muon ``Q(gamma)`` arrays from charged-pion decay."""

    positive_left_dgamma_cm3_s: Any
    positive_right_dgamma_cm3_s: Any
    negative_left_dgamma_cm3_s: Any
    negative_right_dgamma_cm3_s: Any


class PionDecayNeutrinoInjection(NamedTuple):
    """Direct charged-pion neutrino sources per blob-frame energy."""

    muon_neutrino_denergy_cm3_s_erg: Any
    muon_antineutrino_denergy_cm3_s_erg: Any


class MuonDecayElectronPositronInjection(NamedTuple):
    """Muon-decay positron/electron ``Q(gamma)`` arrays."""

    positron_dgamma_cm3_s: Any
    electron_dgamma_cm3_s: Any


class MuonDecayNeutrinoInjection(NamedTuple):
    """Flavor-resolved muon-decay neutrino sources per blob-frame energy."""

    electron_neutrino_denergy_cm3_s_erg: Any
    electron_antineutrino_denergy_cm3_s_erg: Any
    muon_neutrino_denergy_cm3_s_erg: Any
    muon_antineutrino_denergy_cm3_s_erg: Any


def electron_synchrotron(
    photon_energy_erg,
    electron_gamma,
    dn_electron_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return blob-comoving electron/positron ``E L_E`` in erg s^-1.

    Energies are in erg, gamma is total/rest energy, and the input density is
    ``N_e- + N_e+`` in cm^-3 per gamma. The sphere field and radius use G and
    cm. This is optically thin radiation without transport or frame transfer.
    """
    return _electron_synchrotron(
        photon_energy_erg,
        electron_gamma,
        dn_electron_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
    )


def electron_synchrotron_self_absorbed_sed(
    photon_energy_erg,
    electron_gamma,
    dn_electron_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return escaped blob-comoving electron SSA ``E L_E`` in erg s^-1.

    ``photon_energy_erg`` is a positive one-dimensional blob-frame energy
    array. ``electron_gamma`` is a strictly increasing one-dimensional
    total/rest-energy grid with at least three values above one, and the
    matching non-negative density is total electron plus positron ``N(gamma)``
    in cm^-3 per gamma. Magnetic field and homogeneous-sphere radius use G and
    cm. The result is a JAX float64 array with the photon-energy shape: the
    emergent self-absorbed SED, not ``alpha_nu``, internal photon density, or
    an observer-frame quantity. The owner rejects physical population inversion.
    """
    return _electron_synchrotron_self_absorbed_sed(
        photon_energy_erg,
        electron_gamma,
        dn_electron_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
    )


def proton_synchrotron(
    photon_energy_erg,
    proton_gamma,
    dn_proton_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return optically thin blob-comoving proton ``E L_E`` in erg s^-1.

    Photon energy is in erg, proton gamma is total/rest energy, density is
    ``N_p`` in cm^-3 per gamma, magnetic field is in G, and radius is in cm.
    """
    return _synchrotron(
        photon_energy_erg,
        proton_gamma,
        dn_proton_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_P_CGS,
        charge_esu=_E_ESU,
    )


def charged_pion_synchrotron(
    photon_energy_erg,
    pion_gamma,
    dn_charged_pion_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return optically thin blob-comoving charged-pion ``E L_E`` in erg s^-1.

    The density is the sum of pi+ and pi- ``N(gamma)`` in cm^-3 per gamma;
    photon energy, field, and homogeneous-sphere radius use erg, G, and cm.
    """
    return _synchrotron(
        photon_energy_erg,
        pion_gamma,
        dn_charged_pion_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_PI_CGS,
        charge_esu=_E_ESU,
    )


def muon_synchrotron(
    photon_energy_erg,
    muon_gamma,
    dn_muon_dgamma_cm3,
    *,
    magnetic_field_G,
    radius_cm,
):
    """Return optically thin blob-comoving muon ``E L_E`` in erg s^-1.

    The density sums both charges and helicities in cm^-3 per gamma; photon
    energy, magnetic field, and homogeneous-sphere radius use erg, G, and cm.
    """
    return _synchrotron(
        photon_energy_erg,
        muon_gamma,
        dn_muon_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        radius_cm=radius_cm,
        particle_mass_g=_M_MU_CGS,
        charge_esu=_E_ESU,
    )


def inverse_compton(
    scattered_photon_energy_erg,
    electron_gamma,
    dn_electron_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    radius_cm,
):
    """Return isotropic blob-comoving inverse-Compton ``E L_E`` in erg s^-1.

    Electron/positron density is per gamma; the already-comoving target is
    ``dn_ph/dE`` in cm^-3 erg^-1. Both photon grids use erg and radius uses cm.
    Selecting a synchrotron or external target gives SSC or EC without a
    duplicate process implementation.
    """
    return _inverse_compton(
        scattered_photon_energy_erg,
        electron_gamma,
        dn_electron_dgamma_cm3,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        radius_cm=radius_cm,
    )


def internal_photon_density(
    photon_energy_erg,
    produced_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return homogeneous-sphere internal ``dn_ph/dE`` in cm^-3 erg^-1.

    Photon energy and intrinsic blob ``E L_E`` use erg and erg s^-1; the
    matching local absorption coefficient must be non-negative in cm^-1 and
    the sphere radius is in cm. The Dermer volume/direction average is applied.
    This target construction adds no source radiation, feedback, or frame
    transformation; population-inverted transfer is outside its contract.
    """
    return _internal_photon_density(
        photon_energy_erg,
        produced_sed_erg_s,
        absorption_coefficient_cm1,
        radius_cm=radius_cm,
    )


def electron_synchrotron_absorption_coefficient(
    photon_energy_erg,
    electron_gamma,
    dn_electron_dgamma_cm3,
    *,
    magnetic_field_G,
):
    """Return signed blob-comoving electron synchrotron ``alpha_nu`` in cm^-1.

    Photon energy is in erg, gamma is total/rest energy, total electron plus
    positron density is cm^-3 per gamma, and magnetic field is in G. Negative
    values identify population inversion; no transfer attenuation is applied.
    """
    return _synchrotron_absorption_coefficient(
        photon_energy_erg,
        electron_gamma,
        dn_electron_dgamma_cm3,
        magnetic_field_G=magnetic_field_G,
        particle_mass_g=_M_E_CGS,
        charge_esu=_E_ESU,
    )


def gamma_gamma_absorption_coefficient(
    gamma_ray_energy_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return isotropic blob-comoving gamma-gamma absorption in cm^-1.

    Both photon grids are in erg and the target is angle-integrated
    ``dn_ph/dE`` in cm^-3 erg^-1. No geometry, pair source, or frame transform
    is included.
    """
    return _gamma_gamma_absorption_coefficient(
        gamma_ray_energy_erg,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )


def gamma_gamma_optical_depth(
    gamma_ray_energy_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    radius_cm,
):
    """Return dimensionless homogeneous-sphere diameter gamma-gamma depth.

    Photon energies and target ``dn_ph/dE`` are blob-comoving in erg and
    cm^-3 erg^-1; ``radius_cm`` is the sphere radius. The result is
    ``tau_d = 2 R alpha``, not a foreground path depth.
    """
    return _gamma_gamma_optical_depth(
        gamma_ray_energy_erg,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        radius_cm=radius_cm,
    )


def gamma_gamma_absorbed_sed(
    intrinsic_sed_erg_s,
    absorption_coefficient_cm1,
    *,
    radius_cm,
):
    """Return escaped blob ``E L_E`` after mixed-sphere gamma-gamma transfer.

    Intrinsic ``E L_E`` is in erg s^-1, the matching local coefficient is in
    cm^-1, and radius is in cm. The result uses homogeneous mixed emission,
    not foreground-screen attenuation or an observer-frame transform.
    """
    return _gamma_gamma_absorbed_sed(
        intrinsic_sed_erg_s,
        absorption_coefficient_cm1,
        radius_cm=radius_cm,
    )


def photomeson_interaction_rate(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the blob-comoving Sim-B photomeson event rate in s^-1.

    Proton gamma is total/rest energy; the isotropic target uses erg and
    ``dn_ph/dE`` in cm^-3 erg^-1. This is an event rate, not energy loss or a
    proton sink, and the prescription is intended for relativistic protons.
    """
    return _photomeson_interaction_rate(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )


def photomeson_energy_loss_rate(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the blob-comoving fractional photomeson proton loss in s^-1.

    Proton gamma is total/rest energy; target photon energy and ``dn_ph/dE``
    use erg and cm^-3 erg^-1. The output is ``-dot(gamma_p)/gamma_p``, not an
    interaction rate or local particle sink.
    """
    return _photomeson_energy_loss_rate(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )


def photomeson_pion_injection(
    pion_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return charge-resolved blob-comoving photomeson pion injections.

    Parent and daughter gamma are total/rest energies; proton density is
    cm^-3 per gamma and the isotropic photon target is ``dn_ph/dE`` in
    cm^-3 erg^-1. Named outputs are ``Q_pi0``, ``Q_pi+``, and ``Q_pi-`` in
    cm^-3 s^-1 per pion gamma. No transport or decay is included.
    """
    return PionInjection(
        *_photomeson_pion_injection(
            pion_gamma,
            proton_gamma,
            dn_proton_dgamma_cm3,
            target_photon_energy_erg,
            dn_target_photon_denergy_cm3_erg,
        )
    )


def bethe_heitler_proton_energy_loss_rate(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return fractional blob-comoving Bethe-Heitler proton loss in s^-1.

    Proton gamma is total/rest energy and the isotropic target uses erg and
    ``dn_ph/dE`` in cm^-3 erg^-1. The result is continuous
    ``-dot(gamma_p)/gamma_p``; pair injection and legacy fixed loss are absent.
    """
    return _bethe_heitler_loss(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )


def proton_proton_pion_injection_and_loss(
    pion_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    cold_proton_density_cm3,
):
    """Return cold-target pp proton loss and charge-resolved pion injections.

    Both grids are blob-frame total/rest-energy gamma, proton density is
    cm^-3 per gamma, and the stationary target density is cm^-3. The named
    outputs are fractional proton loss in s^-1 and pion ``Q(gamma)`` in
    cm^-3 s^-1 per pion gamma. The fixed grids must remain outside JAX tracing.
    """
    return ProtonProtonResult(
        *_proton_proton_pions_and_loss(
            pion_gamma,
            proton_gamma,
            dn_proton_dgamma_cm3,
            cold_proton_density_cm3,
        )
    )


def neutral_pion_decay_photon_injection(
    photon_energy_erg,
    pion_gamma,
    q_neutral_pion_decay_dgamma_cm3_s,
):
    """Return blob-comoving pi0-decay photon ``Q(E)`` in cm^-3 s^-1 erg^-1.

    Photon energy is in erg, pion gamma is total/rest energy, and the parent
    input is the actual pi0 decay-event rate in cm^-3 s^-1 per pion gamma.
    No pion transport, absorption, or observer transform is included.
    """
    return _neutral_pion_decay_photons(
        photon_energy_erg,
        pion_gamma,
        q_neutral_pion_decay_dgamma_cm3_s,
    )


def charged_pion_decay_muon_injection(
    muon_gamma,
    pion_gamma,
    q_positive_pion_decay_dgamma_cm3_s,
    q_negative_pion_decay_dgamma_cm3_s,
):
    """Return helicity-resolved blob-comoving muon ``Q(gamma)`` arrays.

    Parent inputs are actual pi+ and pi- decay-event rates in cm^-3 s^-1 per
    pion gamma. Named outputs use cm^-3 s^-1 per muon gamma; left and right
    mean helicity -1 and +1. No muon transport or decay is included.
    """
    return ChargedPionMuonInjection(
        *_charged_pion_decay_muons(
            muon_gamma,
            pion_gamma,
            q_positive_pion_decay_dgamma_cm3_s,
            q_negative_pion_decay_dgamma_cm3_s,
        )
    )


def charged_pion_decay_neutrino_injection(
    neutrino_energy_erg,
    pion_gamma,
    q_positive_pion_decay_dgamma_cm3_s,
    q_negative_pion_decay_dgamma_cm3_s,
):
    """Return direct nu_mu and anti-nu_mu injection from charged-pion decay.

    Neutrino energy is blob-comoving erg; parent inputs are actual pi+ and pi-
    decay-event rates per pion gamma. Named outputs are in cm^-3 s^-1 erg^-1
    before muon decay or flavor oscillation.
    """
    return PionDecayNeutrinoInjection(
        *_charged_pion_decay_neutrinos(
            neutrino_energy_erg,
            pion_gamma,
            q_positive_pion_decay_dgamma_cm3_s,
            q_negative_pion_decay_dgamma_cm3_s,
        )
    )


def muon_decay_electron_positron_injection(
    electron_gamma,
    muon_gamma,
    q_positive_left_muon_decay_dgamma_cm3_s,
    q_positive_right_muon_decay_dgamma_cm3_s,
    q_negative_left_muon_decay_dgamma_cm3_s,
    q_negative_right_muon_decay_dgamma_cm3_s,
):
    """Return named blob-comoving positron/electron sources from muon decay.

    The four parent decay-event rates are per muon gamma in cm^-3 s^-1 and
    left/right mean helicity -1/+1. Outputs are ``Q(gamma)`` in cm^-3 s^-1
    per electron gamma. No transport, cooling, or radiation is included.
    """
    return MuonDecayElectronPositronInjection(
        *_muon_decay_leptons(
            electron_gamma,
            muon_gamma,
            q_positive_left_muon_decay_dgamma_cm3_s,
            q_positive_right_muon_decay_dgamma_cm3_s,
            q_negative_left_muon_decay_dgamma_cm3_s,
            q_negative_right_muon_decay_dgamma_cm3_s,
        )
    )


def muon_decay_neutrino_injection(
    neutrino_energy_erg,
    muon_gamma,
    q_positive_left_muon_decay_dgamma_cm3_s,
    q_positive_right_muon_decay_dgamma_cm3_s,
    q_negative_left_muon_decay_dgamma_cm3_s,
    q_negative_right_muon_decay_dgamma_cm3_s,
):
    """Return flavor-resolved blob-comoving neutrino sources from muon decay.

    Neutrino energy uses erg; the four helicity/charge parent decay rates use
    cm^-3 s^-1 per muon gamma. Named nu_e, anti-nu_e, nu_mu, and anti-nu_mu
    outputs are cm^-3 s^-1 erg^-1 before oscillation.
    """
    return MuonDecayNeutrinoInjection(
        *_muon_decay_neutrinos(
            neutrino_energy_erg,
            muon_gamma,
            q_positive_left_muon_decay_dgamma_cm3_s,
            q_positive_right_muon_decay_dgamma_cm3_s,
            q_negative_left_muon_decay_dgamma_cm3_s,
            q_negative_right_muon_decay_dgamma_cm3_s,
        )
    )
