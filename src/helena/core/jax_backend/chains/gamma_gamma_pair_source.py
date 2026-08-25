"""One-way gamma-gamma pair injection from an intrinsic gamma-ray SED."""

import jax.numpy as jnp
import numpy as np

from helena.core.contracts import real_1d as _real_1d
from helena.core.jax_backend.chains.gamma_gamma_internal_photon_field import (
    _gamma_gamma_internal_photon_field_with_absorption_coefficient_jax,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    _contains_tracer,
    gamma_gamma_absorption_coefficient_jax,
)
from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
    gamma_gamma_pair_injection_jax,
)

__all__ = ["gamma_gamma_pair_source_chain_jax"]


def gamma_gamma_pair_source_chain_jax(
    gamma_ray_energy_erg,
    intrinsic_gamma_sed_erg_s,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    lepton_gamma,
    *,
    radius_cm,
    external_target_photon_energy_erg=None,
    dn_external_target_photon_denergy_cm3_erg=None,
    include_external_gamma_gamma_target=False,
    internal_pair_response=None,
    external_pair_response=None,
):
    """Return one-way gamma-gamma positron and electron injection.

    Primary and target photon energies are strictly increasing blob-comoving
    grids in erg. ``intrinsic_gamma_sed_erg_s`` is intrinsic ``E L_E`` in
    erg s^-1, the target field is isotropic ``dn_target/dE`` in
    cm^-3 erg^-1, ``lepton_gamma`` is a strictly increasing daughter
    Lorentz-factor grid greater than one, and ``radius_cm`` is the homogeneous
    sphere radius in cm.

    With ``include_external_gamma_gamma_target=True``, the optional external
    field is additionally treated as isotropic, homogeneous, co-spatial, and
    blob-comoving. Internal and external absorption coefficients are evaluated
    independently and summed before one mixed-sphere transfer. The resulting
    internal gamma density is then passed to the pair kernel once per target,
    and the charge-separated sources are summed. The prescribed external field
    is not depleted. This is not an anisotropic or line-of-sight geometry.

    Per-charge outputs are ``dN/(dV dt dgamma)`` in cm^-3 s^-1. No pair
    multiplicity, target density, residence time, interpolation, transport,
    radiation, feedback, cascade, or frame transform is applied a second time.
    Optional internal/external pair responses are fixed-grid outputs of
    ``prepare_gamma_gamma_pair_response_jax`` and keep traced dynamic spectra
    on the discrete contraction path.
    """
    if not isinstance(include_external_gamma_gamma_target, (bool, np.bool_)):
        raise ValueError("include_external_gamma_gamma_target must be boolean")
    include_external_gamma_gamma_target = bool(include_external_gamma_gamma_target)
    if include_external_gamma_gamma_target and (
        external_target_photon_energy_erg is None
        or dn_external_target_photon_denergy_cm3_erg is None
    ):
        raise ValueError(
            "external target photon energy and density are required when "
            "include_external_gamma_gamma_target=True"
        )
    if not _contains_tracer(lepton_gamma):
        daughter = _real_1d(
            "lepton_gamma",
            lepton_gamma,
            minimum_size=1,
            positive=True,
        )
        if np.any(daughter <= 1.0) or np.any(np.diff(daughter) <= 0.0):
            raise ValueError("lepton_gamma must contain strictly increasing values > 1")

    internal_absorption_coefficient = gamma_gamma_absorption_coefficient_jax(
        gamma_ray_energy_erg,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if include_external_gamma_gamma_target:
        external_absorption_coefficient = gamma_gamma_absorption_coefficient_jax(
            gamma_ray_energy_erg,
            external_target_photon_energy_erg,
            dn_external_target_photon_denergy_cm3_erg,
        )
    else:
        external_absorption_coefficient = jnp.zeros_like(
            internal_absorption_coefficient
        )
    total_absorption_coefficient = (
        internal_absorption_coefficient + external_absorption_coefficient
    )
    internal = _gamma_gamma_internal_photon_field_with_absorption_coefficient_jax(
        gamma_ray_energy_erg,
        intrinsic_gamma_sed_erg_s,
        total_absorption_coefficient,
        radius_cm=radius_cm,
    )
    internal_positron, internal_electron = gamma_gamma_pair_injection_jax(
        lepton_gamma,
        gamma_ray_energy_erg,
        internal["internal_photon_density_cm3_erg"],
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        pair_response=internal_pair_response,
    )
    if include_external_gamma_gamma_target:
        external_positron, external_electron = gamma_gamma_pair_injection_jax(
            lepton_gamma,
            gamma_ray_energy_erg,
            internal["internal_photon_density_cm3_erg"],
            external_target_photon_energy_erg,
            dn_external_target_photon_denergy_cm3_erg,
            pair_response=external_pair_response,
        )
    else:
        external_positron = jnp.zeros_like(internal_positron)
        external_electron = jnp.zeros_like(internal_electron)
    positron = internal_positron + external_positron
    electron = internal_electron + external_electron

    return {
        "volume_cm3": internal["volume_cm3"],
        "absorption_coefficient_cm1": total_absorption_coefficient,
        "gamma_gamma_internal_target_absorption_coefficient_cm1": (
            internal_absorption_coefficient
        ),
        "gamma_gamma_external_target_absorption_coefficient_cm1": (
            external_absorption_coefficient
        ),
        "gamma_gamma_total_absorption_coefficient_cm1": (total_absorption_coefficient),
        "optical_depth": internal["optical_depth"],
        "transfer_factor": internal["transfer_factor"],
        "intrinsic_sed_erg_s": internal["intrinsic_sed_erg_s"],
        "escaped_sed_erg_s": internal["escaped_sed_erg_s"],
        "removed_sed_erg_s": internal["removed_sed_erg_s"],
        "internal_gamma_photon_density_cm3_erg": (
            internal["internal_photon_density_cm3_erg"]
        ),
        "absorbed_photon_rate_cm3_s_erg": (internal["absorbed_photon_rate_cm3_s_erg"]),
        "internal_target_positron_pair_injection_dgamma_cm3_s": (internal_positron),
        "internal_target_electron_pair_injection_dgamma_cm3_s": (internal_electron),
        "external_target_positron_pair_injection_dgamma_cm3_s": (external_positron),
        "external_target_electron_pair_injection_dgamma_cm3_s": (external_electron),
        "positron_pair_injection_dgamma_cm3_s": positron,
        "electron_pair_injection_dgamma_cm3_s": electron,
        "total_pair_injection_dgamma_cm3_s": positron + electron,
    }
