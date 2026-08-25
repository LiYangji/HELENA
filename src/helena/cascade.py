"""Standalone electromagnetic-cascade facade over the production JAX solver."""

from typing import Any, NamedTuple

import jax.numpy as jnp

from helena.core.jax_backend.chains.steady_state_electromagnetic_cascade import (
    steady_state_electromagnetic_cascade_jax as _steady_state_em_cascade,
)

__all__ = ["EMCascadeResult", "solve_em"]


class EMCascadeResult(NamedTuple):
    """Minimal scientific view of a steady blob-frame electromagnetic cascade."""

    electron_density_dgamma_cm3: Any
    positron_density_dgamma_cm3: Any
    intrinsic_gamma_sed_erg_s: Any
    escaped_gamma_sed_erg_s: Any
    gamma_gamma_optical_depth: Any
    converged: Any
    iterations: Any
    residual: Any


def solve_em(
    synchrotron_photon_energy_erg,
    gamma_ray_energy_erg,
    electron_gamma,
    q_electron_injection_dgamma_cm3_s,
    q_positron_injection_dgamma_cm3_s,
    proton_gamma,
    dn_proton_dgamma_cm3,
    q_gamma_injection_denergy_cm3_s,
    external_target_photon_energy_erg,
    dn_external_target_photon_denergy_cm3_erg,
    *,
    magnetic_field_G,
    radius_cm,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
    enable_synchrotron_self_compton=True,
    enable_external_inverse_compton=True,
    include_external_target_in_gamma_gamma=False,
    additional_continuous_loss_rate_dgamma_s=0.0,
):
    """Solve a steady electromagnetic cascade in one homogeneous blob.

    Photon grids are blob-comoving erg; particle gamma is total/rest energy.
    Electron/positron sources are cm^-3 s^-1 per gamma, proton density is
    cm^-3 per gamma, and the prescribed gamma source is cm^-3 s^-1 erg^-1.
    The external target must already be blob-comoving ``dn_ph/dE`` in
    cm^-3 erg^-1. Field, radius, adiabatic/escape rates, and additional
    ``-d gamma/dt`` use G, cm, s^-1, s^-1, and s^-1.

    The result contains final particle densities, intrinsic/escaped gamma
    ``E L_E`` in erg s^-1, diameter optical depth, and compact convergence
    facts. Fixed-grid pair responses are prepared internally on the host;
    their tensors, Picard state, relaxation, and ledgers are not public.
    """
    zero = jnp.zeros_like(jnp.asarray(electron_gamma, dtype=jnp.float64))
    result = _steady_state_em_cascade(
        synchrotron_photon_energy_erg,
        gamma_ray_energy_erg,
        electron_gamma,
        q_electron_injection_dgamma_cm3_s,
        q_positron_injection_dgamma_cm3_s,
        zero,
        zero,
        zero,
        zero,
        proton_gamma,
        dn_proton_dgamma_cm3,
        q_gamma_injection_denergy_cm3_s,
        external_target_photon_energy_erg,
        dn_external_target_photon_denergy_cm3_erg,
        magnetic_field_G,
        radius_cm,
        adiabatic_rate_s_inv,
        escape_rate_s_inv,
        include_external_gamma_gamma_target=(include_external_target_in_gamma_gamma),
        enable_ssc=enable_synchrotron_self_compton,
        enable_ec=enable_external_inverse_compton,
        additional_continuous_loss_rate_dgamma_s=(
            additional_continuous_loss_rate_dgamma_s
        ),
    )
    return EMCascadeResult(
        electron_density_dgamma_cm3=result["electron_density_dgamma_cm3"],
        positron_density_dgamma_cm3=result["positron_density_dgamma_cm3"],
        intrinsic_gamma_sed_erg_s=result["total_gamma_intrinsic_sed_erg_s"],
        escaped_gamma_sed_erg_s=result["escaped_gamma_sed_erg_s"],
        gamma_gamma_optical_depth=result["gamma_gamma_optical_depth"],
        converged=result["converged"],
        iterations=result["iteration_count"],
        residual=result["final_cascade_residual"],
    )
