"""One-way transport of assembled electron and positron injection."""

import jax.numpy as jnp

from helena.core.contracts import (
    electron_gamma as _electron_gamma,
    finite_scalar as _finite_scalar,
    nonnegative_on_grid as _nonnegative_on_grid,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import _contains_tracer
from helena.core.jax_backend.transport.electron_positron_transport import (
    electron_positron_steady_state_transport_jax,
)

__all__ = ["total_lepton_transport_chain_jax"]


def total_lepton_transport_chain_jax(
    electron_gamma,
    primary_electron_injection_dgamma_cm3_s,
    primary_positron_injection_dgamma_cm3_s,
    muon_decay_electron_injection_dgamma_cm3_s,
    muon_decay_positron_injection_dgamma_cm3_s,
    bethe_heitler_electron_injection_dgamma_cm3_s,
    bethe_heitler_positron_injection_dgamma_cm3_s,
    gamma_gamma_electron_injection_dgamma_cm3_s,
    gamma_gamma_positron_injection_dgamma_cm3_s,
    magnetic_field_G,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
    *,
    inverse_compton_loss_rate_dgamma_s=0.0,
    additional_continuous_loss_rate_dgamma_s=0.0,
):
    """Assemble four source classes and run one steady lepton transport.

    ``electron_gamma`` is a strictly increasing blob-comoving Lorentz-factor
    grid greater than one. Every named electron/positron source is a matching
    non-negative ``dN/(dV dt dgamma)`` in cm^-3 s^-1 per unit Lorentz factor.
    ``magnetic_field_G`` is in Gauss. Adiabatic and escape rates are
    non-negative scalars or grid-shaped arrays in s^-1. Escape is a local
    particle sink. The inverse-Compton and additional loss inputs are
    non-negative scalars or grid-shaped continuous losses
    ``b = -d gamma/dt`` in s^-1 and are added without another gamma factor.

    The returned component sources are unchanged, totals are direct
    elementwise sums, transported densities are ``dn/dgamma`` in cm^-3, and
    escape spectra are in cm^-3 s^-1 per unit Lorentz factor. Both charges use
    the existing shared synchrotron, adiabatic, continuous-loss, and escape
    operator. No upstream source kernel, radiation, photon update, feedback,
    cascade, observer transform, interpolation, or normalization is included.
    """
    source_names = (
        "primary_electron_injection_dgamma_cm3_s",
        "primary_positron_injection_dgamma_cm3_s",
        "muon_decay_electron_injection_dgamma_cm3_s",
        "muon_decay_positron_injection_dgamma_cm3_s",
        "bethe_heitler_electron_injection_dgamma_cm3_s",
        "bethe_heitler_positron_injection_dgamma_cm3_s",
        "gamma_gamma_electron_injection_dgamma_cm3_s",
        "gamma_gamma_positron_injection_dgamma_cm3_s",
    )
    sources = (
        primary_electron_injection_dgamma_cm3_s,
        primary_positron_injection_dgamma_cm3_s,
        muon_decay_electron_injection_dgamma_cm3_s,
        muon_decay_positron_injection_dgamma_cm3_s,
        bethe_heitler_electron_injection_dgamma_cm3_s,
        bethe_heitler_positron_injection_dgamma_cm3_s,
        gamma_gamma_electron_injection_dgamma_cm3_s,
        gamma_gamma_positron_injection_dgamma_cm3_s,
    )
    if not _contains_tracer(
        (
            electron_gamma,
            sources,
            magnetic_field_G,
            adiabatic_rate_s_inv,
            escape_rate_s_inv,
            inverse_compton_loss_rate_dgamma_s,
            additional_continuous_loss_rate_dgamma_s,
        )
    ):
        gamma = _electron_gamma(electron_gamma)
        sources = tuple(
            _nonnegative_on_grid(name, source, gamma, allow_scalar=False)
            for name, source in zip(source_names, sources, strict=True)
        )
        _finite_scalar("magnetic_field_G", magnetic_field_G, allow_zero=True)
        _nonnegative_on_grid(
            "adiabatic_rate_s_inv",
            adiabatic_rate_s_inv,
            gamma,
            allow_scalar=True,
        )
        _nonnegative_on_grid(
            "escape_rate_s_inv",
            escape_rate_s_inv,
            gamma,
            allow_scalar=True,
        )
        _nonnegative_on_grid(
            "inverse_compton_loss_rate_dgamma_s",
            inverse_compton_loss_rate_dgamma_s,
            gamma,
            allow_scalar=True,
        )
        _nonnegative_on_grid(
            "additional_continuous_loss_rate_dgamma_s",
            additional_continuous_loss_rate_dgamma_s,
            gamma,
            allow_scalar=True,
        )

    gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    sources = tuple(jnp.asarray(source, dtype=jnp.float64) for source in sources)
    (
        primary_electron,
        primary_positron,
        muon_electron,
        muon_positron,
        bethe_heitler_electron,
        bethe_heitler_positron,
        gamma_gamma_electron,
        gamma_gamma_positron,
    ) = sources
    total_electron = (
        primary_electron + muon_electron + bethe_heitler_electron + gamma_gamma_electron
    )
    total_positron = (
        primary_positron + muon_positron + bethe_heitler_positron + gamma_gamma_positron
    )
    total_additional_loss = jnp.asarray(
        inverse_compton_loss_rate_dgamma_s, dtype=jnp.float64
    ) + jnp.asarray(additional_continuous_loss_rate_dgamma_s, dtype=jnp.float64)
    (
        electron_density,
        positron_density,
        electron_escape,
        positron_escape,
    ) = electron_positron_steady_state_transport_jax(
        gamma,
        total_electron,
        total_positron,
        magnetic_field_G,
        adiabatic_rate_s_inv,
        escape_rate_s_inv,
        additional_continuous_loss_rate_dgamma_s=total_additional_loss,
    )
    return {
        **dict(zip(source_names, sources, strict=True)),
        "total_electron_injection_dgamma_cm3_s": total_electron,
        "total_positron_injection_dgamma_cm3_s": total_positron,
        "total_additional_continuous_loss_rate_dgamma_s": total_additional_loss,
        "electron_density_dgamma_cm3": electron_density,
        "positron_density_dgamma_cm3": positron_density,
        "electron_escape_dgamma_cm3_s": electron_escape,
        "positron_escape_dgamma_cm3_s": positron_escape,
    }
