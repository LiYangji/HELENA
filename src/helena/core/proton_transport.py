"""Steady-state proton transport with continuous losses and explicit escape."""

import numpy as np

from helena.core.contracts import (
    proton_gamma as _proton_gamma,
    proton_injection as _proton_injection,
    rate_on_grid as _rate_on_grid,
)
from helena.core.photomeson import photomeson_energy_loss_rate
from helena.core.proton_timescales import (
    proton_adiabatic_continuous_loss_rate,
    proton_synchrotron_continuous_loss_rate,
)
from helena.core.steady_state_finite_volume import (
    steady_state_finite_volume_solution,
)

__all__ = ["proton_steady_state_transport"]


def proton_steady_state_transport(
    proton_gamma,
    q_proton_injection_dgamma_cm3_s,
    magnetic_field_G,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    adiabatic_rate_s_inv,
    escape_rate_s_inv,
):
    """Return a blob-frame steady proton density and escape-rate spectrum.

    Parameters
    ----------
    proton_gamma : array_like, shape (n_proton,)
        At least two finite, strictly increasing Lorentz factors greater than
        one.
    q_proton_injection_dgamma_cm3_s : array_like, shape (n_proton,)
        Finite non-negative ``dn_p/(d gamma_p dt)`` in cm^-3 s^-1 per unit
        proton Lorentz factor.
    magnetic_field_G : float
        Finite non-negative blob-comoving magnetic field in Gauss.
    target_photon_energy_erg : array_like, shape (n_photon,)
        At least two finite, positive, strictly increasing blob-frame photon
        energies in erg.
    dn_target_photon_denergy_cm3_erg : array_like, shape (n_photon,)
        Matching finite non-negative ``dn_ph/dE`` in cm^-3 erg^-1.
    adiabatic_rate_s_inv, escape_rate_s_inv : float or array_like
        Finite non-negative scalar or proton-grid-shaped rates in s^-1.

    Returns
    -------
    tuple of numpy.ndarray
        ``(N_p, Q_p,esc)`` as float64 arrays. ``N_p = dn_p/d gamma_p`` is in
        cm^-3 and ``Q_p,esc = lambda_p,esc N_p`` is in cm^-3 s^-1, both per
        unit proton Lorentz factor.

    Notes
    -----
    The conservative finite-volume solver receives the summed continuous loss
    ``b_p = b_syn + b_ad + gamma_p t_pgamma,loss^-1`` and the separate local
    escape sink. The photomeson term uses the existing fractional energy-loss
    rate, including its Sim-B inelasticity, exactly once; it is not the event
    interaction rate.

    This first proton wrapper treats total initial-nucleon photomeson energy
    transfer as continuous proton cooling. It does not distinguish proton from
    neutron final states and includes no nucleon conversion, regeneration,
    Bethe--Heitler, pp, acceleration, diffusion, finite age, pion production,
    daughter chain, observer transform, fitting, or JAX backend.
    """
    gamma = _proton_gamma(proton_gamma)
    if gamma.size < 2 or np.any(np.diff(gamma) <= 0.0):
        raise ValueError(
            "proton_gamma must contain at least two strictly increasing values"
        )
    injection = _proton_injection(q_proton_injection_dgamma_cm3_s, gamma)
    synchrotron_loss = proton_synchrotron_continuous_loss_rate(gamma, magnetic_field_G)
    adiabatic_loss = proton_adiabatic_continuous_loss_rate(gamma, adiabatic_rate_s_inv)
    escape_rate = _rate_on_grid("escape_rate_s_inv", escape_rate_s_inv, gamma)
    fractional_photomeson_loss = photomeson_energy_loss_rate(
        gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        photomeson_loss = gamma * fractional_photomeson_loss
        total_loss = synchrotron_loss + adiabatic_loss + photomeson_loss
    if not np.all(np.isfinite(total_loss)) or np.any(total_loss < 0.0):
        raise FloatingPointError("total proton loss exceeds the finite float64 range")

    density = steady_state_finite_volume_solution(
        gamma, injection, total_loss, escape_rate
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        escape = escape_rate * density
    if not np.all(np.isfinite(escape)) or np.any(escape < 0.0):
        raise FloatingPointError(
            "proton escape spectrum exceeds the finite float64 range"
        )
    return density, escape
