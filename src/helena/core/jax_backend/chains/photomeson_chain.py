"""JAX orchestration of the blob-frame pion secondary chain."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.jax_backend.kernels.muon_decay import (
    muon_decay_electron_positron_injection_jax,
    muon_decay_neutrino_injection_jax,
)
from helena.core.jax_backend.transport.muon_transport import (
    helicity_resolved_muon_steady_state_transport_jax,
)
from helena.core.jax_backend.kernels.photomeson import (
    photomeson_energy_loss_rate_jax,
    photomeson_pion_injection_jax,
)
from helena.core.jax_backend.kernels.pion_decay import (
    charged_pion_decay_muon_injection_jax,
    charged_pion_decay_neutrino_injection_jax,
    neutral_pion_decay_photon_injection_jax,
)
from helena.core.jax_backend.transport.pion_transport import (
    charged_pion_steady_state_transport_jax,
)
from helena.core.jax_backend.transport.proton_transport import (
    proton_steady_state_transport_jax,
)

__all__ = ["photomeson_secondary_chain_jax"]


def photomeson_secondary_chain_jax(
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
    proton_density_dgamma_cm3=None,
    additional_pion_injection_dgamma_cm3_s=None,
):
    """Return the complete JAX photomeson forward-production chain.

    Parameters
    ----------
    proton_gamma, pion_gamma, muon_gamma : array_like
        Strictly increasing blob-comoving Lorentz-factor grids greater than
        one. Proton, charged-pion, and helicity-resolved muon distributions
        are differential in their corresponding Lorentz factor.
    q_proton_injection_dgamma_cm3_s : array_like
        Proton injection ``dn_p/(d gamma_p dt)`` in cm^-3 s^-1 per unit
        proton Lorentz factor.
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg : array_like
        Strictly increasing positive blob-frame target photon energies in erg
        and matching isotropic or angle-averaged ``dn_ph/dE`` in
        cm^-3 erg^-1.
    photon_energy_erg, neutrino_energy_erg : array_like
        Positive blob-frame daughter-energy grids in erg. Photon and neutrino
        injections are returned in cm^-3 s^-1 erg^-1.
    electron_gamma : array_like
        Blob-frame electron/positron Lorentz factors greater than one;
        injections are returned in cm^-3 s^-1 per unit electron Lorentz
        factor.
    magnetic_field_G : float
        Positive blob-frame magnetic-field magnitude in Gauss, shared by the
        proton, charged-pion, and muon transport stages.
    adiabatic_rate_s_inv, escape_rate_s_inv : float or array_like
        Non-negative proton adiabatic and escape rates in s^-1, either scalar
        or on ``proton_gamma``.
    proton_density_dgamma_cm3 : array_like or None, optional
        A finite non-negative blob-frame proton density ``dn_p/d gamma_p`` in
        cm^-3 with exactly the ``proton_gamma`` shape. If supplied, this is the
        unique already-transported proton distribution and may include the
        combined effects of synchrotron, adiabatic, photomeson, Bethe--Heitler,
        and escape terms.
    additional_pion_injection_dgamma_cm3_s : tuple or None, optional
        Optional ``(Q_pi0, Q_pi+, Q_pi-)`` arrays on ``pion_gamma`` in
        cm^-3 s^-1 per pion Lorentz factor. They are added once to the
        photomeson sources before the existing pion and muon chain. The
        default is no additional source.

    Returns
    -------
    dict
        A fixed dictionary of JAX float64 arrays containing ``proton_density``;
        charge-resolved ``q_pi0``, ``q_pi_plus``, and ``q_pi_minus``;
        charged-pion densities and actual decay-event rates; ``pi0`` photons;
        ``mu+_L, mu+_R, mu-_L, mu-_R`` injection, density, and actual
        decay-event rates; ``positron`` and ``electron``; direct-pion and
        muon-decay neutrino components; and total ``nu_mu`` and
        ``anti_nu_mu``. The direct and muon-decay neutrinos use the same
        ``neutrino_energy_erg`` grid and cm^-3 s^-1 erg^-1 convention, so the
        totals are direct elementwise sums.

    Notes
    -----
    This function only connects existing JAX physics kernels. Charged-pion
    and muon daughter kernels receive the transport outputs ``lambda N``, not
    the original injection or steady density. Neutral pions decay promptly
    through the existing finite-beta two-photon kernel. Existing branching
    conventions are therefore applied exactly once.

    With the default ``proton_density_dgamma_cm3=None``, the target photon
    field supplies the photomeson proton loss and the function performs the
    existing internal proton transport. When an external density is supplied,
    that loss calculation and proton transport are skipped completely;
    ``q_proton_injection_dgamma_cm3_s``, ``adiabatic_rate_s_inv``, and
    ``escape_rate_s_inv`` do not participate again. The caller is responsible
    for their physical consistency with the prescribed density. The same
    target photon field still supplies pion injection, and ``magnetic_field_G``
    still controls charged-pion and muon cooling.

    All quantities are blob-comoving. The inherited photomeson model is
    Hümmer Sim-B with ``beta_p ~= 1``; charged-pion and muon decay kernels use
    their existing ultrarelativistic scaling approximations. Eager calls use
    the validation of each public stage, while JIT-traced calls require
    already-valid fixed-shape inputs and JAX float64 mode. No additional
    escape, adiabatic loss, prompt switch, helicity evolution, secondary
    transport, oscillation, cascade, observer transform, or gradient wrapper
    is included.
    """
    if proton_density_dgamma_cm3 is None:
        fractional_photomeson_loss = photomeson_energy_loss_rate_jax(
            proton_gamma,
            target_photon_energy_erg,
            dn_target_photon_denergy_cm3_erg,
        )
        proton_density, _ = proton_steady_state_transport_jax(
            proton_gamma,
            q_proton_injection_dgamma_cm3_s,
            magnetic_field_G,
            fractional_photomeson_loss,
            adiabatic_rate_s_inv,
            escape_rate_s_inv,
        )
        pion_proton_density = proton_density
    else:
        proton_density = proton_density_dgamma_cm3
        density_shape = getattr(proton_density, "shape", None)
        proton_shape = getattr(proton_gamma, "shape", None)
        density_dtype = getattr(proton_density, "dtype", None)
        if getattr(density_dtype, "kind", None) == "b":
            raise ValueError("proton_density_dgamma_cm3 must be real-valued")
        if density_shape is None or len(density_shape) != 1:
            raise ValueError("proton_density_dgamma_cm3 must be one-dimensional")
        if proton_shape is not None and density_shape != proton_shape:
            raise ValueError("proton_density_dgamma_cm3 must match proton_gamma shape")
        # Preserve the exact returned input while making closed-over constants
        # traced before the downstream wrapper selects its eager-validation path.
        pion_proton_density = proton_density * 1.0
    q_pi0, q_pi_plus, q_pi_minus = photomeson_pion_injection_jax(
        pion_gamma,
        proton_gamma,
        pion_proton_density,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if additional_pion_injection_dgamma_cm3_s is not None:
        if len(additional_pion_injection_dgamma_cm3_s) != 3:
            raise ValueError("additional pion injection must contain three arrays")
        additional = tuple(
            jnp.asarray(value, dtype=jnp.float64)
            for value in additional_pion_injection_dgamma_cm3_s
        )
        if not any(
            isinstance(value, jax.core.Tracer) for value in (*additional, pion_gamma)
        ):
            pion_shape = np.asarray(pion_gamma).shape
            for value in additional:
                array = np.asarray(value)
                if (
                    array.shape != pion_shape
                    or not np.all(np.isfinite(array))
                    or np.any(array < 0.0)
                ):
                    raise ValueError(
                        "additional pion injections must match pion_gamma and be "
                        "finite and non-negative"
                    )
        q_pi0 = q_pi0 + additional[0]
        q_pi_plus = q_pi_plus + additional[1]
        q_pi_minus = q_pi_minus + additional[2]
    n_pi_plus, n_pi_minus, q_pi_plus_decay, q_pi_minus_decay = (
        charged_pion_steady_state_transport_jax(
            pion_gamma,
            q_pi_plus,
            q_pi_minus,
            magnetic_field_G,
        )
    )
    pi0_decay_photons = neutral_pion_decay_photon_injection_jax(
        photon_energy_erg,
        pion_gamma,
        q_pi0,
    )
    muon_injections = charged_pion_decay_muon_injection_jax(
        muon_gamma,
        pion_gamma,
        q_pi_plus_decay,
        q_pi_minus_decay,
    )
    direct_pion_neutrinos = charged_pion_decay_neutrino_injection_jax(
        neutrino_energy_erg,
        pion_gamma,
        q_pi_plus_decay,
        q_pi_minus_decay,
    )
    muon_transport = helicity_resolved_muon_steady_state_transport_jax(
        muon_gamma,
        *muon_injections,
        magnetic_field_G,
    )
    positron, electron = muon_decay_electron_positron_injection_jax(
        electron_gamma,
        muon_gamma,
        *muon_transport[4:],
    )
    muon_decay_neutrinos = muon_decay_neutrino_injection_jax(
        neutrino_energy_erg,
        muon_gamma,
        *muon_transport[4:],
    )

    return {
        "proton_density": proton_density,
        "q_pi0": q_pi0,
        "q_pi_plus": q_pi_plus,
        "q_pi_minus": q_pi_minus,
        "n_pi_plus": n_pi_plus,
        "n_pi_minus": n_pi_minus,
        "q_pi_plus_decay": q_pi_plus_decay,
        "q_pi_minus_decay": q_pi_minus_decay,
        "pi0_decay_photons": pi0_decay_photons,
        "q_mu_plus_left": muon_injections[0],
        "q_mu_plus_right": muon_injections[1],
        "q_mu_minus_left": muon_injections[2],
        "q_mu_minus_right": muon_injections[3],
        "direct_pion_decay_nu_mu": direct_pion_neutrinos[0],
        "direct_pion_decay_anti_nu_mu": direct_pion_neutrinos[1],
        "n_mu_plus_left": muon_transport[0],
        "n_mu_plus_right": muon_transport[1],
        "n_mu_minus_left": muon_transport[2],
        "n_mu_minus_right": muon_transport[3],
        "q_mu_plus_left_decay": muon_transport[4],
        "q_mu_plus_right_decay": muon_transport[5],
        "q_mu_minus_left_decay": muon_transport[6],
        "q_mu_minus_right_decay": muon_transport[7],
        "positron": positron,
        "electron": electron,
        "muon_decay_nu_e": muon_decay_neutrinos[0],
        "muon_decay_anti_nu_e": muon_decay_neutrinos[1],
        "muon_decay_nu_mu": muon_decay_neutrinos[2],
        "muon_decay_anti_nu_mu": muon_decay_neutrinos[3],
        "total_nu_mu": direct_pion_neutrinos[0] + muon_decay_neutrinos[2],
        "total_anti_nu_mu": direct_pion_neutrinos[1] + muon_decay_neutrinos[3],
    }
