"""Unified internal gamma-gamma operations and explicit migration errors."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
)
from helena.core.constants import (
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.contracts import real_1d as _real_1d
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    _angle_averaged_cross_section_jax,
)
from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
    _contract_pair_response,
    _trapezoid_weights,
    prepare_gamma_gamma_pair_response_jax,
)

__all__ = [
    "gamma_gamma_pair_source_chain_jax",
    "gamma_gamma_total_field_jax",
    "prepare_gamma_gamma_total_field_response_jax",
]


def gamma_gamma_pair_source_chain_jax(*args, **kwargs):
    """Reject retired labelled internal-field orchestration explicitly.

    Separate produced gamma SED and synchrotron target inputs cannot specify
    a unique internal total population. Use the common-state EM solver or
    gamma_gamma_total_field_jax with an explicit dn/dE population instead.
    """
    raise ValueError(
        "The one-way internal gamma-gamma chain is retired; rebuild unique source populations and use the unified EM solver"
    )


def prepare_gamma_gamma_total_field_response_jax(
    photon_energy_erg, lepton_gamma, *, pair_response=None
):
    """Prepare fixed grids for one isotropic internal photon population.

    Both grids must be concrete, finite, strictly increasing float64-compatible
    arrays with at least two nodes. Photon energies are blob-comoving erg;
    daughter Lorentz factors exceed one. No density, radius or physical source
    parameter is cached. This host preparation is outside JIT/AD.

    An optional existing ``pair_response`` must belong to exactly these grids
    (daughter, photon, photon); its provenance is the caller's responsibility.
    It already includes c, isotropic angular averaging and BOTH dE quadrature
    weights, but no same-population event factor of 1/2. No renormalization is
    applied. The NumPy/SciPy kernels remain the independent reference backend.

    Support masks flag photon pairs that CAN place a daughter outside the
    finite daughter grid. They yield conservative bounds, not measured loss;
    finite daughter quadrature error is reported separately by the operator.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX gamma-gamma total fields require jax_enable_x64")
    energy = _real_1d(
        "photon_energy_erg", photon_energy_erg, minimum_size=2, positive=True
    )
    daughter = _real_1d("lepton_gamma", lepton_gamma, minimum_size=2, positive=True)
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("photon_energy_erg must be strictly increasing")
    if np.any(daughter <= 1.0) or np.any(np.diff(daughter) <= 0.0):
        raise ValueError("lepton_gamma must be strictly increasing and > 1")
    scaled = energy / _MEC2_CGS
    if (
        not np.all(np.isfinite(scaled))
        or scaled[-1] > np.sqrt(np.finfo(float).max) / 2.0
    ):
        raise ValueError("photon energy products exceed finite float64 range")
    if pair_response is None:
        pair_response = prepare_gamma_gamma_pair_response_jax(daughter, energy, energy)
    else:
        supplied = np.asarray(pair_response)
        if supplied.shape != (daughter.size, energy.size, energy.size):
            raise ValueError("pair_response must have shape (daughter, photon, photon)")
        if (
            supplied.dtype != np.float64
            or not np.all(np.isfinite(supplied))
            or np.any(supplied < 0)
        ):
            raise ValueError("pair_response must be finite nonnegative float64")

    # Maximize W^2=(1-1/s)*((x+y)^2-4s), 1<=s<=xy, at s=clip((x+y)/2).
    # Rationalize the lower endpoint to avoid subtracting nearly equal numbers.
    x, y = scaled[:, None], scaled[None, :]
    product = x * y
    active = product > 1.0
    total = np.where(active, x + y, 2.0)
    s = np.minimum(np.maximum(total / 2.0, 1.0), np.maximum(product, 1.0))
    beta = np.sqrt(1.0 - 1.0 / s)
    momentum = np.sqrt(np.maximum((x - y) ** 2 + 4.0 * (product - s), 0.0))
    width = beta * momentum
    lower = np.maximum((4.0 * s + total**2 / s - 4.0) / (2.0 * (total + width)), 1.0)
    upper = (total + width) / 2.0
    eps = 64.0 * np.finfo(np.float64).eps
    lower_mask = active & (lower <= daughter[0] + eps * np.maximum(lower, daughter[0]))
    upper_mask = active & (
        upper >= daughter[-1] - eps * np.maximum(total, daughter[-1])
    )
    return {
        "photon_energy_erg": jnp.asarray(energy, dtype=jnp.float64),
        "lepton_gamma": jnp.asarray(daughter, dtype=jnp.float64),
        "photon_integration_weights_erg": jnp.asarray(energy)
        * _trapezoid_weights(jnp.log(energy)),
        "lepton_integration_weights": _trapezoid_weights(daughter),
        "absorption_kernel_cm2": _angle_averaged_cross_section_jax(
            jnp.asarray(product)
        ),
        "pair_response": jnp.asarray(pair_response, dtype=jnp.float64),
        "lower_support_mask": jnp.asarray(lower_mask),
        "upper_support_mask": jnp.asarray(upper_mask),
    }


def gamma_gamma_total_field_jax(dn_photon_denergy_cm3_erg, response):
    """Return frozen-field gamma-gamma collisions for ONE internal population.

    ``response`` is the fixed-grid pytree from the preparation function above.
    The dynamic ``dn/dE`` is isotropic, blob-comoving, cm^-3 erg^-1. It must
    contain each physical photon once; this function does not add S/L/H or
    infer source identities. All supported energies participate above the
    actual pair threshold. No 1 MeV gate, density closure, SSA, photon escape,
    external reservoir, transport or frame conversion is performed here.

    Alpha is cm^-1; differential photon loss is cm^-3 s^-1 erg^-1; each charge's
    Q is cm^-3 s^-1 dgamma^-1. Integrated rates/powers are per cm^3, not luminosity.
    Powers use TOTAL lepton energy gamma*m_e*c^2, including rest mass.

    The response describes labelled photon populations: it contains c and
    angular/energy weights, but no event deduplication. For the same population
    each charge is 0.5*R[n,n]. Photon loss is c*alpha*n WITHOUT that half.
    Both density factors remain differentiable. Returned moment residuals are
    signed (retained/incident - 1); they include quadrature AND finite-support
    effects and must not be interpreted as actual outside-grid leakage.

    Outside-grid quantities are conservative kinematic bounds computed from
    photon event cells, not inferred from those residuals. Tolerances belong to
    the caller. ``input_valid`` flags nonfinite/negative densities using JAX;
    invalid inputs are not clipped. Shape/x64 checks are static. All dynamic
    arithmetic is pure JAX, including zero-field-safe budgets and bounds.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX gamma-gamma total fields require jax_enable_x64")
    density = jnp.asarray(dn_photon_denergy_cm3_erg, dtype=jnp.float64)
    energy = response["photon_energy_erg"]
    if density.shape != energy.shape:
        raise ValueError("photon density must have the prepared photon-grid shape")
    weights = response["photon_integration_weights_erg"]
    weighted = density * weights
    kernel = response["absorption_kernel_cm2"]
    alpha = kernel @ weighted
    loss = _C_CGS * alpha * density
    cells = 0.5 * _C_CGS * kernel * weighted[:, None] * weighted[None, :]
    events = jnp.sum(cells)
    charge_source = (
        0.5 * _contract_pair_response(response["pair_response"], density, density)[0]
    )
    daughter = response["lepton_gamma"]
    daughter_weights = response["lepton_integration_weights"]
    charge_rate = jnp.sum(charge_source * daughter_weights)
    pair_power = 2.0 * _MEC2_CGS * jnp.sum(daughter * charge_source * daughter_weights)
    photon_rate = jnp.sum(loss * weights)
    photon_power = jnp.sum(energy * loss * weights)
    incident_energy = energy[:, None] + energy[None, :]
    lower = response["lower_support_mask"]
    upper = response["upper_support_mask"]
    lower_number = jnp.sum(cells * lower)
    upper_number = jnp.sum(cells * upper)
    lower_power_cells = (
        cells * lower * jnp.minimum(2.0 * _MEC2_CGS * daughter[0], incident_energy)
    )
    upper_power_cells = cells * upper * incident_energy
    outside_number = jnp.sum(cells * (lower | upper))
    outside_power = jnp.sum(
        jnp.minimum(lower_power_cells + upper_power_cells, cells * incident_energy)
    )

    def fraction(numerator, denominator):
        safe = jnp.where(denominator > 0.0, denominator, 1.0)
        return jnp.where(denominator > 0.0, numerator / safe, 0.0)

    return {
        "absorption_coefficient_cm1": alpha,
        "photon_loss_rate_cm3_s_erg": loss,
        "electron_pair_injection_dgamma_cm3_s": charge_source,
        "positron_pair_injection_dgamma_cm3_s": charge_source,
        "event_rate_cm3_s": events,
        "photon_loss_number_rate_cm3_s": photon_rate,
        "photon_loss_power_erg_cm3_s": photon_power,
        "electron_injection_rate_cm3_s": charge_rate,
        "positron_injection_rate_cm3_s": charge_rate,
        "pair_injection_power_erg_cm3_s": pair_power,
        "pair_number_relative_residual": fraction(charge_rate - events, events),
        "pair_energy_relative_residual": fraction(
            pair_power - photon_power, photon_power
        ),
        "lower_outside_per_charge_number_rate_bound_cm3_s": lower_number,
        "upper_outside_per_charge_number_rate_bound_cm3_s": upper_number,
        "lower_outside_pair_power_bound_erg_cm3_s": jnp.sum(lower_power_cells),
        "upper_outside_pair_power_bound_erg_cm3_s": jnp.sum(upper_power_cells),
        "outside_per_charge_number_rate_bound_cm3_s": outside_number,
        "outside_pair_power_bound_erg_cm3_s": outside_power,
        "outside_number_fraction_bound": fraction(outside_number, events),
        "outside_energy_fraction_bound": fraction(outside_power, photon_power),
        "input_valid": jnp.all(jnp.isfinite(density) & (density >= 0.0)),
    }
