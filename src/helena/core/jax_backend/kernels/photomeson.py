"""JAX Hümmer Sim-B isotropic photomeson rate kernels."""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    GEV_TO_ERG as _GEV_CGS,
    MICROBARN_TO_CM2 as _MICROBARN_CGS,
    PHOTOMESON_MULTIPION_INELASTICITY as _MULTIPION_INELASTICITY,
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.contracts import real_1d as _real_1d
from helena.core.tables import (
    PHOTOMESON_DIRECT_INELASTICITY as _DIRECT_INELASTICITY,
    PHOTOMESON_MULTIPION_COMPONENT_CROSS_SECTION_MICROBARN as _MULTIPION_COMPONENT_CROSS_SECTION_MICROBARN,
    PHOTOMESON_MULTIPION_COMPONENT_MAXIMUM_GEV as _MULTIPION_COMPONENT_MAXIMUM_GEV,
    PHOTOMESON_MULTIPION_COMPONENT_MINIMUM_GEV as _MULTIPION_COMPONENT_MINIMUM_GEV,
    PHOTOMESON_PION_COMPONENT_CHI as _PION_COMPONENT_CHI,
    PHOTOMESON_PION_COMPONENT_MULTIPLICITY as _PION_COMPONENT_MULTIPLICITY,
    PHOTOMESON_RESONANCE_INELASTICITY as _RESONANCE_INELASTICITY,
    PION_TO_PROTON_MASS_RATIOS as _PION_TO_PROTON_MASS_RATIO,
)

__all__ = [
    "photomeson_interaction_rate_jax",
    "photomeson_energy_loss_rate_jax",
    "photomeson_cooling_time_jax",
]


def _box_response(y_gev, minimum_gev, maximum_gev, cross_section_microbarn):
    y = jnp.asarray(y_gev, dtype=jnp.float64)
    two_y = 2.0 * y
    active = two_y >= minimum_gev
    contributes = two_y > minimum_gev
    upper = jnp.minimum(two_y, maximum_gev)
    y_safe = jnp.where(active, y, 1.0)
    response = (
        cross_section_microbarn * (upper**2 - minimum_gev**2) / (2.0 * y_safe) ** 2
    )
    return jnp.where(contributes, response, 0.0)


def _resonance_responses(y_gev):
    return jnp.stack(
        [
            _box_response(y_gev, 0.2, 0.5, 200.0),
            _box_response(y_gev, 0.5, 1.2, 90.0),
        ]
    )


def _direct_integral_t1(epsilon_r_gev):
    epsilon = jnp.asarray(epsilon_r_gev, dtype=jnp.float64)
    x = jnp.log10(epsilon / 2.0)
    low = (((40.4699 * x + 102.728) * x + 110.765) * x + 84.0859) * x + 35.9533
    high = (((0.025044 * x - 0.387884) * x + 2.03074) * x + 40.5478) * x + 30.2004
    return jnp.where(epsilon < 0.17, 0.0, jnp.where(epsilon < 0.96, low, high))


def _direct_integral_t2(epsilon_r_gev):
    epsilon = jnp.asarray(epsilon_r_gev, dtype=jnp.float64)
    value = -3.4083 + 16.2864 / epsilon + 40.7160 * jnp.log(epsilon)
    return jnp.where(epsilon < 0.4, 0.0, value)


def _direct_response(y_gev, minimum_gev, maximum_gev, integral):
    y = jnp.asarray(y_gev, dtype=jnp.float64)
    two_y = 2.0 * y
    active = two_y >= minimum_gev
    contributes = two_y > minimum_gev
    upper = jnp.minimum(two_y, maximum_gev)
    upper_safe = jnp.where(active, upper, minimum_gev)
    y_safe = jnp.where(active, y, 1.0)
    response = (integral(upper_safe) - integral(minimum_gev)) / (2.0 * y_safe**2)
    return jnp.where(contributes, response, 0.0)


def _direct_responses(y_gev):
    return jnp.stack(
        [
            _direct_response(y_gev, 0.17, 0.56, _direct_integral_t1),
            _direct_response(y_gev, 0.56, 10.0, _direct_integral_t1),
            _direct_response(y_gev, 10.0, jnp.inf, _direct_integral_t1),
            _direct_response(y_gev, 0.4, 1.58, _direct_integral_t2),
            _direct_response(y_gev, 1.58, 10.0, _direct_integral_t2),
            _direct_response(y_gev, 10.0, jnp.inf, _direct_integral_t2),
        ]
    )


def _pion_component_response(component, y_gev):
    """Return one JAX Sim-B pion-component response in microbarn."""
    if component == 0:
        return _box_response(y_gev, 0.2, 0.5, 200.0)
    if component == 1:
        return _box_response(y_gev, 0.5, 1.2, 90.0)
    if component < 5:
        limits = ((0.17, 0.56), (0.56, 10.0), (10.0, jnp.inf))
        minimum, maximum = limits[component - 2]
        return _direct_response(y_gev, minimum, maximum, _direct_integral_t1)
    if component < 9:
        limits = ((0.4, 1.58), (1.58, 10.0), (10.0, jnp.inf), (0.4, jnp.inf))
        minimum, maximum = limits[component - 5]
        return _direct_response(y_gev, minimum, maximum, _direct_integral_t2)

    multipion_component = component - 9
    return _box_response(
        y_gev,
        _MULTIPION_COMPONENT_MINIMUM_GEV[multipion_component],
        _MULTIPION_COMPONENT_MAXIMUM_GEV[multipion_component],
        _MULTIPION_COMPONENT_CROSS_SECTION_MICROBARN[multipion_component],
    )


def _multipion_response(y_gev):
    y = jnp.asarray(y_gev, dtype=jnp.float64)
    two_y = 2.0 * y
    active = two_y >= 0.5
    x = jnp.log10(jnp.where(active, y, 0.25))
    low = (((67.2251 * x - 59.6965) * x - 98.4187) * x + 120.894) * x + 87.5538
    high = (((0.0493614 * x - 0.858307) * x + 10.612) * x - 25.3296) * x + 131.839
    return jnp.where(active, jnp.where(two_y <= 14.0, low, high), 0.0)


def _event_response(y_gev, *, energy_weighted):
    resonance = _resonance_responses(y_gev)
    direct = _direct_responses(y_gev)
    multipion = _multipion_response(y_gev)
    if energy_weighted:
        return (
            jnp.tensordot(jnp.asarray(_RESONANCE_INELASTICITY), resonance, axes=(0, 0))
            + jnp.tensordot(jnp.asarray(_DIRECT_INELASTICITY), direct, axes=(0, 0))
            + _MULTIPION_INELASTICITY * multipion
        )
    return jnp.sum(resonance, axis=0) + jnp.sum(direct, axis=0) + multipion


def _photomeson_rate_kernel(
    proton_gamma,
    target_photon_energy_erg,
    target_photon_density,
    *,
    energy_weighted,
):
    gamma = jnp.asarray(proton_gamma, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    target_density = jnp.asarray(target_photon_density, dtype=jnp.float64)
    y_gev = gamma[:, None] * target_energy[None, :] / _GEV_CGS
    response_cm2 = (
        _event_response(y_gev, energy_weighted=energy_weighted) * _MICROBARN_CGS
    )
    integrand = target_energy[None, :] * target_density[None, :] * response_cm2
    intervals = (
        jnp.diff(jnp.log(target_energy)) * (integrand[:, 1:] + integrand[:, :-1]) / 2.0
    )
    return _C_CGS * jnp.sum(intervals, axis=1)


def _pion_component_injection_kernel(
    pion_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    output_gamma = jnp.asarray(pion_gamma, dtype=jnp.float64)
    proton_grid = jnp.asarray(proton_gamma, dtype=jnp.float64)
    proton_density = jnp.asarray(dn_proton_dgamma_cm3, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    target_density = jnp.asarray(dn_target_photon_denergy_cm3_erg, dtype=jnp.float64)

    multiplicity = jnp.asarray(_PION_COMPONENT_MULTIPLICITY, dtype=jnp.float64)
    active = multiplicity > 0.0
    chi = jnp.asarray(_PION_COMPONENT_CHI, dtype=jnp.float64)
    chi_safe = jnp.where(active, chi, 1.0)
    mass_ratio = jnp.asarray(_PION_TO_PROTON_MASS_RATIO, dtype=jnp.float64)
    jacobian = jnp.where(active, mass_ratio[:, None] / chi_safe, 0.0)

    mapped_gamma = jacobian[:, :, None] * output_gamma[None, None, :]
    supported = (
        active[:, :, None]
        & (mapped_gamma >= proton_grid[0])
        & (mapped_gamma <= proton_grid[-1])
    )
    mapped_safe = jnp.where(supported, mapped_gamma, proton_grid[0])
    mapped_density = jnp.interp(
        jnp.log(mapped_safe), jnp.log(proton_grid), proton_density
    )
    mapped_density = jnp.where(supported, mapped_density, 0.0)

    y_gev = mapped_safe[:, :, :, None] * target_energy[None, None, None, :] / _GEV_CGS
    response_cm2 = (
        jnp.stack(
            [
                _pion_component_response(component, y_gev[:, component])
                for component in range(_PION_COMPONENT_MULTIPLICITY.shape[1])
            ],
            axis=1,
        )
        * _MICROBARN_CGS
    )
    integrand = (
        target_energy[None, None, None, :]
        * target_density[None, None, None, :]
        * response_cm2
    )
    intervals = (
        jnp.diff(jnp.log(target_energy))[None, None, None, :]
        * (integrand[:, :, :, 1:] + integrand[:, :, :, :-1])
        / 2.0
    )
    component_rate = _C_CGS * jnp.sum(intervals, axis=3)
    return (
        jacobian[:, :, None]
        * multiplicity[:, :, None]
        * mapped_density
        * component_rate
    )


def _photomeson_pion_injection_kernel(
    pion_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    components = _pion_component_injection_kernel(
        pion_gamma,
        proton_gamma,
        dn_proton_dgamma_cm3,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    injection = jnp.sum(components, axis=1)
    return injection[0], injection[1], injection[2]


def _validated_rate(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    energy_weighted,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX photomeson rates require jax_enable_x64")

    arguments = (
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _photomeson_rate_kernel(*arguments, energy_weighted=energy_weighted)

    gamma = _real_1d("proton_gamma", proton_gamma, minimum_size=1, positive=True)
    if np.any(gamma < 1.0):
        raise ValueError("proton_gamma must contain only values >= 1")
    target_energy = _real_1d(
        "target_photon_energy_erg",
        target_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(target_energy) <= 0.0):
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    target_density = _real_1d(
        "dn_target_photon_denergy_cm3_erg",
        dn_target_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=target_energy.shape,
    )
    if not np.any(target_density):
        return jnp.zeros(gamma.shape, dtype=jnp.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        maximum_two_y = 2.0 * np.max(gamma) * np.max(target_energy) / _GEV_CGS
    if not np.isfinite(maximum_two_y):
        raise FloatingPointError(
            "proton and target-photon energy product exceeds the finite float64 range"
        )

    result = _photomeson_rate_kernel(
        gamma,
        target_energy,
        target_density,
        energy_weighted=energy_weighted,
    )
    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(result)
    ):
        return result
    result_array = np.asarray(result)
    if not np.all(np.isfinite(result_array)) or np.any(result_array < 0.0):
        raise FloatingPointError("photomeson rate exceeds the finite float64 range")
    return result


def photomeson_pion_injection_jax(
    pion_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return JAX charge-resolved pion injection from blob-frame protons.

    ``pion_gamma`` is a finite one-dimensional array with values ``>= 1`` in
    any order. ``proton_gamma`` is a finite, strictly increasing grid with at
    least two values ``>= 1``; matching ``dn_proton_dgamma_cm3`` is a finite
    non-negative proton density per unit Lorentz factor in cm^-3. The target
    is a positive strictly increasing photon-energy grid in erg and matching
    finite non-negative isotropic or angle-averaged ``dn_ph/dE`` in
    cm^-3 erg^-1. All quantities are blob-comoving.

    The fixed float64 return order is ``(q_pi0, q_pi_plus, q_pi_minus)`` in
    cm^-3 s^-1 per unit pion Lorentz factor. It mirrors the NumPy Sim-B
    implementation: LR/HR, every direct component including T2b, and all
    multipion L/H components retain their existing charge, multiplicity,
    ``chi``, mass-ratio, logarithmic proton-grid interpolation, and exact
    support conventions. Eager calls validate inputs; JIT-traced calls require
    already-valid fixed-shape inputs. The ``beta_p ~= 1`` approximation is
    physically intended for mapped ``proton_gamma >> 1``.

    This kernel accepts a prescribed steady proton density. It does not solve
    proton or pion transport and includes no pion decay, downstream daughter
    injection, frame transformation, cascade, or observer physics.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX photomeson pion injection requires jax_enable_x64")

    arguments = (
        pion_gamma,
        proton_gamma,
        dn_proton_dgamma_cm3,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if any(isinstance(value, jax.core.Tracer) for value in arguments):
        return _photomeson_pion_injection_kernel(*arguments)

    output_gamma = _real_1d("pion_gamma", pion_gamma, minimum_size=1, positive=True)
    if np.any(output_gamma < 1.0):
        raise ValueError("pion_gamma must contain only values >= 1")
    proton_grid = _real_1d("proton_gamma", proton_gamma, minimum_size=2, positive=True)
    if np.any(proton_grid < 1.0):
        raise ValueError("proton_gamma must contain only values >= 1")
    if np.any(np.diff(proton_grid) <= 0.0):
        raise ValueError("proton_gamma must be strictly increasing")
    proton_density = _real_1d(
        "dn_proton_dgamma_cm3",
        dn_proton_dgamma_cm3,
        minimum_size=2,
        positive=False,
        shape=proton_grid.shape,
    )
    target_energy = _real_1d(
        "target_photon_energy_erg",
        target_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(target_energy) <= 0.0):
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    target_density = _real_1d(
        "dn_target_photon_denergy_cm3_erg",
        dn_target_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=target_energy.shape,
    )

    if not np.any(proton_density) or not np.any(target_density):
        zero = jnp.zeros(output_gamma.shape, dtype=jnp.float64)
        return zero, zero, zero

    active = _PION_COMPONENT_MULTIPLICITY > 0.0
    jacobian = np.zeros_like(_PION_COMPONENT_CHI)
    jacobian[active] = (
        np.broadcast_to(_PION_TO_PROTON_MASS_RATIO[:, None], active.shape)[active]
        / _PION_COMPONENT_CHI[active]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        mapped_gamma = jacobian[:, :, None] * output_gamma[None, None, :]
    if not np.all(np.isfinite(mapped_gamma[active])):
        raise FloatingPointError(
            "pion-to-proton Lorentz-factor mapping exceeds the finite float64 range"
        )
    supported = (
        active[:, :, None]
        & (mapped_gamma >= proton_grid[0])
        & (mapped_gamma <= proton_grid[-1])
    )
    if np.any(supported):
        with np.errstate(over="ignore", invalid="ignore"):
            maximum_two_y = (
                2.0 * np.max(mapped_gamma[supported]) * np.max(target_energy) / _GEV_CGS
            )
        if not np.isfinite(maximum_two_y):
            raise FloatingPointError(
                "proton and target-photon energy product exceeds the finite "
                "float64 range"
            )

    result = _photomeson_pion_injection_kernel(
        output_gamma,
        proton_grid,
        proton_density,
        target_energy,
        target_density,
    )
    result_array = np.asarray(result)
    if not np.all(np.isfinite(result_array)) or np.any(result_array < 0.0):
        raise FloatingPointError(
            "photomeson pion injection exceeds the finite float64 range"
        )
    return result


def photomeson_interaction_rate_jax(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the JAX isotropic Sim-B photomeson event rate in s^-1.

    Inputs are blob-comoving ``proton_gamma >= 1``, a finite positive strictly
    increasing photon-energy grid in erg, and matching finite non-negative
    isotropic or angle-averaged ``dn_ph/dE`` in cm^-3 erg^-1. The float64 JAX
    result follows the proton-grid order.

    The Hümmer et al. Sim-B ``beta_p ~= 1`` response is physically intended
    for ``proton_gamma >> 1``. It counts each LR/HR, T1/T2a, or total
    multipion event once; T2b and pion multiplicities are not extra events.
    Eager calls validate inputs, while JIT-traced calls require valid inputs.
    No frame transform, pion injection, transport, or observer physics is
    included.
    """
    return _validated_rate(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        energy_weighted=False,
    )


def photomeson_energy_loss_rate_jax(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the JAX fractional photomeson energy-loss rate in s^-1.

    Inputs and blob-frame conventions match
    :func:`photomeson_interaction_rate_jax`. The Sim-B event responses use the
    existing resonance/direct inelasticities and ``K_Multi-pi ~= 0.6``. This
    is total initial-nucleon energy transfer, not a species-resolved proton
    loss, and each event is counted once. The float64 JAX result follows
    ``proton_gamma``; eager calls validate values and traced calls require
    already-valid fixed-shape inputs.

    No charge-resolved pion production, proton-neutron conversion, transport,
    cascade, frame transform, or observer physics is included.
    """
    return _validated_rate(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        energy_weighted=True,
    )


def photomeson_cooling_time_jax(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the blob-comoving JAX photomeson cooling time in seconds.

    Inputs, physical range, and Sim-B assumptions match
    :func:`photomeson_energy_loss_rate_jax`. The result is the reciprocal
    total initial-nucleon fractional loss rate and is positive infinity where
    that rate is exactly zero. It is not a species-resolved proton cooling
    time and includes no pion injection, transport, or frame transformation.
    """
    loss_rate = photomeson_energy_loss_rate_jax(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    positive = loss_rate > 0.0
    safe_loss_rate = jnp.where(positive, loss_rate, 1.0)
    return jnp.where(positive, 1.0 / safe_loss_rate, jnp.inf)
