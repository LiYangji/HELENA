"""JAX cold-target proton--proton pion production and proton loss."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    GEV_TO_ERG as _GEV_CGS,
    PROTON_REST_ENERGY_CGS as _PROTON_REST_ENERGY_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.contracts import real_1d as _real_1d
from helena.core.tables import (
    PION_REST_ENERGIES_CGS as _PION_REST_ENERGY_CGS,
    PION_TO_PROTON_MASS_RATIOS as _PION_TO_PROTON_MASS_RATIO,
)

__all__ = [
    "PP_TRANSITION_KINETIC_ENERGY_GEV",
    "proton_proton_inelastic_cross_section_jax",
    "proton_proton_pion_response_jax",
    "proton_proton_fractional_energy_loss_rate_jax",
    "proton_proton_pion_injection_and_loss_jax",
]


PP_TRANSITION_KINETIC_ENERGY_GEV = 100.0
_PP_TRANSITION_END_KINETIC_ENERGY_GEV = 200.0
_PP_MINIMUM_SIBYLL_X = 1.0e-3
_MILLIBARN_CGS = 1.0e-27
_PROTON_REST_ENERGY_GEV = _PROTON_REST_ENERGY_CGS / _GEV_CGS
_PION_REST_ENERGY_GEV = _PION_REST_ENERGY_CGS / _GEV_CGS
_KERNEL_CONFIGURATION = (
    "kelner_2006_sibyll_hybrid",
    PP_TRANSITION_KINETIC_ENERGY_GEV,
    _PP_TRANSITION_END_KINETIC_ENERGY_GEV,
    _PP_MINIMUM_SIBYLL_X,
)


def _trapezoid_weights(coordinate):
    widths = np.diff(coordinate)
    return np.concatenate(
        (widths[:1] / 2.0, (widths[:-1] + widths[1:]) / 2.0, widths[-1:] / 2.0)
    )


def _inelastic_cross_section_cm2(proton_gamma, pion_mass_gev):
    proton = np.asarray(proton_gamma, dtype=np.float64)
    proton_energy_gev = proton * _PROTON_REST_ENERGY_GEV
    threshold_gev = (
        _PROTON_REST_ENERGY_GEV
        + 2.0 * pion_mass_gev
        + pion_mass_gev**2 / (2.0 * _PROTON_REST_ENERGY_GEV)
    )
    active = proton_energy_gev > threshold_gev
    safe_energy = np.where(active, proton_energy_gev, threshold_gev)
    logarithm = np.log(safe_energy / 1.0e3)
    cross_section_mb = (34.3 + 1.88 * logarithm + 0.25 * logarithm**2) * (
        1.0 - (threshold_gev / safe_energy) ** 4
    ) ** 2
    return np.where(active, cross_section_mb * _MILLIBARN_CGS, 0.0)


def _sibyll_pion_redistribution(proton_gamma, x, pion_mass_ratio):
    proton = np.asarray(proton_gamma, dtype=np.float64)
    fraction = np.asarray(x, dtype=np.float64)
    kinetic_energy_gev = (proton - 1.0) * _PROTON_REST_ENERGY_GEV
    active = (
        (kinetic_energy_gev >= PP_TRANSITION_KINETIC_ENERGY_GEV)
        & (fraction >= _PP_MINIMUM_SIBYLL_X)
        & (fraction < 1.0)
        & (fraction * proton >= pion_mass_ratio)
    )
    safe_fraction = np.where(active, fraction, 0.5)
    safe_proton = np.where(
        kinetic_energy_gev >= PP_TRANSITION_KINETIC_ENERGY_GEV,
        proton,
        1.0 + PP_TRANSITION_KINETIC_ENERGY_GEV / _PROTON_REST_ENERGY_GEV,
    )
    logarithm = np.log(safe_proton * _PROTON_REST_ENERGY_GEV / 1.0e3)
    a = 3.67 + 0.83 * logarithm + 0.075 * logarithm**2
    alpha = 0.98 / np.sqrt(a)
    r = 2.6 / np.sqrt(a)
    b_pi = a + 0.25
    x_alpha = safe_fraction**alpha
    one_minus = 1.0 - x_alpha
    denominator = 1.0 + r * x_alpha * one_minus
    shape = (one_minus / denominator) ** 4 * (
        1.0 / one_minus + r * (1.0 - 2.0 * x_alpha) / denominator
    )
    threshold = np.sqrt(
        np.where(active, 1.0 - pion_mass_ratio / (safe_fraction * safe_proton), 0.0)
    )
    result = 4.0 * alpha * b_pi * safe_fraction ** (alpha - 1.0) * shape * threshold
    return np.where(active, result, 0.0)


def _transition_weight(proton_gamma):
    kinetic_energy = (proton_gamma - 1.0) * _PROTON_REST_ENERGY_GEV
    scaled = (kinetic_energy - PP_TRANSITION_KINETIC_ENERGY_GEV) / (
        _PP_TRANSITION_END_KINETIC_ENERGY_GEV - PP_TRANSITION_KINETIC_ENERGY_GEV
    )
    smooth = scaled**2 * (3.0 - 2.0 * scaled)
    return np.where(
        kinetic_energy <= PP_TRANSITION_KINETIC_ENERGY_GEV,
        0.0,
        np.where(kinetic_energy >= _PP_TRANSITION_END_KINETIC_ENERGY_GEV, 1.0, smooth),
    )


def _high_energy_fraction(proton_gamma):
    proton = np.asarray(proton_gamma, dtype=np.float64)
    x = np.geomspace(_PP_MINIMUM_SIBYLL_X, 1.0, 2049)
    fractions = []
    for mass_ratio in _PION_TO_PROTON_MASS_RATIO:
        redistribution = _sibyll_pion_redistribution(
            proton[None, :], x[:, None], mass_ratio
        )
        fractions.append(np.trapezoid(x[:, None] * redistribution, x=x, axis=0))
    return np.asarray(fractions)


def _delta_inelasticity():
    transition_gamma = np.asarray(
        [1.0 + PP_TRANSITION_KINETIC_ENERGY_GEV / _PROTON_REST_ENERGY_GEV]
    )
    energy_fraction = _high_energy_fraction(transition_gamma)[:, 0]
    return energy_fraction * transition_gamma[0] / (transition_gamma[0] - 1.0)


def _delta_response(pion_gamma, proton_gamma, cross_section, delta_inelasticity):
    pion = np.asarray(pion_gamma, dtype=np.float64)
    proton = np.asarray(proton_gamma, dtype=np.float64)
    pion_weights = _trapezoid_weights(pion)
    proton_weights = _trapezoid_weights(proton)
    response = np.zeros((3, pion.size, proton.size), dtype=np.float64)
    for species, mass_ratio in enumerate(_PION_TO_PROTON_MASS_RATIO):
        mapped = delta_inelasticity[species] * (proton - 1.0) / mass_ratio
        for parent, daughter in enumerate(mapped):
            if daughter < pion[0] or daughter > pion[-1]:
                continue
            upper = int(np.searchsorted(pion, daughter, side="right"))
            event_rate = (
                _C_CGS * cross_section[species, parent] * proton_weights[parent]
            )
            if upper == 0:
                response[species, 0, parent] = event_rate / pion_weights[0]
            elif upper == pion.size:
                response[species, -1, parent] = event_rate / pion_weights[-1]
            else:
                lower = upper - 1
                right = (daughter - pion[lower]) / (pion[upper] - pion[lower])
                response[species, lower, parent] = (
                    event_rate * (1.0 - right) / pion_weights[lower]
                )
                response[species, upper, parent] = (
                    event_rate * right / pion_weights[upper]
                )
    return response


def _precompute_proton_proton_kernel(pion_gamma, proton_gamma):
    pion = np.asarray(pion_gamma, dtype=np.float64)
    proton = np.asarray(proton_gamma, dtype=np.float64)
    proton_weights = _trapezoid_weights(proton)
    cross_section = np.asarray(
        [
            _inelastic_cross_section_cm2(proton, pion_mass)
            for pion_mass in _PION_REST_ENERGY_GEV
        ]
    )
    delta_inelasticity = _delta_inelasticity()
    low_response = _delta_response(pion, proton, cross_section, delta_inelasticity)

    x = (
        _PION_TO_PROTON_MASS_RATIO[:, None, None]
        * pion[None, :, None]
        / proton[None, None, :]
    )
    high_response = (
        _C_CGS
        * cross_section[:, None, :]
        * proton_weights[None, None, :]
        * (_PION_TO_PROTON_MASS_RATIO[:, None, None] / proton[None, None, :])
        * _sibyll_pion_redistribution(
            proton[None, None, :],
            x,
            _PION_TO_PROTON_MASS_RATIO[:, None, None],
        )
    )
    weight = _transition_weight(proton)
    response = (1.0 - weight)[None, None, :] * low_response + weight[
        None, None, :
    ] * high_response

    high_fraction = _high_energy_fraction(proton)
    low_fraction = (
        delta_inelasticity[:, None] * (proton - 1.0)[None, :] / proton[None, :]
    )
    energy_fraction = (1.0 - weight)[None, :] * low_fraction + weight[
        None, :
    ] * high_fraction
    fractional_loss_coefficient = _C_CGS * np.sum(
        cross_section * energy_fraction, axis=0
    )
    if (
        not np.all(np.isfinite(response))
        or np.any(response < 0.0)
        or not np.all(np.isfinite(fractional_loss_coefficient))
        or np.any(fractional_loss_coefficient < 0.0)
    ):
        raise FloatingPointError("proton-proton response exceeds float64 range")
    return response, fractional_loss_coefficient


def _grid_key(grid):
    array = np.ascontiguousarray(grid)
    return array.dtype.str, array.shape, array.tobytes()


def _array_from_key(key):
    dtype, shape, values = key
    return np.frombuffer(values, dtype=np.dtype(dtype)).reshape(shape)


@lru_cache(maxsize=3)
def _cached_proton_proton_kernel(pion_key, proton_key, configuration):
    if configuration != _KERNEL_CONFIGURATION:
        raise ValueError("unsupported proton-proton kernel configuration")
    return _precompute_proton_proton_kernel(
        _array_from_key(pion_key), _array_from_key(proton_key)
    )


def _validated_grids(pion_gamma, proton_gamma):
    pion = _real_1d("pion_gamma", pion_gamma, minimum_size=2, positive=True)
    proton = _real_1d("proton_gamma", proton_gamma, minimum_size=2, positive=True)
    if np.any(pion <= 1.0) or np.any(np.diff(pion) <= 0.0):
        raise ValueError("pion_gamma must contain strictly increasing values > 1")
    if np.any(proton <= 1.0) or np.any(np.diff(proton) <= 0.0):
        raise ValueError("proton_gamma must contain strictly increasing values > 1")
    return pion, proton


def proton_proton_inelastic_cross_section_jax(proton_gamma):
    """Return charge-resolved cold-target inelastic pp cross sections in cm^2.

    ``proton_gamma`` is a finite one-dimensional blob-frame Lorentz-factor
    array with values greater than one. The float64 result has shape
    ``(3, n_proton)`` in ``(pi0, pi+, pi-)`` order. It uses Kelner et al.
    (2006) Eq. 79 with the relevant PDG pion mass in the exact total-energy
    threshold. No target density, pion redistribution, transport, or frame
    transform is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX proton-proton physics requires jax_enable_x64")
    proton = _real_1d("proton_gamma", proton_gamma, minimum_size=1, positive=True)
    if np.any(proton <= 1.0):
        raise ValueError("proton_gamma must contain only values > 1")
    return jnp.asarray(
        [
            _inelastic_cross_section_cm2(proton, pion_mass)
            for pion_mass in _PION_REST_ENERGY_GEV
        ],
        dtype=jnp.float64,
    )


def proton_proton_pion_response_jax(pion_gamma, proton_gamma):
    """Return the cached fixed-grid pp pion response in s^-1.

    Inputs are strictly increasing blob-comoving Lorentz-factor grids greater
    than one. The float64 response shape is ``(3, n_pion, n_proton)`` in
    ``(pi0, pi+, pi-)`` order and excludes the cold-proton density. It includes
    linear-grid proton quadrature weights, the Kelner threshold-corrected
    inelastic cross section, the pion/proton mass Jacobian, physical support,
    a Kelner/SIBYLL redistribution for proton kinetic energy at and above
    100 GeV, and the AM3-like delta approximation below it. A fixed smoothstep
    completes the transition from 100 to 200 GeV without renormalization.

    The response is cached only by the two fixed grids and prescription
    constants. Dynamic proton spectra and target density never enter the cache
    key. No pion transport, decay, radiation, or daughter production occurs.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX proton-proton physics requires jax_enable_x64")
    pion, proton = _validated_grids(pion_gamma, proton_gamma)
    response, _ = _cached_proton_proton_kernel(
        _grid_key(pion), _grid_key(proton), _KERNEL_CONFIGURATION
    )
    return jnp.asarray(response, dtype=jnp.float64)


def proton_proton_fractional_energy_loss_rate_jax(
    pion_gamma, proton_gamma, cold_proton_density_cm3
):
    """Return the cold-target pp fractional proton loss rate in s^-1.

    The fixed ``pion_gamma`` and ``proton_gamma`` grids are strictly increasing
    blob-comoving Lorentz factors greater than one; the pion grid selects the
    same cached hybrid response used for production. The target is a finite
    non-negative scalar cold-proton density in cm^-3. The float64 result is
    ``-dot(gamma_p)/gamma_p`` on the proton grid and is the energy moment of
    the delta/SIBYLL prescription, including grid-independent physical support.
    It is differentiable in the target density and includes no proton transport.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX proton-proton physics requires jax_enable_x64")
    if any(isinstance(value, jax.core.Tracer) for value in (pion_gamma, proton_gamma)):
        raise ValueError("proton-proton response grids must be fixed outside tracing")
    pion, proton = _validated_grids(pion_gamma, proton_gamma)
    if not isinstance(cold_proton_density_cm3, jax.core.Tracer):
        target = np.asarray(cold_proton_density_cm3)
        if target.ndim != 0 or target.dtype.kind not in "iuf":
            raise ValueError("cold_proton_density_cm3 must be a real scalar")
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("cold_proton_density_cm3 must be finite and non-negative")
    _, loss_coefficient = _cached_proton_proton_kernel(
        _grid_key(pion), _grid_key(proton), _KERNEL_CONFIGURATION
    )
    return jnp.asarray(cold_proton_density_cm3, dtype=jnp.float64) * jnp.asarray(
        loss_coefficient, dtype=jnp.float64
    )


def proton_proton_pion_injection_and_loss_jax(
    pion_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    cold_proton_density_cm3,
):
    """Return cold-target pp proton loss and charge-resolved pion injection.

    ``pion_gamma`` and ``proton_gamma`` are fixed, strictly increasing
    blob-comoving grids greater than one. ``dn_proton_dgamma_cm3`` is the
    finite non-negative proton density in cm^-3 per proton Lorentz factor and
    ``cold_proton_density_cm3`` is a finite non-negative scalar density of
    stationary target protons in cm^-3.

    The float64 return is ``(t_pp_loss^-1, Q_pi0, Q_pi+, Q_pi-)``. The first
    array follows ``proton_gamma`` and is ``-dot(gamma_p)/gamma_p`` in s^-1;
    pion arrays follow ``pion_gamma`` and are in cm^-3 s^-1 per pion Lorentz
    factor. The loss is the energy moment of the same hybrid redistribution,
    not a fixed-inelasticity approximation. All quantities are blob-comoving.

    Dynamic evaluation is exactly target density times a cached response
    matrix contraction, plus target density times the cached loss coefficient,
    and is differentiable with respect to both dynamic inputs. No dynamic
    result is cached. The SIBYLL redistribution is restricted to ``x >= 1e-3``;
    pion-grid truncation is not repaired by renormalization. No neutron,
    nuclei, kaon, charm, direct gamma, lepton, or neutrino channel is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX proton-proton physics requires jax_enable_x64")
    if any(isinstance(value, jax.core.Tracer) for value in (pion_gamma, proton_gamma)):
        raise ValueError("proton-proton response grids must be fixed outside tracing")
    pion, proton = _validated_grids(pion_gamma, proton_gamma)
    dynamic_traced = any(
        isinstance(value, jax.core.Tracer)
        for value in (dn_proton_dgamma_cm3, cold_proton_density_cm3)
    )
    if not dynamic_traced:
        density = _real_1d(
            "dn_proton_dgamma_cm3",
            dn_proton_dgamma_cm3,
            minimum_size=2,
            positive=False,
            shape=proton.shape,
        )
        target = np.asarray(cold_proton_density_cm3)
        if target.ndim != 0 or target.dtype.kind not in "iuf":
            raise ValueError("cold_proton_density_cm3 must be a real scalar")
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("cold_proton_density_cm3 must be finite and non-negative")
    else:
        density = dn_proton_dgamma_cm3
    response, loss_coefficient = _cached_proton_proton_kernel(
        _grid_key(pion), _grid_key(proton), _KERNEL_CONFIGURATION
    )
    target = jnp.asarray(cold_proton_density_cm3, dtype=jnp.float64)
    injection = target * jnp.matmul(
        jnp.asarray(response, dtype=jnp.float64),
        jnp.asarray(density, dtype=jnp.float64),
    )
    loss = target * jnp.asarray(loss_coefficient, dtype=jnp.float64)
    if not dynamic_traced:
        outputs = (loss, *injection)
        if any(
            not np.all(np.isfinite(np.asarray(value)))
            or np.any(np.asarray(value) < 0.0)
            for value in outputs
        ):
            raise FloatingPointError("proton-proton output exceeds float64 range")
    return loss, injection[0], injection[1], injection[2]
