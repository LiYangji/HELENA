"""Isotropic photomeson rates and charge-resolved pion injection."""

import numpy as np
from scipy.integrate import trapezoid

from helena.core.constants import (
    GEV_TO_ERG as _GEV_CGS,
    MICROBARN_TO_CM2 as _MICROBARN_CGS,
    PHOTOMESON_MULTIPION_INELASTICITY as _MULTIPION_INELASTICITY,
    PROTON_REST_ENERGY_CGS as _PROTON_REST_ENERGY_VALUE_CGS,
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
    PION_REST_ENERGIES_CGS as _PION_REST_ENERGY_VALUES_CGS,
    PION_TO_PROTON_MASS_RATIOS as _PION_TO_PROTON_MASS_RATIO,
)

__all__ = [
    "photomeson_interaction_rate",
    "photomeson_energy_loss_rate",
    "photomeson_cooling_time",
    "photomeson_pion_injection",
]

_PROTON_REST_ENERGY_CGS = _PROTON_REST_ENERGY_VALUE_CGS
_PION_REST_ENERGY_CGS = _PION_REST_ENERGY_VALUES_CGS


def _box_response(y_gev, minimum_gev, maximum_gev, cross_section_microbarn):
    """Return Eq. (27) for a constant cross section over one PRF-energy box."""
    y = np.asarray(y_gev, dtype=float)
    two_y = 2.0 * y
    active = two_y >= minimum_gev
    upper = np.minimum(two_y, maximum_gev)
    y_safe = np.where(active, y, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        response = (
            cross_section_microbarn * (upper**2 - minimum_gev**2) / (2.0 * y_safe) ** 2
        )
    return np.where(active, response, 0.0)


def _resonance_responses(y_gev):
    """Return the Sim-B LR and HR event responses, in microbarn."""
    return np.stack(
        [
            _box_response(y_gev, 0.2, 0.5, 200.0),
            _box_response(y_gev, 0.5, 1.2, 90.0),
        ]
    )


def _direct_integral_t1(epsilon_r_gev):
    epsilon = np.asarray(epsilon_r_gev, dtype=float)
    x = np.log10(epsilon / 2.0)
    low = (((40.4699 * x + 102.728) * x + 110.765) * x + 84.0859) * x + 35.9533
    high = (((0.025044 * x - 0.387884) * x + 2.03074) * x + 40.5478) * x + 30.2004
    return np.where(epsilon < 0.17, 0.0, np.where(epsilon < 0.96, low, high))


def _direct_integral_t2(epsilon_r_gev):
    epsilon = np.asarray(epsilon_r_gev, dtype=float)
    value = -3.4083 + 16.2864 / epsilon + 40.7160 * np.log(epsilon)
    return np.where(epsilon < 0.4, 0.0, value)


def _direct_response(y_gev, minimum_gev, maximum_gev, integral):
    y = np.asarray(y_gev, dtype=float)
    two_y = 2.0 * y
    active = two_y >= minimum_gev
    upper = np.minimum(two_y, maximum_gev)
    upper_safe = np.where(active, upper, minimum_gev)
    y_safe = np.where(active, y, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        response = (integral(upper_safe) - integral(minimum_gev)) / (2.0 * y_safe**2)
    return np.where(active, response, 0.0)


def _direct_responses(y_gev):
    """Return direct T1 and event-counted T2a responses, in microbarn."""
    return np.stack(
        [
            _direct_response(y_gev, 0.17, 0.56, _direct_integral_t1),
            _direct_response(y_gev, 0.56, 10.0, _direct_integral_t1),
            _direct_response(y_gev, 10.0, np.inf, _direct_integral_t1),
            _direct_response(y_gev, 0.4, 1.58, _direct_integral_t2),
            _direct_response(y_gev, 1.58, 10.0, _direct_integral_t2),
            _direct_response(y_gev, 10.0, np.inf, _direct_integral_t2),
        ]
    )


def _pion_component_response(component, y_gev):
    """Return one Sim-B pion-component response in microbarn."""
    if component == 0:
        return _box_response(y_gev, 0.2, 0.5, 200.0)
    if component == 1:
        return _box_response(y_gev, 0.5, 1.2, 90.0)
    if component < 5:
        limits = ((0.17, 0.56), (0.56, 10.0), (10.0, np.inf))
        minimum, maximum = limits[component - 2]
        return _direct_response(y_gev, minimum, maximum, _direct_integral_t1)
    if component < 9:
        limits = ((0.4, 1.58), (1.58, 10.0), (10.0, np.inf), (0.4, np.inf))
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
    """Return the total-cross-section multipion response of Eq. (38)."""
    y = np.asarray(y_gev, dtype=float)
    two_y = 2.0 * y
    active = two_y >= 0.5
    x = np.log10(np.where(active, y, 0.25))
    low = (((67.2251 * x - 59.6965) * x - 98.4187) * x + 120.894) * x + 87.5538
    high = (((0.0493614 * x - 0.858307) * x + 10.612) * x - 25.3296) * x + 131.839
    return np.where(active, np.where(two_y <= 14.0, low, high), 0.0)


def _event_response(y_gev, *, energy_weighted):
    resonance = _resonance_responses(y_gev)
    direct = _direct_responses(y_gev)
    multipion = _multipion_response(y_gev)
    if energy_weighted:
        return (
            np.tensordot(_RESONANCE_INELASTICITY, resonance, axes=(0, 0))
            + np.tensordot(_DIRECT_INELASTICITY, direct, axes=(0, 0))
            + _MULTIPION_INELASTICITY * multipion
        )
    return np.sum(resonance, axis=0) + np.sum(direct, axis=0) + multipion


def _photomeson_rate(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    energy_weighted,
):
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

    result = np.zeros(gamma.shape, dtype=float)
    if not np.any(target_density):
        return result

    log_target_energy = np.log(target_energy)
    target_weight = target_energy * target_density
    for index, gamma_value in enumerate(gamma):
        with np.errstate(over="ignore", invalid="ignore"):
            y_gev = gamma_value * target_energy / _GEV_CGS
            two_y_gev = 2.0 * y_gev
        if not np.all(np.isfinite(two_y_gev)):
            raise FloatingPointError(
                "proton and target-photon energy product exceeds the finite "
                "float64 range"
            )
        response_cm2 = (
            _event_response(y_gev, energy_weighted=energy_weighted) * _MICROBARN_CGS
        )
        result[index] = _C_CGS * trapezoid(
            target_weight * response_cm2,
            x=log_target_energy,
        )

    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("photomeson rate exceeds the finite float64 range")
    return result


def photomeson_interaction_rate(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the isotropic photomeson event rate in the blob frame.

    ``proton_gamma`` is a finite one-dimensional array with values ``>= 1``;
    the numerical API accepts this full range, but the ``beta_p ~= 1`` response
    is physically intended for ``proton_gamma >> 1``. Target photon energies
    are a strictly increasing positive grid in erg, and the matching finite
    non-negative, angle-integrated isotropic or
    angle-averaged ``dn_ph/dE`` is in cm^-3 erg^-1. The returned float64 array
    follows the proton input order and has units s^-1.

    The calculation uses Hümmer et al. (2010) Sim-B with ``beta_p ~= 1``:
    lower and higher resonances, direct one- and two-pion production, and the
    total-cross-section multipion response. Each physical event is counted
    once; pion multiplicities and the second pion component T2b are not extra
    interactions. All inputs must already be in the blob comoving frame.
    No frame transform, secondary production, transport, cascade, or JAX
    calculation is performed.
    """
    return _photomeson_rate(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        energy_weighted=False,
    )


def photomeson_energy_loss_rate(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the initial nucleon's total fractional photomeson loss rate.

    Inputs have the same valid ranges, blob-comoving frame convention, and
    units as :func:`photomeson_interaction_rate`; ``proton_gamma >= 1`` is
    accepted numerically, while the ``beta_p ~= 1`` response is physically
    intended for ``proton_gamma >> 1``. The output is the energy transferred
    from the initial high-energy nucleon to secondaries per unit nucleon energy
    and time, in s^-1, and follows ``proton_gamma``. Sim-B event responses are
    weighted by the Table 4 and 5 inelasticities and by
    ``K_Multi-pi ~= 0.6`` from Appendix B. Inelasticity includes all pions from
    an event, while that event is counted only once.

    The calculation does not distinguish whether the leading final-state
    nucleon is a proton or neutron. It is not a species-resolved proton
    transport cooling term, and proton-to-neutron conversion is not treated as
    escape or neutron reinjection. No secondary spectrum or transport is
    included.
    """
    return _photomeson_rate(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
        energy_weighted=True,
    )


def photomeson_cooling_time(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the blob-comoving photomeson cooling time in seconds.

    Inputs are the same finite ``proton_gamma >= 1``, strictly increasing
    photon-energy grid in erg, and non-negative isotropic blob-frame
    ``dn_ph/dE`` in cm^-3 erg^-1 accepted by
    :func:`photomeson_energy_loss_rate`. The API accepts ``proton_gamma >= 1``,
    but the Sim-B ``beta_p ~= 1`` response is physically intended for
    ``proton_gamma >> 1``. This cooling time is the reciprocal total
    initial-nucleon fractional energy-loss rate and is ``numpy.inf`` where that
    rate is zero; it is not a species-resolved proton transport cooling time.
    No frame transformation, neutron escape or reinjection, secondary physics,
    transport, cascade, fitting, or JAX backend is included.
    """
    loss_rate = photomeson_energy_loss_rate(
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    cooling_time = np.full(loss_rate.shape, np.inf, dtype=float)
    np.divide(1.0, loss_rate, out=cooling_time, where=loss_rate > 0.0)
    return cooling_time


def photomeson_pion_injection(
    pion_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return charge-resolved pion injection from initial blob-frame protons.

    ``pion_gamma`` is a finite one-dimensional array with values ``>= 1`` and
    may be in any order. ``proton_gamma`` is a finite, strictly increasing
    one-dimensional grid with at least two values ``>= 1``; its matching
    ``dn_proton_dgamma_cm3`` is finite, non-negative, and in cm^-3 per unit
    proton Lorentz factor. Target photon energies are a strictly increasing
    positive grid in erg, with finite non-negative isotropic or angle-averaged
    ``dn_ph/dE`` in cm^-3 erg^-1. All inputs are in the blob comoving frame.

    The fixed return order is ``(q_pi0, q_pi_plus, q_pi_minus)``. Each float64
    array follows ``pion_gamma`` and is the local injection rate per unit pion
    Lorentz factor, in cm^-3 s^-1. The Hümmer et al. (2010) Sim-B
    ``beta_p ~= 1`` response includes LR/HR, all Table 5 direct components
    (including the second-pion T2b component), and all Table 6 multipion L/H
    components for an initial proton. Proton density is interpolated linearly
    in ``ln(proton_gamma)`` without extrapolation; values mapped outside the
    supplied proton support are exactly zero.

    No pion evolution, cooling, decay, downstream secondary spectrum,
    neutron-initiated interaction, transport, cascade, fitting, frame
    transform, or JAX backend is included. The ultrarelativistic response is
    physically intended for mapped ``proton_gamma >> 1``.
    """
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

    injection = np.zeros((3, output_gamma.size), dtype=float)
    if not np.any(proton_density) or not np.any(target_density):
        return tuple(injection)

    log_proton_grid = np.log(proton_grid)
    log_target_energy = np.log(target_energy)
    target_weight = target_energy * target_density
    for species in range(3):
        mass_ratio = _PION_TO_PROTON_MASS_RATIO[species]
        for component, multiplicity in enumerate(_PION_COMPONENT_MULTIPLICITY[species]):
            if multiplicity == 0.0:
                continue
            jacobian = mass_ratio / _PION_COMPONENT_CHI[species, component]
            with np.errstate(over="ignore", invalid="ignore"):
                mapped_gamma = jacobian * output_gamma
            if not np.all(np.isfinite(mapped_gamma)):
                raise FloatingPointError(
                    "pion-to-proton Lorentz-factor mapping exceeds the finite "
                    "float64 range"
                )
            supported = (mapped_gamma >= proton_grid[0]) & (
                mapped_gamma <= proton_grid[-1]
            )
            if not np.any(supported):
                continue

            mapped_supported = mapped_gamma[supported]
            density_supported = np.interp(
                np.log(mapped_supported),
                log_proton_grid,
                proton_density,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                y_gev = (
                    mapped_supported[:, np.newaxis]
                    * target_energy[np.newaxis, :]
                    / _GEV_CGS
                )
                two_y_gev = 2.0 * y_gev
            if not np.all(np.isfinite(two_y_gev)):
                raise FloatingPointError(
                    "proton and target-photon energy product exceeds the finite "
                    "float64 range"
                )
            response_cm2 = _pion_component_response(component, y_gev) * _MICROBARN_CGS
            component_rate = _C_CGS * trapezoid(
                target_weight[np.newaxis, :] * response_cm2,
                x=log_target_energy,
                axis=1,
            )
            injection[species, supported] += (
                jacobian * multiplicity * density_supported * component_rate
            )

    if not np.all(np.isfinite(injection)) or np.any(injection < 0.0):
        raise FloatingPointError(
            "photomeson pion injection exceeds the finite float64 range"
        )
    return tuple(injection)
