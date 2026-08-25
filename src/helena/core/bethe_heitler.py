"""Bethe--Heitler pair injection and proton energy loss for isotropic fields."""

from decimal import Decimal, localcontext

import numpy as np
from astropy.constants import alpha, m_e, m_p, sigma_T
from scipy.integrate import trapezoid
from scipy.special import zeta


from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.contracts import real_1d as _real_1d

__all__ = [
    "bethe_heitler_pair_injection",
    "bethe_heitler_proton_energy_loss_rate",
]

_ALPHA = float(alpha.value)
_R_E_SQUARED_CGS = 3.0 * float(sigma_T.cgs.value) / (8.0 * np.pi)
_ELECTRON_TO_PROTON_MASS_RATIO = float(m_e.cgs.value / m_p.cgs.value)
_CROSS_SECTION_SCALE_CGS = _ALPHA * _R_E_SQUARED_CGS
_BETHE_HEITLER_PAIR_OMEGA_MAX_MEC2 = 600.0

_DEFAULT_QUADRATURE_ORDER = 32
_GAUSS_NODE, _GAUSS_WEIGHT = np.polynomial.legendre.leggauss(_DEFAULT_QUADRATURE_ORDER)

_LN2 = np.log(2.0)
_PHI_D = np.array(
    [
        -170.0
        + 84.0 * _LN2
        - 16.0 * _LN2**2
        + np.pi**2 * (10.0 - 4.0 * _LN2) / 3.0
        + 8.0 * zeta(3.0, 1.0),
        88.0 - 40.0 * _LN2 + 8.0 * _LN2**2 - 4.0 * np.pi**2 / 3.0,
        -20.0 + 8.0 * _LN2,
        8.0 / 3.0,
    ]
)


def _blumenthal_scalar_decimal(omega_value, energy_minus_value, cosine_value):
    """Re-evaluate one cancellation-limited Eq. (10) point at high precision."""
    with localcontext() as context:
        context.prec = 50
        one = Decimal(1)
        two = Decimal(2)
        three = Decimal(3)
        four = Decimal(4)
        omega = Decimal.from_float(float(omega_value))
        energy_minus = Decimal.from_float(float(energy_minus_value))
        cosine = Decimal.from_float(float(cosine_value))
        energy_plus = omega - energy_minus
        momentum_minus = ((energy_minus - one) * (energy_minus + one)).sqrt()
        momentum_plus = ((energy_plus - one) * (energy_plus + one)).sqrt()
        sine_squared = (one - cosine) * (one + cosine)
        delta_minus = energy_minus - momentum_minus * cosine
        transfer_squared = momentum_plus**2 + two * omega * delta_minus
        transfer = transfer_squared.sqrt()
        y = (
            two
            / momentum_minus**2
            * (
                (energy_plus * energy_minus + momentum_plus * momentum_minus + one)
                / omega
            ).ln()
        )
        y_plus = two * (energy_plus + momentum_plus).ln() / momentum_plus
        delta_plus_transfer = (
            (transfer + momentum_plus) / (transfer - momentum_plus)
        ).ln()

        terms = (
            -four
            * sine_squared
            * (two * energy_minus**2 + one)
            / (momentum_minus**2 * delta_minus**4),
            (Decimal(5) * energy_minus**2 - two * energy_plus * energy_minus + three)
            / (momentum_minus**2 * delta_minus**2),
            (momentum_minus**2 - omega**2) / (transfer_squared * delta_minus**2),
            two * energy_plus / (momentum_minus**2 * delta_minus),
            y
            / (momentum_minus * momentum_plus)
            * (
                two
                * energy_minus
                * sine_squared
                * (three * omega + momentum_minus**2 * energy_plus)
                / delta_minus**4
                + (
                    two * energy_minus**2 * (energy_minus**2 + energy_plus**2)
                    - Decimal(7) * energy_minus**2
                    - three * energy_plus * energy_minus
                    - energy_plus**2
                    + one
                )
                / delta_minus**2
                + omega
                * (energy_minus**2 - energy_minus * energy_plus - one)
                / delta_minus
            ),
            -delta_plus_transfer
            / (momentum_plus * transfer)
            * (
                two / delta_minus**2
                - three * omega / delta_minus
                - omega
                * (momentum_minus**2 - omega**2)
                / (transfer_squared * delta_minus)
            ),
            -two * y_plus / delta_minus,
        )
        bracket = sum(terms, Decimal(0))
        scale = Decimal.from_float(_CROSS_SECTION_SCALE_CGS)
        return float(
            scale * momentum_minus * momentum_plus / (two * omega**3) * bracket
        )


def _blumenthal_differential_cross_section(
    photon_energy, electron_energy, electron_cosine
):
    """Return Blumenthal (1970) Eq. (10) in cm^2 per dE_- dcos(theta_-)."""
    omega, energy_minus, cosine = np.broadcast_arrays(
        np.asarray(photon_energy, dtype=float),
        np.asarray(electron_energy, dtype=float),
        np.asarray(electron_cosine, dtype=float),
    )
    shape = omega.shape
    omega = omega.ravel()
    energy_minus = energy_minus.ravel()
    cosine = cosine.ravel()
    result = np.zeros(omega.shape, dtype=float)

    energy_plus = omega - energy_minus
    active = (
        (omega > 2.0)
        & (energy_minus > 1.0)
        & (energy_plus > 1.0)
        & (np.abs(cosine) < 1.0)
    )
    if not np.any(active):
        return result.reshape(shape)

    omega = omega[active]
    energy_minus = energy_minus[active]
    energy_plus = energy_plus[active]
    cosine = cosine[active]
    momentum_minus = np.sqrt((energy_minus - 1.0) * (energy_minus + 1.0))
    momentum_plus = np.sqrt((energy_plus - 1.0) * (energy_plus + 1.0))
    sine_squared = (1.0 - cosine) * (1.0 + cosine)
    delta_minus = energy_minus - momentum_minus * cosine
    transfer_squared = momentum_plus**2 + 2.0 * omega * delta_minus
    transfer = np.sqrt(transfer_squared)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        y = (
            2.0
            / momentum_minus**2
            * np.log(
                (energy_plus * energy_minus + momentum_plus * momentum_minus + 1.0)
                / omega
            )
        )
        y_plus = 2.0 * np.arccosh(energy_plus) / momentum_plus
        delta_plus_transfer = 2.0 * np.arctanh(momentum_plus / transfer)

        first = (
            -4.0
            * sine_squared
            * (2.0 * energy_minus**2 + 1.0)
            / (momentum_minus**2 * delta_minus**4)
        )
        second = (5.0 * energy_minus**2 - 2.0 * energy_plus * energy_minus + 3.0) / (
            momentum_minus**2 * delta_minus**2
        )
        third = (momentum_minus**2 - omega**2) / (transfer_squared * delta_minus**2)
        fourth = 2.0 * energy_plus / (momentum_minus**2 * delta_minus)
        y_term = (
            y
            / (momentum_minus * momentum_plus)
            * (
                2.0
                * energy_minus
                * sine_squared
                * (3.0 * omega + momentum_minus**2 * energy_plus)
                / delta_minus**4
                + (
                    2.0 * energy_minus**2 * (energy_minus**2 + energy_plus**2)
                    - 7.0 * energy_minus**2
                    - 3.0 * energy_plus * energy_minus
                    - energy_plus**2
                    + 1.0
                )
                / delta_minus**2
                + omega
                * (energy_minus**2 - energy_minus * energy_plus - 1.0)
                / delta_minus
            )
        )
        transfer_term = (
            -delta_plus_transfer
            / (momentum_plus * transfer)
            * (
                2.0 / delta_minus**2
                - 3.0 * omega / delta_minus
                - omega
                * (momentum_minus**2 - omega**2)
                / (transfer_squared * delta_minus)
            )
        )
        bracket = np.zeros(first.shape, dtype=float)
        compensation = np.zeros(first.shape, dtype=float)
        for term in (
            first,
            second,
            third,
            fourth,
            y_term,
            transfer_term,
            -2.0 * y_plus / delta_minus,
        ):
            increment = term - compensation
            updated = bracket + increment
            compensation = (updated - bracket) - increment
            bracket = updated
        values = (
            _CROSS_SECTION_SCALE_CGS
            * momentum_minus
            * momentum_plus
            / (2.0 * omega**3)
            * bracket
        )

    unstable = ~np.isfinite(values) | (values < 0.0)
    for index in np.flatnonzero(unstable):
        values[index] = _blumenthal_scalar_decimal(
            omega[index], energy_minus[index], cosine[index]
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(
            "Blumenthal differential cross section is not finite and non-negative; "
            f"minimum={np.nanmin(values):.6e}, maximum={np.nanmax(values):.6e}, "
            f"negative_count={np.count_nonzero(values < 0.0)}"
        )
    result[active] = values
    return result.reshape(shape)


def _quadrature_nodes(order):
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("quadrature_order must be an integer >= 2")
    if order == _DEFAULT_QUADRATURE_ORDER:
        return _GAUSS_NODE, _GAUSS_WEIGHT
    return np.polynomial.legendre.leggauss(order)


def _bethe_heitler_single_charge_cross_response(
    electron_gamma,
    proton_gamma,
    target_photon_epsilon,
    *,
    quadrature_order=_DEFAULT_QUADRATURE_ORDER,
):
    """Return the Born-domain KA (2008) Eq. (62) response in cm^2/dgamma."""
    daughter = np.asarray(electron_gamma, dtype=float)
    result = np.zeros(daughter.shape, dtype=float)
    nodes, weights = _quadrature_nodes(quadrature_order)
    with np.errstate(over="ignore", invalid="ignore"):
        u = proton_gamma * target_photon_epsilon
    if not np.isfinite(u):
        raise FloatingPointError(
            "proton and target-photon energy product exceeds float64 range"
        )
    if u <= 1.0:
        return result

    omega_maximum = 2.0 * min(u, 0.5 * _BETHE_HEITLER_PAIR_OMEGA_MAX_MEC2)
    root = np.sqrt(omega_maximum * (omega_maximum - 2.0))
    factor = omega_maximum - 1.0 + root
    upper = proton_gamma * factor
    lower = proton_gamma / factor
    supported = (daughter > lower) & (daughter < upper)
    for index in np.flatnonzero(supported):
        ratio = daughter[index] / proton_gamma
        energy_minimum = 0.5 * (ratio + 1.0 / ratio)
        omega_minimum = energy_minimum + 1.0
        if omega_minimum >= omega_maximum:
            continue
        omega_half_width = 0.5 * (omega_maximum - omega_minimum)
        omega = omega_minimum + omega_half_width * (nodes + 1.0)
        omega_weight = omega_half_width * weights

        energy_half_width = 0.5 * (omega - 1.0 - energy_minimum)
        energy_minus = energy_minimum + energy_half_width[:, None] * (
            nodes[None, :] + 1.0
        )
        energy_weight = energy_half_width[:, None] * weights[None, :]
        momentum_minus = np.sqrt((energy_minus - 1.0) * (energy_minus + 1.0))
        cosine = (energy_minus - ratio) / momentum_minus
        differential = _blumenthal_differential_cross_section(
            omega[:, None], energy_minus, cosine
        )
        inner = np.sum(energy_weight * differential / momentum_minus, axis=1)
        result[index] = np.sum(omega_weight * omega * inner) / (
            2.0 * proton_gamma * u**2
        )

    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "Bethe-Heitler single-charge response is not finite and non-negative"
        )
    return result


def _chodorowski_phi(kappa):
    """Return Chodorowski et al. (1992) Eqs. (3.14)--(3.19)."""
    value = np.asarray(kappa, dtype=float)
    result = np.zeros(value.shape, dtype=float)
    low = (value >= 2.0) & (value < 25.0)
    x = value[low] - 2.0
    result[low] = (
        (np.pi / 12.0)
        * x**4
        / (1.0 + 0.8048 * x + 0.1459 * x**2 + 1.137e-3 * x**3 - 3.879e-6 * x**4)
    )

    high = value >= 25.0
    logarithm = np.log(value[high])
    asymptote = value[high] * (
        _PHI_D[0]
        + logarithm * (_PHI_D[1] + logarithm * (_PHI_D[2] + logarithm * _PHI_D[3]))
    )
    result[high] = asymptote / (
        1.0 - 2.910 / value[high] - 78.35 / value[high] ** 2 - 1837.0 / value[high] ** 3
    )
    return result


def _chodorowski_psi(kappa):
    """Return the Chodorowski et al. (1992) Eqs. (2.4)--(2.5) event response."""
    value = np.asarray(kappa, dtype=float)
    result = np.zeros(value.shape, dtype=float)
    low = (value > 2.0) & (value < 4.0)
    k = value[low]
    result[low] = (
        2.0
        * np.pi
        / 3.0
        * (
            347.0 / (40.0 * k)
            - 130903.0 / 1440.0
            - 19151.0 * _LN2 / 48.0
            - 15163.0 * k / 480.0
            + 2593.0 * k**2 / 1920.0
            + 3904.0 / (9.0 * (2.0 + k) ** 3)
            - 9688.0 / (15.0 * (2.0 + k) ** 2)
            + 10676.0 / (15.0 * (2.0 + k))
            + 1007.0 * np.log(k) / 48.0
            + 189.0 * np.log(2.0 + k)
        )
    )

    high = value >= 4.0
    k = value[high]
    logarithm = np.log(2.0 * k)
    result[high] = (
        -67.0 / (1728.0 * k**4)
        + 29.0 * logarithm / (144.0 * k**4)
        + 7.0 / (4.0 * k**2)
        + 3.0 * logarithm / (2.0 * k**2)
        - 2.71245
        + 2.0 * (np.pi**2 / 3.0 - 7.0 + 4.0 * zeta(3.0, 1.0)) * logarithm
        + 2.0 * (6.0 - np.pi**2 / 3.0) * logarithm**2
        - 4.0 * logarithm**3 / 3.0
        + 2.0 * logarithm**4 / 3.0
        - 130.0 * k**2 / 27.0
        + 14.0 * k**2 * logarithm / 9.0
    )
    return result


def bethe_heitler_pair_injection(
    electron_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return isotropic Bethe--Heitler positron and electron injection.

    ``electron_gamma`` contains finite blob-comoving Lorentz factors greater
    than one in any order. ``proton_gamma`` contains at least two finite,
    strictly increasing blob-comoving Lorentz factors greater than one, with
    matching non-negative ``dn_p/dgamma_p`` in cm^-3. Target energies are at
    least two finite, positive, strictly increasing blob-comoving values in
    erg, with matching non-negative isotropic ``dn_ph/dE`` in cm^-3 erg^-1.

    The fixed return order is ``(Q_positron, Q_electron)``. Each float64 array
    follows ``electron_gamma`` and is one single-charge spectrum in cm^-3 s^-1
    per daughter Lorentz factor; Born charge symmetry makes them identical.

    The reference evaluates Kelner & Aharonian (2008) Eq. (62) with the
    corrected Blumenthal (1970) Eq. (10) Born cross section and fixed 32-point
    Gauss--Legendre quadrature in each inner coordinate. Following the
    conservative AM3 Born-domain prescription, it includes only
    ``2 <= omega < 600``, where ``omega`` is the proton-rest-frame photon
    energy in units of ``m_e c^2``. It assumes ``gamma_p >> 1``, soft photons,
    an unpolarized isotropic field, and negligible proton recoil. Finite-recoil
    Bethe--Heitler pairs at ``omega >= 600`` are not modeled; the bound is not
    a physical reaction cutoff.

    The independent continuous proton-cooling provider is unchanged and does
    not use this cutoff, so this pair source alone does not imply strict
    closure with the complete Bethe--Heitler proton-loss power. This function
    performs no interpolation, extrapolation, transport, target depletion,
    cascade, observer transformation, lookup-table, surrogate, Monte Carlo,
    or JAX calculation.
    """
    daughter = _real_1d("electron_gamma", electron_gamma, minimum_size=1, positive=True)
    if np.any(daughter <= 1.0):
        raise ValueError("electron_gamma must contain only values > 1")
    proton = _real_1d("proton_gamma", proton_gamma, minimum_size=2, positive=True)
    if np.any(proton <= 1.0) or np.any(np.diff(proton) <= 0.0):
        raise ValueError(
            "proton_gamma must contain at least two strictly increasing values > 1"
        )
    proton_density = _real_1d(
        "dn_proton_dgamma_cm3",
        dn_proton_dgamma_cm3,
        minimum_size=2,
        positive=False,
        shape=proton.shape,
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

    result = np.zeros(daughter.shape, dtype=float)
    if not np.any(proton_density) or not np.any(target_density):
        return result.copy(), result

    epsilon = target_energy / _MEC2_CGS
    target_weight = target_energy * target_density
    proton_integrand = np.zeros((daughter.size, proton.size), dtype=float)
    for proton_index, proton_value in enumerate(proton):
        if proton_density[proton_index] == 0.0:
            continue
        target_integrand = np.zeros((daughter.size, target_energy.size), dtype=float)
        for target_index, epsilon_value in enumerate(epsilon):
            if target_density[target_index] == 0.0:
                continue
            target_integrand[:, target_index] = target_weight[
                target_index
            ] * _bethe_heitler_single_charge_cross_response(
                daughter, proton_value, epsilon_value
            )
        proton_integrand[:, proton_index] = (
            proton_value
            * proton_density[proton_index]
            * trapezoid(target_integrand, x=np.log(target_energy), axis=1)
        )

    result = _C_CGS * trapezoid(proton_integrand, x=np.log(proton), axis=1)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "Bethe-Heitler pair injection exceeds the finite float64 range"
        )
    return result.copy(), result


def bethe_heitler_proton_energy_loss_rate(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the blob-frame fractional Bethe--Heitler proton loss rate.

    ``proton_gamma`` contains at least two finite, strictly increasing
    blob-comoving Lorentz factors greater than one. Target energies are at
    least two finite, positive, strictly increasing blob-comoving values in
    erg, with matching finite non-negative isotropic ``dn_ph/dE`` in cm^-3
    erg^-1. The returned float64 array follows the proton grid and is
    ``-dot(gamma_p)/gamma_p`` in s^-1, not an event or proton-destruction rate.

    This evaluates Chodorowski, Zdziarski & Sikora (1992) Eq. (3.11), using
    their Eqs. (3.14)--(3.19) for ``phi`` and exact threshold
    ``kappa = 2 gamma_p E_ph/(m_e c^2) >= 2``. It assumes ``gamma_p >> 1``,
    soft isotropic photons, the unpolarized Born approximation, and negligible
    proton recoil. No transport, interpolation, extrapolation, target
    depletion, cascade, or JAX calculation is included.
    """
    proton = _real_1d("proton_gamma", proton_gamma, minimum_size=2, positive=True)
    if np.any(proton <= 1.0) or np.any(np.diff(proton) <= 0.0):
        raise ValueError(
            "proton_gamma must contain at least two strictly increasing values > 1"
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

    result = np.zeros(proton.shape, dtype=float)
    if not np.any(target_density):
        return result
    epsilon = target_energy / _MEC2_CGS
    with np.errstate(over="ignore", invalid="ignore"):
        kappa = 2.0 * proton[:, None] * epsilon[None, :]
    if not np.all(np.isfinite(kappa)):
        raise FloatingPointError(
            "proton and target-photon energy product exceeds float64 range"
        )
    phi = _chodorowski_phi(kappa)
    response = np.zeros(kappa.shape, dtype=float)
    active = kappa >= 2.0
    response[active] = phi[active] / kappa[active] ** 2
    result = (
        2.0
        * _CROSS_SECTION_SCALE_CGS
        * _C_CGS
        * _ELECTRON_TO_PROTON_MASS_RATIO
        * trapezoid(
            target_energy[None, :] * target_density[None, :] * response,
            x=np.log(target_energy),
            axis=1,
        )
    )
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError(
            "Bethe-Heitler proton loss exceeds the finite float64 range"
        )
    return result
