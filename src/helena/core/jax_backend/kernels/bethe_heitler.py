"""JAX Bethe--Heitler pair injection and proton energy loss."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.contracts import real_1d as _real_1d
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    _checked_result,
    _contains_tracer,
)

__all__ = [
    "BH_PAIR_RESPONSE_VERSION",
    "bethe_heitler_pair_injection_jax",
    "bethe_heitler_proton_energy_loss_rate_jax",
    "prepare_bethe_heitler_pair_response_jax",
]

BH_PAIR_RESPONSE_VERSION = "ka2008_born_omega_lt_600_v1"

_CROSS_SECTION_SCALE_CGS = 5.79467271183577e-28
_ELECTRON_TO_PROTON_MASS_RATIO = 0.0005446170214901457
_FLOAT64_MAX = 1.7976931348623157e308
_BETHE_HEITLER_PAIR_OMEGA_MAX_MEC2 = 600.0

_GAUSS_NODE = (
    -0.9972638618494816,
    -0.9856115115452683,
    -0.9647622555875064,
    -0.9349060759377397,
    -0.8963211557660521,
    -0.84936761373257,
    -0.7944837959679424,
    -0.7321821187402897,
    -0.6630442669302152,
    -0.5877157572407623,
    -0.5068999089322294,
    -0.42135127613063533,
    -0.33186860228212767,
    -0.23928736225213706,
    -0.1444719615827965,
    -0.04830766568773832,
    0.04830766568773832,
    0.1444719615827965,
    0.23928736225213706,
    0.33186860228212767,
    0.42135127613063533,
    0.5068999089322294,
    0.5877157572407623,
    0.6630442669302152,
    0.7321821187402897,
    0.7944837959679424,
    0.84936761373257,
    0.8963211557660521,
    0.9349060759377397,
    0.9647622555875064,
    0.9856115115452683,
    0.9972638618494816,
)
_GAUSS_WEIGHT = (
    0.007018610009470362,
    0.016274394730906486,
    0.02539206530926273,
    0.03427386291302163,
    0.042835898022226926,
    0.05099805926237596,
    0.05868409347853546,
    0.0658222227763616,
    0.0723457941088482,
    0.07819389578707012,
    0.08331192422694662,
    0.08765209300440369,
    0.09117387869576371,
    0.09384439908080439,
    0.0956387200792746,
    0.09654008851472758,
    0.09654008851472758,
    0.0956387200792746,
    0.09384439908080439,
    0.09117387869576371,
    0.08765209300440369,
    0.08331192422694662,
    0.07819389578707012,
    0.0723457941088482,
    0.0658222227763616,
    0.05868409347853546,
    0.05099805926237596,
    0.042835898022226926,
    0.03427386291302163,
    0.02539206530926273,
    0.016274394730906486,
    0.007018610009470362,
)
_PHI_D = (-86.06919977855736, 50.958264354161976, -14.454822555520437, 8.0 / 3.0)
_PAIR_KERNEL_CONFIGURATION = (
    _C_CGS,
    _MEC2_CGS,
    _CROSS_SECTION_SCALE_CGS,
    _BETHE_HEITLER_PAIR_OMEGA_MAX_MEC2,
    _GAUSS_NODE,
    _GAUSS_WEIGHT,
)


def _blumenthal_differential_cross_section_jax(
    photon_energy,
    electron_energy,
    electron_cosine,
):
    """Return Blumenthal (1970) Eq. (10) in cm^2 per dE_- dcos(theta_-)."""
    omega, energy_minus, cosine = jnp.broadcast_arrays(
        jnp.asarray(photon_energy, dtype=jnp.float64),
        jnp.asarray(electron_energy, dtype=jnp.float64),
        jnp.asarray(electron_cosine, dtype=jnp.float64),
    )
    energy_plus = omega - energy_minus
    active = (
        (omega > 2.0)
        & (energy_minus > 1.0)
        & (energy_plus > 1.0)
        & (jnp.abs(cosine) < 1.0)
    )
    omega_safe = jnp.where(active, omega, 3.0)
    energy_minus_safe = jnp.where(active, energy_minus, 1.5)
    cosine_safe = jnp.where(active, cosine, 0.0)
    energy_plus_safe = omega_safe - energy_minus_safe

    momentum_minus = jnp.sqrt((energy_minus_safe - 1.0) * (energy_minus_safe + 1.0))
    momentum_plus = jnp.sqrt((energy_plus_safe - 1.0) * (energy_plus_safe + 1.0))
    sine_squared = (1.0 - cosine_safe) * (1.0 + cosine_safe)
    delta_minus = energy_minus_safe - momentum_minus * cosine_safe
    transfer_squared = momentum_plus**2 + 2.0 * omega_safe * delta_minus
    transfer = jnp.sqrt(transfer_squared)

    logarithm_increment = (
        (energy_plus_safe - 1.0) * (energy_minus_safe - 1.0)
        + momentum_plus * momentum_minus
    ) / omega_safe
    y = 2.0 / momentum_minus**2 * jnp.log1p(logarithm_increment)
    y_plus = 2.0 * jnp.arccosh(energy_plus_safe) / momentum_plus
    delta_plus_transfer = 2.0 * jnp.arctanh(momentum_plus / transfer)

    first = (
        -4.0
        * sine_squared
        * (2.0 * energy_minus_safe**2 + 1.0)
        / (momentum_minus**2 * delta_minus**4)
    )
    second = (
        5.0 * energy_minus_safe**2 - 2.0 * energy_plus_safe * energy_minus_safe + 3.0
    ) / (momentum_minus**2 * delta_minus**2)
    third = (momentum_minus**2 - omega_safe**2) / (transfer_squared * delta_minus**2)
    fourth = 2.0 * energy_plus_safe / (momentum_minus**2 * delta_minus)
    y_term = (
        y
        / (momentum_minus * momentum_plus)
        * (
            2.0
            * energy_minus_safe
            * sine_squared
            * (3.0 * omega_safe + momentum_minus**2 * energy_plus_safe)
            / delta_minus**4
            + (
                2.0
                * energy_minus_safe**2
                * (energy_minus_safe**2 + energy_plus_safe**2)
                - 7.0 * energy_minus_safe**2
                - 3.0 * energy_plus_safe * energy_minus_safe
                - energy_plus_safe**2
                + 1.0
            )
            / delta_minus**2
            + omega_safe
            * (energy_minus_safe**2 - energy_minus_safe * energy_plus_safe - 1.0)
            / delta_minus
        )
    )
    transfer_term = (
        -delta_plus_transfer
        / (momentum_plus * transfer)
        * (
            2.0 / delta_minus**2
            - 3.0 * omega_safe / delta_minus
            - omega_safe
            * (momentum_minus**2 - omega_safe**2)
            / (transfer_squared * delta_minus)
        )
    )

    def compensated_add(total, compensation, term):
        increment = term - compensation
        updated = total + increment
        return updated, (updated - total) - increment

    bracket = jnp.zeros_like(omega_safe)
    compensation = jnp.zeros_like(omega_safe)
    for term in (
        first,
        second,
        third,
        fourth,
        y_term,
        transfer_term,
        -2.0 * y_plus / delta_minus,
    ):
        bracket, compensation = compensated_add(bracket, compensation, term)
    values = (
        _CROSS_SECTION_SCALE_CGS
        * momentum_minus
        * momentum_plus
        / (2.0 * omega_safe**3)
        * bracket
    )
    return jnp.where(active, values, 0.0)


def _bethe_heitler_single_charge_cross_response_jax(
    electron_gamma,
    proton_gamma,
    target_photon_epsilon,
):
    """Return the Born-domain KA (2008) Eq. (62) response in cm^2/dgamma."""
    daughter = jnp.asarray(electron_gamma, dtype=jnp.float64)
    proton = jnp.asarray(proton_gamma, dtype=jnp.float64)
    epsilon = jnp.asarray(target_photon_epsilon, dtype=jnp.float64)
    u = proton * epsilon
    collision = u > 1.0
    safe_u = jnp.where(collision, u, 2.0)
    omega_maximum = 2.0 * jnp.minimum(safe_u, 0.5 * _BETHE_HEITLER_PAIR_OMEGA_MAX_MEC2)
    root = jnp.sqrt(omega_maximum * (omega_maximum - 2.0))
    factor = omega_maximum - 1.0 + root
    lower = proton / factor
    upper = proton * factor
    nodes = jnp.asarray(_GAUSS_NODE, dtype=jnp.float64)
    weights = jnp.asarray(_GAUSS_WEIGHT, dtype=jnp.float64)

    def integrate_daughter(daughter_value):
        kinematic_support = (
            collision & (daughter_value > lower) & (daughter_value < upper)
        )
        candidate_ratio = jnp.where(kinematic_support, daughter_value / proton, 1.0)
        candidate_energy_minimum = 0.5 * (candidate_ratio + 1.0 / candidate_ratio)
        supported = kinematic_support & (candidate_energy_minimum + 1.0 < omega_maximum)
        ratio = jnp.where(supported, candidate_ratio, 1.0)
        energy_minimum = 0.5 * (ratio + 1.0 / ratio)
        omega_minimum = energy_minimum + 1.0
        omega_half_width = 0.5 * (omega_maximum - omega_minimum)
        omega = omega_minimum + omega_half_width * (nodes + 1.0)
        omega_weight = omega_half_width * weights

        energy_half_width = 0.5 * (omega - 1.0 - energy_minimum)
        energy_minus = energy_minimum + energy_half_width[:, None] * (
            nodes[None, :] + 1.0
        )
        energy_weight = energy_half_width[:, None] * weights[None, :]
        momentum_minus = jnp.sqrt((energy_minus - 1.0) * (energy_minus + 1.0))
        cosine = (energy_minus - ratio) / momentum_minus
        differential = _blumenthal_differential_cross_section_jax(
            omega[:, None],
            energy_minus,
            cosine,
        )
        inner = jnp.sum(energy_weight * differential / momentum_minus, axis=1)
        response = jnp.sum(omega_weight * omega * inner) / (2.0 * proton * safe_u**2)
        return jnp.where(supported, response, 0.0)

    return jax.lax.map(integrate_daughter, daughter)


def _pair_injection_kernel(
    electron_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    daughter = jnp.asarray(electron_gamma, dtype=jnp.float64)
    proton = jnp.asarray(proton_gamma, dtype=jnp.float64)
    proton_density = jnp.asarray(dn_proton_dgamma_cm3, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    target_density = jnp.asarray(
        dn_target_photon_denergy_cm3_erg,
        dtype=jnp.float64,
    )
    epsilon = target_energy / _MEC2_CGS
    log_target_energy = jnp.log(target_energy)

    def integrate_proton(_, proton_values):
        proton_value, density_value = proton_values

        def integrate_target(_, target_values):
            energy_value, epsilon_value, target_density_value = target_values
            response = _bethe_heitler_single_charge_cross_response_jax(
                daughter,
                proton_value,
                epsilon_value,
            )
            return None, energy_value * target_density_value * response

        _, target_integrand = jax.lax.scan(
            integrate_target,
            None,
            (target_energy, epsilon, target_density),
        )
        target_integral = jnp.trapezoid(
            target_integrand,
            x=log_target_energy,
            axis=0,
        )
        return None, proton_value * density_value * target_integral

    _, proton_integrand = jax.lax.scan(
        integrate_proton,
        None,
        (proton, proton_density),
    )
    result = _C_CGS * jnp.trapezoid(
        proton_integrand,
        x=jnp.log(proton),
        axis=0,
    )
    return result, result


def _trapezoid_weights(coordinate):
    coordinate = jnp.asarray(coordinate, dtype=jnp.float64)
    widths = jnp.diff(coordinate)
    return jnp.concatenate(
        (widths[:1] / 2.0, (widths[:-1] + widths[1:]) / 2.0, widths[-1:] / 2.0)
    )


def _precompute_pair_kernel(electron_gamma, proton_gamma, target_photon_energy_erg):
    """Return the fixed ``(proton, photon, daughter)`` single-charge kernel."""
    daughter = jnp.asarray(electron_gamma, dtype=jnp.float64)
    proton = jnp.asarray(proton_gamma, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    epsilon = target_energy / _MEC2_CGS
    proton_weight = _C_CGS * proton * _trapezoid_weights(jnp.log(proton))
    target_weight = target_energy * _trapezoid_weights(jnp.log(target_energy))

    def scan_proton(_, proton_values):
        proton_value, proton_factor = proton_values

        def scan_target(_, target_values):
            epsilon_value, target_factor = target_values
            response = _bethe_heitler_single_charge_cross_response_jax(
                daughter, proton_value, epsilon_value
            )
            return None, proton_factor * target_factor * response

        _, kernel = jax.lax.scan(scan_target, None, (epsilon, target_weight))
        return None, kernel

    return jax.lax.scan(scan_proton, None, (proton, proton_weight))[1]


def _contract_pair_kernel(kernel, proton_density, target_photon_density):
    """Contract a fixed kernel with current spectra; return positrons, electrons."""
    target_contraction = jnp.sum(
        jnp.asarray(kernel, dtype=jnp.float64)
        * jnp.asarray(target_photon_density, dtype=jnp.float64)[None, :, None],
        axis=1,
    )
    result = jnp.sum(
        jnp.asarray(proton_density, dtype=jnp.float64)[:, None] * target_contraction,
        axis=0,
    )
    return result, result


def _grid_key(grid):
    array = np.ascontiguousarray(grid)
    return array.dtype.str, array.shape, array.tobytes()


@lru_cache(maxsize=3)
def _cached_pair_kernel(daughter_key, proton_key, target_key, configuration):
    if configuration != _PAIR_KERNEL_CONFIGURATION:
        raise ValueError("unsupported Bethe-Heitler pair-kernel configuration")

    def array(key):
        dtype, shape, values = key
        return np.frombuffer(values, dtype=np.dtype(dtype)).reshape(shape)

    return _checked_result(
        _precompute_pair_kernel(
            array(daughter_key), array(proton_key), array(target_key)
        ),
        "Bethe-Heitler pair response",
    )


def prepare_bethe_heitler_pair_response_jax(
    electron_gamma,
    proton_gamma,
    target_photon_energy_erg,
):
    """Return a cached fixed-grid Bethe--Heitler pair response.

    Inputs are concrete blob-comoving daughter Lorentz factors greater than
    one, a strictly increasing proton Lorentz-factor grid greater than one,
    and a strictly increasing positive target-photon energy grid in erg. The
    float64 result has shape ``(n_proton, n_target, n_daughter)``, units
    cm^3 erg s^-1 per daughter Lorentz factor, and contains the unchanged
    Born-domain response, 32x32 quadrature, speed-of-light prefactor, and both
    caller-grid logarithmic trapezoid weights. Dynamic proton and photon
    densities are excluded.

    Response construction is host-side and not an autodiff operation. Pass
    the result to ``bethe_heitler_pair_injection_jax`` for JIT, reverse-mode,
    JVP, or VJP execution with dynamic spectra.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX Bethe-Heitler pair injection requires jax_enable_x64")
    if _contains_tracer((electron_gamma, proton_gamma, target_photon_energy_erg)):
        raise ValueError("Bethe-Heitler pair response grids must be concrete")

    daughter = _real_1d("electron_gamma", electron_gamma, minimum_size=1, positive=True)
    if (daughter <= 1.0).any():
        raise ValueError("electron_gamma must contain only values > 1")
    proton = _real_1d("proton_gamma", proton_gamma, minimum_size=2, positive=True)
    if (proton <= 1.0).any() or (proton[1:] <= proton[:-1]).any():
        raise ValueError(
            "proton_gamma must contain at least two strictly increasing values > 1"
        )
    target_energy = _real_1d(
        "target_photon_energy_erg",
        target_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if (target_energy[1:] <= target_energy[:-1]).any():
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    maximum_epsilon = target_energy[-1] / _MEC2_CGS
    if maximum_epsilon > 1.0 and proton[-1] > _FLOAT64_MAX / maximum_epsilon:
        raise FloatingPointError(
            "proton and target-photon energy product exceeds float64 range"
        )
    return _cached_pair_kernel(
        _grid_key(daughter),
        _grid_key(proton),
        _grid_key(target_energy),
        _PAIR_KERNEL_CONFIGURATION,
    )


def _chodorowski_phi_jax(kappa):
    """Return Chodorowski et al. (1992) Eqs. (3.14)--(3.19)."""
    value = jnp.asarray(kappa, dtype=jnp.float64)
    low = (value >= 2.0) & (value < 25.0)
    x = jnp.where(low, value - 2.0, 0.0)
    low_result = (
        (jnp.pi / 12.0)
        * x**4
        / (1.0 + 0.8048 * x + 0.1459 * x**2 + 1.137e-3 * x**3 - 3.879e-6 * x**4)
    )

    high = value >= 25.0
    high_value = jnp.where(high, value, 25.0)
    logarithm = jnp.log(high_value)
    asymptote = high_value * (
        _PHI_D[0]
        + logarithm * (_PHI_D[1] + logarithm * (_PHI_D[2] + logarithm * _PHI_D[3]))
    )
    high_result = asymptote / (
        1.0 - 2.910 / high_value - 78.35 / high_value**2 - 1837.0 / high_value**3
    )
    return jnp.where(low, low_result, jnp.where(high, high_result, 0.0))


def _proton_energy_loss_kernel(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    proton = jnp.asarray(proton_gamma, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    target_density = jnp.asarray(
        dn_target_photon_denergy_cm3_erg,
        dtype=jnp.float64,
    )
    kappa = 2.0 * proton[:, None] * target_energy[None, :] / _MEC2_CGS
    active = kappa >= 2.0
    safe_kappa = jnp.where(active, kappa, 2.0)
    response = jnp.where(active, _chodorowski_phi_jax(kappa) / safe_kappa**2, 0.0)
    return (
        2.0
        * _CROSS_SECTION_SCALE_CGS
        * _C_CGS
        * _ELECTRON_TO_PROTON_MASS_RATIO
        * jnp.trapezoid(
            target_energy[None, :] * target_density[None, :] * response,
            x=jnp.log(target_energy),
            axis=1,
        )
    )


def bethe_heitler_pair_injection_jax(
    electron_gamma,
    proton_gamma,
    dn_proton_dgamma_cm3,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    pair_response=None,
):
    """Return isotropic JAX Bethe--Heitler positron and electron injection.

    ``electron_gamma`` contains finite blob-comoving Lorentz factors greater
    than one in any order. ``proton_gamma`` contains at least two finite,
    strictly increasing blob-comoving Lorentz factors greater than one, with
    matching non-negative ``dn_p/dgamma_p`` in cm^-3. Target energies are at
    least two finite, positive, strictly increasing blob-comoving values in
    erg, with matching non-negative isotropic ``dn_ph/dE`` in cm^-3 erg^-1.

    The fixed float64 return order is ``(Q_positron, Q_electron)``. Each array
    follows ``electron_gamma`` and is in cm^-3 s^-1 per daughter Lorentz
    factor. This mirrors the NumPy KA/Blumenthal single-charge reference with
    exact threshold and support, fixed 32x32 Gauss--Legendre quadrature, and
    caller-grid logarithmic trapezoids. Following the conservative AM3
    Born-domain prescription, it includes only ``2 <= omega < 600``, where
    ``omega`` is the proton-rest-frame photon energy in units of ``m_e c^2``.
    Finite-recoil Bethe--Heitler pairs at ``omega >= 600`` are not modeled;
    the bound is not a physical reaction cutoff.

    The independent continuous proton-cooling provider is unchanged and does
    not use this cutoff, so this pair source alone does not imply strict
    closure with the complete Bethe--Heitler proton-loss power. The pair
    calculation assumes an unpolarized isotropic blob-comoving soft-photon
    field, ``gamma_p >> 1``, and negligible proton recoil.

    Eager calls validate inputs and outputs, cache the quadrature-integrated
    response by the three fixed grids and provider constants, then contract it
    with the current proton and photon spectra. Dynamic spectra are never part
    of the cache key. Traced calls require the fixed float64 ``pair_response``
    returned by :func:`prepare_bethe_heitler_pair_response_jax` and execute only
    the discrete contraction; they never silently fall back to direct
    quadrature. No transport, radiation, target depletion, cascade,
    interpolation, observer transform, or empirical normalization is applied.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX Bethe-Heitler pair injection requires jax_enable_x64")

    arguments = (
        electron_gamma,
        proton_gamma,
        dn_proton_dgamma_cm3,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if _contains_tracer(arguments):
        if pair_response is None:
            raise ValueError(
                "traced Bethe-Heitler pair injection requires a prepared response"
            )
        expected_shape = (
            np.shape(proton_gamma)[0],
            np.shape(target_photon_energy_erg)[0],
            np.shape(electron_gamma)[0],
        )
        if np.shape(pair_response) != expected_shape:
            raise ValueError(f"pair_response must have shape {expected_shape}")
        if jnp.asarray(pair_response).dtype != jnp.float64:
            raise ValueError("pair_response must have dtype float64")
        return _contract_pair_kernel(
            pair_response,
            dn_proton_dgamma_cm3,
            dn_target_photon_denergy_cm3_erg,
        )

    daughter = _real_1d("electron_gamma", electron_gamma, minimum_size=1, positive=True)
    if (daughter <= 1.0).any():
        raise ValueError("electron_gamma must contain only values > 1")
    proton = _real_1d("proton_gamma", proton_gamma, minimum_size=2, positive=True)
    if (proton <= 1.0).any() or (proton[1:] <= proton[:-1]).any():
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
    if (target_energy[1:] <= target_energy[:-1]).any():
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    target_density = _real_1d(
        "dn_target_photon_denergy_cm3_erg",
        dn_target_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=target_energy.shape,
    )
    expected_shape = (proton.size, target_energy.size, daughter.size)
    response = None
    if pair_response is not None:
        response_values = np.asarray(pair_response)
        if response_values.shape != expected_shape:
            raise ValueError(f"pair_response must have shape {expected_shape}")
        if response_values.dtype != np.float64:
            raise ValueError("pair_response must have dtype float64")
        if not np.all(np.isfinite(response_values)) or np.any(response_values < 0.0):
            raise ValueError("pair_response must be finite and non-negative")
        response = jnp.asarray(pair_response, dtype=jnp.float64)
    if not proton_density.any() or not target_density.any():
        zero = jnp.zeros(daughter.shape, dtype=jnp.float64)
        return zero, zero
    maximum_epsilon = target_energy[-1] / _MEC2_CGS
    if maximum_epsilon > 1.0 and proton[-1] > _FLOAT64_MAX / maximum_epsilon:
        raise FloatingPointError(
            "proton and target-photon energy product exceeds float64 range"
        )

    if response is None:
        response = prepare_bethe_heitler_pair_response_jax(
            daughter,
            proton,
            target_energy,
        )
    result = _contract_pair_kernel(
        response,
        proton_density,
        target_density,
    )
    return _checked_result(result, "Bethe-Heitler pair injection")


def bethe_heitler_proton_energy_loss_rate_jax(
    proton_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return the JAX fractional Bethe--Heitler proton loss rate in s^-1.

    ``proton_gamma`` is a finite, strictly increasing blob-comoving grid with
    at least two values greater than one. Target energies are finite, positive,
    strictly increasing blob-comoving values in erg with matching finite non-
    negative isotropic ``dn_ph/dE`` in cm^-3 erg^-1. The float64 result follows
    the proton grid and is ``-dot(gamma_p)/gamma_p`` in s^-1, not an event rate
    or proton sink.

    This mirrors the NumPy Chodorowski response with exact ``kappa >= 2``
    threshold and the unchanged branch at ``kappa = 25``. It assumes
    ultrarelativistic protons, an unpolarized isotropic soft-photon field, the
    Born approximation, and negligible recoil. Eager calls validate inputs and
    outputs; traced calls require valid fixed-shape inputs. No transport,
    interpolation, target depletion, cascade, radiation, or frame transform is
    included. A transport loss coefficient would be ``gamma_p`` times this
    fractional rate.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX Bethe-Heitler proton loss requires jax_enable_x64")

    arguments = (
        proton_gamma,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if _contains_tracer(arguments):
        return _proton_energy_loss_kernel(*arguments)

    proton = _real_1d("proton_gamma", proton_gamma, minimum_size=2, positive=True)
    if (proton <= 1.0).any() or (proton[1:] <= proton[:-1]).any():
        raise ValueError(
            "proton_gamma must contain at least two strictly increasing values > 1"
        )
    target_energy = _real_1d(
        "target_photon_energy_erg",
        target_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if (target_energy[1:] <= target_energy[:-1]).any():
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    target_density = _real_1d(
        "dn_target_photon_denergy_cm3_erg",
        dn_target_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=target_energy.shape,
    )
    if not target_density.any():
        return jnp.zeros(proton.shape, dtype=jnp.float64)
    maximum_kappa_factor = 2.0 * target_energy[-1] / _MEC2_CGS
    if maximum_kappa_factor > 1.0 and proton[-1] > _FLOAT64_MAX / maximum_kappa_factor:
        raise FloatingPointError(
            "proton and target-photon energy product exceeds float64 range"
        )

    result = _proton_energy_loss_kernel(proton, target_energy, target_density)
    return _checked_result(result, "Bethe-Heitler proton loss")
