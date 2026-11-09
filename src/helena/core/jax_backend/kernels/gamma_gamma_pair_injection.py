"""JAX differential pair injection from isotropic photon-photon collisions."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
    SPEED_OF_LIGHT_CGS as _C_CGS,
    THOMSON_CROSS_SECTION_CGS as _SIGMA_T_CGS,
)
from helena.core.contracts import real_1d as _real_1d
from helena.core.tables import (
    GAMMA_GAMMA_ANGLE_T as _ANGLE_T,
    GAMMA_GAMMA_ANGLE_WEIGHT as _ANGLE_WEIGHT,
)
from helena.core.jax_backend.kernels.gamma_gamma_absorption import (
    _checked_result,
    _contains_tracer,
)

__all__ = [
    "gamma_gamma_pair_injection_jax",
    "prepare_gamma_gamma_pair_response_jax",
]

_PAIR_RESPONSE_CONFIGURATION = (
    "daughter-support-sin2-128-v1",
    _C_CGS,
    _MEC2_CGS,
    _SIGMA_T_CGS,
    tuple(_ANGLE_T),
    tuple(_ANGLE_WEIGHT),
)


def _differential_cross_section_dgamma_jax(electron_gamma, x, eta, s):
    """Return the fixed-collision cross section in cm^2 per gamma."""
    gamma, x, eta, s = jnp.broadcast_arrays(
        jnp.asarray(electron_gamma, dtype=jnp.float64),
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(eta, dtype=jnp.float64),
        jnp.asarray(s, dtype=jnp.float64),
    )
    momentum_squared = (x + eta) ** 2 - 4.0 * s
    collision = (s > 1.0) & (momentum_squared > 0.0)
    collision_s = jnp.where(collision, s, 2.0)
    collision_beta = jnp.sqrt(1.0 - 1.0 / collision_s)
    collision_momentum = jnp.sqrt(jnp.where(collision, momentum_squared, 1.0))
    lower = 0.5 * (x + eta - collision_beta * collision_momentum)
    upper = 0.5 * (x + eta + collision_beta * collision_momentum)
    active = collision & (gamma >= lower) & (gamma <= upper)

    safe_s = jnp.where(active, s, 2.0)
    w = jnp.sqrt(safe_s)
    beta = jnp.sqrt(1.0 - 1.0 / safe_s)
    momentum = jnp.sqrt(jnp.where(active, momentum_squared, 1.0))
    c_1 = (x - gamma) ** 2 - 1.0
    c_2 = (eta - gamma) ** 2 - 1.0
    d_1 = x**2 + x * eta + gamma * (eta - x)
    d_2 = eta**2 + x * eta - gamma * (eta - x)
    a_1 = jnp.where(active, x * eta + safe_s * c_1, 1.0)
    a_2 = jnp.where(active, x * eta + safe_s * c_2, 1.0)
    g_sum = 1.0 / jnp.sqrt(a_1) + 1.0 / jnp.sqrt(a_2)
    f_sum = (d_1 - 2.0 * safe_s) / a_1**1.5 + (d_2 - 2.0 * safe_s) / a_2**1.5
    values = (
        3.0
        * _SIGMA_T_CGS
        / (8.0 * w)
        * (
            -1.0 / (momentum * w)
            + (3.0 - beta**4) * g_sum / 4.0
            - f_sum / (8.0 * safe_s)
        )
    )
    return jnp.where(active, values, 0.0)


def _isotropic_differential_pair_kernel_jax(
    electron_gamma,
    gamma_ray_energy_erg,
    target_photon_energy_erg,
):
    """Return cm^2 per daughter gamma, integrating its exact angular support.

    The unchanged differential cross section is integrated with 128-point
    Gauss--Legendre quadrature after a sin^2 map of the allowed invariant-s
    interval. This resolves narrow daughter support near gamma=1.
    """
    gamma = jnp.asarray(electron_gamma, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    x = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64) / _MEC2_CGS
    eta = target_energy / _MEC2_CGS
    y = x * eta
    # Daughter support requires 4 s^2 - 4 A s + (x+eta)^2 <= 0,
    # A=1+gamma*(x+eta-gamma). Rationalize the smaller root to avoid
    # subtracting nearly equal terms for hard/soft incident photons.
    daughter = gamma[:, None]
    total = x + eta[None, :]
    a = 1.0 + daughter * (total - daughter)
    discriminant = (daughter - 1.0) * (total - daughter - 1.0) * (a + total)
    possible = (y[None, :] > 1.0) & (daughter > 1.0) & (total - daughter > 1.0)
    possible = possible & (discriminant > 0.0)
    root = jnp.sqrt(jnp.where(possible, discriminant, 1.0))
    denominator = jnp.where(possible, a + root, 1.0)
    lower = jnp.maximum(1.0, total**2 / (2.0 * denominator))
    upper = jnp.minimum(y[None, :], denominator / 2.0)
    active = possible & (upper > lower)
    interval = jnp.where(active, upper - lower, 0.0)
    # sin^2 maps both kinematic endpoints smoothly without altering support.
    angle_t = jnp.asarray(_ANGLE_T, dtype=jnp.float64)
    angle_weight = jnp.asarray(_ANGLE_WEIGHT, dtype=jnp.float64)
    s = jnp.where(
        active[:, :, None],
        lower[:, :, None] + interval[:, :, None] * jnp.sin(jnp.pi * angle_t / 2.0) ** 2,
        1.0,
    )
    jacobian = interval[:, :, None] * (jnp.pi / 2.0) * jnp.sin(jnp.pi * angle_t)
    differential = _differential_cross_section_dgamma_jax(
        gamma[:, None, None], x, eta[None, :, None], s
    )
    integral = jnp.sum(angle_weight * s * differential * jacobian, axis=-1)
    safe_y = jnp.where(y > 1.0, y, 1.0)
    response = 2.0 * integral / safe_y[None, :] ** 2
    return jnp.where(active, response, 0.0)


def _pair_injection_kernel(
    electron_gamma,
    gamma_ray_energy_erg,
    dn_gamma_ray_denergy_cm3_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    daughter = jnp.asarray(electron_gamma, dtype=jnp.float64)
    gamma_ray_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
    gamma_ray_density = jnp.asarray(dn_gamma_ray_denergy_cm3_erg, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    target_density = jnp.asarray(dn_target_photon_denergy_cm3_erg, dtype=jnp.float64)
    log_target_energy = jnp.log(target_energy)
    target_weight = target_energy * target_density

    def integrate_primary(_, primary):
        energy, density = primary
        response = _isotropic_differential_pair_kernel_jax(
            daughter,
            energy,
            target_energy,
        )
        target_integral = jnp.trapezoid(
            target_weight[None, :] * response,
            x=log_target_energy,
            axis=1,
        )
        return None, energy * density * target_integral

    _, primary_integrand = jax.lax.scan(
        integrate_primary,
        None,
        (gamma_ray_energy, gamma_ray_density),
    )
    result = _C_CGS * jnp.trapezoid(
        primary_integrand,
        x=jnp.log(gamma_ray_energy),
        axis=0,
    )
    return result, result


def _trapezoid_weights(coordinate):
    coordinate = jnp.asarray(coordinate, dtype=jnp.float64)
    widths = jnp.diff(coordinate)
    return jnp.concatenate(
        (widths[:1] / 2.0, (widths[:-1] + widths[1:]) / 2.0, widths[-1:] / 2.0)
    )


def _precompute_pair_response(
    electron_gamma,
    gamma_ray_energy_erg,
    target_photon_energy_erg,
):
    """Return the fixed ``(daughter, primary, target)`` weighted response."""
    daughter = jnp.asarray(electron_gamma, dtype=jnp.float64)
    primary_energy = jnp.asarray(gamma_ray_energy_erg, dtype=jnp.float64)
    target_energy = jnp.asarray(target_photon_energy_erg, dtype=jnp.float64)
    primary_weight = (
        _C_CGS
        * primary_energy
        * _trapezoid_weights(jnp.log(primary_energy))
    )
    target_weight = target_energy * _trapezoid_weights(jnp.log(target_energy))

    def build_primary(_, values):
        energy, weight = values
        response = _isotropic_differential_pair_kernel_jax(
            daughter,
            energy,
            target_energy,
        )
        return None, weight * response * target_weight[None, :]

    response = jax.lax.scan(
        build_primary,
        None,
        (primary_energy, primary_weight),
    )[1]
    return jnp.transpose(response, (1, 0, 2))


def _contract_pair_response(
    response,
    gamma_ray_density,
    target_photon_density,
):
    """Contract one fixed response with both dynamic photon spectra."""
    target_contraction = jnp.sum(
        jnp.asarray(response, dtype=jnp.float64)
        * jnp.asarray(target_photon_density, dtype=jnp.float64)[None, None, :],
        axis=2,
    )
    result = jnp.sum(
        target_contraction
        * jnp.asarray(gamma_ray_density, dtype=jnp.float64)[None, :],
        axis=1,
    )
    return result, result


def _grid_key(grid):
    array = np.ascontiguousarray(grid)
    return array.dtype.str, array.shape, array.tobytes()


@lru_cache(maxsize=3)
def _cached_pair_response(daughter_key, primary_key, target_key, configuration):
    if configuration != _PAIR_RESPONSE_CONFIGURATION:
        raise ValueError("unsupported gamma-gamma pair-response configuration")

    def array(key):
        dtype, shape, values = key
        return np.frombuffer(values, dtype=np.dtype(dtype)).reshape(shape)

    response = _precompute_pair_response(
        array(daughter_key),
        array(primary_key),
        array(target_key),
    )
    return _checked_result(response, "gamma-gamma pair response")


def prepare_gamma_gamma_pair_response_jax(
    electron_gamma,
    gamma_ray_energy_erg,
    target_photon_energy_erg,
):
    """Return a cached fixed-grid differential pair response.

    The inputs are concrete blob-comoving daughter Lorentz factors greater
    than one, strictly increasing high-energy photon energies in erg, and
    strictly increasing target-photon energies in erg. The float64 result has
    shape ``(n_daughter, n_gamma_ray, n_target)`` and units
    cm^3 erg^2 s^-1 per daughter Lorentz factor. It contains the support-mapped
    128-point angular response, speed-of-light prefactor, and both caller-grid
    logarithmic trapezoid weights. Dynamic photon densities are excluded.

    This host-side content cache requires concrete fixed grids. Pass the
    returned response to ``gamma_gamma_pair_injection_jax`` when tracing or
    differentiating dynamic spectra; response construction itself is not an
    autodiff operation.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX gamma-gamma pair injection requires jax_enable_x64")
    if _contains_tracer(
        (electron_gamma, gamma_ray_energy_erg, target_photon_energy_erg)
    ):
        raise ValueError("gamma-gamma pair response grids must be concrete")

    daughter = _real_1d("electron_gamma", electron_gamma, minimum_size=1, positive=True)
    if np.any(daughter <= 1.0):
        raise ValueError("electron_gamma must contain only values > 1")
    primary_energy = _real_1d(
        "gamma_ray_energy_erg",
        gamma_ray_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(primary_energy) <= 0.0):
        raise ValueError("gamma_ray_energy_erg must be strictly increasing")
    target_energy = _real_1d(
        "target_photon_energy_erg",
        target_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(target_energy) <= 0.0):
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    with np.errstate(over="ignore", invalid="ignore"):
        maximum_y = np.max(primary_energy) * np.max(target_energy) / _MEC2_CGS**2
    if not np.isfinite(maximum_y):
        raise FloatingPointError(
            "gamma-ray and target-photon energy product exceeds float64 range"
        )
    return _cached_pair_response(
        _grid_key(daughter),
        _grid_key(primary_energy),
        _grid_key(target_energy),
        _PAIR_RESPONSE_CONFIGURATION,
    )


def gamma_gamma_pair_injection_jax(
    electron_gamma,
    gamma_ray_energy_erg,
    dn_gamma_ray_denergy_cm3_erg,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
    *,
    pair_response=None,
):
    """Return JAX isotropic gamma-gamma positron and electron injection.

    ``electron_gamma`` contains finite blob-comoving Lorentz factors greater
    than one in arbitrary order. Primary and target photon energies are finite,
    positive, strictly increasing blob-comoving grids in erg with at least two
    points. Their matching inputs are actual finite non-negative ``dn/dE``
    photon densities in cm^-3 erg^-1, not injection rates or SEDs.

    The float64 return order is ``(Q_positron, Q_electron)`` in cm^-3 s^-1 per
    unit daughter Lorentz factor. It directly mirrors the NumPy reference's
    full unpolarized Boettcher--Schlickeiser differential QED kernel, exact
    threshold and daughter support, support-mapped 128-point angular average,
    and caller-grid log-energy trapezoids.

    Eager calls validate inputs, prepare/cache the fixed response when omitted,
    and contract it with the two current photon densities. Traced calls must
    receive the fixed float64 ``pair_response`` returned by
    :func:`prepare_gamma_gamma_pair_response_jax`; this keeps JIT, reverse
    gradients, JVP, and VJP on the accelerated contraction rather than silently
    rebuilding the direct angular quadrature. No photon-density closure,
    absorption coupling, transport, cascade, or frame transform is included.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX gamma-gamma pair injection requires jax_enable_x64")

    arguments = (
        electron_gamma,
        gamma_ray_energy_erg,
        dn_gamma_ray_denergy_cm3_erg,
        target_photon_energy_erg,
        dn_target_photon_denergy_cm3_erg,
    )
    if _contains_tracer(arguments):
        if pair_response is None:
            raise ValueError(
                "traced gamma-gamma pair injection requires a prepared response"
            )
        expected_shape = (
            np.shape(electron_gamma)[0],
            np.shape(gamma_ray_energy_erg)[0],
            np.shape(target_photon_energy_erg)[0],
        )
        if np.shape(pair_response) != expected_shape:
            raise ValueError(f"pair_response must have shape {expected_shape}")
        if jnp.asarray(pair_response).dtype != jnp.float64:
            raise ValueError("pair_response must have dtype float64")
        return _contract_pair_response(
            pair_response,
            dn_gamma_ray_denergy_cm3_erg,
            dn_target_photon_denergy_cm3_erg,
        )

    daughter = _real_1d("electron_gamma", electron_gamma, minimum_size=1, positive=True)
    if np.any(daughter <= 1.0):
        raise ValueError("electron_gamma must contain only values > 1")
    gamma_ray_energy = _real_1d(
        "gamma_ray_energy_erg",
        gamma_ray_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(gamma_ray_energy) <= 0.0):
        raise ValueError("gamma_ray_energy_erg must be strictly increasing")
    gamma_ray_density = _real_1d(
        "dn_gamma_ray_denergy_cm3_erg",
        dn_gamma_ray_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=gamma_ray_energy.shape,
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
    expected_shape = (daughter.size, gamma_ray_energy.size, target_energy.size)
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
    if not np.any(gamma_ray_density) or not np.any(target_density):
        zero = jnp.zeros(daughter.shape, dtype=jnp.float64)
        return zero, zero

    with np.errstate(over="ignore", invalid="ignore"):
        maximum_y = np.max(gamma_ray_energy) * np.max(target_energy) / _MEC2_CGS**2
    if not np.isfinite(maximum_y):
        raise FloatingPointError(
            "gamma-ray and target-photon energy product exceeds float64 range"
        )

    if response is None:
        response = prepare_gamma_gamma_pair_response_jax(
            daughter,
            gamma_ray_energy,
            target_energy,
        )
    result = _contract_pair_response(
        response,
        gamma_ray_density,
        target_density,
    )
    return _checked_result(result, "gamma-gamma pair injection")
