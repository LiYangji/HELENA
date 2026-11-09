"""Small-grid checks for the single-population gamma-gamma operator.

This tests the fixed-state operator, not photon closure or a converged cascade.
The independent opacity quadrature and photon/pair moment comparison supplement
NumPy/JAX same-formula checks. No response cache or model run is required.
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as MEC2,
)
from helena.core.constants import (
    SPEED_OF_LIGHT_CGS as C,
)
from helena.core.constants import (
    THOMSON_CROSS_SECTION_CGS as SIGMA_T,
)
from helena.core.gamma_gamma_pair_injection import gamma_gamma_pair_injection
from helena.core.jax_backend.chains.gamma_gamma_pair_source import (
    gamma_gamma_pair_source_chain_jax,
    gamma_gamma_total_field_jax,
    prepare_gamma_gamma_total_field_response_jax,
)

jax.config.update("jax_enable_x64", True)

Q = "electron_pair_injection_dgamma_cm3_s"
QP = "positron_pair_injection_dgamma_cm3_s"
ALPHA = "absorption_coefficient_cm1"
LOSS = "photon_loss_rate_cm3_s_erg"


def _weights(coordinate):
    """Independent trapezoid weights, used with E dlnE and with dgamma."""
    delta = np.diff(coordinate)
    return np.concatenate(([delta[0]], delta[:-1] + delta[1:], [delta[-1]])) / 2


def _angle_average_quad(y):
    """Independent adaptive Breit--Wheeler angular integral (cm^2).

    Integrate dimensionless sigma/sigma_T to avoid an absolute quadrature
    tolerance accidentally accepting a cross section of order 1e-25 as zero.
    """
    if y <= 1:
        return 0.0

    def integrand(s):
        beta = np.sqrt(1 - 1 / s)
        sigma = (
            3
            / (16 * s)
            * (
                (3 - beta**4) * np.log((1 + beta) / (1 - beta))
                - 2 * beta * (2 - beta**2)
            )
        )
        return s * sigma

    integral, _ = quad(integrand, 1.0, y, epsabs=2e-12, epsrel=2e-12)
    return 2 * SIGMA_T * integral / y**2


@pytest.fixture(autouse=True)
def _cpu_default():
    # Keep small reference/AD checks on CPU even on a CUDA workstation.
    # Only the explicitly parametrized GPU test switches device.
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.fixture(scope="module")
def bath():
    energy = MEC2 * np.array([0.1, 0.4, 1.0, 1.8, 3.0, 10.0, 30.0])
    daughter = np.geomspace(1.000001, 60.01, 513)
    weight = energy * _weights(np.log(energy))
    with jax.default_device(jax.devices("cpu")[0]):
        response = prepare_gamma_gamma_total_field_response_jax(energy, daughter)
    return energy, daughter, weight, response


def _line_density(weight, index, number=10.0):
    result = np.zeros_like(weight)
    result[index] = number / weight[index]
    return result


def test_zero_threshold_and_invalid_input(bath):
    energy, daughter, weight, response = bath
    zero = np.zeros_like(energy)
    result = gamma_gamma_total_field_jax(zero, response)
    assert bool(result["input_valid"])
    for name, value in result.items():
        if name != "input_valid":
            np.testing.assert_array_equal(value, np.zeros_like(value))

    # All occupied photons obey E <= mec^2, including the exact threshold.
    # Opacity at unoccupied high-energy nodes need not vanish.
    density = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]) / weight
    threshold = gamma_gamma_total_field_jax(density, response)
    for key in (Q, QP, LOSS, "event_rate_cm3_s"):
        np.testing.assert_array_equal(threshold[key], np.zeros_like(threshold[key]))
    for bad in (np.full_like(zero, -1.0), np.full_like(zero, np.nan)):
        assert not bool(gamma_gamma_total_field_jax(bad, response)["input_valid"])

    # At n=0 the quadratic pair/loss tangent is exactly zero, while opacity is
    # linear. This exercises strict zero entries without taking log(0).
    direction = 1 / weight
    _, tangent = jax.jvp(
        lambda n: tuple(
            gamma_gamma_total_field_jax(n, response)[k] for k in (ALPHA, Q, LOSS)
        ),
        (jnp.asarray(zero),),
        (jnp.asarray(direction),),
    )
    np.testing.assert_allclose(
        tangent[0],
        gamma_gamma_total_field_jax(direction, response)[ALPHA],
        rtol=3e-14,
        atol=0,
    )
    np.testing.assert_array_equal(tangent[1], np.zeros_like(daughter))
    np.testing.assert_array_equal(tangent[2], zero)


def test_opacity_independent_quadrature_and_numpy_pair(bath):
    energy, daughter, weight, response = bath
    density = np.array([2.0, 1.0, 0.0, 3.0, 2.0, 0.5, 0.2]) / weight
    result = gamma_gamma_total_field_jax(density, response)
    kernel = np.array(
        [[_angle_average_quad(x * y / MEC2**2) for y in energy] for x in energy]
    )
    alpha = kernel @ (density * weight)
    np.testing.assert_allclose(result[ALPHA], alpha, rtol=5e-7, atol=0)
    np.testing.assert_allclose(
        result["photon_loss_number_rate_cm3_s"],
        C * np.sum(alpha * density * weight),
        rtol=5e-7,
        atol=0,
    )
    np.testing.assert_allclose(
        result["photon_loss_power_erg_cm3_s"],
        C * np.sum(energy * alpha * density * weight),
        rtol=5e-7,
        atol=0,
    )
    reference, _ = gamma_gamma_pair_injection(
        daughter, energy, density, energy, density
    )
    np.testing.assert_allclose(result[Q], 0.5 * reference, rtol=3e-11, atol=0)
    np.testing.assert_array_equal(result[Q], result[QP])
    np.testing.assert_allclose(
        result["photon_loss_number_rate_cm3_s"],
        2 * result["event_rate_cm3_s"],
        rtol=3e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        response["absorption_kernel_cm2"],
        np.asarray(response["absorption_kernel_cm2"]).T,
        rtol=3e-14,
        atol=0,
    )


@pytest.mark.parametrize(
    "first,second",
    [(3, 3), (3, 4), (0, 6)],
    ids=["below_1MeV_self", "comparable", "hard_soft"],
)
def test_event_count_and_independent_number_energy_moments(bath, first, second):
    energy, daughter, weight, response = bath
    a = _line_density(weight, first)
    b = _line_density(weight, second, 7.0)
    a_result = gamma_gamma_total_field_jax(a, response)
    if first == second:
        # One population: one event removes two photons and makes one e+e- pair.
        measured = a_result
        event = 0.5 * C * 10.0**2 * _angle_average_quad((energy[first] / MEC2) ** 2)
    else:
        full = gamma_gamma_total_field_jax(a + b, response)
        b_result = gamma_gamma_total_field_jax(b, response)
        measured = {
            k: full[k] - a_result[k] - b_result[k]
            for k in (
                Q,
                QP,
                "event_rate_cm3_s",
                "photon_loss_number_rate_cm3_s",
                "photon_loss_power_erg_cm3_s",
            )
        }
        event = (
            C
            * 10.0
            * 7.0
            * _angle_average_quad(energy[first] * energy[second] / MEC2**2)
        )
        # Distinct-population cross term has no 1/2. Swapping labels preserves Q.
        ab, _ = gamma_gamma_pair_injection(daughter, energy, a, energy, b)
        ba, _ = gamma_gamma_pair_injection(daughter, energy, b, energy, a)
        np.testing.assert_allclose(ab, ba, rtol=3e-11, atol=0)
        np.testing.assert_allclose(measured[Q], ab, rtol=3e-11, atol=1e-14 * np.max(ab))
    assert event > 0
    np.testing.assert_allclose(measured["event_rate_cm3_s"], event, rtol=5e-7, atol=0)
    np.testing.assert_allclose(
        measured["photon_loss_number_rate_cm3_s"], 2 * event, rtol=5e-7, atol=0
    )
    expected_power = event * (energy[first] + energy[second])
    np.testing.assert_allclose(
        measured["photon_loss_power_erg_cm3_s"], expected_power, rtol=5e-7, atol=0
    )
    q = np.asarray(measured[Q])
    number = np.trapezoid(q, daughter)
    power = 2 * MEC2 * np.trapezoid(daughter * q, daughter)
    # This is a coarse daughter-grid discretization check, not an exact energy
    # renormalization. The upper grid includes all kinematic support of cases.
    np.testing.assert_allclose(number, event, rtol=1e-2, atol=0)
    np.testing.assert_allclose(power, expected_power, rtol=1e-2, atol=0)
    print(
        "gamma_gamma_moments",
        json.dumps(
            {
                "first_mec2": float(energy[first] / MEC2),
                "second_mec2": float(energy[second] / MEC2),
                "event_rate_cm3_s": float(event),
                "per_charge_number_relative_error": float(number / event - 1),
                "pair_power_relative_error": float(power / expected_power - 1),
            },
            sort_keys=True,
        ),
    )


def test_source_repartition_does_not_change_fixed_population(bath):
    energy, _, weight, response = bath
    density = np.arange(1.0, energy.size + 1) / weight
    expected = gamma_gamma_total_field_jax(density, response)
    # Source labels and a bookkeeping split cannot gate collisions in this
    # operator. This does not assert that the entire production solver is split
    # independent: its source construction and SSA/IC policies are separate.
    for split in (0.8 * MEC2, 1.0e6 * 1.602176634e-12, 5 * MEC2):
        low = np.where(energy <= split, density, 0)
        high = np.where(energy > split, density, 0)
        actual = gamma_gamma_total_field_jax(low + high, response)
        for key in (ALPHA, Q, QP, LOSS, "event_rate_cm3_s"):
            np.testing.assert_array_equal(actual[key], expected[key])


def test_density_scaling_jvp_vjp_and_multistep_difference(bath):
    energy, _, weight, response = bath
    density = np.arange(1.0, energy.size + 1) / weight
    baseline = gamma_gamma_total_field_jax(density, response)
    keys = (ALPHA, LOSS, Q)
    scales = [
        max(float(np.max(np.abs(baseline[k]))), np.finfo(float).tiny) for k in keys
    ]

    def vector(log_amplitudes):
        output = gamma_gamma_total_field_jax(
            density * jnp.exp(log_amplitudes), response
        )
        return jnp.concatenate([output[k] / scale for k, scale in zip(keys, scales)])

    coordinate = jnp.zeros(energy.size, dtype=jnp.float64)
    direction = jnp.linspace(-0.3, 0.4, energy.size)
    value, tangent = jax.jvp(vector, (coordinate,), (direction,))
    jacobian = jax.jacfwd(vector)(coordinate)
    np.testing.assert_allclose(tangent, jacobian @ direction, rtol=3e-12, atol=3e-13)
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        difference = (
            vector(coordinate + step * direction)
            - vector(coordinate - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(difference, tangent, rtol=2e-6, atol=2e-9)
        errors.append(float(np.max(np.abs(np.asarray(difference - tangent)))))
    cotangent = jnp.linspace(-0.7, 0.5, value.size)
    _, pullback = jax.vjp(vector, coordinate)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent),
        jnp.vdot(pullback(cotangent)[0], direction),
        rtol=3e-12,
        atol=3e-12,
    )
    # A common density scaling changes both collision sides: Q'(1)=2Q(1).
    for key, degree in ((ALPHA, 1), (LOSS, 2), (Q, 2), (QP, 2)):
        _, derivative = jax.jvp(
            lambda scale, field=key: gamma_gamma_total_field_jax(
                scale * density, response
            )[field],
            (jnp.float64(1.0),),
            (jnp.float64(1.0),),
        )
        np.testing.assert_allclose(
            derivative, degree * baseline[key], rtol=3e-13, atol=0
        )
    print("gamma_gamma_ad", json.dumps({"max_absolute_fd_errors": errors}))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_eager_jit_backend_equivalence(bath, backend, record_property):
    try:
        device = jax.devices(backend)[0]
    except RuntimeError:
        pytest.skip(f"{backend} backend unavailable")
    energy, _, weight, response = bath
    density = np.arange(1.0, energy.size + 1) / weight
    expected = gamma_gamma_total_field_jax(density, response)
    on_device = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), response)
    n = jax.device_put(density, device)
    with jax.default_device(device):
        eager = gamma_gamma_total_field_jax(n, on_device)
        compiled = jax.jit(
            lambda values: gamma_gamma_total_field_jax(values, on_device)
        )
        started = time.perf_counter()
        actual = jax.block_until_ready(compiled(n))
        first_seconds = time.perf_counter() - started
        started = time.perf_counter()
        repeated = jax.block_until_ready(compiled(n))
        warm_seconds = time.perf_counter() - started

        # Compile a small forward and reverse derivative on this device too;
        # the more detailed vector/difference checks above deliberately use CPU.
        def pair_sum(scale):
            return jnp.sum(gamma_gamma_total_field_jax(scale * n, on_device)[Q])

        scaling_jvp = jax.jit(
            lambda scale: jax.jvp(pair_sum, (scale,), (jnp.ones_like(scale),))[1]
        )
        scaling_vjp = jax.jit(jax.grad(pair_sum))
        unit = jax.device_put(np.float64(1), device)
        forward_derivative = jax.block_until_ready(scaling_jvp(unit))
        reverse_derivative = jax.block_until_ready(scaling_vjp(unit))
        expected_derivative = 2 * np.sum(np.asarray(expected[Q]))
        np.testing.assert_allclose(
            forward_derivative, expected_derivative, rtol=3e-12, atol=0
        )
        np.testing.assert_allclose(
            reverse_derivative, expected_derivative, rtol=3e-12, atol=0
        )
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=3e-12, atol=1e-40)
        np.testing.assert_allclose(actual[key], eager[key], rtol=3e-12, atol=1e-40)
        np.testing.assert_array_equal(actual[key], repeated[key])
    record_property(f"{backend}_compile_first_seconds", first_seconds)
    record_property(f"{backend}_warm_seconds", warm_seconds)
    print(
        "gamma_gamma_timing",
        json.dumps(
            {
                "backend": backend,
                "compile_first_seconds": first_seconds,
                "warm_seconds": warm_seconds,
            }
        ),
    )


def test_retired_one_way_contract_is_explicitly_rejected(bath):
    energy, daughter, weight, response = bath
    target = np.arange(1.0, energy.size + 1) / weight
    intrinsic = 1e34 * np.exp(-np.log(energy / (3 * MEC2)) ** 2)
    # A produced SED plus a separate target cannot identify one unique bath.
    # Do not silently add the arrays or retain a fallback production solver.
    with pytest.raises(ValueError, match="retired|total|unified"):
        gamma_gamma_pair_source_chain_jax(
            energy,
            intrinsic,
            energy,
            target,
            daughter,
            radius_cm=1e15,
            internal_pair_response=response["pair_response"],
        )


def test_insufficient_daughter_upper_support_exceeds_declared_budget(bath):
    energy, _, weight, _ = bath
    # The caller declares the acceptable outside-support budget before seeing
    # the result. This 1.8 x 1.8 case has genuine daughters above gamma=1.2.
    maximum_outside_energy_fraction = 1e-5
    daughter = np.geomspace(1.000001, 1.2, 65)
    response = prepare_gamma_gamma_total_field_response_jax(energy, daughter)
    result = gamma_gamma_total_field_jax(_line_density(weight, 3), response)
    photon_power = float(result["photon_loss_power_erg_cm3_s"])
    event_rate = float(result["event_rate_cm3_s"])
    assert photon_power > 0 and event_rate > 0
    upper_fraction_bound = (
        float(result["upper_outside_pair_power_bound_erg_cm3_s"]) / photon_power
    )
    assert upper_fraction_bound > maximum_outside_energy_fraction
    assert (
        float(result["outside_energy_fraction_bound"]) > maximum_outside_energy_fraction
    )
    # A positive conservative bound alone is not enough: the represented
    # injection actually misses a substantial number and energy fraction.
    number_fraction = float(result["electron_injection_rate_cm3_s"]) / event_rate
    energy_fraction = float(result["pair_injection_power_erg_cm3_s"]) / photon_power
    assert number_fraction < 0.9
    assert energy_fraction < 0.9
    assert float(result["pair_number_relative_residual"]) < -0.1
    assert float(result["pair_energy_relative_residual"]) < -0.1
    print(
        "gamma_gamma_insufficient_support",
        json.dumps(
            {
                "daughter_gamma_max": float(daughter[-1]),
                "declared_maximum_outside_energy_fraction": maximum_outside_energy_fraction,
                "upper_outside_energy_fraction_bound": upper_fraction_bound,
                "represented_per_charge_number_fraction": number_fraction,
                "represented_pair_energy_fraction": energy_fraction,
            },
            sort_keys=True,
        ),
    )
