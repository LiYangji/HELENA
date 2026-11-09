"""Support-mapped pair quadrature against independent CM/Lorentz references.

Reference values were evaluated by adaptive invariant-s integration of the
independent azimuthal CM cross section with 4096 azimuth nodes. They are saved
in M05_gamma_gamma_precision_round4/support_reference_angular.json, alongside
1024-node convergence checks; they do not use the BS differential expression.
This test isolates angular integration, with no density or photon-grid weights.
"""

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.constants import ELECTRON_REST_ENERGY_CGS as MEC2
from helena.core.gamma_gamma_pair_injection import _isotropic_differential_pair_kernel
from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
    _PAIR_RESPONSE_CONFIGURATION,
    _isotropic_differential_pair_kernel_jax,
    gamma_gamma_pair_injection_jax,
    prepare_gamma_gamma_pair_response_jax,
)


def test_daughter_support_mapped_quadrature_matches_independent_cm_reference():
    cases = np.asarray(
        [
            [1.8, 1.8, 1.000001, 1.0040816934105469e-28],
            [1.8, 1.8, 1.8, 1.758909217172011e-25],
            [2.0, 3.0, 1.000001, 3.0761119910570026e-29],
            [2.0, 3.0, 2.5, 5.938201986237758e-26],
            [1000.0, 0.01, 25.670661200672473, 9.235303357713111e-32],
            [1000.0, 0.01, 500.005, 5.890066292809928e-29],
            [1.0e6, 1.0, 1.01, 3.5844936065157077e-32],
            [0.9273756048427304, 1.3194982558478183, 1.000001, 5.717075893939151e-28],
        ]
    )
    numpy_values = np.asarray(
        [
            _isotropic_differential_pair_kernel([g], x * MEC2, [y * MEC2])[0, 0]
            for x, y, g, _ in cases
        ]
    )

    def evaluate(row):
        x, y, gamma = row
        return _isotropic_differential_pair_kernel_jax(
            jnp.asarray([gamma]), x * MEC2, jnp.asarray([y * MEC2])
        )[0, 0]

    eager = jax.vmap(evaluate)(jnp.asarray(cases[:, :3]))
    compiled = jax.jit(jax.vmap(evaluate))(jnp.asarray(cases[:, :3]))
    swapped = jax.jit(jax.vmap(evaluate))(jnp.asarray(cases[:, [1, 0, 2]]))
    # Independent angular reference changes <6e-13 between1024 and4096 nodes;
    # 2e-8 allows differential-formula roundoff for the hard/soft cases.
    for values in (numpy_values, eager, compiled, swapped):
        np.testing.assert_allclose(values, cases[:, 3], rtol=2e-8, atol=0)
        assert np.all(np.isfinite(values)) and np.all(np.asarray(values) > 0)


def test_mapped_quadrature_keeps_true_threshold_and_daughter_support():
    gamma = np.asarray([1.000001, 1.2, 10.0])
    for x, targets in [(0.5, [0.5, 1.0, 2.0]), (2.0, [0.5, 0.75, 1.0])]:
        expected = _isotropic_differential_pair_kernel(
            gamma, x * MEC2, np.asarray(targets) * MEC2
        )
        actual = jax.jit(_isotropic_differential_pair_kernel_jax)(
            jnp.asarray(gamma), x * MEC2, jnp.asarray(targets) * MEC2
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
        assert np.all(np.isfinite(actual))
        np.testing.assert_array_equal(
            np.asarray(actual)[:, x * np.asarray(targets) <= 1], 0
        )
        np.testing.assert_array_equal(
            np.asarray(actual)[gamma > x + max(targets) - 1], 0
        )


def test_new_response_identity_and_dynamic_density_derivatives():
    assert _PAIR_RESPONSE_CONFIGURATION[0] == "daughter-support-sin2-128-v1"
    energy = jnp.asarray([0.7, 1.8, 3.0]) * MEC2
    gamma = jnp.asarray([1.000001, 1.1, 1.5, 2.5, 4.0, 6.1])
    response = prepare_gamma_gamma_pair_response_jax(gamma, energy, energy)
    density = jnp.asarray([2.0e8, 3.0e8, 1.0e8])

    def source(n):
        return (
            0.5
            * gamma_gamma_pair_injection_jax(
                gamma, energy, n, energy, n, pair_response=response
            )[0]
        )

    value, tangent = jax.jvp(jax.jit(source), (density,), (density,))
    # Both density sides participate: scaling n gives the exact degree-two JVP.
    np.testing.assert_allclose(tangent, 2 * value, rtol=2e-13, atol=0)
    cotangent = jnp.arange(1, gamma.size + 1, dtype=jnp.float64)
    _, pullback = jax.vjp(source, density)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent),
        jnp.vdot(pullback(cotangent)[0], density),
        rtol=2e-13,
        atol=0,
    )
    zero, zero_tangent = jax.jvp(source, (jnp.zeros_like(density),), (density,))
    np.testing.assert_array_equal(zero, 0)
    np.testing.assert_array_equal(zero_tangent, 0)
