"""Regression for the narrow upper endpoint of the original mono+PL collision."""

import numpy as np

from helena.core.constants import ELECTRON_REST_ENERGY_CGS as MEC2, SPEED_OF_LIGHT_CGS
from helena.core.gamma_gamma import _angle_averaged_cross_section
from helena.core.jax_backend.kernels.gamma_gamma_cascade_response import (
    prepare_cascade_pair_response_jax,
)


def test_local_endpoint_spectrum_and_single_charge_moments():
    x, eta = 245855025.8808568, 1.015630004681144e-6
    gamma = np.geomspace(1.5e5, 3e8, 128)
    primary = np.array([5e5, x]) * MEC2
    target = np.array([0.8 * eta, eta]) * MEC2
    response = np.asarray(prepare_cascade_pair_response_jax(gamma, primary, target))
    np.testing.assert_array_equal(response[:, 0, :], 0.0)
    assert np.all(response >= 0.0)
    photon_weight = SPEED_OF_LIGHT_CGS * (primary[1] * np.diff(np.log(primary))[0] / 2)
    photon_weight *= target[1] * np.diff(np.log(target))[0] / 2
    mass = response[:, 1, 1] * np.gradient(gamma) / photon_weight
    mass[[0, -1]] *= 0.5
    sigma = _angle_averaged_cross_section(x * eta)
    np.testing.assert_allclose(mass.sum(), sigma, rtol=1e-6, atol=0)
    np.testing.assert_allclose(2 * mass @ gamma, (x + eta) * sigma, rtol=1e-6, atol=0)

    # Independent NumPy angular kernel; order-64 integration of the low- and
    # high-energy halves separately against each local linear basis function.
    # The final nonzero node brackets the physical upper endpoint 245608626.30.
    fractions = [0.0567805422676164, 0.19558259724560967, 0.12153505417341054]
    np.testing.assert_allclose(mass[122:125] / mass.sum(), fractions, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(mass[125:], 0.0)


def test_hard_hard_collision_resolves_incoming_energy_peaks():
    eta = 35999.35
    x = np.array([eta, 1.0287e7])
    gamma = np.geomspace(1.000001, 1.1e7, 160)
    primary = x * MEC2
    target = np.array([eta, eta * 1.01]) * MEC2
    response = np.asarray(prepare_cascade_pair_response_jax(gamma, primary, target))
    mass = response[:, :, 0] * np.gradient(gamma)[:, None]
    mass[[0, -1]] *= 0.5
    fractions = mass / mass.sum(axis=0)
    # Independent angular-kernel quadrature used additive subpanels about
    # both incident energies. An endpoint-only log rule misses these peaks.
    np.testing.assert_allclose(
        fractions[[102, 103], 0], [0.14375768789964444, 0.7032528031362273], rtol=1e-6
    )
    np.testing.assert_allclose(
        fractions[[103, 158], 1], [0.27646901416304, 0.3238452899584153], rtol=1e-6
    )


def test_head_on_angular_transition_inside_a_daughter_interval():
    # The head-on cap changes at g=17.09594467 inside the last active interval.
    gamma = np.r_[
        np.geomspace(1.000001, 16.0, 64), 16.642227280997783, 17.69129051895561, 19.0
    ]
    primary = np.array([15.0, 17.349795445894955]) * MEC2
    target = np.array([0.8, 0.9457808537628666]) * MEC2
    response = np.asarray(prepare_cascade_pair_response_jax(gamma, primary, target))
    mass = response[:, 1, 1] * np.gradient(gamma)
    mass[[0, -1]] *= 0.5
    # Reference fractions are independent GL64 integrals split at the angular cap.
    np.testing.assert_allclose(
        mass[-3:-1] / mass.sum(), [0.11692398437887179, 0.03094460060935586], rtol=1e-6
    )
