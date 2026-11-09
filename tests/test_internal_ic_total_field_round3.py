"""Round3 internal-IC wiring on frozen fields and a tiny real coupled map.

Frozen tests replace only photon transfer with a prescribed source population;
this is an interface test, not a new equilibrium solution or kernel validation.
Emission, cooling and particle transport remain production functions. The final
small feedback test leaves the complete EM map unpatched.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helena.core.constants import ELECTRON_REST_ENERGY_CGS as MEC2
from helena.core.jax_backend.chains.gamma_gamma_pair_source import (
    prepare_gamma_gamma_total_field_response_jax,
)
from helena.core.jax_backend.kernels.inverse_compton_cooling import (
    electron_inverse_compton_loss_rate_dgamma_s_jax,
)
from helena.core.jax_backend.radiation.inverse_compton import inverse_compton_sed_jax

em = importlib.import_module(
    "helena.core.jax_backend.chains.steady_state_electromagnetic_cascade"
)
jax.config.update("jax_enable_x64", True)
NAMES = (
    "electron_synchrotron",
    "proton_synchrotron",
    "pion_synchrotron",
    "muon_synchrotron",
    "internal_ic",
    "external_compton",
    "pi0",
)


@pytest.fixture(scope="module")
def small_problem():
    energy = MEC2 * np.geomspace(1e-11, 1e3, 43)
    gamma = np.geomspace(10.0, 2.1e5, 97)
    external = MEC2 * np.geomspace(1e-10, 1e-3, 15)
    primary = 3e-6 * np.exp(-np.log(gamma / 1e3) ** 2)
    zero = np.zeros_like(gamma)
    args = (
        energy,
        energy,
        gamma,
        primary,
        zero,
        zero,
        zero,
        zero,
        zero,
        np.array([1.01, 10.0, 100.0]),
        np.zeros(3),
        np.zeros_like(energy),
        external,
        np.zeros_like(external),
    )
    response = prepare_gamma_gamma_total_field_response_jax(energy, gamma)
    frozen = 1e19 * np.exp(-np.log(energy / (1e-7 * MEC2)) ** 2)
    return args, response, frozen


def run(problem, *, internal=True, external=False, iterations=32):
    args, response, _ = problem
    return em.steady_state_electromagnetic_cascade_jax(
        *args,
        magnetic_field_G=0.0,
        radius_cm=1e14,
        adiabatic_rate_s_inv=1e-6,
        escape_rate_s_inv=3e-4,
        enable_ssc=internal,
        enable_ec=external,
        enable_internal_gamma_gamma=False,
        include_external_gamma_gamma_target=False,
        gamma_gamma_pair_response=response,
        relaxation_omega=1.0,
        relative_tolerance=1e-10,
        maximum_iterations=iterations,
    )


def freeze_transfer(monkeypatch, fields):
    monkeypatch.setattr(
        em,
        "_internal_photon_density_kernel",
        lambda energy, sources, alpha, radius: fields,
    )


def assert_backend_close(actual, expected):
    """Established round2 peak-normalized backend budget (KN cancellation)."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = float(np.max(np.abs(expected)))
    if scale == 0:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert np.max(np.abs(actual - expected)) / scale <= 1e-10


def assert_ic_reads_total(state, problem):
    energy, _, gamma = map(jnp.asarray, problem[0][:3])
    total = state["internal_photon_density_cm3_erg"]
    leptons = (
        state["electron_density_dgamma_cm3"] + state["positron_density_dgamma_cm3"]
    )
    expected_cooling = electron_inverse_compton_loss_rate_dgamma_s_jax(
        gamma, energy, total
    )
    expected_sed = inverse_compton_sed_jax(
        energy, gamma, leptons, energy, total, radius_cm=1e14
    )
    np.testing.assert_array_equal(
        state["internal_ic_seed_photon_density_cm3_erg"], total
    )
    assert_backend_close(
        state["internal_inverse_compton_loss_rate_dgamma_s"], expected_cooling
    )
    np.testing.assert_allclose(
        state["internal_ic_intrinsic_sed_erg_s"], expected_sed, rtol=2e-13, atol=0
    )
    np.testing.assert_allclose(
        sum(state["photon_components_internal_density_cm3_erg"].values()),
        total,
        rtol=2e-13,
        atol=0,
    )
    np.testing.assert_allclose(
        sum(state["photon_components_intrinsic_sed_erg_s"].values()),
        state["intrinsic_photon_sed_erg_s"],
        rtol=2e-13,
        atol=0,
    )
    assert set(state["photon_components_intrinsic_sed_erg_s"]) == set(NAMES)
    assert "ssc" not in state["photon_components_intrinsic_sed_erg_s"]


def test_synchrotron_only_matches_ssc_emission_and_current_cooling(
    small_problem, monkeypatch
):
    base = jnp.asarray(small_problem[2])
    fields = jnp.zeros((7, base.size)).at[0].set(base)
    freeze_transfer(monkeypatch, fields)
    state = run(small_problem)
    assert bool(state["converged"])
    assert_ic_reads_total(state, small_problem)
    np.testing.assert_allclose(
        state["internal_ic_intrinsic_sed_erg_s"],
        state["ssc_intrinsic_sed_erg_s"],
        rtol=2e-13,
        atol=0,
    )
    assert_backend_close(
        state["internal_inverse_compton_loss_rate_dgamma_s"],
        state["ssc_inverse_compton_loss_rate_dgamma_s"],
    )


@pytest.mark.parametrize(
    "added_source", [4, 5, 6], ids=["internal_ic", "ec_photons", "pi0_photons"]
)
def test_each_nonsynchrotron_source_feeds_emission_and_cooling(
    small_problem, monkeypatch, added_source
):
    base = jnp.asarray(small_problem[2])
    fields = jnp.zeros((7, base.size)).at[0].set(base).at[added_source].set(0.4 * base)
    freeze_transfer(monkeypatch, fields)
    state = run(small_problem)
    assert bool(state["converged"])
    assert_ic_reads_total(state, small_problem)
    # Compare at the same resulting particles: no cooling-induced N change is hidden.
    np.testing.assert_allclose(
        state["internal_ic_intrinsic_sed_erg_s"],
        1.4 * state["ssc_intrinsic_sed_erg_s"],
        rtol=2e-12,
        atol=1e-260,
    )
    assert_backend_close(
        state["internal_inverse_compton_loss_rate_dgamma_s"],
        1.4 * state["ssc_inverse_compton_loss_rate_dgamma_s"],
    )
    assert np.max(np.asarray(state["internal_ic_intrinsic_sed_erg_s"])) > 0


def test_storage_repartition_and_fixed_external_are_counted_once(
    small_problem, monkeypatch
):
    args, response, base = small_problem
    changed = list(args)
    changed[13] = 1e18 * np.exp(-np.log(np.asarray(args[12]) / (1e-7 * MEC2)) ** 2)
    problem = tuple(changed), response, base
    first_fields = jnp.zeros((7, len(base))).at[0].set(base).at[4].set(0.4 * base)
    freeze_transfer(monkeypatch, first_fields)
    first = run(problem, external=True)
    second_fields = (
        first_fields.at[4].set(0).at[5].set(0.15 * base).at[6].set(0.25 * base)
    )
    freeze_transfer(monkeypatch, second_fields)
    second = run(problem, external=True)
    for state in (first, second):
        assert bool(state["converged"])
        assert_ic_reads_total(state, problem)
        np.testing.assert_array_equal(
            state["external_photon_density_cm3_erg"], changed[13]
        )
        np.testing.assert_allclose(
            state["total_inverse_compton_loss_rate_dgamma_s"],
            state["internal_inverse_compton_loss_rate_dgamma_s"]
            + state["external_inverse_compton_loss_rate_dgamma_s"],
            rtol=2e-13,
            atol=0,
        )
        expected_ext = electron_inverse_compton_loss_rate_dgamma_s_jax(
            jnp.asarray(args[2]), jnp.asarray(args[12]), jnp.asarray(changed[13])
        )
        np.testing.assert_allclose(
            state["external_inverse_compton_loss_rate_dgamma_s"],
            expected_ext,
            rtol=2e-13,
            atol=0,
        )
        # Prescribed external photons are not appended to the internal seed state.
        np.testing.assert_allclose(
            state["internal_ic_seed_photon_density_cm3_erg"],
            1.4 * base,
            rtol=2e-13,
            atol=0,
        )
    disabled = run(problem, internal=False, external=True)
    assert bool(disabled["converged"])
    for key in [
        "internal_ic_intrinsic_sed_erg_s",
        "internal_inverse_compton_loss_rate_dgamma_s",
        "ssc_intrinsic_sed_erg_s",
        "ssc_inverse_compton_loss_rate_dgamma_s",
    ]:
        np.testing.assert_array_equal(disabled[key], 0)
    assert np.any(
        np.asarray(
            disabled["photon_components_intrinsic_sed_erg_s"]["external_compton"]
        )
        > 0
    )
    np.testing.assert_allclose(
        disabled["total_inverse_compton_loss_rate_dgamma_s"],
        disabled["external_inverse_compton_loss_rate_dgamma_s"],
        rtol=2e-13,
        atol=0,
    )
    for key in [
        "internal_ic_intrinsic_sed_erg_s",
        "internal_inverse_compton_loss_rate_dgamma_s",
        "electron_density_dgamma_cm3",
        "escaped_photon_sed_erg_s",
    ]:
        np.testing.assert_allclose(first[key], second[key], rtol=2e-12, atol=1e-260)


def test_frozen_density_direction_jvp_and_finite_differences(
    small_problem, record_property
):
    import time

    energy = jnp.asarray(small_problem[0][0])
    gamma = jnp.asarray(small_problem[0][2])
    particles = 3e-3 * jnp.exp(-jnp.log(gamma / 1e3) ** 2)
    base = jnp.asarray(small_problem[2])
    weights = jnp.asarray([1.0, 0.1, 0.05, 0.02, 0.4, 0.2, 0.3])[:, None]

    def evaluate(log_amplitude):
        density = jnp.sum(weights * base * jnp.exp(log_amplitude), axis=0)
        sed = inverse_compton_sed_jax(
            energy, gamma, particles, energy, density, radius_cm=1e14
        )
        cooling = electron_inverse_compton_loss_rate_dgamma_s_jax(
            gamma, energy, density
        )
        return jnp.log(
            jnp.array([jnp.trapezoid(sed, jnp.log(energy)), jnp.sum(cooling)])
        )

    compiled = jax.jit(evaluate)
    point = jnp.float64(0)
    eager = evaluate(point)
    started = time.perf_counter()
    primal = compiled(point)
    jax.block_until_ready(primal)
    record_property("compile_and_first_seconds", time.perf_counter() - started)
    started = time.perf_counter()
    repeated = compiled(point)
    jax.block_until_ready(repeated)
    record_property("warm_seconds", time.perf_counter() - started)
    assert_backend_close(eager, primal)
    np.testing.assert_array_equal(primal, repeated)
    value, tangent = jax.jvp(compiled, (point,), (jnp.float64(1),))
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(tangent))
    np.testing.assert_allclose(tangent, 1.0, rtol=2e-13, atol=0)
    for step in [1e-3, 3e-4, 1e-4]:
        difference = (compiled(point + step) - compiled(point - step)) / (2 * step)
        np.testing.assert_allclose(tangent, difference, rtol=1e-5, atol=1e-7)


def test_new_internal_ic_photons_feed_the_next_real_em_map(small_problem):
    args, response, base = small_problem
    args = list(args)
    # A soft supplied pi0 tail seeds IC; B=0 and no external field leave true SSC=0.
    args[11] = 1e7 * np.exp(-np.log(np.asarray(args[0]) / (1e-7 * MEC2)) ** 2)
    problem = tuple(args), response, base
    second = run(problem, iterations=2)
    third = run(problem, iterations=3)
    assert np.any(np.asarray(second["internal_ic_intrinsic_sed_erg_s"]) > 0)
    np.testing.assert_array_equal(
        second["photon_components_internal_density_cm3_erg"]["internal_ic"], 0
    )
    assert np.any(
        np.asarray(third["photon_components_internal_density_cm3_erg"]["internal_ic"])
        > 0
    )
    assert_ic_reads_total(third, problem)
    np.testing.assert_array_equal(third["ssc_intrinsic_sed_erg_s"], 0)
    np.testing.assert_array_equal(third["ssc_inverse_compton_loss_rate_dgamma_s"], 0)
    assert np.any(np.asarray(third["internal_inverse_compton_loss_rate_dgamma_s"]) > 0)
    # These deliberately truncated maps test ordering; neither is claimed converged.
    assert not bool(second["converged"])
    assert not bool(third["converged"])


def test_frozen_density_cpu_gpu_equivalence(small_problem):
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CUDA backend is unavailable")
    results = []
    for device in [jax.devices("cpu")[0], gpu]:
        with jax.default_device(device):
            energy = jax.device_put(small_problem[0][0], device)
            gamma = jax.device_put(small_problem[0][2], device)
            density = jax.device_put(small_problem[2], device)
            particles = 3e-3 * jnp.exp(-jnp.log(gamma / 1e3) ** 2)

            def evaluate(
                amplitude,
                energy=energy,
                gamma=gamma,
                density=density,
                particles=particles,
            ):
                seed = density * jnp.exp(amplitude)
                return (
                    inverse_compton_sed_jax(
                        energy, gamma, particles, energy, seed, radius_cm=1e14
                    ),
                    electron_inverse_compton_loss_rate_dgamma_s_jax(
                        gamma, energy, seed
                    ),
                )

            point = jax.device_put(np.float64(0), device)
            primal, tangent = jax.jvp(
                jax.jit(evaluate), (point,), (jnp.ones_like(point),)
            )
            jax.block_until_ready((primal, tangent))
            results.append(tuple(np.asarray(x) for x in (*primal, *tangent)))
    for cpu, gpu_value in zip(*results, strict=True):
        assert_backend_close(gpu_value, cpu)


def test_log_seed_quadrature_matches_analytic_thomson_power():
    from helena.core.constants import SPEED_OF_LIGHT_CGS, THOMSON_CROSS_SECTION_CGS
    from helena.core.inverse_compton_cooling import (
        electron_inverse_compton_loss_rate_dgamma_s,
    )

    energy = MEC2 * np.geomspace(1e-25, 1e-18, 17)
    gamma = np.array([10.0, 1e3, 1e4])
    energy_per_log_interval = 1e-10
    density = energy_per_log_interval / energy**2
    energy_density = energy_per_log_interval * np.log(energy[-1] / energy[0])
    prefactor = 4 * THOMSON_CROSS_SECTION_CGS * SPEED_OF_LIGHT_CGS / (3 * MEC2)
    expected = prefactor * gamma**2 * energy_density
    reference = electron_inverse_compton_loss_rate_dgamma_s(gamma, energy, density)
    inputs = tuple(map(jnp.asarray, (gamma, energy, density)))
    eager = electron_inverse_compton_loss_rate_dgamma_s_jax(*inputs)
    compiled = jax.jit(electron_inverse_compton_loss_rate_dgamma_s_jax)(*inputs)
    for value in [reference, eager, compiled]:
        np.testing.assert_allclose(value, expected, rtol=1e-10, atol=0)
    # Historical direct-dE trapezoids over 1/E have a finite coarse-log-grid bias.
    old_direct = prefactor * gamma**2 * np.trapezoid(energy * density, energy)
    assert np.max(np.abs(old_direct / expected - 1)) > 0.01
