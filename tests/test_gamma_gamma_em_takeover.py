"""M05 round 2: one accepted photon state, external reservoir, and finite AD.

These small-grid integration checks run the production EM owner, never an
alternate solver. Large saved-state checks live in the round-2 run directory.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helena.core.constants import ELECTRON_REST_ENERGY_CGS as MEC2
from helena.core.jax_backend.chains.gamma_gamma_pair_source import (
    prepare_gamma_gamma_total_field_response_jax,
)
from helena.core.jax_backend.chains.steady_state_electromagnetic_cascade import (
    steady_state_electromagnetic_cascade_jax,
)
from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
    prepare_gamma_gamma_pair_response_jax,
)
from helena.core.photon_fields import produced_sed_to_internal_photon_density

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def photon_problem():
    energy = MEC2 * np.array([0.01, 0.1, 0.4, 0.7, 1.0, 1.8, 3.0, 10.0, 30.0, 100.0])
    daughter = np.geomspace(1.000001, 210.0, 257)
    external = MEC2 * np.array([0.2, 0.5, 1.0, 2.0, 3.0])
    response = prepare_gamma_gamma_total_field_response_jax(energy, daughter)
    external_response = prepare_gamma_gamma_pair_response_jax(
        daughter, energy, external
    )
    response["external_target_photon_energy_erg"] = jnp.asarray(external)
    q_gamma = 1e10 * np.exp(-np.log(energy / (3 * MEC2)) ** 2)
    external_density = 1e15 * np.exp(-np.log(external / MEC2) ** 2)
    zero = jnp.zeros_like(jnp.asarray(daughter))
    args = (
        energy,
        energy,
        daughter,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([1.01, 10.0, 100.0]),
        jnp.zeros(3),
        q_gamma,
        external,
        external_density,
    )
    return args, response, external_response


def solve(
    problem,
    *,
    internal=True,
    external=False,
    log_radius=0.0,
    iterations=96,
    meson_sources=None,
):
    args, response, external_response = problem
    return steady_state_electromagnetic_cascade_jax(
        *args,
        magnetic_field_G=0.0,
        radius_cm=1e14 * jnp.exp(log_radius),
        adiabatic_rate_s_inv=1e-6,
        escape_rate_s_inv=3e-4,
        enable_ssc=False,
        enable_ec=False,
        enable_internal_gamma_gamma=internal,
        include_external_gamma_gamma_target=external,
        gamma_gamma_pair_response=response,
        external_gamma_gamma_pair_response=external_response,
        relaxation_omega=0.7,
        relative_tolerance=1e-9,
        maximum_iterations=iterations,
        meson_synchrotron_intrinsic_seds_erg_s=meson_sources,
    )


@pytest.mark.parametrize(
    "internal,external", [(False, False), (True, False), (False, True), (True, True)]
)
def test_common_state_and_fixed_external_switches(
    photon_problem, internal, external, record_property
):
    result = solve(photon_problem, internal=internal, external=external)
    jax.block_until_ready(result)
    assert bool(result["converged"])
    assert float(result["final_photon_map_number_residual"]) <= 1e-9
    assert float(result["final_photon_map_energy_residual"]) <= 1e-9
    np.testing.assert_array_equal(
        result["external_photon_density_cm3_erg"], photon_problem[0][13]
    )
    density = result["internal_photon_density_cm3_erg"]
    np.testing.assert_allclose(
        sum(result["photon_components_internal_density_cm3_erg"].values()),
        density,
        rtol=3e-13,
        atol=0,
    )
    np.testing.assert_allclose(
        sum(result["photon_components_escaped_sed_erg_s"].values()),
        result["escaped_photon_sed_erg_s"],
        rtol=3e-13,
        atol=0,
    )
    produced = sum(result["photon_components_intrinsic_sed_erg_s"].values())
    reference = produced_sed_to_internal_photon_density(
        np.asarray(result["photon_energy_erg"]),
        np.asarray(produced),
        np.asarray(result["gamma_gamma_absorption_coefficient_cm1"]),
        radius_cm=1e14,
    )
    # B=0 makes SSA zero; this is the independent NumPy sphere transfer check.
    np.testing.assert_allclose(density, reference, rtol=2e-8, atol=1e-280)
    internal_rate = float(result["internal_gamma_gamma_budget"]["event_rate_cm3_s"])
    external_rate = float(result["external_gamma_gamma_event_rate_cm3_s"])
    external_budget = result["external_gamma_gamma_budget"]
    for key, value in external_budget.items():
        if np.ndim(value) == 0:
            record_property("external_" + key, float(value))
    reservoir_loss = external_budget["reservoir_photon_loss_rate_cm3_s_erg"]
    external_energy = jnp.asarray(photon_problem[0][12])
    reservoir_number = jnp.trapezoid(
        reservoir_loss * external_energy, jnp.log(external_energy)
    )
    reservoir_power = jnp.trapezoid(
        reservoir_loss * external_energy**2, jnp.log(external_energy)
    )
    np.testing.assert_allclose(reservoir_number, external_rate, rtol=3e-12, atol=0)
    np.testing.assert_allclose(
        reservoir_power,
        external_budget["reservoir_photon_loss_power_erg_cm3_s"],
        rtol=3e-12,
        atol=0,
    )
    weights = photon_problem[1]["photon_integration_weights_erg"]
    lost_photons = float(jnp.sum(result["photon_loss_rate_cm3_s_erg"] * weights))
    np.testing.assert_allclose(
        lost_photons, 2 * internal_rate + external_rate, rtol=3e-12, atol=0
    )
    if external:
        assert abs(float(external_budget["pair_number_relative_residual"])) <= 5e-3
        assert abs(float(external_budget["pair_energy_relative_residual"])) <= 5e-3
    assert (internal_rate > 0) if internal else (internal_rate == 0)
    assert (external_rate > 0) if external else (external_rate == 0)
    if external:
        assert float(result["external_reservoir_loss_power_erg_cm3_s"]) > 0
    if not (internal or external):
        np.testing.assert_array_equal(result["photon_loss_rate_cm3_s_erg"], 0)
        np.testing.assert_array_equal(
            result["high_resolution_electron_pair_injection_dgamma_cm3_s"], 0
        )
    for value in jax.tree_util.tree_leaves(result):
        if hasattr(value, "dtype") and np.issubdtype(value.dtype, np.number):
            assert np.all(np.isfinite(value))


def test_exhausted_iterations_are_not_success(photon_problem):
    result = solve(photon_problem, iterations=1)
    assert not bool(result["converged"])
    assert float(result["final_cascade_residual"]) > 1e-9


def test_radius_feedback_jvp_matches_multiple_difference_steps(photon_problem):
    energy = jnp.asarray(photon_problem[0][0])

    def target(log_radius):
        result = solve(photon_problem, log_radius=log_radius)
        number = jnp.trapezoid(
            result["internal_photon_density_cm3_erg"] * energy, jnp.log(energy)
        )
        rate = result["internal_gamma_gamma_budget"]["event_rate_cm3_s"]
        escaped = jnp.trapezoid(result["escaped_photon_sed_erg_s"], jnp.log(energy))
        return jnp.log(jnp.array([number, rate, escaped]))

    compiled = jax.jit(target)
    point = jnp.float64(0)
    primal, tangent = jax.jvp(compiled, (point,), (jnp.float64(1),))
    jax.block_until_ready((primal, tangent))
    assert np.all(np.isfinite(tangent))
    assert np.any(np.abs(tangent) > 0.1)
    for step in (3e-4, 1e-4, 3e-5):
        difference = (compiled(point + step) - compiled(point - step)) / (2 * step)
        np.testing.assert_allclose(tangent, difference, rtol=1e-4, atol=1e-6)


def test_stale_external_response_identity_is_rejected(photon_problem):
    args, response, external_response = photon_problem
    stale = list(args)
    stale[12] = 2 * np.asarray(stale[12])
    with pytest.raises(ValueError, match="(?i)external"):
        solve((tuple(stale), response, external_response), external=True)


def test_hard_external_daughter_support_is_reported(photon_problem):
    args, response, _ = photon_problem
    hard = list(args)
    hard[12] = MEC2 * np.array([2e3, 5e3, 1e4, 3e4, 1e5])
    hard[13] = np.full(5, 1e9)
    current = {**response, "external_target_photon_energy_erg": jnp.asarray(hard[12])}
    external_response = prepare_gamma_gamma_pair_response_jax(
        args[2], args[0], hard[12]
    )
    result = solve(
        (tuple(hard), current, external_response), internal=False, external=True
    )
    budget = result["external_gamma_gamma_budget"]
    assert float(budget["event_rate_cm3_s"]) > 0
    assert float(budget["outside_number_fraction_bound"]) > 0.9
    assert float(budget["outside_energy_fraction_bound"]) > 0.9


def test_source_storage_partition_does_not_select_collisions(photon_problem):
    energy = np.asarray(photon_problem[0][0])
    source = 1e40 * np.exp(-np.log(energy / (3 * MEC2)) ** 2)
    zero = np.zeros_like(source)
    first = solve(
        photon_problem, meson_sources={"pion_total": source, "muon_total": zero}
    )
    # A storage partition through the old MeV region must not select collisions.
    lower = energy < 1.4 * MEC2
    second = solve(
        photon_problem,
        meson_sources={
            "pion_total": np.where(lower, source, 0),
            "muon_total": np.where(lower, 0, source),
        },
    )
    assert bool(first["converged"]) and bool(second["converged"])
    for name in [
        "internal_photon_density_cm3_erg",
        "gamma_gamma_absorption_coefficient_cm1",
        "high_resolution_electron_pair_injection_dgamma_cm3_s",
        "escaped_photon_sed_erg_s",
    ]:
        np.testing.assert_allclose(first[name], second[name], rtol=3e-12, atol=0)


def test_joint_em_backend_equivalence(photon_problem, record_property):
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CUDA backend is unavailable")
    import time

    records = {}
    for backend, device in [("cpu", jax.devices("cpu")[0]), ("gpu", gpu)]:
        with jax.default_device(device):
            local = jax.device_put(photon_problem, device)

            def evaluate(log_radius, local=local):
                result = solve(local, log_radius=log_radius, external=True)
                return tuple(
                    result[name]
                    for name in [
                        "internal_photon_density_cm3_erg",
                        "gamma_gamma_absorption_coefficient_cm1",
                        "high_resolution_electron_pair_injection_dgamma_cm3_s",
                        "escaped_photon_sed_erg_s",
                    ]
                )

            point = jax.device_put(jnp.float64(0), device)
            eager = evaluate(point)
            compiled = jax.jit(evaluate)
            start = time.perf_counter()
            value = compiled(point)
            jax.block_until_ready(value)
            record_property(
                backend + "_first_compile_execution_seconds",
                time.perf_counter() - start,
            )
            start = time.perf_counter()
            repeated = compiled(point)
            jax.block_until_ready(repeated)
            record_property(backend + "_warm_seconds", time.perf_counter() - start)
            _, tangent = jax.jvp(
                compiled, (point,), (jax.device_put(jnp.float64(1), device),)
            )
            jax.block_until_ready(tangent)
            records[backend] = (
                tuple(np.asarray(v) for v in value),
                tuple(np.asarray(v) for v in tangent),
            )
            for a, b, c in zip(eager, value, repeated, strict=True):
                scale = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
                assert np.max(np.abs(np.asarray(a) - np.asarray(b))) / scale < 1e-10
                np.testing.assert_array_equal(b, c)
    for cpu, gpu_values in zip(records["cpu"], records["gpu"], strict=True):
        for a, b in zip(cpu, gpu_values, strict=True):
            scale = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
            assert np.max(np.abs(a - b)) / scale < 1e-10


def test_optional_radiation_sources_share_the_accepted_state():
    energy = MEC2 * np.array(
        [
            1e-12,
            1e-10,
            1e-8,
            1e-6,
            1e-4,
            0.01,
            0.1,
            0.4,
            0.7,
            1.0,
            1.8,
            3.0,
            10.0,
            30.0,
            100.0,
        ]
    )
    gamma = np.geomspace(1.000001, 210.0, 257)
    external = MEC2 * np.array([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    external_density = 1e17 * np.exp(-np.log(external / (1e-5 * MEC2)) ** 2)
    response = prepare_gamma_gamma_total_field_response_jax(energy, gamma)
    zero = np.zeros_like(gamma)
    primary = 1e-6 * np.exp(-np.log(gamma / 30.0) ** 2)
    arguments = (
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
        np.array([10.0, 3.0, 1.0]),
        1e10 * np.exp(-np.log(energy / (3 * MEC2)) ** 2),
        external,
        external_density,
    )
    sources = {
        "pion_total": 1e38 * np.exp(-np.log(energy / (1e-6 * MEC2)) ** 2),
        "muon_total": 2e38 * np.exp(-np.log(energy / (1e-5 * MEC2)) ** 2),
    }

    def run(enabled):
        return steady_state_electromagnetic_cascade_jax(
            *arguments,
            magnetic_field_G=10.0,
            radius_cm=1e14,
            adiabatic_rate_s_inv=1e-6,
            escape_rate_s_inv=3e-4,
            enable_ssc=enabled,
            enable_ec=enabled,
            meson_synchrotron_intrinsic_seds_erg_s=sources,
            gamma_gamma_pair_response=response,
            relative_tolerance=1e-9,
            relaxation_omega=0.7,
            maximum_iterations=128,
        )

    on = run(True)
    off = run(False)
    assert bool(on["converged"]) and bool(off["converged"])
    assert set(on["photon_components_intrinsic_sed_erg_s"]) == {
        "electron_synchrotron",
        "proton_synchrotron",
        "pion_synchrotron",
        "muon_synchrotron",
        "internal_ic",
        "external_compton",
        "pi0",
    }
    for value in on["photon_components_intrinsic_sed_erg_s"].values():
        assert np.any(np.asarray(value) > 0)
    for state in [on, off]:
        np.testing.assert_allclose(
            sum(state["photon_components_internal_density_cm3_erg"].values()),
            state["internal_photon_density_cm3_erg"],
            rtol=3e-13,
            atol=0,
        )
    for name in ["internal_ic", "external_compton"]:
        np.testing.assert_array_equal(
            off["photon_components_intrinsic_sed_erg_s"][name], 0
        )
    np.testing.assert_array_equal(off["ssc_inverse_compton_loss_rate_dgamma_s"], 0)
    np.testing.assert_array_equal(off["external_inverse_compton_loss_rate_dgamma_s"], 0)


def test_jit_rejects_negative_ssa_hidden_by_positive_gamma_gamma(
    photon_problem, monkeypatch
):
    import importlib

    em = importlib.import_module(
        "helena.core.jax_backend.chains.steady_state_electromagnetic_cascade"
    )
    args, _, _ = photon_problem
    energy = jnp.asarray(args[0])
    external = jnp.asarray(args[12])
    external_density = jnp.asarray(args[13])
    ext_alpha = em._angle_averaged_cross_section_jax(
        energy[:, None] * external[None, :] / MEC2**2
    ) @ (external_density * external * em._trapezoid_weights(jnp.log(external)))
    negative_ssa = jnp.where(jnp.arange(energy.size) == 5, -0.5 * ext_alpha, 0.0)
    assert float(negative_ssa[5]) < 0
    monkeypatch.setattr(
        em,
        "synchrotron_absorption_coefficient_jax",
        lambda *args, **kwargs: negative_ssa,
    )
    # Patch precedes a distinct first tracing; no old compiled solve is reused.
    result = jax.jit(lambda: solve(photon_problem, external=True))()
    jax.block_until_ready(result)
    assert np.all(np.asarray(result["photon_absorption_coefficient_cm1"]) >= 0)
    assert np.any(
        np.asarray(result["electron_synchrotron_absorption_coefficient_cm1"]) < 0
    )
    assert not bool(result["converged"])
    assert bool(result["iteration_limit_reached"])
    assert np.isinf(result["final_cascade_residual"])
