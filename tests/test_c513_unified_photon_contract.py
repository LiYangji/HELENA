"""C513 grid and accepted-output contracts after M05 total-field takeover."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from helena.core.constants import ELECTRON_REST_ENERGY_CGS
from helena.models.c513 import (
    _canonical_broadband_escaped_photon_components,
    _canonical_broadband_escaped_photon_sed,
    build_c513_configuration,
)


def test_full_grid_and_two_photon_daughter_support():
    with jax.default_device(jax.devices("cpu")[0]):
        setup = build_c513_configuration()
    configuration = setup["production_configuration"]
    initial = setup["initial_internal_photon_state_blob"]
    energy = np.asarray(configuration["gamma_ray_energy_erg"])
    np.testing.assert_array_equal(
        configuration["synchrotron_photon_energy_erg"], energy
    )
    np.testing.assert_array_equal(initial["energy_erg"], energy)
    assert energy.size == 249
    assert np.all(np.diff(energy) > 0)
    assert set(initial) == {"energy_erg", "density_cm3_erg"}
    assert "synchrotron_split_energy_erg" not in configuration
    assert (
        float(configuration["electron_gamma"][-1])
        >= 2 * energy[-1] / ELECTRON_REST_ENERGY_CGS
    )
    transport = np.asarray(configuration["electron_gamma"])
    daughter = np.asarray(configuration["pair_daughter_gamma"])
    assert transport.size == 513 and daughter.size == 537
    np.testing.assert_array_equal(daughter[[0, -1]], transport[[0, -1]])
    np.testing.assert_array_equal(
        daughter[np.searchsorted(daughter, transport)], transport
    )
    added = daughter[~np.isin(daughter, transport)]
    assert added.size == 24
    assert np.all((added > transport[0]) & (added < transport[1]))
    kinetic = np.r_[transport[0], added, transport[1]] - 1.0
    np.testing.assert_allclose(
        kinetic[1:] / kinetic[:-1], kinetic[1] / kinetic[0], rtol=2e-10
    )
    assert setup["metadata"]["grid_checks"]["pair_daughter_same_endpoints"]


def test_canonical_components_read_the_accepted_state_without_retransfer():
    energy = np.array([1e-12, 1e-6, 1.0])
    names = (
        "electron_synchrotron",
        "proton_synchrotron",
        "pion_synchrotron",
        "muon_synchrotron",
        "internal_ic",
        "external_compton",
        "pi0",
    )
    components = {
        name: (index + 1) * np.array([1.0, 2.0, 3.0])
        for index, name in enumerate(names)
    }
    inner = {
        "photon_energy_erg": energy,
        "photon_components_escaped_sed_erg_s": components,
        "escaped_photon_sed_erg_s": sum(components.values()),
    }
    actual_energy, total = _canonical_broadband_escaped_photon_sed({}, inner)
    component_energy, actual = _canonical_broadband_escaped_photon_components(
        {}, {"inner_electromagnetic_cascade": inner}
    )
    assert actual_energy is energy and component_energy is energy
    assert total is inner["escaped_photon_sed_erg_s"]
    assert (
        actual["transported_lepton_synchrotron"] is components["electron_synchrotron"]
    )
    assert actual["external_compton"] is components["external_compton"]
    assert (
        actual["transported_lepton_internal_inverse_compton"]
        is components["internal_ic"]
    )
    assert "transported_lepton_ssc" not in actual
    np.testing.assert_array_equal(sum(actual.values()), total)


@pytest.mark.parametrize(("maximum", "accepted"), [(2, False), (4, True)])
def test_prediction_rejects_outer_limit_and_requires_consecutive_passes(
    monkeypatch, maximum, accepted
):
    """Control-flow proof only; scientific forward is validated separately."""
    from helena.fitting import public

    with jax.default_device(jax.devices("cpu")[0]):
        setup = build_c513_configuration()
        setup["metadata"]["numerical_strategy"].update(
            max_outer_iterations=maximum, outer_rtol=1e-8, required_consecutive_passes=2
        )
        energy = setup["initial_internal_photon_state_blob"]["energy_erg"]
        monkeypatch.setattr(
            public,
            "_earth_broadband_photon_support",
            lambda *_: (
                np.asarray(energy),
                np.asarray(energy),
                np.ones(energy.shape),
                np.arange(energy.size),
                np.ones(energy.shape),
            ),
        )
        monkeypatch.setattr(
            public,
            "electromagnetic_ledger_passed_jax",
            lambda *_: jax.numpy.asarray(True),
        )

        def fixed_point(state, _external, configuration, **_kwargs):
            return {
                "proton_density_dgamma_cm3": jax.numpy.ones_like(
                    configuration["proton_gamma"]
                ),
                "next_internal_photon_state": {
                    "energy_erg": energy,
                    "density_cm3_erg": jax.numpy.ones_like(energy),
                },
                "inner_electromagnetic_cascade": {
                    "converged": jax.numpy.asarray(True),
                    "photon_energy_erg": energy,
                    "escaped_photon_sed_erg_s": 42 * jax.numpy.ones_like(energy),
                },
            }

        monkeypatch.setattr(public, "outer_hadronic_picard_step", fixed_point)
        model = SimpleNamespace(
            _setup=setup,
            config=None,
            _get_prepared_responses=lambda: {
                "bethe_heitler_pair_responses": {},
                "gamma_gamma_pair_response": {},
            },
        )
        selection = SimpleNamespace(baseline_values=np.zeros(8), unpack=lambda x: x)
        _, prediction = public._model_prediction(model, selection)
        result = np.asarray(jax.jit(prediction)(jax.numpy.zeros(8)))
    if accepted:
        np.testing.assert_array_equal(result, np.full(energy.shape, 42.0))
    else:
        assert np.isnan(result).all()


@pytest.mark.parametrize("changed_axes", ["both_config", "synchrotron_only"])
def test_same_shape_changed_photon_values_rejected_before_prepare_or_forward(
    monkeypatch, tmp_path, changed_axes
):
    from helena.models import c513

    with jax.default_device(jax.devices("cpu")[0]):
        setup = build_c513_configuration()
        configuration = dict(setup["production_configuration"])
        changed = configuration["gamma_ray_energy_erg"] * 1.001
        configuration["synchrotron_photon_energy_erg"] = changed
        if changed_axes == "both_config":
            configuration["gamma_ray_energy_erg"] = changed
        assert (
            changed.shape
            == setup["initial_internal_photon_state_blob"]["energy_erg"].shape
        )

        def forbidden(*_args, **_kwargs):
            raise AssertionError(
                "mismatched grid reached response preparation or model"
            )

        monkeypatch.setattr(
            c513, "load_or_prepare_bethe_heitler_pair_response_jax", forbidden
        )
        monkeypatch.setattr(c513, "solve_steady_state_hadronic_cascade", forbidden)
        bad_setup = {**setup, "production_configuration": configuration}
        with pytest.raises(ValueError, match="grids must match exactly"):
            c513.prepare_c513_responses(bad_setup, cache_dir=tmp_path)
        with pytest.raises(ValueError, match="grids must match exactly"):
            c513.run_c513_forward(
                setup=setup,
                production_configuration=configuration,
                prepared_responses={},
                observer_options={},
            )


def test_photon_setup_guard_rejects_tracers_before_numpy_conversion():
    from helena.models.c513 import _validate_common_photon_grid

    @jax.jit
    def attempt(energy):
        _validate_common_photon_grid(
            {
                "initial_internal_photon_state_blob": {"energy_erg": energy},
                "production_configuration": {
                    "synchrotron_photon_energy_erg": energy,
                    "gamma_ray_energy_erg": energy,
                },
            }
        )
        return energy

    with (
        jax.default_device(jax.devices("cpu")[0]),
        pytest.raises(ValueError, match="grids must be concrete"),
    ):
        attempt(jax.numpy.array([1e-12, 1e-6, 1.0]))
