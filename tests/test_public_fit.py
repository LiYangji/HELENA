"""Focused P4 checks for the one public fitting entry and result."""

from dataclasses import FrozenInstanceError, replace

import jax.numpy as jnp
import numpy as np
import pytest

from helena import FitResult, ModelConfig, OneZoneModel, SEDDataset, fit
from helena.core.observer import transform_comoving_sed
from helena.fitting import public as public_fitting
from helena.parameters import DIFFERENTIABLE_PARAMETER_NAMES


def _config():
    return ModelConfig(
        magnetic_field_G=0.1,
        radius_cm=1.0e16,
        redshift=0.1,
        doppler_factor=30.0,
        luminosity_distance_cm=1.0e27,
        electron_kinetic_injection_luminosity_erg_s=3.7e40,
        electron_injection_index=2.0,
        electron_gamma_max=3.0e5,
        proton_kinetic_injection_luminosity_erg_s=2.8e46,
        proton_injection_index=2.0,
        proton_gamma_max=1.0e7,
    )


def _dataset(observed):
    observed = np.asarray(observed)
    size = observed.size
    return SEDDataset(
        energy_erg=np.arange(1, size + 1, dtype=float),
        sed_flux_erg_cm2_s=observed,
        energy_error_lower_erg=np.full(size, 0.1),
        energy_error_upper_erg=np.full(size, 0.1),
        sed_flux_error_lower_erg_cm2_s=0.1 * observed,
        sed_flux_error_upper_erg_cm2_s=0.1 * observed,
        instrument=np.full(size, "synthetic"),
        statistic_type=np.full(size, "gaussian"),
        scientific_use=np.full(size, "fit"),
    )


def _install_linear_prediction(monkeypatch, truth_by_name):
    names = tuple(truth_by_name)
    design = jnp.asarray(
        np.vstack(
            [
                np.linspace(0.4 + index, 1.1 + index, len(names) + 2)
                for index in range(len(names))
            ]
        ).T
    )
    truth = jnp.asarray([truth_by_name[name] for name in names])
    observed = np.exp(np.asarray(design @ truth))

    def model_prediction(_model, selection):
        columns = jnp.asarray([names.index(name) for name in selection.free_names])

        def prediction(values):
            canonical_values = jnp.zeros(len(names)).at[columns].set(values)
            return jnp.exp(design @ canonical_values)

        return np.arange(1, observed.size + 1, dtype=float), prediction

    monkeypatch.setattr(public_fitting, "_model_prediction", model_prediction)
    return _dataset(observed)


def test_root_imports_selection_validation_and_eight_parameter_setup(monkeypatch):
    assert SEDDataset and fit and FitResult
    model = OneZoneModel(_config())
    dataset = _dataset([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="at least one"):
        public_fitting._prepare_fit_request(model, dataset, [], "lbfgsb")
    with pytest.raises(ValueError, match="unknown parameter"):
        public_fitting._prepare_fit_request(
            model, dataset, ["not_a_parameter"], "lbfgsb"
        )
    with pytest.raises(ValueError, match="unique"):
        public_fitting._prepare_fit_request(
            model, dataset, ["log_Le", "log_Le"], "lbfgsb"
        )
    with pytest.raises(ValueError, match="at most eight"):
        public_fitting._prepare_fit_request(
            model,
            dataset,
            [*DIFFERENTIABLE_PARAMETER_NAMES, "ninth"],
            "lbfgsb",
        )
    with pytest.raises(ValueError, match="unknown parameter"):
        public_fitting._prepare_fit_request(model, dataset, ["enable_ssc"], "lbfgsb")

    monkeypatch.setattr(public_fitting.jax, "default_backend", lambda: "gpu")
    selection = public_fitting._prepare_fit_request(
        model, dataset, DIFFERENTIABLE_PARAMETER_NAMES, "lbfgsb"
    )
    assert selection.free_names == DIFFERENTIABLE_PARAMETER_NAMES
    assert selection.pack(selection.baseline_values).shape == (8,)
    assert selection.unpack(selection.pack(selection.baseline_values)).shape == (8,)


def test_cpu_high_dimensional_request_warns_without_changing_selection(monkeypatch):
    model = OneZoneModel(_config())
    dataset = _dataset([1.0, 1.0, 1.0])
    monkeypatch.setattr(public_fitting.jax, "default_backend", lambda: "cpu")

    with pytest.warns(RuntimeWarning, match="more than two"):
        selection = public_fitting._prepare_fit_request(
            model, dataset, ["log_B", "log_Le", "p_e"], "migrad"
        )

    assert selection.free_names == ("log_B", "log_Le", "p_e")
    assert selection.free_bounds is None


def test_fit_uses_canonical_broadband_support_around_electron_synchrotron_peak(
    monkeypatch,
):
    config = replace(_config(), redshift=0.0)
    model = OneZoneModel(config)
    monkeypatch.setattr(
        OneZoneModel,
        "_get_prepared_responses",
        lambda _self: {
            "bethe_heitler_pair_responses": {},
            "gamma_gamma_pair_response": None,
        },
    )

    def outer_step(state, _external, dynamic, **_kwargs):
        amplitude = jnp.sum(dynamic["primary_electron_injection_dgamma_cm3_s"])
        return {
            "next_internal_photon_state": state,
            "proton_density_dgamma_cm3": jnp.ones_like(dynamic["proton_gamma"]),
            "inner_electromagnetic_cascade": {
                "converged": jnp.asarray(True),
                "photon_energy_erg": dynamic["gamma_ray_energy_erg"],
                "escaped_photon_sed_erg_s": amplitude
                * jnp.ones_like(dynamic["gamma_ray_energy_erg"]),
            },
        }

    monkeypatch.setattr(public_fitting, "outer_hadronic_picard_step", outer_step)
    monkeypatch.setattr(
        public_fitting,
        "electromagnetic_ledger_passed_jax",
        lambda *_: jnp.asarray(True),
    )

    frequency_hz = 1.33e17
    energy_erg = np.asarray([6.62607015e-27 * frequency_hz])

    def dataset(flux, energy=energy_erg):
        flux = np.asarray([flux])
        return SEDDataset(
            energy_erg=energy,
            sed_flux_erg_cm2_s=flux,
            energy_error_lower_erg=np.zeros(1),
            energy_error_upper_erg=np.zeros(1),
            sed_flux_error_lower_erg_cm2_s=0.1 * flux,
            sed_flux_error_upper_erg_cm2_s=0.1 * flux,
            instrument=np.asarray(["synthetic"]),
            statistic_type=np.asarray(["gaussian"]),
            scientific_use=np.asarray(["fit"]),
        )

    placeholder = dataset(1.0)
    selection = public_fitting._prepare_fit_request(
        model, placeholder, ["log_Le"], "lbfgsb"
    )
    model_energy, prediction = public_fitting._model_prediction(model, selection)
    high_energy = np.asarray(
        model._setup["production_configuration"]["gamma_ray_energy_erg"]
    )
    high_energy_observer, _ = transform_comoving_sed(
        high_energy,
        np.ones_like(high_energy),
        doppler_factor=config.doppler_factor,
        redshift=config.redshift,
    )
    # The full common grid still includes the synchrotron peak below the old
    # native gamma-grid minimum; no low/high or 1 MeV output selection remains.
    old_gamma_min_observer = (
        1.0e2 * 1.602176634e-12 * config.doppler_factor / (1.0 + config.redshift)
    )
    assert model_energy[0] < energy_erg[0] < old_gamma_min_observer
    np.testing.assert_array_equal(model_energy, high_energy_observer)

    coordinates = selection.pack(selection.baseline_values)
    truth_flux = float(np.asarray(prediction(jnp.asarray(coordinates)))[0])
    result = fit(
        model,
        dataset(truth_flux),
        parameters=["log_Le"],
        optimizer="lbfgsb",
    )
    assert isinstance(result, FitResult)
    assert result.success
    assert result.free_parameters == ("log_Le",)
    assert result.best_parameters["log_Le"] == pytest.approx(
        config.electron_kinetic_injection_luminosity_erg_s
    )

    with pytest.raises(ValueError, match="inside the model domain"):
        fit(
            model,
            dataset(truth_flux, energy=np.asarray([0.5 * model_energy[0]])),
            parameters=["log_Le"],
            optimizer="lbfgsb",
        )


def test_one_parameter_lbfgsb_fit_reports_physical_result(monkeypatch):
    truth = 0.12
    dataset = _install_linear_prediction(monkeypatch, {"log_Le": truth})
    config = _config()
    model = OneZoneModel(config)

    result = fit(
        model,
        dataset,
        parameters=["log_Le"],
        optimizer="lbfgsb",
        options={"maxiter": 30},
    )

    assert isinstance(result, FitResult)
    assert result.success
    assert result.free_parameters == ("log_Le",)
    assert result.joint_dimension == 1
    assert result.optimizer == "lbfgsb"
    assert result.uncertainties is None and result.covariance is None
    assert result.best_parameters["log_Le"] == pytest.approx(
        config.electron_kinetic_injection_luminosity_erg_s * np.exp(truth),
        rel=1.0e-8,
    )
    assert (
        result.best_config.electron_kinetic_injection_luminosity_erg_s
        == pytest.approx(result.best_parameters["log_Le"])
    )
    assert model.config is config
    with pytest.raises(FrozenInstanceError):
        result.nll = 0.0


@pytest.mark.parametrize("optimizer", ("lbfgsb", "migrad"))
def test_two_parameter_recovery_shared_optimizer_contract(monkeypatch, optimizer):
    truth = {"log_B": -0.08, "log_Le": 0.15}
    dataset = _install_linear_prediction(monkeypatch, truth)
    config = _config()
    model = OneZoneModel(config)

    result = fit(
        model,
        dataset,
        parameters=["log_B", "log_Le"],
        optimizer=optimizer,
        options=(
            {"maxiter": 50} if optimizer == "lbfgsb" else {"ncall": 100, "iterate": 1}
        ),
    )

    assert result.success
    assert result.free_parameters == ("log_B", "log_Le")
    assert result.joint_dimension == 2
    assert result.optimizer == optimizer
    assert result.nll == pytest.approx(
        4.0 * (np.log(0.2) + 0.5 * np.log(np.pi / 2.0)), abs=1.0e-10
    )
    assert result.best_parameters["log_B"] == pytest.approx(
        config.magnetic_field_G * np.exp(truth["log_B"]), rel=1.0e-7
    )
    assert result.best_parameters["log_Le"] == pytest.approx(
        config.electron_kinetic_injection_luminosity_erg_s * np.exp(truth["log_Le"]),
        rel=1.0e-7,
    )
    assert result.fixed_parameters["log_Lp"] == (
        config.proton_kinetic_injection_luminosity_erg_s
    )
    assert result.best_config.proton_kinetic_injection_luminosity_erg_s == (
        config.proton_kinetic_injection_luminosity_erg_s
    )
    assert model.config is config
    np.testing.assert_array_equal(result.prediction["indices"], np.arange(4))
    assert result.residuals["upper_limits"]["indices"].size == 0
