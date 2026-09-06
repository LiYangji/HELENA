"""Focused P6 checks for public local Gauss-Newton diagnostics."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import helena.diagnostics as diagnostics
from helena import SEDDataset
from helena.diagnostics import GaussNewtonResult, gauss_newton
from helena.fitting.core import (
    LogSEDDataset,
    _local_identifiability_diagnostics,
    add_postfit_diagnostics,
)
from helena.sensitivity import SensitivityResult


def _dataset(energy, *, statistics=None, uses=None):
    energy = np.asarray(energy, dtype=float)
    size = energy.size
    statistics = np.asarray(statistics or ["gaussian"] * size)
    uses = np.asarray(uses or ["fit"] * size)
    flux = np.linspace(1.0, 2.0, size)
    lower = 0.1 * flux
    upper = lower.copy()
    upper_limit = statistics == "upper_limit"
    asymmetric = statistics == "asymmetric_gaussian"
    lower[upper_limit] = upper[upper_limit] = 0.0
    upper[asymmetric] *= 1.5
    return SEDDataset(
        energy_erg=energy,
        sed_flux_erg_cm2_s=flux,
        energy_error_lower_erg=0.01 * energy,
        energy_error_upper_erg=0.01 * energy,
        sed_flux_error_lower_erg_cm2_s=lower,
        sed_flux_error_upper_erg_cm2_s=upper,
        instrument=np.full(size, "synthetic"),
        statistic_type=statistics,
        scientific_use=uses,
    )


def _jacobian(values, *, direction=None):
    return SensitivityResult(
        energy_erg=np.asarray((1.0, 2.0, 4.0)),
        values=np.asarray(values),
        parameters=("log_B", "log_Le"),
        support_indices=np.asarray((0, 1, 2)),
        direction=direction,
    )


def test_diagnostics_exports_only_the_frozen_scientific_surface():
    assert diagnostics.__all__ == ["GaussNewtonResult", "gauss_newton"]


def test_gauss_newton_known_matrix_log_interpolation_and_inactive_rows():
    model_jacobian = np.asarray(((1.0, 0.0), (0.0, 2.0), (1.0, 1.0)))
    result = _jacobian(model_jacobian)
    dataset = _dataset(
        (1.0, np.sqrt(2.0), 2.0, 4.0, 3.0),
        statistics=(
            "gaussian",
            "gaussian",
            "asymmetric_gaussian",
            "gaussian",
            "upper_limit",
        ),
        uses=("fit", "constraint", "context", "fit", "exclude"),
    )
    diagnostics = gauss_newton(result, dataset)

    sampled = np.column_stack(
        [
            np.interp(
                np.log(dataset.energy_erg[dataset.likelihood_mask]),
                np.log(result.energy_erg),
                column,
            )
            for column in model_jacobian.T
        ]
    )
    weighted = sampled / 0.1
    curvature = weighted.T @ weighted
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    covariance = np.asarray(((200.0, -150.0), (-150.0, 225.0))) / 22500.0
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)

    assert isinstance(diagnostics, GaussNewtonResult)
    assert diagnostics.parameters == ("log_B", "log_Le")
    assert diagnostics.status == "locally_identifiable"
    assert diagnostics.rank == 2
    assert "local Gauss-Newton" in diagnostics.covariance_label
    np.testing.assert_array_equal(diagnostics.active_indices, (0, 1, 3))
    np.testing.assert_allclose(diagnostics.weighted_jacobian, weighted)
    np.testing.assert_allclose(diagnostics.curvature, curvature)
    np.testing.assert_allclose(diagnostics.covariance, covariance)
    np.testing.assert_allclose(diagnostics.correlation, correlation)
    np.testing.assert_allclose(diagnostics.singular_values, singular_values)
    assert diagnostics.condition_number == pytest.approx(
        singular_values[0] / singular_values[-1]
    )
    assert not diagnostics.curvature.flags.writeable
    with pytest.raises(FrozenInstanceError):
        diagnostics.rank = 0


def test_rank_deficiency_is_reported_without_regularization():
    result = _jacobian(((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)))
    diagnostics = gauss_newton(result, _dataset((1.0, 2.0, 4.0)))

    assert diagnostics.status == "rank_deficient"
    assert diagnostics.rank == 1
    assert np.isinf(diagnostics.condition_number)
    np.testing.assert_allclose(diagnostics.covariance, np.ones((2, 2)) / 5600.0)
    np.testing.assert_allclose(diagnostics.correlation, np.ones((2, 2)))


@pytest.mark.parametrize("statistic", ("asymmetric_gaussian", "upper_limit"))
def test_active_unsupported_statistics_fail_clearly(statistic):
    result = _jacobian(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    dataset = _dataset(
        (1.0, 2.0, 4.0),
        statistics=("gaussian", statistic, "gaussian"),
    )
    with pytest.raises(ValueError, match="unsupported active types"):
        gauss_newton(result, dataset)


def test_jvp_empty_selection_and_out_of_domain_fail_clearly():
    jvp_result = _jacobian((1.0, 2.0, 3.0), direction=(0.3, -0.2))
    with pytest.raises(ValueError, match="not a JVP"):
        gauss_newton(jvp_result, _dataset((1.0, 2.0, 4.0)))

    jacobian = _jacobian(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    with pytest.raises(ValueError, match="select at least one"):
        gauss_newton(jacobian, _dataset((1.0, 2.0, 4.0), uses=("context",) * 3))
    with pytest.raises(ValueError, match="inside the Jacobian domain"):
        gauss_newton(jacobian, _dataset((0.5, 2.0, 4.0)))


def _n01_case(case):
    """Known singular directions provide an analytic oracle, not Gram pinv."""
    if case == "rank_one":
        return np.array(((1., 1.), (2., 2.), (3., 3.))), np.ones((2, 2)) / 56., 1
    singular_values = np.array((1., .1, .01, .001, 1e-8))
    if case == "exact_null":
        singular_values[-1] = 0.
    rotation = np.eye(5)
    if case == "mixed":
        v = np.arange(1., 6.)
        rotation -= 2 * np.outer(v, v) / (v @ v)
    jacobian = np.zeros((48, 5))
    jacobian[:5] = np.diag(singular_values) @ rotation.T
    keep = singular_values > 48 * np.finfo(np.float64).eps * singular_values[0]
    covariance = (rotation[:, keep] / singular_values[keep] ** 2) @ rotation[:, keep].T
    return jacobian, covariance, int(keep.sum())


def _assert_inverse_geometry(covariance, correlation, expected, *, tolerance=1e-12):
    assert np.linalg.norm(covariance - expected) / np.linalg.norm(expected) <= tolerance
    scale = np.sqrt(np.diag(expected))
    denominator = np.outer(scale, scale)
    reference = np.full(expected.shape, np.nan)
    np.divide(expected, denominator, out=reference, where=denominator > 0.)
    np.testing.assert_array_equal(np.isfinite(correlation), np.isfinite(reference))
    np.testing.assert_allclose(correlation, reference, rtol=0., atol=tolerance)


@pytest.mark.parametrize("case", ("axis", "mixed", "exact_null", "rank_one"))
def test_n01_shared_rank_and_inverse_use_the_same_directions(case):
    jacobian, expected, rank = _n01_case(case)
    status, result, finite, _ = _local_identifiability_diagnostics(jacobian)
    assert finite
    assert result["jacobian_rank"] == rank
    assert status == ("rank_deficient" if rank < jacobian.shape[1] else "locally_identifiable")
    if rank < jacobian.shape[1]:
        assert np.isinf(result["condition_number"])
    else:
        assert result["condition_number"] == pytest.approx(1e8, rel=1e-7)
    _assert_inverse_geometry(
        result["local_gauss_newton_covariance"],
        result["normalized_parameter_correlation_matrix"],
        expected,
        tolerance=1e-7 if case in ("axis", "mixed") else 1e-12,
    )
    if case == "axis":
        assert result["local_gauss_newton_covariance"][4, 4] == pytest.approx(1e16, rel=1e-12)


@pytest.mark.parametrize("route", ("public", "postfit"))
@pytest.mark.parametrize("case", ("axis", "mixed", "exact_null", "rank_one"))
def test_n01_wrappers_execute_shared_diagnostics_and_preserve_validity(route, case):
    jacobian, expected, rank = _n01_case(case)
    energy = np.arange(1., jacobian.shape[0] + 1.)
    names = tuple(f"coordinate_{i}" for i in range(jacobian.shape[1]))
    if route == "public":
        dataset = _dataset(energy)
        sigma = dataset.sed_flux_error_lower_erg_cm2_s / dataset.sed_flux_erg_cm2_s
        sensitivity = SensitivityResult(
            energy_erg=energy,
            values=jacobian * sigma[:, None],
            parameters=names,
            support_indices=np.arange(energy.size),
        )
        result = gauss_newton(sensitivity, dataset)
        assert result.parameters == names
        np.testing.assert_allclose(result.weighted_jacobian, jacobian, rtol=1e-14)
        covariance, correlation = result.covariance, result.correlation
        actual_rank, condition, status = result.rank, result.condition_number, result.status
    else:
        dataset = LogSEDDataset(
            energy_erg=energy, observed_sed=np.ones(energy.size),
            sigma_log_sed=np.ones(energy.size), quantity="earth_energy_flux",
        )
        upstream = OptimizeResult(x=np.zeros(len(names)), numerically_finite=True)
        result = add_postfit_diagnostics(
            upstream, dataset, np.ones(energy.size), model_quantity="earth_energy_flux",
            weighted_jacobian=jacobian, parameter_names=names,
        )
        assert result is upstream
        assert result.numerically_finite
        values = result.identifiability_diagnostics
        covariance = values["local_gauss_newton_covariance"]
        correlation = values["normalized_parameter_correlation_matrix"]
        actual_rank = values["jacobian_rank"]
        condition = values["condition_number"]
        status = result.identifiability_status
        assert all(set(pair["parameters"]) <= set(names)
                   for pair in values["strongest_absolute_parameter_correlations"])
    assert actual_rank == rank
    assert status == ("rank_deficient" if rank < len(names) else "locally_identifiable")
    if rank < len(names):
        assert np.isinf(condition)
    else:
        assert condition == pytest.approx(1e8, rel=1e-7)
    _assert_inverse_geometry(
        covariance, correlation, expected,
        tolerance=1e-7 if case in ("axis", "mixed") else 1e-12,
    )


@pytest.mark.parametrize(
    "jacobian,expected,rank,status,finite",
    [
        (np.zeros((3, 2)), np.zeros((2, 2)), 0, "rank_deficient", True),
        (np.empty((0, 2)), np.zeros((2, 2)), 0, "rank_deficient", True),
        (np.empty((3, 0)), np.empty((0, 0)), 0, "non_finite", False),
        (np.empty((0, 0)), np.empty((0, 0)), 0, "non_finite", False),
        (np.array(((3.,), (4.,))), np.array(((.04,),)), 1, "locally_identifiable", True),
        (np.array(((1., 0., 0.), (0., 2., 0.))), np.diag((1., .25, 0.)),
         2, "rank_deficient", True),
        (np.diag((1., np.finfo(np.float64).eps)), np.diag((1., 0.)),
         1, "rank_deficient", True),
    ],
    ids=("zero", "empty_rows", "empty_columns", "empty_both", "one_parameter",
         "underdetermined", "below_cutoff"),
)
def test_n01_general_shapes_and_retention_threshold(jacobian, expected, rank, status, finite):
    # Last case has s_min = 0.5*tau*s_max, so it must NOT retain every direction.
    actual_status, result, actual_finite, _ = _local_identifiability_diagnostics(jacobian)
    assert (actual_status, actual_finite, result["jacobian_rank"]) == (status, finite, rank)
    np.testing.assert_allclose(result["local_gauss_newton_covariance"], expected,
                               rtol=1e-12, atol=1e-12)
    valid = np.diag(expected) > 0.
    np.testing.assert_array_equal(
        np.isfinite(result["normalized_parameter_correlation_matrix"]),
        np.outer(valid, valid),
    )
    if rank < jacobian.shape[1] or jacobian.shape[1] == 0:
        assert np.isinf(result["condition_number"])
    else:
        assert result["condition_number"] == 1.


@pytest.mark.parametrize("value", (np.nan, np.inf))
def test_n01_nonfinite_input_preserves_failure_contract(value):
    status, result, finite, warnings = _local_identifiability_diagnostics([[value]])
    assert (status, result, finite) == ("non_finite", None, False)
    assert warnings == ["weighted Jacobian contains non-finite values"]


def test_n01_svd_failure_is_not_silently_converted(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("N01 injected SVD failure")

    monkeypatch.setattr(np.linalg, "svd", fail)
    with pytest.raises(np.linalg.LinAlgError, match="N01 injected SVD failure"):
        _local_identifiability_diagnostics(np.eye(2))


def test_n01_saved_txs_full_precision_and_figure7_printing():
    # Small portable snapshot of archived diagnostics; hashes/provenance are in JSON.
    fixture = json.loads(
        (Path(__file__).parent / "data" / "n01_txs_diagnostics.json").read_text()
    )
    names = tuple(fixture["parameters"])
    assert names == ("log_Le", "log_Lp", "log_B", "log_R", "p_e")
    jacobian = np.asarray(fixture["weighted_jacobian"])
    assert jacobian.shape == (48, 5)
    status, result, finite, _ = _local_identifiability_diagnostics(jacobian, names)
    assert (status, finite, result["jacobian_rank"]) == ("locally_identifiable", True, 5)
    covariance = np.asarray(fixture["archived_covariance"])
    rho = np.asarray(fixture["archived_correlation"])
    current_rho = result["normalized_parameter_correlation_matrix"]
    assert np.linalg.norm(result["local_gauss_newton_covariance"] - covariance) / np.linalg.norm(covariance) <= 1e-9
    np.testing.assert_array_equal(np.isfinite(current_rho), np.isfinite(rho))
    assert np.max(np.abs(current_rho - rho)) <= 1e-9
    singular_values = fixture["archived_singular_values"]
    for current, archived, fmt in zip(
        result["singular_values"], singular_values, (".2f", ".2f", ".2f", ".2f", ".3f")
    ):
        assert format(current, fmt) == format(archived, fmt)
    assert format(result["condition_number"], ".2f") == format(singular_values[0] / singular_values[-1], ".2f")
    assert [f"{x:.3f}" for x in current_rho.flat] == [f"{x:.3f}" for x in rho.flat]
