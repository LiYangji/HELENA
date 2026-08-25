"""Focused P6 checks for public local Gauss-Newton diagnostics."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import helena.diagnostics as diagnostics
from helena import SEDDataset
from helena.diagnostics import GaussNewtonResult, gauss_newton
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
    covariance = np.linalg.pinv(
        curvature,
        rcond=max(weighted.shape) * np.finfo(np.float64).eps,
        hermitian=True,
    )
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
    np.testing.assert_allclose(
        diagnostics.covariance, np.linalg.pinv(diagnostics.curvature, hermitian=True)
    )


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
