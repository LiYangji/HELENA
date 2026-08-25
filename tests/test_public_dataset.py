"""Focused P4 checks for the public observed-SED dataset boundary."""

import numpy as np
import pytest

from helena import SEDDataset
from helena.fitting import mixed_sed_nll_terms


def _dataset(**updates):
    arguments = {
        "energy_erg": [1.0, 2.0, 3.0, 4.0],
        "sed_flux_erg_cm2_s": [10.0, 8.0, 5.0, 4.0],
        "energy_error_lower_erg": [0.1, 0.1, 0.1, 0.1],
        "energy_error_upper_erg": [0.1, 0.1, 0.1, 0.1],
        "sed_flux_error_lower_erg_cm2_s": [2.0, 1.0, 0.0, 1.0],
        "sed_flux_error_upper_erg_cm2_s": [2.0, 3.0, 0.0, 1.0],
        "instrument": ["A", "B", "C", "D"],
        "statistic_type": [
            "gaussian",
            "asymmetric_gaussian",
            "upper_limit",
            "gaussian",
        ],
        "scientific_use": ["fit", "fit", "constraint", "context"],
    }
    arguments.update(updates)
    return SEDDataset(**arguments)


def test_public_dataset_preserves_observation_and_scientific_use_semantics():
    dataset = _dataset()

    np.testing.assert_array_equal(
        dataset.statistic_type,
        ["gaussian", "asymmetric_gaussian", "upper_limit", "gaussian"],
    )
    np.testing.assert_array_equal(
        dataset.scientific_use, ["fit", "fit", "constraint", "context"]
    )
    np.testing.assert_array_equal(dataset.likelihood_mask, [True, True, True, False])
    assert dataset.selected_count == 3

    excluded = dataset.with_scientific_use(
        np.asarray([False, True, False, False]), "exclude"
    )
    assert isinstance(excluded, SEDDataset)
    np.testing.assert_array_equal(excluded.statistic_type, dataset.statistic_type)
    np.testing.assert_array_equal(
        excluded.scientific_use, ["fit", "exclude", "constraint", "context"]
    )

    terms = np.asarray(
        mixed_sed_nll_terms(
            [0.5, 1.0, 2.0, 3.0, 4.0],
            [10.0, 10.0, 8.0, 4.0, 4.0],
            excluded,
        )
    )
    assert terms[1] == 0.0
    assert terms[3] == 0.0
    assert np.all(np.isfinite(terms[:3]))


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"energy_erg": [1.0, 2.0]}, "share"),
        (
            {"sed_flux_error_lower_erg_cm2_s": [2.0, -1.0, 0.0, 1.0]},
            "non-negative",
        ),
        (
            {"sed_flux_error_upper_erg_cm2_s": [2.0, 3.0, 1.0, 1.0]},
            "upper limits",
        ),
        (
            {"scientific_use": ["fit", "fit", "constraint", "unknown"]},
            "scientific uses",
        ),
    ),
)
def test_public_dataset_rejects_shape_uncertainty_limit_and_use_errors(
    updates, message
):
    with pytest.raises(ValueError, match=message):
        _dataset(**updates)
