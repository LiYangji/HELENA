"""Focused provenance and selection checks for the TXS canonical example."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.table import Table

DATA = Path(__file__).resolve().parents[1] / "examples/txs0506_2017/data"


def test_dense_public_data_are_frozen_and_roles_are_explicit():
    with (DATA / "txs0506_2017_curated.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 780
    assert {
        "instrument",
        "observation",
        "mjd_start",
        "mjd_stop",
        "frequency_hz",
        "energy_ev",
        "sed_flux_erg_cm2_s",
        "sed_flux_error_lower_erg_cm2_s",
        "sed_flux_error_upper_erg_cm2_s",
        "upper_limit_flag",
        "reference",
        "source_archive",
        "scientific_use",
        "selection_reason",
    } <= rows[0].keys()
    assert Counter(row["scientific_use"] for row in rows) == {
        "fit": 49,
        "constraint": 10,
        "context": 714,
        "exclude": 7,
    }
    assert len(Table.read(DATA / "txs0506_public_raw.ecsv")) == 729


def test_campaign_fit_and_vhe_constraint_selection():
    with (DATA / "txs0506_2017_curated.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fit = [row for row in rows if row["scientific_use"] == "fit"]
    nustar = [row for row in fit if row["instrument"] == "NuSTAR"]
    assert len(nustar) == 44
    assert {float(row["mjd_start"]) for row in nustar} == {58025.094594907}
    assert {float(row["mjd_stop"]) for row in nustar} == {58025.750798611}
    assert {row["instrument"] for row in fit} == {
        "Swift-UVOT",
        "Swift-XRT",
        "NuSTAR",
        "AGILE",
        "Fermi-LAT",
        "MAGIC",
    }
    xrt = next(row for row in fit if row["instrument"] == "Swift-XRT")
    assert "Phi^-1(0.95)" in xrt["note"]
    assert float(xrt["sed_flux_error_lower_erg_cm2_s"]) > 0.0
    assert float(xrt["sed_flux_error_upper_erg_cm2_s"]) > 0.0
    constraints = [
        row for row in rows if row["scientific_use"] == "constraint"
    ]
    assert Counter(row["instrument"] for row in constraints) == {
        "H.E.S.S.": 6,
        "VERITAS": 4,
    }
    assert all(row["statistic_type"] == "upper_limit" for row in constraints)
    assert max(float(row["energy_ev"]) for row in constraints) < 100e12
    assert all(np.isfinite(float(row["sed_flux_erg_cm2_s"])) for row in rows)


def test_provenance_declares_statistical_boundary():
    text = (DATA / "PROVENANCE.md").read_text(encoding="utf-8")
    for marker in (
        "MJD 58003--58033",
        "729 points",
        "44 machine-readable NuSTAR",
        "post-fit consistency check",
        "not an IceCube event",
    ):
        assert marker in text
