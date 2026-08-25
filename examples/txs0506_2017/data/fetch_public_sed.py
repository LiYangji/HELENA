"""Freeze public TXS 0506+056 SED data used by the canonical example."""

from __future__ import annotations

import csv
import json
import math
import urllib.request
from pathlib import Path
from statistics import NormalDist

import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
SSDC_URL = (
    "https://tools.ssdc.asi.it/SED-REST/rest/getData"
    "?ra=77.358185156&dec=5.693148283"
)
CDS_XRT_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/864/84/table1.dat"
PLANCK_EV_S = 4.135667696e-15
TEV_TO_ERG = 1.602176634
TWO_SIDED_90_PERCENT_Z = NormalDist().inv_cdf(0.95)

COLUMNS = (
    "energy_ev",
    "frequency_hz",
    "sed_flux_erg_cm2_s",
    "energy_error_lower_ev",
    "energy_error_upper_ev",
    "sed_flux_error_lower_erg_cm2_s",
    "sed_flux_error_upper_erg_cm2_s",
    "instrument",
    "band",
    "observation",
    "mjd_start",
    "mjd_stop",
    "epoch",
    "statistic_type",
    "upper_limit_flag",
    "scientific_use",
    "reference",
    "source_archive",
    "source_key",
    "selection_reason",
    "note",
)


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "HELENA/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _row(
    frequency_hz,
    flux,
    error_lower,
    error_upper,
    instrument,
    band,
    observation,
    mjd_start,
    mjd_stop,
    epoch,
    statistic_type,
    scientific_use,
    reference,
    source_archive,
    source_key,
    selection_reason,
    note="",
    frequency_error_hz=0.0,
):
    frequency_hz = float(frequency_hz)
    return {
        "energy_ev": frequency_hz * PLANCK_EV_S,
        "frequency_hz": frequency_hz,
        "sed_flux_erg_cm2_s": float(flux),
        "energy_error_lower_ev": float(frequency_error_hz) * PLANCK_EV_S,
        "energy_error_upper_ev": float(frequency_error_hz) * PLANCK_EV_S,
        "sed_flux_error_lower_erg_cm2_s": float(error_lower),
        "sed_flux_error_upper_erg_cm2_s": float(error_upper),
        "instrument": instrument,
        "band": band,
        "observation": observation,
        "mjd_start": mjd_start,
        "mjd_stop": mjd_stop,
        "epoch": epoch,
        "statistic_type": statistic_type,
        "upper_limit_flag": str(statistic_type == "upper_limit").lower(),
        "scientific_use": scientific_use,
        "reference": reference,
        "source_archive": source_archive,
        "source_key": source_key,
        "selection_reason": selection_reason,
        "note": note,
    }


def _ssdc_rows(payload):
    rows = []
    raw = []
    for dataset in payload["Datasets"]:
        reference = dataset["Reference"]
        bibliography = reference.get("Bibliography", [])
        citation = bibliography[0].get("URL", "") if bibliography else ""
        for point in dataset["Data"]:
            raw.append(
                {
                    "reference_id": reference["Id"],
                    "reference_name": reference["Name"],
                    "band": reference["Band"],
                    "kind": reference["Kind"],
                    "bibliography_url": citation,
                    **point,
                }
            )
            frequency = point.get("Frequency")
            flux = point.get("Nufnu")
            error = point.get("NufnuError", 0.0)
            if not all(math.isfinite(float(value)) for value in (frequency, flux, error)):
                continue
            if float(frequency) <= 0.0 or float(flux) <= 0.0 or float(error) < 0.0:
                continue
            is_nustar = (
                reference["Id"] == 14923
                and abs(float(point.get("StartTime", -1.0)) - 58025.094594907) < 1e-8
                and abs(float(point.get("StopTime", -1.0)) - 58025.750798611) < 1e-8
            )
            info = str(point.get("Info", ""))
            statistic = "upper_limit" if "Upper Limit" in info else "gaussian"
            use = (
                "fit"
                if is_nustar
                else "exclude"
                if statistic == "gaussian"
                and (float(error) == 0.0 or float(error) >= float(flux))
                else "context"
            )
            rows.append(
                _row(
                    frequency,
                    flux,
                    0.0 if statistic == "upper_limit" else error,
                    0.0 if statistic == "upper_limit" else error,
                    "NuSTAR" if is_nustar else reference["Name"],
                    reference["Band"],
                    f"SSDC dataset {reference['Id']}",
                    point.get("StartTime", ""),
                    point.get("StopTime", ""),
                    "IceCube-170922A campaign" if is_nustar else "archival",
                    statistic,
                    use,
                    citation or f"SSDC dataset {reference['Id']}",
                    SSDC_URL,
                    f"ssdc:{reference['Id']}",
                    (
                        "Machine-readable NuSTAR spectrum within Keivani Epoch 1"
                        if is_nustar
                        else "Uncertainty is unavailable or incompatible with the log-SED contract; retained only in raw provenance"
                        if use == "exclude"
                        else "Public archival context; not asserted simultaneous"
                    ),
                    info,
                    point.get("FrequencyError", 0.0),
                )
            )
    Table(rows=raw).write(HERE / "txs0506_public_raw.ecsv", overwrite=True)
    return rows


def _power_law_pivot(flux, index, energy_kev=1.0):
    lower, upper = 0.3, 10.0
    if np.isclose(index, 2.0):
        normalization = flux / np.log(upper / lower)
    else:
        normalization = flux * (2.0 - index) / (
            upper ** (2.0 - index) - lower ** (2.0 - index)
        )
    return normalization * energy_kev ** (2.0 - index)


def _xrt_rows(raw_table):
    rows = []
    reference = "Keivani et al. 2018, ApJ 864, 84; CDS J/ApJ/864/84/table1"
    for line in raw_table.decode("ascii").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split("|")]
        mjd, mjd_error, _, index, index_lo, index_hi, *_, flux, flux_lo, flux_hi = values
        flux *= 1e-12
        candidates = [
            _power_law_pivot(candidate_flux, candidate_index)
            for candidate_flux in (flux - flux_lo * 1e-12, flux + flux_hi * 1e-12)
            for candidate_index in (index - index_lo, index + index_hi)
        ]
        pivot = _power_law_pivot(flux, index)
        simultaneous = abs(mjd - 58023.8535) < 1e-6
        rows.append(
            _row(
                2.417989242e17,
                pivot,
                (pivot - min(candidates)) / TWO_SIDED_90_PERCENT_Z,
                (max(candidates) - pivot) / TWO_SIDED_90_PERCENT_Z,
                "Swift-XRT",
                "X-ray",
                "Swift observation",
                mjd - mjd_error,
                mjd + mjd_error,
                "Keivani Epoch 1" if 58003 <= mjd <= 58033 else "2017 monitoring",
                "asymmetric_gaussian",
                "fit" if simultaneous else "context",
                reference,
                CDS_XRT_URL,
                f"cds:xrt:{mjd:.4f}",
                (
                    "Quasi-simultaneous 2017 campaign XRT spectrum"
                    if simultaneous
                    else "2017 monitoring context; not simultaneous with the alert"
                ),
                "1 keV nuFnu derived from the published 0.3-10 keV power law; published two-sided 90% ranges divided by Phi^-1(0.95) for the Gaussian likelihood",
            )
        )
    return rows


def _published_campaign_rows():
    rows = [
        _row(
            5.25e9, 2.967825e-14, 1.48484091998e-15, 1.48484091998e-15,
            "VLA", "Radio", "5.25 GHz", 58031.6429, 58031.6429,
            "IceCube-170922A campaign", "gaussian", "context",
            "IceCube Collaboration et al. 2018, Science 361, eaat1378",
            "https://doi.org/10.1126/science.aat1378", "science:vla",
            "Campaign context; compact one-zone radio is self-absorbed/resolved",
        ),
        _row(
            1.0e15, 3.480e-11, 3.130e-12, 3.130e-12,
            "Swift-UVOT", "Optical/UV", "Extinction-corrected aggregate", 58023.8470, 58023.8470,
            "Keivani Epoch 1", "gaussian", "fit",
            "Keivani et al. 2018, ApJ 864, 84",
            "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/864/84/", "keivani:uvot:58023.8470",
            "Published extinction-corrected campaign value at 1e15 Hz",
        ),
        _row(
            2.0e8 / PLANCK_EV_S, 8.8e-11, 3.5e-11, 3.5e-11,
            "AGILE", "Gamma-ray", "200 MeV pivot", 58006.0, 58019.0,
            "IceCube-170922A campaign", "gaussian", "fit",
            "IceCube Collaboration et al. 2018, Science 361, eaat1378",
            "https://doi.org/10.1126/science.aat1378", "science:agile",
            "Published 13-day campaign E^2 dN/dE value",
        ),
        _row(
            1.44e9 / PLANCK_EV_S, 4.71762832493e-11, 3.65450081509e-12, 3.65450081509e-12,
            "Fermi-LAT", "Gamma-ray", "Epoch-1 1.44 GeV pivot", 58003.0, 58033.0,
            "Keivani Epoch 1", "gaussian", "fit",
            "Keivani et al. 2018, ApJ 864, 84",
            "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/864/84/", "keivani:lat:epoch1",
            "Published Epoch-1 power-law normalization at its pivot",
        ),
        _row(
            1.30e11 / PLANCK_EV_S, 5.41535702292e-12, 1.45712354934e-12, 1.45712354934e-12,
            "MAGIC", "VHE", "130 GeV spectrum pivot", 58024.0, 58030.0,
            "IceCube-170922A campaign", "gaussian", "fit",
            "IceCube Collaboration et al. 2018, Science 361, eaat1378",
            "https://doi.org/10.1126/science.aat1378", "science:magic:130gev",
            "Published observed differential-spectrum normalization; EBL required",
            frequency_error_hz=1.95e10 / PLANCK_EV_S,
        ),
    ]
    hess = (
        (0.16, 0.28, 6.6e-11), (0.28, 0.48, 2.1e-11),
        (0.48, 0.85, 4.5e-12), (0.85, 1.50, 1.8e-12),
        (1.50, 2.63, 5.9e-13), (2.63, 4.62, 3.3e-13),
    )
    veritas = (
        (0.141, 0.316, 5.4e-11), (0.316, 0.708, 6.4e-12),
        (0.708, 1.585, 5.3e-13), (1.585, 3.548, 8.0e-14),
    )
    for instrument, bins, window in (
        ("H.E.S.S.", hess, "MJD 58019.0-58020.0"),
        ("VERITAS", veritas, "MJD 58019.4-58027.4"),
    ):
        for index, (lower, upper, differential_limit) in enumerate(bins, start=1):
            energy_tev = math.sqrt(lower * upper)
            frequency = energy_tev * 1e12 / PLANCK_EV_S
            rows.append(
                _row(
                    frequency,
                    energy_tev**2 * differential_limit * TEV_TO_ERG,
                    0.0, 0.0, instrument, "VHE", f"95% differential upper-limit bin {index}",
                    window, window, "IceCube-170922A campaign", "upper_limit", "constraint",
                    "IceCube Collaboration et al. 2018 supplementary material",
                    "https://icecube.wisc.edu/wp-content/uploads/2021/01/detection-flaring-blazar.pdf",
                    f"science:{instrument.lower()}:{index}",
                    "Published campaign upper limit within validated EBL support",
                    frequency_error_hz=max(energy_tev - lower, upper - energy_tev) * 1e12 / PLANCK_EV_S,
                )
            )
    return rows


def main():
    payload = json.loads(_download(SSDC_URL))
    rows = _ssdc_rows(payload)
    rows.extend(_xrt_rows(_download(CDS_XRT_URL)))
    rows.extend(_published_campaign_rows())
    rows.sort(key=lambda row: (float(row["frequency_hz"]), str(row["instrument"])))
    with (HERE / "txs0506_2017_curated.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {use: sum(row["scientific_use"] == use for row in rows) for use in ("fit", "constraint", "context", "exclude")}
    assert len(payload["Datasets"]) > 50 and sum(len(item["Data"]) for item in payload["Datasets"]) > 500
    assert counts["fit"] >= 45 and counts["constraint"] == 10
    print(f"wrote {len(rows)} curated rows: {counts}")


if __name__ == "__main__":
    main()
