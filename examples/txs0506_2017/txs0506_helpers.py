"""Data, plotting, and IceCube integration helpers for the TXS example."""

import os
from csv import DictReader
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import log_ndtr

EV_TO_ERG = 1.602176634e-12
GEV_TO_ERG = 1.0e9 * EV_TO_ERG
PLANCK_EV_S = 4.135667696e-15
HALF_YEAR_SECONDS = 0.5 * 365.25 * 86400.0
EVENT_ENERGY_RANGE_GEV = (1.83e5, 4.3e6)
DATA_DIR = Path(__file__).resolve().parent / "data"


def default_cache_dir():
    configured = os.environ.get("HELENA_CACHE_DIR")
    return Path(configured).expanduser() if configured else None


def _load_csv(path, numeric_columns):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    result = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        result[name] = (
            np.asarray(values, dtype=np.float64)
            if name in numeric_columns
            else np.asarray(values, dtype=str)
        )
    return result


def load_em_data(path=DATA_DIR / "txs0506_2017_curated.csv"):
    return _load_csv(
        path,
        {
            "energy_ev",
            "sed_flux_erg_cm2_s",
            "energy_error_lower_ev",
            "energy_error_upper_ev",
            "sed_flux_error_lower_erg_cm2_s",
            "sed_flux_error_upper_erg_cm2_s",
        },
    )


def load_icecube_effective_area(path=DATA_DIR / "icecube_aeff_ic86c.csv"):
    return _load_csv(
        path,
        {
            "log10_energy_min_GeV",
            "log10_energy_max_GeV",
            "effective_area_m2",
        },
    )


def expected_events(
    energy_erg,
    muon_neutrino_sed_erg_cm2_s,
    effective_area,
    exposure_seconds=HALF_YEAR_SECONDS,
    energy_range_gev=None,
):
    """Integrate Aeff * dN/dE for combined Earth nu_mu + anti_nu_mu."""

    energy = np.asarray(energy_erg, dtype=np.float64)
    sed = np.asarray(muon_neutrino_sed_erg_cm2_s, dtype=np.float64)
    if (
        energy.ndim != 1
        or energy.shape != sed.shape
        or energy.size < 2
        or np.any(~np.isfinite(energy))
        or np.any(~np.isfinite(sed))
        or np.any(energy <= 0.0)
        or np.any(sed < 0.0)
        or np.any(np.diff(energy) <= 0.0)
    ):
        raise ValueError("neutrino energy/SED must be finite, ordered, and non-negative")
    positive = sed > 0.0
    if np.count_nonzero(positive) < 2:
        return 0.0
    log_energy = np.log(energy[positive])
    log_sed = np.log(sed[positive])

    lower = 10.0 ** effective_area["log10_energy_min_GeV"] * GEV_TO_ERG
    upper = 10.0 ** effective_area["log10_energy_max_GeV"] * GEV_TO_ERG
    area = effective_area["effective_area_m2"] * 1.0e4
    if not (
        lower.shape == upper.shape == area.shape
        and np.all(area >= 0.0)
        and np.all(upper > lower)
        and np.allclose(lower[1:], upper[:-1])
    ):
        raise ValueError("effective-area bins must be finite, ordered, and contiguous")
    if energy_range_gev is not None:
        range_lower, range_upper = np.asarray(energy_range_gev) * GEV_TO_ERG
        if not 0.0 < range_lower < range_upper:
            raise ValueError("energy_range_gev must be an increasing positive pair")
    else:
        range_lower, range_upper = 0.0, np.inf

    rate = 0.0
    for bin_lower, bin_upper, bin_area in zip(lower, upper, area, strict=True):
        lo = max(bin_lower, range_lower, energy[positive][0])
        hi = min(bin_upper, range_upper, energy[positive][-1])
        if lo >= hi or bin_area == 0.0:
            continue
        internal = (energy[positive] > lo) & (energy[positive] < hi)
        coordinate = np.concatenate(
            ([np.log(lo)], log_energy[internal], [np.log(hi)])
        )
        values = np.exp(np.interp(coordinate, log_energy, log_sed))
        rate += bin_area * np.trapezoid(values / np.exp(coordinate), x=coordinate)
    return float(exposure_seconds * rate)


def print_dataset(data, dataset):
    print(dataset.audit())
    print("\nInstrument                                      total fit constraint context exclude")
    for instrument in np.unique(data["instrument"]):
        selected = data["instrument"] == instrument
        counts = [
            np.count_nonzero(selected & (data["scientific_use"] == role))
            for role in ("fit", "constraint", "context", "exclude")
        ]
        print(
            f"{instrument[:45]:45s} {np.count_nonzero(selected):5d} "
            + " ".join(f"{count:10d}" for count in counts)
        )


def print_fit_result(initial_config, fit_result):
    rows = (
        ("log_Le", "erg/s", "electron_kinetic_injection_luminosity_erg_s"),
        ("log_Lp", "erg/s", "proton_kinetic_injection_luminosity_erg_s"),
        ("log_B", "G", "magnetic_field_G"),
        ("log_R", "cm", "radius_cm"),
        ("p_e", "", "electron_injection_index"),
    )
    print("\nParameter  Unit       Initial          5D best fit")
    for name, unit, field in rows:
        print(
            f"{name:9s} {unit:8s} {getattr(initial_config, field):15.6e} "
            f"{getattr(fit_result.best_config, field):15.6e}"
        )
    print(
        f"\nsuccess={fit_result.success}; status={fit_result.status}; "
        f"NLL={fit_result.nll:.12f}; iterations={fit_result.iterations}; "
        f"evaluations={fit_result.evaluations}; "
        f"runtime={fit_result.runtime_seconds:.3f} s\n{fit_result.message}"
    )


FIT_INSTRUMENT_STYLES = {
    "Swift-UVOT": {"marker": "s", "color": "#CC79A7"},
    "Swift-XRT": {"marker": "D", "color": "#009E73"},
    "NuSTAR": {"marker": "o", "color": "#0072B2"},
    "AGILE": {"marker": "P", "color": "#E69F00"},
    "Fermi-LAT": {"marker": "^", "color": "#D55E00"},
    "MAGIC": {"marker": "*", "color": "black"},
}

CONSTRAINT_STYLES = {
    "H.E.S.S.": {"marker": "v", "color": "#6A3D9A"},
    "VERITAS": {"marker": "d", "color": "#8C564B"},
}


COMPONENT_FIELDS = {
    "primary_electron_synchrotron": r"Primary $e^-$ synchrotron",
    "primary_electron_inverse_compton": r"Primary $e^-$ inverse Compton",
    "bethe_heitler_pair_synchrotron": r"BH-pair synchrotron",
    "bethe_heitler_pair_inverse_compton": r"BH-pair inverse Compton",
    "photomeson_secondary_pair_synchrotron": (
        r"Photomeson-secondary pair synchrotron"
    ),
    "photomeson_secondary_pair_inverse_compton": (
        r"Photomeson-secondary pair inverse Compton"
    ),
    "gamma_gamma_reprocessed_pair_synchrotron": (
        r"$\gamma\gamma$ reprocessed-pair synchrotron"
    ),
    "gamma_gamma_reprocessed_pair_inverse_compton": (
        r"$\gamma\gamma$ reprocessed-pair inverse Compton"
    ),
    "proton_synchrotron": "Proton synchrotron",
    "neutral_pion_decay": r"$\pi^0$-decay photons",
    "charged_pion_synchrotron": r"$\pi^\pm$ synchrotron",
    "muon_synchrotron": r"$\mu^\pm$ synchrotron",
}


def component_audit(components, total_sed):
    total = np.asarray(total_sed, dtype=np.float64)
    values = {
        name: np.asarray(
            components[
                f"{name}_broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s"
            ],
            dtype=np.float64,
        )
        for name in COMPONENT_FIELDS
    }
    summed = sum(values.values())
    support = total > np.max(total) * 1.0e-20
    closure = np.max(np.abs(summed[support] - total[support]) / total[support])
    peaks = {name: float(np.max(value)) for name, value in values.items()}
    fractions = {
        name: float(np.max(value[support] / total[support]))
        for name, value in values.items()
    }
    return values, float(closure), peaks, fractions


def print_component_audit(components, total_sed, diagnostics):
    _, closure, peaks, fractions = component_audit(components, total_sed)
    print("\nEscaped Earth-observed component audit")
    for name, label in COMPONENT_FIELDS.items():
        print(
            f"{label:36s} peak={peaks[name]:.6e} erg cm^-2 s^-1; "
            f"max/total={fractions[name]:.6e}"
        )
    print(f"maximum additive closure residual={closure:.6e}")
    print(
        "frozen particle/synchrotron/IC closure maxima="
        f"{diagnostics['source_resolved_particle_electron_maximum_relative_error']:.3e}, "
        f"{diagnostics['source_resolved_particle_positron_maximum_relative_error']:.3e}, "
        f"{diagnostics['source_resolved_synchrotron_maximum_relative_error']:.3e}, "
        f"{diagnostics['source_resolved_inverse_compton_maximum_relative_error']:.3e}"
    )
    print(
        "Gamma-gamma reprocessed pairs retain aggregate mixed-parent ancestry; "
        "their synchrotron and IC mechanisms are separated."
    )
    return closure


def _log_sed_at(model_energy, model_sed, observation_energy):
    energy = np.asarray(model_energy, dtype=np.float64)
    sed = np.asarray(model_sed, dtype=np.float64)
    valid = np.isfinite(energy) & np.isfinite(sed) & (energy > 0.0) & (sed > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("model SED has insufficient positive plotting support")
    return np.exp(
        np.interp(
            np.log(observation_energy),
            np.log(energy[valid]),
            np.log(sed[valid]),
        )
    )


def _standardized_detection_residual(data, prediction, mask):
    observed = data["sed_flux_erg_cm2_s"][mask]
    gaussian = data["statistic_type"][mask] == "gaussian"
    lower = np.where(
        gaussian,
        data["sed_flux_error_lower_erg_cm2_s"][mask] / observed,
        np.log(observed)
        - np.log(observed - data["sed_flux_error_lower_erg_cm2_s"][mask]),
    )
    upper = np.where(
        gaussian,
        data["sed_flux_error_upper_erg_cm2_s"][mask] / observed,
        np.log(observed + data["sed_flux_error_upper_erg_cm2_s"][mask])
        - np.log(observed),
    )
    numerator = np.log(observed) - np.log(prediction)
    return numerator / np.where(prediction < observed, lower, upper)


def manual_log_sed_nll(data, model_energy_erg, model_sed):
    """Independent spot calculation of the documented Public log-SED NLL."""
    selected = np.isin(data["scientific_use"], ("fit", "constraint"))
    prediction = _log_sed_at(
        model_energy_erg,
        model_sed,
        data["energy_ev"][selected] * EV_TO_ERG,
    )
    observed = data["sed_flux_erg_cm2_s"][selected]
    statistic = data["statistic_type"][selected]
    limit = statistic == "upper_limit"
    result = 0.0
    if np.any(~limit):
        detection_data = {name: values[selected] for name, values in data.items()}
        detection_mask = ~limit
        residual = _standardized_detection_residual(
            detection_data, prediction[detection_mask], detection_mask
        )
        gaussian = statistic[detection_mask] == "gaussian"
        lower_error = detection_data["sed_flux_error_lower_erg_cm2_s"][detection_mask]
        upper_error = detection_data["sed_flux_error_upper_erg_cm2_s"][detection_mask]
        detected = observed[detection_mask]
        lower_sigma = np.where(
            gaussian,
            lower_error / detected,
            np.log(detected) - np.log(detected - lower_error),
        )
        upper_sigma = np.where(
            gaussian,
            upper_error / detected,
            np.log(detected + upper_error) - np.log(detected),
        )
        result += np.sum(
            0.5 * residual**2
            + np.log(lower_sigma + upper_sigma)
            + 0.5 * np.log(np.pi / 2.0)
        )
    if np.any(limit):
        quantile = NormalDist().inv_cdf(0.95)
        sigma = observed[limit] / quantile
        result -= np.sum(log_ndtr((observed[limit] - prediction[limit]) / sigma))
    return float(result)


def plot_broadband_fit_and_neutrino(
    data,
    best_photon_energy_erg,
    best_photon_sed,
    best_components,
    *,
    neutrino_energy_erg,
    all_flavor_neutrino_sed,
    muon_neutrino_sed,
    effective_area,
    expected_count,
    expected_count_alert_band,
    probability_at_least_one,
):
    """Plot the EM fit, likelihood residuals, and post-fit neutrinos."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )

    fig = plt.figure(figsize=(12.4, 8.8))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(2.15, 1.0),
        hspace=0.30,
        wspace=0.25,
    )
    photon_ax = fig.add_subplot(grid[0, :])
    residual_ax = fig.add_subplot(grid[1, 0])
    neutrino_ax = fig.add_subplot(grid[1, 1])

    best_energy = np.asarray(best_photon_energy_erg, dtype=np.float64)
    best_sed = np.asarray(best_photon_sed, dtype=np.float64)
    photon_floor = 0.02 * np.min(data["sed_flux_erg_cm2_s"])
    best = (
        np.isfinite(best_sed)
        & (best_sed >= photon_floor)
        & (best_sed >= np.max(best_sed) * 1.0e-12)
    )
    photon_ax.loglog(
        best_energy[best] / (PLANCK_EV_S * EV_TO_ERG),
        best_sed[best],
        color="black",
        linewidth=2.7,
        label="Best-fit escaped total",
        zorder=5,
    )

    component_specs = (
        ("primary_electron_synchrotron", "#0072B2", "-"),
        ("primary_electron_inverse_compton", "#E69F00", "-"),
        ("bethe_heitler_pair_synchrotron", "#009E73", "--"),
        ("bethe_heitler_pair_inverse_compton", "#009E73", ":"),
        ("photomeson_secondary_pair_synchrotron", "#56B4E9", "--"),
        ("photomeson_secondary_pair_inverse_compton", "#56B4E9", ":"),
        ("gamma_gamma_reprocessed_pair_synchrotron", "#9467BD", "-."),
        ("gamma_gamma_reprocessed_pair_inverse_compton", "#9467BD", ":"),
        ("proton_synchrotron", "#D55E00", ":"),
        ("neutral_pion_decay", "#CC79A7", "--"),
        ("charged_pion_synchrotron", "#8C564B", ":"),
        ("muon_synchrotron", "#17BECF", ":"),
    )
    component_energy = np.asarray(
        best_components["broadband_energy_earth_erg"], dtype=np.float64
    )
    component_values, _, component_peaks, _ = component_audit(
        best_components, best_sed
    )
    for name, color, linestyle in component_specs:
        label = COMPONENT_FIELDS[name]
        component_flux = component_values[name]
        peak = component_peaks[name]
        support = (
            np.isfinite(component_flux)
            & (component_flux >= photon_floor)
            & (component_flux >= peak * 1.0e-8)
        )
        if peak == 0.0 or np.count_nonzero(support) < 2:
            continue
        photon_ax.loglog(
            component_energy[support] / (PLANCK_EV_S * EV_TO_ERG),
            component_flux[support],
            color=color,
            linestyle=linestyle,
            linewidth=1.25,
            alpha=0.95,
            label=label,
            zorder=2,
        )

    detection = data["statistic_type"] != "upper_limit"
    context_detection = (data["scientific_use"] == "context") & detection
    photon_ax.scatter(
        data["energy_ev"][context_detection] / PLANCK_EV_S,
        data["sed_flux_erg_cm2_s"][context_detection],
        marker=".",
        s=8,
        color="0.55",
        alpha=0.24,
        linewidths=0.0,
            label="Archival/context (not fitted)",
        zorder=1,
    )
    for instrument, style in FIT_INSTRUMENT_STYLES.items():
        mask = (
            (data["scientific_use"] == "fit")
            & detection
            & (data["instrument"] == instrument)
        )
        photon_ax.errorbar(
            data["energy_ev"][mask] / PLANCK_EV_S,
            data["sed_flux_erg_cm2_s"][mask],
            xerr=np.vstack(
                (
                    data["energy_error_lower_ev"][mask] / PLANCK_EV_S,
                    data["energy_error_upper_ev"][mask] / PLANCK_EV_S,
                )
            ),
            yerr=np.vstack(
                (
                    data["sed_flux_error_lower_erg_cm2_s"][mask],
                    data["sed_flux_error_upper_erg_cm2_s"][mask],
                )
            ),
            fmt=style["marker"],
            color=style["color"],
            mfc=style["color"],
            mec="black",
            mew=0.45,
            label=instrument,
            linestyle="none",
            markersize=7.5 if instrument == "MAGIC" else (3.7 if instrument == "NuSTAR" else 5.5),
            capsize=1.5,
            elinewidth=0.75,
            zorder=9,
        )
    for instrument, style in CONSTRAINT_STYLES.items():
        mask = (
            (data["scientific_use"] == "constraint")
            & ~detection
            & (data["instrument"] == instrument)
        )
        photon_ax.errorbar(
            data["energy_ev"][mask] / PLANCK_EV_S,
            data["sed_flux_erg_cm2_s"][mask],
            xerr=np.vstack(
                (
                    data["energy_error_lower_ev"][mask] / PLANCK_EV_S,
                    data["energy_error_upper_ev"][mask] / PLANCK_EV_S,
                )
            ),
            yerr=0.25 * data["sed_flux_erg_cm2_s"][mask],
            uplims=True,
            fmt=style["marker"],
            mfc="white",
            mec=style["color"],
            color=style["color"],
            linestyle="none",
            label=f"{instrument} constraint (95% UL)",
            zorder=9,
        )
    context_limit = (data["scientific_use"] == "context") & ~detection
    if np.any(context_limit):
        photon_ax.errorbar(
            data["energy_ev"][context_limit] / PLANCK_EV_S,
            data["sed_flux_erg_cm2_s"][context_limit],
            yerr=0.18 * data["sed_flux_erg_cm2_s"][context_limit],
            uplims=True,
            fmt="v",
            markersize=3.0,
            mfc="white",
            mec="0.55",
            color="0.55",
            alpha=0.35,
            linestyle="none",
            label="_nolegend_",
            zorder=2,
        )
    photon_ax.set(
        xlabel="Observer frequency [Hz]",
        ylabel=r"$\nu F_\nu$ [erg cm$^{-2}$ s$^{-1}$]",
    )
    photon_ax.set_ylim(
        photon_floor * 0.5,
        5.0 * max(np.max(best_sed), np.max(data["sed_flux_erg_cm2_s"])),
    )
    photon_ax.legend(fontsize=6.2, ncol=4, loc="best")
    photon_ax.text(
        0.012,
        0.965,
        "(a) Photon SED fit",
        transform=photon_ax.transAxes,
        va="top",
        fontweight="bold",
    )

    residual_ax.axhline(0.0, color="0.6", linewidth=0.9)
    for instrument, style in FIT_INSTRUMENT_STYLES.items():
        mask = (
            (data["scientific_use"] == "fit")
            & detection
            & (data["instrument"] == instrument)
        )
        prediction = _log_sed_at(
            best_energy,
            best_sed,
            data["energy_ev"][mask] * EV_TO_ERG,
        )
        residual = _standardized_detection_residual(data, prediction, mask)
        residual_ax.scatter(
            data["energy_ev"][mask] / PLANCK_EV_S,
            residual,
            marker=style["marker"],
            s=48 if instrument == "MAGIC" else (19 if instrument == "NuSTAR" else 30),
            facecolor=style["color"],
            edgecolor="black",
            linewidth=0.45,
            label=instrument,
            zorder=4,
        )
    residual_ax.set(
        xlabel="Observer frequency [Hz]",
        ylabel=r"$[\ln F_{\rm obs}-\ln F_{\rm model}]/\sigma_{\ln}$",
    )
    residual_ax.set_xscale("log")
    residual_ax.legend(fontsize=7.2, ncol=2)
    residual_ax.text(
        0.025,
        0.95,
        "(b) EM likelihood residuals",
        transform=residual_ax.transAxes,
        va="top",
        fontweight="bold",
    )

    neutrino_energy = np.asarray(neutrino_energy_erg, dtype=np.float64) / GEV_TO_ERG
    all_flavor = np.asarray(all_flavor_neutrino_sed, dtype=np.float64)
    muon = np.asarray(muon_neutrino_sed, dtype=np.float64)
    neutrino_floor = max(np.max(all_flavor), np.max(muon)) * 1.0e-12
    all_support = np.isfinite(all_flavor) & (all_flavor >= neutrino_floor)
    muon_support = np.isfinite(muon) & (muon >= neutrino_floor)
    aeff_lower = 10.0 ** np.min(effective_area["log10_energy_min_GeV"])
    aeff_upper = 10.0 ** np.max(effective_area["log10_energy_max_GeV"])
    neutrino_ax.axvspan(
        aeff_lower,
        aeff_upper,
        color="#56B4E9",
        alpha=0.10,
        label=r"IC86c $A_{\rm eff}$ support",
    )
    neutrino_ax.axvspan(
        EVENT_ENERGY_RANGE_GEV[0],
        EVENT_ENERGY_RANGE_GEV[1],
        color="#E69F00",
        alpha=0.22,
        label="Event-energy interval",
    )
    neutrino_ax.loglog(
        neutrino_energy[all_support],
        all_flavor[all_support],
        color="black",
        linewidth=2.3,
        label=r"All flavor $\nu+\bar\nu$",
    )
    neutrino_ax.loglog(
        neutrino_energy[muon_support],
        muon[muon_support],
        "--",
        color="#0072B2",
        linewidth=1.8,
        label=r"$\nu_\mu+\bar\nu_\mu$",
    )
    neutrino_ax.set(
        xlabel="Observer neutrino energy [GeV]",
        ylabel=r"$E_\nu^2\Phi_\nu$ [erg cm$^{-2}$ s$^{-1}$]",
    )
    neutrino_ax.set_xlim(
        min(aeff_lower, np.min(neutrino_energy[all_support])),
        max(aeff_upper, np.max(neutrino_energy[all_support])),
    )
    neutrino_ax.legend(fontsize=7.6, loc="upper right")
    neutrino_ax.text(
        0.025,
        0.95,
        "(c) Post-fit neutrino prediction",
        transform=neutrino_ax.transAxes,
        va="top",
        fontweight="bold",
    )
    neutrino_ax.text(
        0.03,
        0.05,
        rf"$N_{{\rm full}}={expected_count:.4f}$" + "\n"
        + rf"$N_{{\rm event\ band}}={expected_count_alert_band:.4f}$" + "\n"
        + rf"$P(N\geq1)={probability_at_least_one:.4f}$" + "\n"
        + "not in EM likelihood",
        transform=neutrino_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    for axis in (photon_ax, residual_ax, neutrino_ax):
        axis.grid(True, which="both", color="0.9", linewidth=0.45)
    fig.suptitle(
        "TXS 0506+056: broadband fit and neutrino prediction",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Best-fit total and source-resolved components are escaped, EBL-attenuated "
        "Earth fluxes from one converged solution. Pair synchrotron and IC use the "
        "same frozen final transport and target; gamma-gamma reprocessing retains "
        "aggregate mixed-parent ownership.\nPlot-only masks omit numerical tails; "
        "negligible groups are reported in the component audit.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        wrap=True,
    )
    fig.subplots_adjust(top=0.93, bottom=0.10)
    return fig


def plot_local_diagnostics(jacobian_result, gauss_newton_result):
    """Plot the Public broadband Jacobian and local GN correlation."""

    frequency_hz = jacobian_result.energy_erg / (PLANCK_EV_S * EV_TO_ERG)
    support = (frequency_hz >= 1.0e12) & (frequency_hz <= 1.0e31)
    if np.count_nonzero(support) < 2:
        raise ValueError("Jacobian has no scientific plotting support")

    figure, (sensitivity_axis, correlation_axis) = plt.subplots(
        1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": (1.65, 1.0)}
    )
    for column, parameter in enumerate(jacobian_result.parameters):
        sensitivity_axis.semilogx(
            frequency_hz[support],
            jacobian_result.values[support, column],
            linewidth=1.7,
            label=parameter,
        )
    sensitivity_axis.axhline(0.0, color="black", linewidth=0.8)
    sensitivity_axis.set(
        xlabel="Observer frequency [Hz]",
        ylabel=r"$\partial\ln(\nu F_\nu^{\rm Earth})/\partial\theta$",
        title="5D broadband sensitivity",
    )
    sensitivity_axis.grid(True, which="both", color="0.9", linewidth=0.45)
    sensitivity_axis.legend(fontsize=8)

    image = correlation_axis.imshow(
        gauss_newton_result.correlation,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    positions = np.arange(len(gauss_newton_result.parameters))
    correlation_axis.set_xticks(
        positions,
        gauss_newton_result.parameters,
        rotation=35,
        ha="right",
    )
    correlation_axis.set_yticks(positions, gauss_newton_result.parameters)
    for row in positions:
        for column in positions:
            correlation_axis.text(
                column,
                row,
                f"{gauss_newton_result.correlation[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    correlation_axis.set_title("5D Gauss–Newton correlation")
    figure.colorbar(image, ax=correlation_axis, shrink=0.85, label="Local correlation")
    figure.tight_layout()
    return figure


def print_gauss_newton_summary(jacobian_result, gauss_newton_result):
    correlations = sorted(
        (
            (
                abs(gauss_newton_result.correlation[row, column]),
                gauss_newton_result.correlation[row, column],
                gauss_newton_result.parameters[row],
                gauss_newton_result.parameters[column],
            )
            for row in range(len(gauss_newton_result.parameters))
            for column in range(row + 1, len(gauss_newton_result.parameters))
        ),
        reverse=True,
    )
    print(f"Jacobian target: {jacobian_result.target}")
    print(f"Jacobian shape: {jacobian_result.values.shape}")
    print(f"GN status/rank: {gauss_newton_result.status}/{gauss_newton_result.rank}")
    print(f"GN condition number: {gauss_newton_result.condition_number:.6e}")
    print(
        "GN singular values: "
        + np.array2string(gauss_newton_result.singular_values, precision=6)
    )
    print(
        "Strongest correlations: "
        + "; ".join(
            f"{left}-{right}={value:+.3f}"
            for _, value, left, right in correlations[:3]
        )
    )
    print(
        "Gauss-Newton is a local identifiability diagnostic at the 5D best-fit "
        "point, not a posterior covariance or global uncertainty estimate."
    )


def _self_check():
    em = load_em_data()
    assert len(em["energy_ev"]) == 780
    assert np.count_nonzero(em["scientific_use"] == "fit") == 49
    assert np.count_nonzero(em["scientific_use"] == "constraint") == 10
    area = load_icecube_effective_area()
    assert len(area["effective_area_m2"]) == 70
    energy = np.geomspace(1.0, 10.0, 10001) * GEV_TO_ERG
    sed = np.full(energy.shape, 1.0e-12)
    one_bin = {
        "log10_energy_min_GeV": np.asarray([0.0]),
        "log10_energy_max_GeV": np.asarray([1.0]),
        "effective_area_m2": np.asarray([2.0]),
    }
    calculated = expected_events(energy, sed, one_bin, exposure_seconds=1.0)
    expected = 2.0e4 * 1.0e-12 * (1.0 / energy[0] - 1.0 / energy[-1])
    assert np.isclose(calculated, expected, rtol=1.0e-7)
    component = np.asarray([1.0, 2.0])
    components = {
        f"{name}_broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s": (
            component
        )
        for name in COMPONENT_FIELDS
    }
    _, closure, _, _ = component_audit(
        components, len(COMPONENT_FIELDS) * component
    )
    assert closure == 0.0
    assert {
        name: style["marker"] for name, style in FIT_INSTRUMENT_STYLES.items()
    } == {
        "Swift-UVOT": "s",
        "Swift-XRT": "D",
        "NuSTAR": "o",
        "AGILE": "P",
        "Fermi-LAT": "^",
        "MAGIC": "*",
    }


if __name__ == "__main__":
    _self_check()
    print("TXS 0506+056 helper self-check: PASS")
