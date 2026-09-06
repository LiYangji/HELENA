"""Small presentation helpers shared by the CPU and GPU quickstarts."""

import os
import time
from html import escape
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib import font_manager
from matplotlib.text import Text

REQUESTED_FONT = "DejaVu Serif"
REFERENCE_FILE = Path(__file__).with_name("ssc_benchmark_reference.npz")
SYNCOOL_REFERENCE_FILE = Path(__file__).with_name("syncool_am3_reference.npz")
PGAMMA_REFERENCE_FILE = Path(__file__).with_name("pgamma_plpl_reference.npz")
PS_REFERENCE_FILE = Path(__file__).with_name("ps_benchmark_reference.npz")
LEHA_REFERENCE_FILE = Path(__file__).with_name("leha_benchmark_reference.npz")
RADIUS_CM = 1.0e15
DOPPLER_FACTOR = 30.0
C_CGS = 2.99792458e10
H_ERG_S = 6.62607015e-27
ELECTRON_REST_ENERGY_ERG = 8.1871057769e-7
THOMSON_CROSS_SECTION_CM2 = 6.6524587321e-25
MEGAPARSEC_CM = 3.085677581491367e24
BROADBAND_FREQUENCIES_HZ = (
    1.0e13,
    1.0e15,
    3.0e16,
    1.3e17,
    5.0e17,
    1.0e19,
    1.0e20,
    1.0e21,
    1.0e22,
    1.0e24,
    1.0e25,
    1.0e26,
    1.0e28,
    1.0e29,
    1.0e30,
    7.5e30,
    3.0e31,
    1.0e32,
)

_TABLE_STYLE = """
<style>
.helena-table { font-family: "DejaVu Serif", serif; }
.helena-table { border-collapse: collapse; margin: 0.5rem 0 1rem 0; }
.helena-table th, .helena-table td {
  border-bottom: 1px solid #b8b8b8; padding: 0.28rem 0.65rem;
  text-align: left; vertical-align: top;
}
.helena-table th { font-weight: 700; }
</style>
"""


def setup_quickstart_style():
    """Use Matplotlib's portable bundled serif faces and STIX math text."""
    variants = {}
    try:
        for name, (style, weight) in {
            "regular": ("normal", "normal"),
            "bold": ("normal", "bold"),
            "italic": ("italic", "normal"),
            "bold_italic": ("italic", "bold"),
        }.items():
            path = font_manager.findfont(
                font_manager.FontProperties(
                    family=REQUESTED_FONT, style=style, weight=weight
                ),
                fallback_to_default=False,
            )
            family = font_manager.FontProperties(fname=path).get_name()
            if family != REQUESTED_FONT:
                raise RuntimeError(f"resolved {family!r}, expected {REQUESTED_FONT!r}")
            variants[name] = {"family": family, "path": path, "file": Path(path).name}
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"{REQUESTED_FONT} regular, bold, italic, and bold-italic faces are required"
        ) from error
    if len({item["path"] for item in variants.values()}) != 4:
        raise RuntimeError(f"Four distinct {REQUESTED_FONT} font faces are required")

    mpl.rcParams.update(
        {
            "font.family": REQUESTED_FONT,
            "font.serif": [REQUESTED_FONT],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
            "axes.unicode_minus": False,
        }
    )
    markdown_css = Path(__file__).with_name("notebook_markdown.css").read_text(encoding="utf-8")
    display(HTML(_TABLE_STYLE + "<style>" + markdown_css + "</style>"))
    return {"family": REQUESTED_FONT, "mathtext": "stix", "variants": variants}


def benchmark_cases():
    """Return the two published inputs and their compact numerical arrays."""
    cases = {}
    for name, magnetic_field_G, gamma_max in (
        ("SSC-TH", 0.1, 1.0e4),
        ("SSC-KN", 0.01, 1.0e6),
    ):
        electron_gamma = np.geomspace(1.0 + 1.0e-8, 100.0 * gamma_max, 240)
        electron_density = (
            1.0e4 * electron_gamma**-1.9 * np.exp(-electron_gamma / gamma_max)
        )
        frequency_blob_hz = np.geomspace(
            1.0e6 / DOPPLER_FACTOR, 1.0e33 / DOPPLER_FACTOR, 240
        )
        cases[name] = {
            "radius_cm": RADIUS_CM,
            "magnetic_field_G": magnetic_field_G,
            "doppler_factor": DOPPLER_FACTOR,
            "gamma_min": 1.0,
            "gamma_max": gamma_max,
            "electron_index": 1.9,
            "electron_normalization_cm3": 1.0e4,
            "cooling": "disabled",
            "escape": "R/c (quoted; fixed density)",
            "electron_gamma": electron_gamma,
            "electron_density_cm3": electron_density,
            "frequency_blob_hz": frequency_blob_hz,
            "photon_energy_blob_erg": H_ERG_S * frequency_blob_hz,
        }
    return cases


def configured_cache_dir():
    """Return only the explicit cache override until cache fetch is implemented."""
    configured = os.environ.get("HELENA_CACHE_DIR")
    return Path(configured).expanduser() if configured else None


def observer_components(frequency_blob_hz, synchrotron_blob, ssc_blob, delta):
    """Apply the validated observer-frame transform to both components."""
    return (
        np.asarray(frequency_blob_hz) * delta,
        np.asarray(synchrotron_blob) * delta**4,
        np.asarray(ssc_blob) * delta**4,
    )


def load_benchmark_references():
    """Load the local, dependency-free AM3 and LeHaMoC total curves."""
    with np.load(REFERENCE_FILE, allow_pickle=False) as archive:
        if (
            str(archive["frame"]) != "observer"
            or str(archive["frequency_unit"]) != "Hz"
            or str(archive["luminosity_unit"]) != "erg s^-1"
            or bool(archive["am3_public_runtime_dependency"])
            or bool(archive["lehamoc_public_runtime_dependency"])
        ):
            raise RuntimeError("unexpected SSC reference metadata")
        references = {}
        for scenario, slug in (("SSC-TH", "ssc_th"), ("SSC-KN", "ssc_kn")):
            references[scenario] = {}
            for code, code_slug in (("AM3", "am3"), ("LeHaMoC", "lehamoc")):
                frequency = np.array(
                    archive[f"{slug}_{code_slug}_frequency_hz"], copy=True
                )
                luminosity = np.array(
                    archive[f"{slug}_{code_slug}_total_nu_lnu_erg_s"], copy=True
                )
                if frequency.shape != luminosity.shape or np.any(np.diff(frequency) <= 0.0):
                    raise RuntimeError(f"invalid {scenario} {code} reference curve")
                references[scenario][code] = (frequency, luminosity)
    return references


def load_syncool_reference():
    """Load the local, dependency-free AM3 SYN-cool reference."""
    with np.load(SYNCOOL_REFERENCE_FILE, allow_pickle=False) as archive:
        if (
            str(archive["scenario"]) != "SYN-cool"
            or str(archive["source_code"]) != "AM3"
            or str(archive["electron_quantity"]) != "dn/dgamma"
            or str(archive["electron_frame"]) != "blob"
            or str(archive["photon_frame"]) != "observer"
            or bool(archive["inverse_compton_enabled"])
            or bool(archive["am3_public_runtime_dependency"])
        ):
            raise RuntimeError("unexpected SYN-cool reference metadata")
        reference = {
            key: np.array(archive[key], copy=True)
            for key in (
                "electron_gamma",
                "steady_electron_dndgamma_cm3",
                "photon_frequency_blob_hz",
                "photon_frequency_observer_hz",
                "synchrotron_observer_nu_lnu_erg_s",
            )
        }
    for grid, values in (
        (reference["electron_gamma"], reference["steady_electron_dndgamma_cm3"]),
        (
            reference["photon_frequency_observer_hz"],
            reference["synchrotron_observer_nu_lnu_erg_s"],
        ),
    ):
        if (
            grid.shape != values.shape
            or np.any(np.diff(grid) <= 0.0)
            or np.any(~np.isfinite(grid))
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise RuntimeError("invalid SYN-cool AM3 reference curve")
    return reference


def syncool_inputs(array_module):
    """Prepare the published SYN-cool grids without hiding HELENA calls."""
    reference = load_syncool_reference()
    radius_cm = 1.0e15
    magnetic_field_G = 50.0
    gamma_min = 1.0
    gamma_max = 1.0e4
    index = 1.9
    injection_compactness = 10.0**-4.5
    thomson_cross_section_cm2 = 6.6524587051e-25
    electron_mass_g = 9.1093837139e-28

    electron_gamma = array_module.geomspace(gamma_min + 1.0e-8, 1.0e8, 960)
    normalization_integral = (
        (gamma_max ** (2.0 - index) - gamma_min ** (2.0 - index))
        / (2.0 - index)
        - (gamma_max ** (1.0 - index) - gamma_min ** (1.0 - index))
        / (1.0 - index)
    )
    injection_normalization = (
        3.0
        * C_CGS
        * injection_compactness
        / (thomson_cross_section_cm2 * radius_cm**2 * normalization_integral)
    )
    injection = (
        injection_normalization
        * electron_gamma**-index
        * array_module.exp(-electron_gamma / gamma_max)
    )
    cooling = (
        thomson_cross_section_cm2
        * magnetic_field_G**2
        * electron_gamma**2
        / (6.0 * array_module.pi * electron_mass_g * C_CGS)
    )
    return {
        "reference": reference,
        "radius_cm": radius_cm,
        "magnetic_field_G": magnetic_field_G,
        "doppler_factor": 30.0,
        "electron_gamma": electron_gamma,
        "injection": injection,
        "cooling": cooling,
        "escape": array_module.full_like(electron_gamma, C_CGS / radius_cm),
        "photon_energy_blob_erg": (
            H_ERG_S * array_module.asarray(reference["photon_frequency_blob_hz"])
        ),
    }


def syncool_observer_result(inputs, steady_electrons, synchrotron_blob):
    """Convert SYN-cool arrays to the two displayed observer quantities."""
    reference = inputs["reference"]
    doppler_factor = inputs["doppler_factor"]
    return {
        "electron_gamma": inputs["electron_gamma"],
        "steady_electron_dndgamma_cm3": steady_electrons,
        "photon_frequency_observer_hz": (
            reference["photon_frequency_blob_hz"] * doppler_factor
        ),
        "synchrotron_observer_nu_lnu_erg_s": (
            synchrotron_blob * doppler_factor**4
        ),
    }


def pgamma_plpl_case():
    """Return the published pγ-PLPL inputs and compact numerical grids."""
    proton_gamma_cutoff = 1.0e8
    proton_gamma = np.geomspace(1.0, 100.0 * proton_gamma_cutoff, 480)
    proton_density = (
        8490.0
        * proton_gamma**-1.9
        * np.exp(-proton_gamma / proton_gamma_cutoff)
    )

    target_epsilon = np.geomspace(1.0e-6, 1.0e-1, 320)
    target_energy = ELECTRON_REST_ENERGY_ERG * target_epsilon
    target_energy_density = (
        3.0
        * ELECTRON_REST_ENERGY_ERG
        * 1.0e-5
        / (THOMSON_CROSS_SECTION_CM2 * RADIUS_CM)
    )
    target_normalization = target_energy_density / np.log(
        target_epsilon[-1] / target_epsilon[0]
    )

    secondary_energy = np.geomspace(1.0e4, 1.0e12, 640)
    return {
        "radius_cm": RADIUS_CM,
        "proton_gamma_min": 1.0,
        "proton_gamma_cutoff": proton_gamma_cutoff,
        "proton_index": 1.9,
        "proton_injection_luminosity_erg_s": 8.5e43,
        "proton_density_normalization_cm3": 8490.0,
        "target_compactness": 1.0e-5,
        "target_index": 2.0,
        "target_epsilon_min": target_epsilon[0],
        "target_epsilon_max": target_epsilon[-1],
        "proton_gamma": proton_gamma,
        "proton_density_cm3": proton_density,
        "target_photon_energy_erg": target_energy,
        "target_photon_density_cm3_erg": target_normalization / target_energy**2,
        "pion_gamma": np.geomspace(1.0 + 1.0e-8, 1.0e10, 640),
        "muon_gamma": np.geomspace(1.0 + 1.0e-8, 1.0e10, 640),
        "secondary_energy_dimensionless": secondary_energy,
        "secondary_energy_erg": ELECTRON_REST_ENERGY_ERG * secondary_energy,
    }


def load_pgamma_reference():
    """Load the dependency-free AM3 and LeHaMoC pγ-PLPL references."""
    with np.load(PGAMMA_REFERENCE_FILE, allow_pickle=False) as archive:
        if (
            str(archive["scenario"]) != "pgamma-PLPL"
            or str(archive["frame"]) != "blob"
            or str(archive["quantity"])
            != "energy_squared_secondary_injection_rate"
            or str(archive["energy_axis"]) != "E_over_me_c2"
            or str(archive["injection_unit"]) != "cm^-3 s^-1"
            or bool(archive["proton_cooling_enabled"])
            or bool(archive["am3_public_runtime_dependency"])
            or bool(archive["lehamoc_public_runtime_dependency"])
        ):
            raise RuntimeError("unexpected pγ-PLPL reference metadata")
        reference = {code: {} for code in ("AM3", "LeHaMoC")}
        for channel in ("photon", "lepton", "neutrino"):
            for code, slug in (("AM3", "am3"), ("LeHaMoC", "lehamoc")):
                energy = np.array(
                    archive[f"{channel}_{slug}_energy_dimensionless"], copy=True
                )
                injection = np.array(
                    archive[f"{channel}_{slug}_energy2_injection_cm3_s"], copy=True
                )
                if (
                    energy.shape != injection.shape
                    or np.any(np.diff(energy) <= 0.0)
                    or np.any(~np.isfinite(energy))
                    or np.any(~np.isfinite(injection))
                    or np.any(injection < 0.0)
                ):
                    raise RuntimeError(f"invalid pγ-PLPL {code} {channel} curve")
                reference[code][channel] = (energy, injection)
    return reference


def pgamma_secondary_result(
    case,
    pi0_photons,
    leptons,
    pion_neutrinos,
    muon_neutrinos,
):
    """Prepare the three displayed pγ secondary-injection quantities."""
    energy = case["secondary_energy_dimensionless"]
    return {
        "energy_dimensionless": energy,
        "photon": energy**2 * ELECTRON_REST_ENERGY_ERG * pi0_photons,
        "lepton": energy**2 * sum(leptons),
        "neutrino": (
            energy**2
            * ELECTRON_REST_ENERGY_ERG
            * (sum(pion_neutrinos) + sum(muon_neutrinos))
        ),
    }


def ps_case():
    """Return the verified Cerruti PS benchmark inputs."""
    return {
        "radius_cm": 1.0e15,
        "magnetic_field_G": 10.0,
        "doppler_factor": 30.0,
        "redshift": 0.0,
        "luminosity_distance_cm": 44.4 * MEGAPARSEC_CM,
        "electron_gamma_min": 1.0,
        "electron_gamma_max": 1.0e3,
        "electron_index": 1.9,
        "electron_injection_luminosity_erg_s": 1.571009e37,
        "proton_gamma_min": 1.0,
        "proton_gamma_max": 1.0e8,
        "proton_index": 1.9,
        "proton_injection_luminosity_erg_s": 1.000202e43,
        "electron_escape": "R/c",
        "proton_escape": "R/c",
        "cache_dir": configured_cache_dir(),
    }


def load_ps_reference():
    """Load observer-frame Figure 9 photon and all-flavor-neutrino curves."""
    with np.load(PS_REFERENCE_FILE, allow_pickle=False) as archive:
        if (
            str(archive["scenario"]) != "PS"
            or str(archive["frame"]) != "observer"
            or str(archive["photon_quantity"]) != "escaped_broadband_nu_L_nu"
            or str(archive["neutrino_quantity"])
            != "all_flavor_neutrino_antineutrino_nu_L_nu"
            or str(archive["frequency_unit"]) != "Hz"
            or str(archive["luminosity_unit"]) != "erg s^-1"
            or bool(archive["am3_public_runtime_dependency"])
            or bool(archive["lehamoc_public_runtime_dependency"])
        ):
            raise RuntimeError("unexpected PS reference metadata")
        reference = {code: {} for code in ("AM3", "LeHaMoC")}
        for channel in ("photon", "neutrino"):
            for code, slug in (("AM3", "am3"), ("LeHaMoC", "lehamoc")):
                frequency = np.array(
                    archive[f"{channel}_{slug}_frequency_observer_hz"], copy=True
                )
                luminosity = np.array(
                    archive[f"{channel}_{slug}_nu_lnu_observer_erg_s"], copy=True
                )
                if (
                    frequency.shape != luminosity.shape
                    or np.any(np.diff(frequency) <= 0.0)
                    or np.any(~np.isfinite(frequency))
                    or np.any(~np.isfinite(luminosity))
                    or np.any(frequency <= 0.0)
                    or np.any(luminosity < 0.0)
                ):
                    raise RuntimeError(f"invalid PS {code} {channel} curve")
                reference[code][channel] = (frequency, luminosity)
    return reference


def leha_case():
    """Return the verified Cerruti LeHa benchmark inputs."""
    return {
        "radius_cm": 1.0e16,
        "magnetic_field_G": 0.1,
        "doppler_factor": 30.0,
        "redshift": 0.0,
        "luminosity_distance_cm": 44.4 * MEGAPARSEC_CM,
        "electron_gamma_min": 1.0,
        "electron_gamma_max": 3.0e5,
        "electron_index": 2.0,
        "electron_injection_luminosity_erg_s": 3.7e40,
        "proton_gamma_min": 1.0,
        "proton_gamma_max": 1.0e7,
        "proton_index": 2.0,
        "proton_injection_luminosity_erg_s": 2.8e46,
        "electron_escape": "R/c",
        "proton_escape": "R/c",
        "cache_dir": configured_cache_dir(),
    }


def load_leha_reference():
    """Load observer-frame Figure 10 photon and all-flavor-neutrino curves."""
    with np.load(LEHA_REFERENCE_FILE, allow_pickle=False) as archive:
        if (
            str(archive["scenario"]) != "LeHa"
            or str(archive["frame"]) != "observer"
            or str(archive["photon_quantity"]) != "escaped_broadband_nu_L_nu"
            or str(archive["neutrino_quantity"])
            != "all_flavor_neutrino_antineutrino_nu_L_nu"
            or str(archive["frequency_unit"]) != "Hz"
            or str(archive["luminosity_unit"]) != "erg s^-1"
            or bool(archive["am3_public_runtime_dependency"])
            or bool(archive["lehamoc_public_runtime_dependency"])
        ):
            raise RuntimeError("unexpected LeHa reference metadata")
        reference = {code: {} for code in ("AM3", "LeHaMoC")}
        for channel in ("photon", "neutrino"):
            for code, slug in (("AM3", "am3"), ("LeHaMoC", "lehamoc")):
                frequency = np.array(
                    archive[f"{channel}_{slug}_frequency_observer_hz"], copy=True
                )
                luminosity = np.array(
                    archive[f"{channel}_{slug}_nu_lnu_observer_erg_s"], copy=True
                )
                if (
                    frequency.shape != luminosity.shape
                    or np.any(np.diff(frequency) <= 0.0)
                    or np.any(~np.isfinite(frequency))
                    or np.any(~np.isfinite(luminosity))
                    or np.any(frequency <= 0.0)
                    or not np.any(luminosity > 0.0)
                ):
                    raise RuntimeError(f"invalid LeHa {code} {channel} curve")
                reference[code][channel] = (frequency, luminosity)
    return reference


def observer_multimessenger_result(forward_result, luminosity_distance_cm):
    """Extract displayed observer-frame photon and all-flavor-neutrino SEDs."""
    luminosity_scale = 4.0 * 3.141592653589793 * luminosity_distance_cm**2
    return {
        "photon_frequency_hz": (
            forward_result.sed["broadband_energy_observer_erg"] / H_ERG_S
        ),
        "photon_nu_lnu_erg_s": (
            forward_result.sed[
                "broadband_intrinsic_energy_flux_observer_erg_cm2_s"
            ]
            * luminosity_scale
        ),
        "neutrino_frequency_hz": (
            forward_result.neutrinos["energy_observer_erg"] / H_ERG_S
        ),
        "neutrino_nu_lnu_erg_s": (
            forward_result.neutrinos[
                "unmixed_flavor_energy_flux_observer_erg_cm2_s"
            ]["all_flavor"]
            * luminosity_scale
        ),
    }


def synchronize(value):
    if isinstance(value, dict):
        for item in value.values():
            synchronize(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            synchronize(item)
    else:
        block = getattr(value, "block_until_ready", None)
        if block is not None:
            block()


def timed_call(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize(result)
    return result, time.perf_counter() - started


def physical_support(frequency_hz, values):
    frequency = np.asarray(frequency_hz)
    array = np.asarray(values)
    return (
        np.isfinite(frequency)
        & np.isfinite(array)
        & (frequency > 0.0)
        & (array > 0.0)
    )


def broadband_photon_curve(forward_result):
    """Return the finite-positive Public observer broadband photon curve."""
    frequency = np.asarray(
        forward_result.sed["broadband_energy_observer_erg"] / H_ERG_S
    )
    flux = np.asarray(
        forward_result.sed[
            "broadband_intrinsic_energy_flux_observer_erg_cm2_s"
        ]
    )
    support = physical_support(frequency, flux)
    return frequency[support], flux[support]


def _log_interpolate(target, grid, values):
    return np.exp(np.interp(np.log(target), np.log(grid), np.log(values)))


def broadband_mock_data(
    truth_forward,
    *,
    random_seed=20260824,
    relative_uncertainty=0.10,
):
    """Prepare the shared deterministic 18-point broadband mock dataset."""
    frequency_hz = np.asarray(BROADBAND_FREQUENCIES_HZ, dtype=np.float64)
    truth_frequency_hz, truth_flux = broadband_photon_curve(truth_forward)
    truth_flux_at_data = _log_interpolate(
        frequency_hz, truth_frequency_hz, truth_flux
    )
    if truth_flux_at_data.min() <= 1.0e-4 * truth_flux.max():
        raise RuntimeError("broadband mock point entered a numerical tail")
    scatter = np.random.default_rng(random_seed).normal(
        0.0, relative_uncertainty, frequency_hz.size
    )
    observed_flux = truth_flux_at_data * (1.0 + scatter)
    if np.any(observed_flux <= 0.0):
        raise RuntimeError("broadband mock observation must be positive")
    return {
        "frequency_hz": frequency_hz,
        "energy_erg": H_ERG_S * frequency_hz,
        "truth_frequency_hz": truth_frequency_hz,
        "truth_flux": truth_flux,
        "truth_flux_at_data": truth_flux_at_data,
        "observed_flux": observed_flux,
        "flux_uncertainty": relative_uncertainty * truth_flux_at_data,
        "energy_error_erg": np.zeros(frequency_hz.size),
        "instrument": np.full(frequency_hz.size, "synthetic"),
        "statistic_type": np.full(frequency_hz.size, "gaussian"),
        "scientific_use": np.full(frequency_hz.size, "fit"),
        "random_seed": random_seed,
        "relative_uncertainty": relative_uncertainty,
    }


def broadband_fit_diagnostics(mock_data, initial_forward, best_forward):
    """Prepare displayed fit curves and standardized log residuals."""
    initial_frequency_hz, initial_flux = broadband_photon_curve(initial_forward)
    best_frequency_hz, best_flux = broadband_photon_curve(best_forward)
    initial_at_data = _log_interpolate(
        mock_data["frequency_hz"], initial_frequency_hz, initial_flux
    )
    best_at_data = _log_interpolate(
        mock_data["frequency_hz"], best_frequency_hz, best_flux
    )
    sigma_ln = mock_data["flux_uncertainty"] / mock_data["observed_flux"]
    initial_residual = (
        np.log(mock_data["observed_flux"]) - np.log(initial_at_data)
    ) / sigma_ln
    best_residual = (
        np.log(mock_data["observed_flux"]) - np.log(best_at_data)
    ) / sigma_ln
    return {
        "initial_frequency_hz": initial_frequency_hz,
        "initial_flux": initial_flux,
        "best_frequency_hz": best_frequency_hz,
        "best_flux": best_flux,
        "initial_residual": initial_residual,
        "best_residual": best_residual,
        "initial_residual_rms": float(np.sqrt(np.mean(initial_residual**2))),
        "best_residual_rms": float(np.sqrt(np.mean(best_residual**2))),
    }


def log_ratio_error(value, reference):
    return abs(np.log(value / reference))


def validate_jacobian(result, parameters):
    """Check the displayed Public Jacobian contract without changing values."""
    if result.parameters != tuple(parameters):
        raise RuntimeError("unexpected Jacobian parameter order")
    if result.values.shape != (result.energy_erg.size, len(parameters)):
        raise RuntimeError("unexpected Jacobian shape")
    if np.any(~np.isfinite(result.values)) or not np.any(
        np.abs(result.values) > 1.0e-8
    ):
        raise RuntimeError("Jacobian must be finite and nonzero")


def plot_leptonic_benchmark(results, references):
    """Draw one clean two-panel comparison with no relative peak floor."""
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.25), sharex=True, sharey=True)
    for axis, scenario in zip(axes, ("SSC-TH", "SSC-KN"), strict=True):
        curves = (
            (
                "HELENA",
                results[scenario]["frequency_hz"],
                results[scenario]["total_nu_lnu_erg_s"],
                "-",
                2.5,
                1.0,
            ),
            ("AM3", *references[scenario]["AM3"], "--", 1.65, 0.82),
            ("LeHaMoC", *references[scenario]["LeHaMoC"], "-.", 1.65, 0.82),
        )
        for label, frequency, luminosity, linestyle, linewidth, alpha in curves:
            support = physical_support(frequency, luminosity)
            if not np.any(support):
                raise RuntimeError(f"{scenario} {label} has no finite-positive support")
            axis.plot(
                np.asarray(frequency)[support],
                np.asarray(luminosity)[support],
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )
        axis.set_title(scenario)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1.0e8, 1.0e31)
        axis.set_ylim(1.0e32, 5.0e43)
        axis.set_xlabel(r"Observer-frame frequency $\nu$ [Hz]")
        axis.grid(which="both", alpha=0.16, linewidth=0.55)
        axis.legend(loc="best")
    axes[0].set_ylabel(r"$\nu L_{\nu}$ [erg s$^{-1}$]")
    figure.tight_layout()
    audit_figure_font(figure)
    return figure


def plot_syncool_transport(result, reference):
    """Draw the steady electrons and observer-frame synchrotron comparison."""
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    for label, source, linestyle, linewidth in (
        ("HELENA", result, "-", 2.5),
        ("AM3", reference, "--", 1.7),
    ):
        support = physical_support(
            source["electron_gamma"], source["steady_electron_dndgamma_cm3"]
        )
        axes[0].plot(
            np.asarray(source["electron_gamma"])[support],
            np.asarray(source["steady_electron_dndgamma_cm3"])[support],
            label=label,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        support = physical_support(
            source["photon_frequency_observer_hz"],
            source["synchrotron_observer_nu_lnu_erg_s"],
        )
        axes[1].plot(
            np.asarray(source["photon_frequency_observer_hz"])[support],
            np.asarray(source["synchrotron_observer_nu_lnu_erg_s"])[support],
            label=label,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    axes[0].set_title("Steady-state electrons")
    axes[0].set_xlabel(r"Electron Lorentz factor $\gamma$")
    axes[0].set_ylabel(r"$N_e(\gamma)$ [cm$^{-3}$ per $\gamma$]")
    axes[0].set_xlim(1.0, 1.0e5)
    axes[0].set_ylim(1.0e-15, 3.0e4)
    axes[1].set_title("Synchrotron SED")
    axes[1].set_xlabel(r"Observer-frame frequency $\nu$ [Hz]")
    axes[1].set_ylabel(r"$\nu L_{\nu}$ [erg s$^{-1}$]")
    axes[1].set_xlim(1.0e9, 1.0e22)
    axes[1].set_ylim(1.0e34, 2.0e45)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(which="both", alpha=0.16, linewidth=0.55)
        axis.legend(loc="best")
    figure.tight_layout()
    audit_figure_font(figure)
    return figure


def plot_pgamma_plpl(result, reference):
    """Draw the three pγ secondary-injection comparisons."""
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    panels = (
        (
            "photon",
            r"$\pi^0$-decay photons",
            r"$\epsilon_\gamma=E_\gamma/(m_ec^2)$",
            r"$\epsilon_\gamma^2 Q_\gamma$ [cm$^{-3}$ s$^{-1}$]",
        ),
        (
            "lepton",
            r"Charged-pion/muon secondary $e^\pm$",
            r"$\gamma_e=E_e/(m_ec^2)$",
            r"$\gamma_e^2 Q_{e^++e^-}$ [cm$^{-3}$ s$^{-1}$]",
        ),
        (
            "neutrino",
            "All-flavor neutrinos",
            r"$\epsilon_\nu=E_\nu/(m_ec^2)$",
            r"$\epsilon_\nu^2 Q_{\nu+\bar{\nu}}$ [cm$^{-3}$ s$^{-1}$]",
        ),
    )
    for axis, (channel, title, xlabel, ylabel) in zip(axes, panels, strict=True):
        curves = (
            (
                "HELENA",
                result["energy_dimensionless"],
                result[channel],
                "-",
                2.5,
            ),
            ("AM3", *reference["AM3"][channel], "--", 1.7),
            ("LeHaMoC", *reference["LeHaMoC"][channel], "-.", 1.7),
        )
        for label, energy, injection, linestyle, linewidth in curves:
            support = physical_support(energy, injection)
            if not np.any(support):
                raise RuntimeError(f"pγ-PLPL {label} {channel} has no support")
            axis.plot(
                np.asarray(energy)[support],
                np.asarray(injection)[support],
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1.0e4, 1.0e12)
        axis.set_ylim(1.0e-9, 1.0e-1)
        axis.grid(which="both", alpha=0.16, linewidth=0.55)
        axis.legend(loc="best")
    figure.tight_layout()
    audit_figure_font(figure)
    return figure


def plot_ps_benchmark(result, reference):
    """Draw observer-frame PS photon and all-flavor-neutrino SEDs."""
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    panels = (
        ("photon", "Photon SED", (1.0e9, 1.0e35), (1.0e35, 3.0e43)),
        (
            "neutrino",
            "All-flavor neutrino SED",
            (1.0e23, 1.0e36),
            (1.0e34, 3.0e40),
        ),
    )
    for axis, (channel, title, xlim, ylim) in zip(axes, panels, strict=True):
        curves = (
            (
                "HELENA",
                result[f"{channel}_frequency_hz"],
                result[f"{channel}_nu_lnu_erg_s"],
                "-",
                2.5,
            ),
            ("AM3", *reference["AM3"][channel], "--", 1.7),
            ("LeHaMoC", *reference["LeHaMoC"][channel], "-.", 1.7),
        )
        for label, frequency, luminosity, linestyle, linewidth in curves:
            support = physical_support(frequency, luminosity)
            if not np.any(support):
                raise RuntimeError(f"PS {label} {channel} has no support")
            axis.plot(
                np.asarray(frequency)[support],
                np.asarray(luminosity)[support],
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        axis.set_title(title)
        axis.set_xlabel(r"Observer-frame frequency $\nu$ [Hz]")
        axis.set_ylabel(r"$\nu L_{\nu}$ [erg s$^{-1}$]")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.grid(which="both", alpha=0.16, linewidth=0.55)
        axis.legend(loc="best")
    figure.tight_layout()
    audit_figure_font(figure)
    return figure


def plot_leha_benchmark(result, reference):
    """Draw observer-frame LeHa photon and all-flavor-neutrino SEDs."""
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    panels = (
        ("photon", "Photon SED", (1.0e9, 1.0e35), (1.0e30, 1.0e45)),
        (
            "neutrino",
            "All-flavor neutrino SED",
            (1.0e24, 1.0e35),
            (1.0e32, 3.0e44),
        ),
    )
    for axis, (channel, title, xlim, ylim) in zip(axes, panels, strict=True):
        curves = (
            (
                "HELENA",
                result[f"{channel}_frequency_hz"],
                result[f"{channel}_nu_lnu_erg_s"],
                "-",
                2.5,
            ),
            ("AM3", *reference["AM3"][channel], "--", 1.7),
            ("LeHaMoC", *reference["LeHaMoC"][channel], "-.", 1.7),
        )
        for label, frequency, luminosity, linestyle, linewidth in curves:
            support = physical_support(frequency, luminosity)
            if not np.any(support):
                raise RuntimeError(f"LeHa {label} {channel} has no support")
            axis.plot(
                np.asarray(frequency)[support],
                np.asarray(luminosity)[support],
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        axis.set_title(title)
        axis.set_xlabel(r"Observer-frame frequency $\nu$ [Hz]")
        axis.set_ylabel(r"$\nu L_{\nu}$ [erg s$^{-1}$]")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.grid(which="both", alpha=0.16, linewidth=0.55)
        axis.legend(loc="best")
    figure.tight_layout()
    audit_figure_font(figure)
    return figure


def plot_broadband_fit(
    mock_data,
    diagnostics,
    *,
    initial_label="Initial model",
    best_label="Best-fit model",
):
    """Draw the shared full-broadband fit and log-residual figure."""
    figure, (sed_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.0)},
    )
    plot_limits = (5.0e12, 2.0e32)
    for label, frequency, flux, linestyle, linewidth in (
        (
            "Truth model",
            mock_data["truth_frequency_hz"],
            mock_data["truth_flux"],
            "-",
            2.2,
        ),
        (
            initial_label,
            diagnostics["initial_frequency_hz"],
            diagnostics["initial_flux"],
            "--",
            1.8,
        ),
        (
            best_label,
            diagnostics["best_frequency_hz"],
            diagnostics["best_flux"],
            "-.",
            1.8,
        ),
    ):
        visible = (frequency >= plot_limits[0]) & (frequency <= plot_limits[1])
        sed_axis.plot(
            frequency[visible],
            flux[visible],
            label=label,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    sed_axis.errorbar(
        mock_data["frequency_hz"],
        mock_data["observed_flux"],
        yerr=mock_data["flux_uncertainty"],
        fmt="o",
        color="black",
        capsize=2.5,
        label="Broadband synthetic observations",
    )
    sed_axis.set_xscale("log")
    sed_axis.set_yscale("log")
    sed_axis.set_xlim(*plot_limits)
    sed_axis.set_ylabel(r"$\nu F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]")
    sed_axis.grid(which="both", alpha=0.16, linewidth=0.55)
    sed_axis.legend(loc="best")

    residual_axis.plot(
        mock_data["frequency_hz"],
        diagnostics["initial_residual"],
        "o--",
        linewidth=1.4,
        label="Initial",
    )
    residual_axis.plot(
        mock_data["frequency_hz"],
        diagnostics["best_residual"],
        "s-.",
        linewidth=1.4,
        label="Best fit",
    )
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.set_xscale("log")
    residual_axis.set_xlabel(r"Observer-frame frequency $\nu$ [Hz]")
    residual_axis.set_ylabel(r"$r_{\ln F}$")
    residual_axis.grid(which="both", alpha=0.16, linewidth=0.55)
    residual_axis.legend(loc="best")
    figure.tight_layout()
    audit_figure_font(figure)
    return figure


def plot_ad_diagnostics(jacobian_result, gauss_newton_result):
    """Draw broadband sensitivities and the Gauss-Newton correlation matrix."""
    sensitivity_frequency_hz = jacobian_result.energy_erg / H_ERG_S
    plot_mask = (
        (sensitivity_frequency_hz >= 5.0e12)
        & (sensitivity_frequency_hz <= 2.0e32)
    )
    if not np.any(plot_mask):
        raise RuntimeError("Jacobian plot mask has no support")

    figure, (sensitivity_axis, correlation_axis) = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        gridspec_kw={"width_ratios": (1.65, 1.0)},
    )
    for column, name in enumerate(jacobian_result.parameters):
        sensitivity_axis.semilogx(
            sensitivity_frequency_hz[plot_mask],
            jacobian_result.values[plot_mask, column],
            linewidth=1.8,
            label=name,
        )
    sensitivity_axis.axhline(0.0, color="black", linewidth=0.8)
    sensitivity_axis.set_xlabel(r"Observer-frame frequency $\nu$ [Hz]")
    sensitivity_axis.set_ylabel(
        r"$\partial \ln(\nu F_{\nu}) / \partial \theta$"
    )
    sensitivity_axis.set_title("Broadband parameter sensitivity")
    sensitivity_axis.grid(which="both", alpha=0.16, linewidth=0.55)
    sensitivity_axis.legend(loc="best")

    correlation_image = correlation_axis.imshow(
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
            )
    correlation_axis.set_title("4D Gauss–Newton correlation")
    colorbar = figure.colorbar(correlation_image, ax=correlation_axis, shrink=0.86)
    colorbar.set_label("Local correlation")
    figure.tight_layout()
    audit_figure_font(figure)
    return figure


def display_figure(figure):
    display(figure)
    plt.close(figure)


def _table_html(headers, rows):
    header = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<table class="helena-table"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table>"
    )


def show_fit_recovery_table(rows, *, first_header="Parameter"):
    """Display physical initial/truth/best values without log-coordinate labels."""
    body = "".join(
        "<tr>"
        f"<td>{label}</td><td>{initial:.6e}</td><td>{truth:.6e}</td>"
        f"<td>{best:.6e}</td><td>{abs(best / truth - 1.0):.3e}</td>"
        "</tr>"
        for label, initial, truth, best in rows
    )
    display(
        HTML(
            '<table class="helena-table"><thead><tr>'
            f"<th>{first_header}</th><th>Initial</th><th>Truth</th>"
            "<th>Best fit</th><th>Relative recovery</th>"
            f"</tr></thead><tbody>{body}</tbody></table>"
        )
    )


def show_gauss_newton_summary(
    jacobian_result,
    gauss_newton_result,
    *,
    jacobian_first_seconds,
    jacobian_warm_seconds,
    gauss_newton_seconds,
):
    """Display the unchanged Public AD/GN diagnostics and timings."""
    correlation_pairs = sorted(
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
    print(f"Jacobian shape: {jacobian_result.values.shape}")
    print(
        f"Jacobian first / warm: {jacobian_first_seconds:.2f} / "
        f"{jacobian_warm_seconds:.2f} s"
    )
    print(f"Gauss-Newton runtime: {gauss_newton_seconds:.4f} s")
    print(
        f"Status / rank: {gauss_newton_result.status} / "
        f"{gauss_newton_result.rank}"
    )
    print(f"Condition number: {gauss_newton_result.condition_number:.6e}")
    print(
        "Singular values: "
        + np.array2string(gauss_newton_result.singular_values, precision=6)
    )
    print(
        "Strongest correlations: "
        + "; ".join(
            f"{left}–{right}={value:+.3f}"
            for _, value, left, right in correlation_pairs[:3]
        )
    )
    print(
        "Gauss-Newton is a local identifiability diagnostic around the "
        "best-fit point, not a posterior or global uncertainty estimate."
    )
    print("AD evaluation point: 4D broadband best-fit model")


def show_parameter_table():
    display(
        HTML(
            _table_html(
                ["Parameter", "SSC-TH", "SSC-KN"],
                [
                    ("R [cm]", "1.0e15", "1.0e15"),
                    ("B [G]", "0.1", "0.01"),
                    ("delta", "30", "30"),
                    ("gamma_min", "1", "1"),
                    ("gamma_max", "1.0e4", "1.0e6"),
                    ("s_e", "1.9", "1.9"),
                    ("n0 [cm^-3]", "1.0e4", "1.0e4"),
                    ("cooling", "disabled", "disabled"),
                    ("escape", "R/c (quoted)", "R/c (quoted)"),
                ],
            )
        )
    )


def show_pgamma_parameter_table():
    display(
        HTML(
            _table_html(
                ["pγ-PLPL parameter", "Value"],
                [
                    ("R [cm]", "1.0e15"),
                    ("gamma_p,min / gamma_p,max", "1 / 1.0e8"),
                    ("s_p", "1.9"),
                    ("L_p,inj [erg s^-1]", "8.5e43"),
                    ("n_p^ss(gamma=1) [cm^-3]", "8490"),
                    ("proton escape", "R/c"),
                    ("proton / pion / muon cooling", "disabled"),
                    ("target compactness / index", "1.0e-5 / 2.0"),
                    ("target E/(m_e c^2)", "1.0e-6 to 1.0e-1"),
                    ("output", "blob-frame x^2 Q(x) [cm^-3 s^-1]"),
                ],
            )
        )
    )


def show_timing_table(timings, notebook_total_seconds):
    rows = [
        (
            scenario,
            f"{timings[scenario]['first_seconds']:.4f}",
            f"{timings[scenario]['warm_seconds']:.4f}",
        )
        for scenario in ("SSC-TH", "SSC-KN")
    ]
    rows.append(("Total notebook", f"{notebook_total_seconds:.4f}", "—"))
    display(HTML(_table_html(["Scenario", "First [s]", "Warm [s]"], rows)))


def audit_figure_font(figure):
    files = set()
    for artist in figure.findobj(match=Text):
        if not artist.get_text():
            continue
        path = font_manager.findfont(
            artist.get_fontproperties(), fallback_to_default=False
        )
        if font_manager.FontProperties(fname=path).get_name() != REQUESTED_FONT:
            raise RuntimeError(f"figure contains a non-{REQUESTED_FONT} text artist")
        files.add(Path(path).name)
    if not files:
        raise RuntimeError("figure font audit found no text")
    return sorted(files)


def _self_check():
    support = physical_support([1.0, 2.0, np.nan], [1.0, 0.0, 3.0])
    assert support.tolist() == [True, False, False]
    assert len(BROADBAND_FREQUENCIES_HZ) == 18
    assert np.all(np.diff(BROADBAND_FREQUENCIES_HZ) > 0.0)
    assert log_ratio_error(1.0, 1.0) == 0.0
    assert set(benchmark_cases()) == {"SSC-TH", "SSC-KN"}
    references = load_benchmark_references()
    assert all(set(references[name]) == {"AM3", "LeHaMoC"} for name in references)
    assert load_syncool_reference()["electron_gamma"].ndim == 1
    pgamma_case = pgamma_plpl_case()
    assert pgamma_case["target_epsilon_min"] == 1.0e-6
    assert pgamma_case["target_epsilon_max"] == 1.0e-1
    assert all(
        set(channels) == {"photon", "lepton", "neutrino"}
        for channels in load_pgamma_reference().values()
    )
    ps = ps_case()
    assert ps["electron_injection_luminosity_erg_s"] == 1.571009e37
    assert ps["proton_injection_luminosity_erg_s"] == 1.000202e43
    assert all(
        set(channels) == {"photon", "neutrino"}
        for channels in load_ps_reference().values()
    )
    leha = leha_case()
    assert leha["electron_injection_luminosity_erg_s"] == 3.7e40
    assert leha["proton_injection_luminosity_erg_s"] == 2.8e46
    assert all(
        set(channels) == {"photon", "neutrino"}
        for channels in load_leha_reference().values()
    )
    assert "helena-table" in _table_html(["A"], [["B"]])


if __name__ == "__main__":
    _self_check()
    print("quickstart_helpers self-check: PASS")
