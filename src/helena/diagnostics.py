"""Scientific local curvature and host-side production diagnostics."""

from dataclasses import dataclass

import numpy as np

from helena.core.constants import (
    CHARGED_PION_MASS_CGS as _CHARGED_PION_MASS_G,
)
from helena.core.constants import (
    MUON_MASS_CGS as _MUON_MASS_G,
)
from helena.core.constants import (
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.muon_timescales import muon_synchrotron_cooling_time
from helena.core.pion_timescales import charged_pion_synchrotron_cooling_time
from helena.core.proton_timescales import (
    proton_adiabatic_continuous_loss_rate,
    proton_synchrotron_continuous_loss_rate,
)
from helena.core.steady_state_finite_volume import _finite_volume_coefficients

__all__ = [
    "GaussNewtonResult",
    "gauss_newton",
]

_MESON_SYNCHROTRON_POWER_RESIDUAL_LIMIT = 1.0e-2


def _immutable_array(value, *, dtype):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class GaussNewtonResult:
    """Local Gaussian curvature diagnostics at active observations."""

    energy_erg: np.ndarray
    active_indices: np.ndarray
    parameters: tuple[str, ...]
    weighted_jacobian: np.ndarray
    curvature: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float
    status: str

    def __post_init__(self):
        for name, dtype in (
            ("energy_erg", np.float64),
            ("active_indices", np.int64),
            ("weighted_jacobian", np.float64),
            ("curvature", np.float64),
            ("covariance", np.float64),
            ("correlation", np.float64),
            ("singular_values", np.float64),
        ):
            object.__setattr__(
                self, name, _immutable_array(getattr(self, name), dtype=dtype)
            )
        object.__setattr__(self, "parameters", tuple(self.parameters))

    @property
    def covariance_label(self):
        """State the deliberately local scope of ``covariance``."""

        return "local Gauss-Newton approximation; not a global uncertainty"


def gauss_newton(jacobian_result, dataset):
    """Return local ``H_GN = J.T @ W @ J`` diagnostics.

    ``T`` is the ordinary transpose. Only caller-active independent symmetric
    Gaussian detections are accepted, with
    ``W = diag(1 / sigma_log**2)`` and ``sigma_log = sigma_F / F``.
    Context and excluded rows remain inactive under the maintained P4 mask.
    """

    # Local imports avoid pulling the fitting facade into low-level physics imports.
    from helena.fitting.core import _local_identifiability_diagnostics
    from helena.fitting.public import SEDDataset
    from helena.sensitivity import SensitivityResult

    if not isinstance(jacobian_result, SensitivityResult):
        raise TypeError("jacobian_result must be a SensitivityResult")
    if jacobian_result.direction is not None:
        raise ValueError("gauss_newton requires a Jacobian result, not a JVP")
    if not isinstance(dataset, SEDDataset):
        raise TypeError("dataset must be a SEDDataset")
    active = dataset.likelihood_mask
    if not np.any(active):
        raise ValueError("dataset must select at least one fit or constraint point")
    unsupported = set(dataset.statistic_type[active]) - {"gaussian"}
    if unsupported:
        raise ValueError(
            "Gauss-Newton diagnostics require active independent symmetric "
            f"Gaussian detections; unsupported active types: {sorted(unsupported)}"
        )
    energy = dataset.energy_erg[active]
    model_energy = jacobian_result.energy_erg
    if np.any(energy < model_energy[0]) or np.any(energy > model_energy[-1]):
        raise ValueError("active dataset energies must lie inside the Jacobian domain")
    sampled = np.column_stack(
        [
            np.interp(np.log(energy), np.log(model_energy), column)
            for column in jacobian_result.values.T
        ]
    )
    sigma_log = (
        dataset.sed_flux_error_lower_erg_cm2_s[active]
        / dataset.sed_flux_erg_cm2_s[active]
    )
    weighted = sampled / sigma_log[:, None]
    status, diagnostics, finite, _ = _local_identifiability_diagnostics(
        weighted, jacobian_result.parameters
    )
    if not finite or diagnostics is None:
        raise FloatingPointError("Gauss-Newton diagnostics are not finite")
    return GaussNewtonResult(
        energy_erg=energy,
        active_indices=np.flatnonzero(active),
        parameters=jacobian_result.parameters,
        weighted_jacobian=weighted,
        curvature=diagnostics["gauss_newton_matrix"],
        covariance=diagnostics["local_gauss_newton_covariance"],
        correlation=diagnostics["normalized_parameter_correlation_matrix"],
        singular_values=diagnostics["singular_values"],
        rank=diagnostics["jacobian_rank"],
        condition_number=diagnostics["condition_number"],
        status=status,
    )


def _value(configuration, name, default=None):
    if name in configuration:
        return configuration[name]
    if default is not None:
        return default
    raise ValueError(f"production_configuration is missing {name!r}")


def _integral(values, coordinate):
    return float(np.trapezoid(np.asarray(values), x=np.asarray(coordinate)))


def _synchrotron_power_ledger(
    gamma,
    density,
    particle_mass,
    cooling_time,
    low_photon_energy,
    low_intrinsic_sed,
    high_photon_energy,
    high_intrinsic_sed,
    volume,
    enabled,
):
    gamma = np.asarray(gamma)
    density = np.asarray(density)
    loss_power = volume * _integral(
        gamma * density * particle_mass * _C_CGS**2 * gamma / cooling_time,
        np.log(gamma),
    )
    photon_power = _integral(low_intrinsic_sed, np.log(low_photon_energy)) + _integral(
        high_intrinsic_sed, np.log(high_photon_energy)
    )
    scale = max(abs(loss_power), abs(photon_power), np.finfo(np.float64).tiny)
    near_zero = max(abs(loss_power), abs(photon_power)) <= np.finfo(np.float64).tiny
    residual = 0.0 if near_zero else (photon_power - loss_power) / scale
    return {
        "enabled": enabled,
        "transport_synchrotron_loss_power_erg_s": loss_power,
        "photon_injection_power_erg_s": photon_power,
        "near_zero": near_zero,
        "power_residual": None if not enabled else residual,
        "threshold": _MESON_SYNCHROTRON_POWER_RESIDUAL_LIMIT,
        "passed": (not enabled)
        or (near_zero and photon_power == loss_power)
        or abs(residual) <= _MESON_SYNCHROTRON_POWER_RESIDUAL_LIMIT,
    }


def hadronic_cascade_meson_synchrotron_power_ledgers(
    configuration,
    chain,
    sources,
):
    """Return post-state pion/muon synchrotron conservation ledgers."""
    pion_gamma = np.asarray(_value(configuration, "pion_gamma"))
    muon_gamma = np.asarray(_value(configuration, "muon_gamma"))
    low_photon_energy = np.asarray(
        _value(configuration, "synchrotron_photon_energy_erg")
    )
    high_photon_energy = np.asarray(_value(configuration, "gamma_ray_energy_erg"))
    magnetic_field = float(_value(configuration, "magnetic_field_G"))
    radius = float(_value(configuration, "radius_cm"))
    volume = 4.0 * np.pi * radius**3 / 3.0
    pion_cooling = charged_pion_synchrotron_cooling_time(pion_gamma, magnetic_field)
    muon_cooling = muon_synchrotron_cooling_time(muon_gamma, magnetic_field)
    low_seds = sources["low_energy_intrinsic_seds_erg_s"]
    high_seds = sources["high_energy_intrinsic_seds_erg_s"]
    ledgers = {
        "pion_plus": _synchrotron_power_ledger(
            pion_gamma,
            chain["n_pi_plus"],
            _CHARGED_PION_MASS_G,
            pion_cooling,
            low_photon_energy,
            low_seds["pion_plus"],
            high_photon_energy,
            high_seds["pion_plus"],
            volume,
            sources["pion_enabled"],
        ),
        "pion_minus": _synchrotron_power_ledger(
            pion_gamma,
            chain["n_pi_minus"],
            _CHARGED_PION_MASS_G,
            pion_cooling,
            low_photon_energy,
            low_seds["pion_minus"],
            high_photon_energy,
            high_seds["pion_minus"],
            volume,
            sources["pion_enabled"],
        ),
        "muon_plus": _synchrotron_power_ledger(
            muon_gamma,
            chain["n_mu_plus_left"] + chain["n_mu_plus_right"],
            _MUON_MASS_G,
            muon_cooling,
            low_photon_energy,
            low_seds["muon_plus"],
            high_photon_energy,
            high_seds["muon_plus"],
            volume,
            sources["muon_enabled"],
        ),
        "muon_minus": _synchrotron_power_ledger(
            muon_gamma,
            chain["n_mu_minus_left"] + chain["n_mu_minus_right"],
            _MUON_MASS_G,
            muon_cooling,
            low_photon_energy,
            low_seds["muon_minus"],
            high_photon_energy,
            high_seds["muon_minus"],
            volume,
            sources["muon_enabled"],
        ),
    }

    def charge_sum(left, right, enabled):
        loss_power = sum(
            ledgers[name]["transport_synchrotron_loss_power_erg_s"]
            for name in (left, right)
        )
        photon_power = sum(
            ledgers[name]["photon_injection_power_erg_s"] for name in (left, right)
        )
        scale = max(abs(loss_power), abs(photon_power), np.finfo(np.float64).tiny)
        near_zero = max(abs(loss_power), abs(photon_power)) <= np.finfo(np.float64).tiny
        residual = 0.0 if near_zero else (photon_power - loss_power) / scale
        return {
            "enabled": enabled,
            "transport_synchrotron_loss_power_erg_s": loss_power,
            "photon_injection_power_erg_s": photon_power,
            "near_zero": near_zero,
            "power_residual": None if not enabled else residual,
            "threshold": _MESON_SYNCHROTRON_POWER_RESIDUAL_LIMIT,
            "passed": (not enabled)
            or (near_zero and photon_power == loss_power)
            or abs(residual) <= _MESON_SYNCHROTRON_POWER_RESIDUAL_LIMIT,
        }

    ledgers["pion_total"] = charge_sum(
        "pion_plus", "pion_minus", sources["pion_enabled"]
    )
    ledgers["muon_total"] = charge_sum(
        "muon_plus", "muon_minus", sources["muon_enabled"]
    )
    return ledgers


def _number_ledger(gamma, injection, density, sink_spectrum, continuous_loss):
    gamma = np.asarray(gamma)
    injection = np.asarray(injection)
    density = np.asarray(density)
    sink_spectrum = np.asarray(sink_spectrum)
    continuous_loss = np.asarray(continuous_loss)
    widths = _finite_volume_coefficients(gamma, continuous_loss, np.ones_like(gamma))[0]
    injected = float(np.sum(injection * widths))
    removed = float(np.sum(sink_spectrum * widths) + continuous_loss[0] * density[0])
    absolute_residual = abs(removed - injected)
    near_zero = abs(injected) <= np.finfo(np.float64).tiny
    residual = None if near_zero else absolute_residual / abs(injected)
    return {
        "injected_number_cm3_s": injected,
        "removed_number_cm3_s": removed,
        "near_zero": near_zero,
        "absolute_residual_cm3_s": absolute_residual,
        "number_residual": residual,
        "threshold": 2.0e-13,
        "passed": (
            absolute_residual <= np.finfo(np.float64).tiny
            if near_zero
            else residual < 2.0e-13
        ),
    }


def hadronic_cascade_particle_ledgers(
    configuration,
    loss_rate,
    bh_loss,
    pp_loss,
    proton_density,
    proton_escape,
    chain,
):
    """Return post-state proton, pion, and muon number ledgers."""
    proton_gamma = np.asarray(_value(configuration, "proton_gamma"))
    pion_gamma = np.asarray(_value(configuration, "pion_gamma"))
    muon_gamma = np.asarray(_value(configuration, "muon_gamma"))
    magnetic_field = float(_value(configuration, "magnetic_field_G"))
    bh_loss = np.asarray(bh_loss)
    pp_loss = np.asarray(pp_loss)
    proton_continuous_loss = (
        proton_synchrotron_continuous_loss_rate(proton_gamma, magnetic_field)
        + proton_adiabatic_continuous_loss_rate(
            proton_gamma,
            _value(configuration, "proton_adiabatic_rate_s_inv"),
        )
        + proton_gamma * (np.asarray(loss_rate) + bh_loss + pp_loss)
    )
    ledgers = {
        "proton": _number_ledger(
            proton_gamma,
            _value(configuration, "primary_proton_injection_dgamma_cm3_s"),
            proton_density,
            proton_escape,
            proton_continuous_loss,
        )
    }
    pion_loss = pion_gamma / charged_pion_synchrotron_cooling_time(
        pion_gamma, magnetic_field
    )
    for charge in ("plus", "minus"):
        ledgers[f"pi_{charge}"] = _number_ledger(
            pion_gamma,
            chain[f"q_pi_{charge}"],
            chain[f"n_pi_{charge}"],
            chain[f"q_pi_{charge}_decay"],
            pion_loss,
        )
    muon_loss = muon_gamma / muon_synchrotron_cooling_time(muon_gamma, magnetic_field)
    for charge in ("plus", "minus"):
        for helicity in ("left", "right"):
            name = f"mu_{charge}_{helicity}"
            ledgers[name] = _number_ledger(
                muon_gamma,
                chain[f"q_{name}"],
                chain[f"n_{name}"],
                chain[f"q_{name}_decay"],
                muon_loss,
            )
    return ledgers


def validate_finite_volume_coefficients(gamma, loss_rate, sink_rate):
    """Run the NumPy reference coefficient checks for an eager JAX call."""
    _finite_volume_coefficients(gamma, loss_rate, sink_rate)
