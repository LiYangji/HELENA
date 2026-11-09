"""Thin public dataset and maximum-likelihood fitting boundary."""

import time
import warnings
from dataclasses import dataclass, replace
from math import exp, log
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.jax_backend.chains.steady_state_hadronic_cascade import (
    electromagnetic_ledger_passed_jax,
    energy_weighted_spectral_defect_jax,
    outer_hadronic_picard_step,
)
from helena.core.propagation import apply_ebl_attenuation
from helena.fitting.core import ParameterSelection, fit_maximum_likelihood
from helena.fitting.real_data import (
    SEDData,
    interpolate_sed_to_observations,
    mixed_sed_diagnostics,
    mixed_sed_nll,
)
from helena.model import ModelConfig, OneZoneModel, _earth_broadband_photon_support
from helena.models.c513 import (
    _canonical_broadband_escaped_photon_sed,
    c513_configuration_from_injection_shape_offsets,
    c513_configuration_from_log_offsets,
)
from helena.parameters import DIFFERENTIABLE_PARAMETER_NAMES

__all__ = ["FitResult", "SEDDataset", "fit"]

_PARAMETER_FIELDS = MappingProxyType(
    dict(
        zip(
            DIFFERENTIABLE_PARAMETER_NAMES,
            (
                (
                    "electron_kinetic_injection_luminosity_erg_s",
                    "electron_kinetic_injection_luminosity_erg_s",
                    "log",
                ),
                (
                    "proton_kinetic_injection_luminosity_erg_s",
                    "proton_kinetic_injection_luminosity_erg_s",
                    "log",
                ),
                ("magnetic_field_G", "magnetic_field_G", "log"),
                ("radius_cm", "radius_cm", "log"),
                ("electron_injection_index", "electron_index", "add"),
                ("proton_injection_index", "proton_index", "add"),
                ("electron_gamma_max", "electron_gamma_cutoff", "log"),
                ("proton_gamma_max", "proton_gamma_cutoff", "log"),
            ),
            strict=True,
        )
    )
)


class SEDDataset(SEDData):
    """Immutable Earth-observed ``E F_E`` data for the maintained likelihood."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class FitResult:
    """Stable physical and statistical result from :func:`fit`."""

    best_parameters: MappingProxyType
    best_config: ModelConfig
    free_parameters: tuple[str, ...]
    fixed_parameters: MappingProxyType
    joint_dimension: int
    nll: float
    optimizer: str
    success: bool
    status: int
    message: str
    iterations: int | None
    evaluations: int
    prediction: MappingProxyType
    residuals: MappingProxyType
    uncertainties: None
    covariance: None
    runtime_seconds: float
    backend: str
    device: str
    x64: bool


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({name: _freeze(item) for name, item in value.items()})
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    return value


def _physical_coordinates(model):
    reference = model._setup["metadata"]["physical_parameters"]
    values = []
    for name in DIFFERENTIABLE_PARAMETER_NAMES:
        field, reference_name, coordinate = _PARAMETER_FIELDS[name]
        value = getattr(model.config, field)
        reference_value = reference[reference_name]
        values.append(
            log(value / reference_value)
            if coordinate == "log"
            else value - reference_value
        )
    return np.asarray(values, dtype=np.float64)


def _physical_value(model, name, coordinate):
    field, reference_name, coordinate_type = _PARAMETER_FIELDS[name]
    reference = model._setup["metadata"]["physical_parameters"][reference_name]
    value = (
        reference * exp(coordinate)
        if coordinate_type == "log"
        else reference + coordinate
    )
    return field, float(value)


def _prepare_fit_request(model, dataset, parameters, optimizer):
    if not isinstance(model, OneZoneModel):
        raise TypeError("model must be a OneZoneModel")
    if not isinstance(dataset, SEDDataset):
        raise TypeError("dataset must be a SEDDataset")
    if dataset.selected_count == 0:
        raise ValueError("dataset must select at least one fit or constraint point")
    if isinstance(parameters, str):
        raise TypeError("parameters must be a sequence of parameter names")
    try:
        names = tuple(parameters)
    except TypeError as error:
        raise TypeError("parameters must be a sequence of parameter names") from error
    if not names:
        raise ValueError("parameters must contain at least one name")
    if len(names) > len(DIFFERENTIABLE_PARAMETER_NAMES):
        raise ValueError("parameters may contain at most eight names")
    selection = ParameterSelection(names, _physical_coordinates(model), None)
    if optimizer not in ("lbfgsb", "migrad"):
        raise ValueError("optimizer must be 'lbfgsb' or 'migrad'")
    if jax.default_backend() == "cpu" and len(names) > 2:
        warnings.warn(
            "joint fitting with more than two free parameters is supported on "
            "CPU but may be very slow or memory intensive for the production "
            "model; GPU execution is recommended",
            RuntimeWarning,
            stacklevel=2,
        )
    return selection


def _model_prediction(model, selection):
    """Differentiate the bounded, convergence-frozen production Picard program.

    The same core, outer tolerances and consecutive-pass rule as forward are
    used. Failed inner/EM gates or exhausted outer iterations return NaNs.
    Host-only particle/meson power audits remain forward diagnostics; these
    are not duplicated in this JAX predictor. The derivative is of this finite
    program, not implicit differentiation of an exact steady solution.
    """
    setup = model._setup
    configuration = setup["production_configuration"]
    responses = model._get_prepared_responses()
    _, energy_observer, flux_scale, earth_indices, transmission = (
        _earth_broadband_photon_support(model.config, configuration)
    )
    baseline = jnp.asarray(selection.baseline_values)
    initial = setup["initial_internal_photon_state_blob"]
    external = setup["fixed_external_photon_state_blob"]
    strategy = setup["metadata"]["numerical_strategy"]
    maximum = int(strategy["max_outer_iterations"])
    outer_tolerance = float(strategy["outer_rtol"])
    required_passes = int(strategy["required_consecutive_passes"])
    if (
        maximum < 2
        or not np.isfinite(outer_tolerance)
        or outer_tolerance <= 0
        or required_passes < 1
    ):
        raise ValueError(
            "outer loop requires at least two iterations and positive tolerances/pass count"
        )

    def prediction(free_values):
        offsets = selection.unpack(free_values) - baseline
        dynamic = c513_configuration_from_log_offsets(configuration, offsets[:4])
        dynamic = c513_configuration_from_injection_shape_offsets(dynamic, offsets[4:])

        def active(previous):
            state = {
                "energy_erg": initial["energy_erg"],
                "density_cm3_erg": previous["density_cm3_erg"],
            }
            result = outer_hadronic_picard_step(
                state,
                external,
                dynamic,
                bethe_heitler_pair_responses=responses["bethe_heitler_pair_responses"],
                gamma_gamma_pair_response=responses["gamma_gamma_pair_response"],
                physics_only=True,
            )
            inner = result["inner_electromagnetic_cascade"]
            following = result["next_internal_photon_state"]
            proton = result["proton_density_dgamma_cm3"]
            valid = (
                previous["valid"]
                & inner["converged"]
                & electromagnetic_ledger_passed_jax(
                    inner, dynamic.get("inner_relative_tolerance", 1e-4)
                )
                & jnp.all(jnp.isfinite(proton) & (proton >= 0))
                & jnp.all(
                    jnp.isfinite(following["density_cm3_erg"])
                    & (following["density_cm3_erg"] >= 0)
                )
            )
            defect = jnp.maximum(
                energy_weighted_spectral_defect_jax(
                    proton, previous["proton_density"], dynamic["proton_gamma"]
                ),
                energy_weighted_spectral_defect_jax(
                    following["density_cm3_erg"],
                    previous["density_cm3_erg"],
                    following["energy_erg"],
                ),
            )
            passes = jnp.where(
                valid & (previous["iteration"] > 0) & (defect < outer_tolerance),
                previous["passes"] + 1,
                0,
            )
            _, escaped = _canonical_broadband_escaped_photon_sed(configuration, inner)
            return {
                "density_cm3_erg": following["density_cm3_erg"],
                "proton_density": proton,
                "escaped": escaped,
                "valid": valid,
                "converged": valid & (passes >= required_passes),
                "passes": passes,
                "iteration": previous["iteration"] + 1,
            }

        def outer_round(previous, _):
            following = jax.lax.cond(
                previous["valid"] & ~previous["converged"],
                active,
                lambda state: state,
                previous,
            )
            return following, None

        final, _ = jax.lax.scan(
            outer_round,
            {
                "density_cm3_erg": initial["density_cm3_erg"],
                "proton_density": jnp.zeros_like(configuration["proton_gamma"]),
                "escaped": jnp.zeros_like(initial["energy_erg"]),
                "valid": jnp.asarray(True),
                "converged": jnp.asarray(False),
                "passes": jnp.asarray(0, dtype=jnp.int32),
                "iteration": jnp.asarray(0, dtype=jnp.int32),
            },
            xs=None,
            length=maximum,
        )
        escaped = jnp.where(
            final["valid"] & final["converged"], final["escaped"], jnp.nan
        )
        return apply_ebl_attenuation(
            escaped[jnp.asarray(earth_indices)]
            * jnp.asarray(flux_scale)[jnp.asarray(earth_indices)],
            jnp.asarray(transmission),
        )

    return energy_observer[earth_indices], prediction


def _earth_photon_log_sed_target(model, parameters):
    """Build the maintained model-bound natural-log Earth photon target.

    The target values are ``ln[(E F_E) / (1 erg cm^-2 s^-1)]`` on the fixed
    finite, strictly positive support of the supplied model's baseline Earth
    photon SED. Coordinates use the shared differentiable-parameter registry;
    the returned parameter axis preserves ``parameters`` order. No floor,
    clipping, or dynamic support is applied.
    """
    if not isinstance(model, OneZoneModel):
        raise TypeError("model must be a OneZoneModel")
    if isinstance(parameters, str):
        raise TypeError("parameters must be a sequence of parameter names")
    try:
        names = tuple(parameters)
    except TypeError as error:
        raise TypeError("parameters must be a sequence of parameter names") from error
    if not names:
        raise ValueError("parameters must contain at least one name")
    if len(names) > len(DIFFERENTIABLE_PARAMETER_NAMES):
        raise ValueError("parameters may contain at most eight names")

    selection = ParameterSelection(names, _physical_coordinates(model), None)
    energy, prediction = _model_prediction(model, selection)
    coordinates = selection.pack(selection.baseline_values)
    baseline_flux = np.asarray(jax.block_until_ready(prediction(coordinates)))
    support = np.flatnonzero(np.isfinite(baseline_flux) & (baseline_flux > 0.0))
    if support.size == 0:
        raise FloatingPointError(
            "baseline Earth photon SED has no finite strictly positive support"
        )

    fixed_support = jnp.asarray(support)

    def target(values):
        flux = prediction(values)[fixed_support]
        return jnp.log(flux / jnp.float64(1.0))

    energy = np.asarray(energy)[support]
    energy.setflags(write=False)
    support.setflags(write=False)
    return energy, support, coordinates, target


def fit(model, dataset, *, parameters, optimizer="lbfgsb", options=None):
    """Fit one ordered 1--8 parameter subset to Earth-frame photon SED data.

    ``model.config`` is the initial physical model. Names in ``parameters`` are
    jointly free; every other registry parameter stays fixed at its input
    physical value. The input model is never mutated. Both optimizers minimize
    the same maintained mixed detection/upper-limit NLL in the existing
    reference-relative coordinates. Photon predictions use the canonical
    broadband escaped Earth SED returned by the forward model, including its
    configured fixed EBL mapping.
    """

    started = time.perf_counter()
    selection = _prepare_fit_request(model, dataset, parameters, optimizer)
    model_energy, prediction = _model_prediction(model, selection)
    selected_energy = dataset.energy_erg[dataset.likelihood_mask]
    if np.any(selected_energy < model_energy[0]) or np.any(
        selected_energy > model_energy[-1]
    ):
        raise ValueError("selected dataset energies must lie inside the model domain")

    def nll(free_values):
        return mixed_sed_nll(model_energy, prediction(free_values), dataset)

    def details(free_values):
        def value_with_prediction(values):
            model_sed = prediction(values)
            value = mixed_sed_nll(model_energy, model_sed, dataset)
            return value, (value, model_sed)

        gradient, (value, model_sed) = jax.jacfwd(value_with_prediction, has_aux=True)(
            free_values
        )
        return value, gradient, model_sed

    compiled_details = jax.jit(details)

    def value_and_grad(free_values):
        value, gradient, _ = compiled_details(free_values)
        return value, gradient

    initial = np.asarray(selection.pack(selection.baseline_values))
    optimizer_result = fit_maximum_likelihood(
        nll,
        initial,
        bounds=selection.free_bounds,
        options=options,
        value_and_grad=value_and_grad,
        optimizer=optimizer,
    )
    final_value, _, final_model_sed = compiled_details(jnp.asarray(optimizer_result.x))
    final_value.block_until_ready()
    final_model_sed.block_until_ready()
    final_model_sed = np.asarray(final_model_sed)

    best_values = {}
    updates = {}
    for name, coordinate in zip(selection.free_names, optimizer_result.x, strict=True):
        field, value = _physical_value(model, name, coordinate)
        best_values[name] = value
        updates[field] = value
    best_config = replace(model.config, **updates)
    fixed_values = {
        name: getattr(model.config, _PARAMETER_FIELDS[name][0])
        for name in selection.fixed_names
    }
    selected_indices = np.flatnonzero(dataset.likelihood_mask)
    selected_prediction = np.asarray(
        interpolate_sed_to_observations(
            model_energy,
            final_model_sed,
            dataset.energy_erg[selected_indices],
        )
    )
    prediction_result = _freeze(
        {
            "indices": selected_indices,
            "energy_erg": dataset.energy_erg[selected_indices],
            "earth_energy_flux_erg_cm2_s": selected_prediction,
        }
    )
    residuals = _freeze(mixed_sed_diagnostics(model_energy, final_model_sed, dataset))
    backend = jax.default_backend()
    return FitResult(
        best_parameters=MappingProxyType(best_values),
        best_config=best_config,
        free_parameters=selection.free_names,
        fixed_parameters=MappingProxyType(fixed_values),
        joint_dimension=len(selection.free_names),
        nll=float(final_value),
        optimizer=optimizer_result.optimizer,
        success=bool(optimizer_result.success),
        status=int(optimizer_result.status),
        message=str(optimizer_result.message),
        iterations=optimizer_result.iterations,
        evaluations=optimizer_result.function_evaluations,
        prediction=prediction_result,
        residuals=residuals,
        uncertainties=None,
        covariance=None,
        runtime_seconds=time.perf_counter() - started,
        backend=backend,
        device=str(jax.devices(backend)[0]),
        x64=bool(jax.config.jax_enable_x64),
    )
