"""Minimal likelihood, optimization, and post-fit tools for SED models."""

import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from helena.parameters import (
    DIFFERENTIABLE_PARAMETER_NAMES as CANONICAL_PARAMETER_NAMES,
)

DEFAULT_LBFGSB_OPTIONS = {"maxiter": 100, "ftol": 1.0e-12, "gtol": 1.0e-7}
DEFAULT_MIGRAD_OPTIONS = {
    "ncall": 100,
    "iterate": 1,
    "use_simplex": False,
    "tol": 0.1,
    "strategy": 0,
}

_LOG_SED_QUANTITIES = frozenset(
    (
        "comoving_energy_luminosity",
        "observer_intrinsic_energy_flux",
        "earth_energy_flux",
    )
)


@dataclass(frozen=True)
class ParameterSelection:
    """Select named free offsets from the canonical C513 fitting vector.

    ``baseline_values`` contains all eight dimensionless offsets in
    :data:`CANONICAL_PARAMETER_NAMES` order. ``bounds_by_name`` maps names to
    finite optimizer-coordinate bounds; ``None`` preserves the maintained
    unbounded optimizer path. Packing and unpacking preserve order, and
    unpacking remains differentiable only with respect to the free values;
    fixed entries are exact baseline constants.
    """

    free_names: tuple[str, ...]
    baseline_values: np.ndarray
    bounds_by_name: Mapping | None
    _free_indices: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.free_names, str):
            raise TypeError("free_names must be a sequence of parameter names")
        free_names = tuple(self.free_names)
        unknown = set(free_names) - set(CANONICAL_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"unknown parameter names: {', '.join(sorted(unknown))}")
        if len(set(free_names)) != len(free_names):
            raise ValueError("free parameter names must be unique")
        baseline = np.array(self.baseline_values, dtype=np.float64, copy=True)
        if baseline.shape != (len(CANONICAL_PARAMETER_NAMES),):
            raise ValueError("baseline values must have canonical 8D shape")
        if not np.all(np.isfinite(baseline)):
            raise ValueError("baseline values must be finite")
        unbounded = self.bounds_by_name is None
        bounds = {}
        for name, pair in dict(self.bounds_by_name or {}).items():
            if name not in CANONICAL_PARAMETER_NAMES:
                raise ValueError(f"unknown bound name: {name}")
            values = tuple(pair)
            if (
                len(values) != 2
                or not np.all(np.isfinite(values))
                or values[0] >= values[1]
            ):
                raise ValueError(f"bounds for {name} must be finite and increasing")
            bounds[name] = (float(values[0]), float(values[1]))
        missing = [name for name in free_names if name not in bounds]
        if missing and not unbounded:
            raise ValueError(f"missing bounds for: {', '.join(missing)}")
        baseline.setflags(write=False)
        object.__setattr__(self, "free_names", free_names)
        object.__setattr__(self, "baseline_values", baseline)
        object.__setattr__(self, "bounds_by_name", MappingProxyType(bounds))
        object.__setattr__(
            self,
            "_free_indices",
            tuple(CANONICAL_PARAMETER_NAMES.index(name) for name in free_names),
        )

    @property
    def fixed_names(self):
        """Return canonical names not selected as free."""

        return tuple(
            name for name in CANONICAL_PARAMETER_NAMES if name not in self.free_names
        )

    @property
    def free_bounds(self):
        """Return bounds in the requested free-parameter order."""

        if not self.bounds_by_name:
            return None
        return tuple(self.bounds_by_name[name] for name in self.free_names)

    def pack(self, full_values):
        """Select free values from one canonical 8D vector."""

        full = jnp.asarray(full_values, dtype=jnp.float64)
        if full.shape != (len(CANONICAL_PARAMETER_NAMES),):
            raise ValueError("full values must have canonical 8D shape")
        return full[jnp.asarray(self._free_indices)]

    def unpack(self, free_values):
        """Reconstruct one canonical 8D vector from the selected free values."""

        free = jnp.asarray(free_values, dtype=jnp.float64)
        if free.shape != (len(self.free_names),):
            raise ValueError("free values must match free_names")
        return (
            jnp.asarray(self.baseline_values)
            .at[jnp.asarray(self._free_indices)]
            .set(free)
        )


def _load_minuit():
    try:
        from iminuit import Minuit
    except ImportError as error:
        raise ImportError(
            "optimizer='migrad' requires iminuit; install helena[migrad]"
        ) from error
    return Minuit


@dataclass(frozen=True)
class LogSEDDataset:
    """Positive SED detections for a Gaussian likelihood in natural-log space.

    ``energy_erg`` is the photon energy in the frame named by ``quantity``.
    ``observed_sed`` is positive finite ``E L_E`` for
    ``quantity="comoving_energy_luminosity"`` or ``E F_E`` for
    ``quantity="observer_intrinsic_energy_flux"`` before line-of-sight
    attenuation or ``quantity="earth_energy_flux"`` after propagation;
    ``sigma_log_sed`` is its dimensionless one-sigma uncertainty in natural-log
    space. ``mask`` optionally selects detections without changing array shape.

    ``quantity=None`` preserves the frozen pre-E1 synthetic-baseline call only.
    New fitting handoffs must label both the dataset and model quantity.

    Only independent, symmetric detections are supported. Upper limits,
    asymmetric errors, covariance, and systematics are outside this first API.
    """

    energy_erg: np.ndarray
    observed_sed: np.ndarray
    sigma_log_sed: np.ndarray
    mask: np.ndarray | None = None
    quantity: str | None = None

    def __post_init__(self):
        energy = np.asarray(self.energy_erg, dtype=np.float64)
        observed = np.asarray(self.observed_sed, dtype=np.float64)
        uncertainty = np.asarray(self.sigma_log_sed, dtype=np.float64)
        if (
            energy.ndim != 1
            or observed.shape != energy.shape
            or uncertainty.shape != energy.shape
        ):
            raise ValueError(
                "energy, observed SED, and uncertainty must share one-dimensional shape"
            )
        if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
            raise ValueError("energy must contain positive finite values in erg")
        if not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
            raise ValueError("observed SED must contain positive finite detections")
        if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty <= 0.0):
            raise ValueError("log-SED uncertainty must contain positive finite values")
        mask = (
            np.ones(energy.shape, dtype=bool)
            if self.mask is None
            else np.asarray(self.mask)
        )
        if mask.shape != energy.shape or mask.dtype != np.bool_:
            raise ValueError("mask must be a boolean array with the dataset shape")
        if not np.any(mask):
            raise ValueError("mask must select at least one detection")
        if self.quantity is not None and (
            not isinstance(self.quantity, str)
            or self.quantity not in _LOG_SED_QUANTITIES
        ):
            raise ValueError(
                "quantity must be 'comoving_energy_luminosity', "
                "'observer_intrinsic_energy_flux', or 'earth_energy_flux'"
            )
        for name, value in (
            ("energy_erg", energy),
            ("observed_sed", observed),
            ("sigma_log_sed", uncertainty),
            ("mask", mask),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def selected_count(self):
        """Return the number of active dimensionless likelihood residuals."""

        return int(np.count_nonzero(self.mask))


def _validate_log_sed_quantity(dataset, model_quantity):
    if not isinstance(dataset, LogSEDDataset):
        raise TypeError("dataset must be LogSEDDataset")
    if dataset.quantity is None and model_quantity is None:
        return
    if model_quantity != dataset.quantity:
        raise ValueError("model_quantity must exactly match the LogSEDDataset quantity")


def gaussian_log_likelihood_log_sed(
    model_sed,
    dataset,
    *,
    model_quantity=None,
    normalized=True,
):
    """Return the independent Gaussian log likelihood in natural-log SED space.

    ``model_sed`` must have the dataset shape. For all new calls,
    ``model_quantity`` must exactly match the dataset's explicit
    ``comoving_energy_luminosity``, ``observer_intrinsic_energy_flux``, or
    ``earth_energy_flux`` contract; this prevents shape-compatible ``E L_E``
    and ``E F_E`` or pre/post-propagation fluxes from being interchanged.
    The model is assumed positive on selected points.
    ``dataset.sigma_log_sed`` is dimensionless. When ``normalized`` is true,
    the result includes ``-0.5 sum(log(2 pi sigma^2))``; otherwise it contains
    only ``-0.5 sum(residual^2)``. The scalar output is dimensionless.
    """

    _validate_log_sed_quantity(dataset, model_quantity)
    model = jnp.asarray(model_sed, dtype=jnp.float64)
    if model.shape != dataset.observed_sed.shape:
        raise ValueError("model SED must have the dataset shape")
    mask = dataset.mask
    sigma = jnp.asarray(dataset.sigma_log_sed[mask], dtype=jnp.float64)
    residual = (
        jnp.log(model[mask])
        - jnp.log(jnp.asarray(dataset.observed_sed[mask], dtype=jnp.float64))
    ) / sigma
    value = -0.5 * jnp.sum(residual**2)
    if normalized:
        value -= 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * sigma**2))
    return value


def gaussian_nll_log_sed(
    model_sed,
    dataset,
    *,
    model_quantity=None,
    normalized=True,
):
    """Return dimensionless negative Gaussian log likelihood for an SED model.

    Inputs, assumptions, units, masking, and the normalization convention are
    identical to :func:`gaussian_log_likelihood_log_sed`.
    """

    return -gaussian_log_likelihood_log_sed(
        model_sed,
        dataset,
        model_quantity=model_quantity,
        normalized=normalized,
    )


def fit_maximum_likelihood(
    nll,
    initial_parameters,
    *,
    bounds=None,
    options=None,
    value_and_grad=None,
    checkpoint_path=None,
    checkpoint_metadata=None,
    optimizer="lbfgsb",
):
    """Minimize a scalar JAX NLL in dimensionless transformed coordinates.

    ``nll(theta)`` must return one finite dimensionless scalar for a fixed-shape
    parameter vector. ``initial_parameters`` and optional ``bounds`` are in the
    caller's dimensionless transformed coordinates. ``optimizer`` selects the
    default ``"lbfgsb"`` backend or optional ``"migrad"`` backend. Both use
    one exact JAX value-and-gradient executable and return SciPy's
    ``OptimizeResult`` with common optimizer, numerical-health, and post-fit
    fields. When ``checkpoint_path`` is supplied, the optimizer result is
    atomically saved and verified immediately after the optimizer returns,
    before any post-fit diagnostics.
    """

    initial = np.asarray(initial_parameters, dtype=np.float64)
    if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
        raise ValueError("initial parameters must be a non-empty finite vector")
    compiled = (
        jax.jit(jax.value_and_grad(nll)) if value_and_grad is None else value_and_grad
    )

    def evaluate(parameters):
        value, gradient = compiled(jnp.asarray(parameters, dtype=jnp.float64))
        value.block_until_ready()
        gradient.block_until_ready()
        value_float = float(value)
        gradient_array = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(value_float) or not np.all(np.isfinite(gradient_array)):
            raise FloatingPointError("NLL and gradient must remain finite")
        return value_float, gradient_array

    optimizer_key = str(optimizer).lower()
    if optimizer_key == "lbfgsb":
        optimizer_options = DEFAULT_LBFGSB_OPTIONS | (options or {})
        started = time.perf_counter()
        result = minimize(
            evaluate,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=optimizer_options,
        )
        optimizer_wall = time.perf_counter() - started
        result.update(
            optimizer="lbfgsb",
            optimizer_name="L-BFGS-B",
            algorithm="L-BFGS-B",
            optimizer_settings=optimizer_options,
            optimizer_limit_reached=(
                int(result.status) == 1 or "LIMIT" in str(result.message).upper()
            ),
            unique_expensive_evaluations=int(result.nfev),
            memoized_bridge_hits=0,
            analytic_gradient=True,
        )
    elif optimizer_key == "migrad":
        Minuit = _load_minuit()
        supplied_options = options or {}
        unknown = set(supplied_options) - set(DEFAULT_MIGRAD_OPTIONS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported MIGRAD options: {names}")
        migrad_options = DEFAULT_MIGRAD_OPTIONS | supplied_options
        cache = {"theta": None, "value": None, "gradient": None}
        counters = {"executions": 0, "hits": 0}

        def memoized(parameters, *, count_hit=True):
            parameters = np.asarray(parameters, dtype=np.float64)
            if cache["theta"] is not None and np.array_equal(
                parameters, cache["theta"]
            ):
                if count_hit:
                    counters["hits"] += 1
            else:
                cache["value"], cache["gradient"] = evaluate(parameters)
                cache["theta"] = parameters.copy()
                counters["executions"] += 1
            return cache["value"], cache["gradient"]

        def minuit_fcn(*parameters):
            return memoized(parameters)[0]

        def minuit_gradient(*parameters):
            return memoized(parameters)[1]

        names = tuple(f"parameter_{index}" for index in range(initial.size))
        minuit = Minuit(
            minuit_fcn,
            *initial,
            grad=minuit_gradient,
            name=names,
        )
        minuit.errordef = Minuit.LIKELIHOOD
        minuit.tol = migrad_options["tol"]
        minuit.strategy = migrad_options["strategy"]
        minuit.throw_nan = True
        if bounds is not None:
            minuit.limits = bounds
        optimizer_options = {
            "errordef": float(minuit.errordef),
            "errordef_name": "Minuit.LIKELIHOOD",
            "tol": float(minuit.tol),
            "strategy": minuit.strategy.strategy,
            "ncall": int(migrad_options["ncall"]),
            "iterate": int(migrad_options["iterate"]),
            "use_simplex": bool(migrad_options["use_simplex"]),
            "throw_nan": bool(minuit.throw_nan),
            "analytic_gradient": True,
            "initial_errors": [float(error) for error in minuit.errors],
            "has_bounds": bounds is not None,
        }
        started = time.perf_counter()
        minuit.migrad(
            ncall=optimizer_options["ncall"],
            iterate=optimizer_options["iterate"],
            use_simplex=optimizer_options["use_simplex"],
        )
        optimizer_wall = time.perf_counter() - started
        final_theta = np.asarray(tuple(minuit.values), dtype=np.float64)
        final_value, final_gradient = memoized(final_theta, count_hit=False)
        fmin = minuit.fmin
        success = bool(
            minuit.valid
            and fmin.is_valid
            and not fmin.is_above_max_edm
            and not fmin.has_reached_call_limit
        )
        if success:
            message = "MIGRAD valid minimum with accepted EDM"
        elif fmin.has_reached_call_limit:
            message = "MIGRAD call limit reached"
        elif fmin.is_above_max_edm:
            message = "MIGRAD EDM above goal"
        else:
            message = "MIGRAD did not return a valid minimum"
        result = OptimizeResult(
            x=final_theta,
            fun=final_value,
            jac=final_gradient,
            success=success,
            status=0 if success else 1,
            message=message,
            nit=None,
            nfev=int(minuit.nfcn),
            njev=int(minuit.ngrad),
        )
        result.update(
            optimizer="migrad",
            optimizer_name="MIGRAD",
            algorithm=str(fmin.algorithm),
            optimizer_settings=optimizer_options,
            optimizer_limit_reached=bool(fmin.has_reached_call_limit),
            m_valid=bool(minuit.valid),
            fmin_is_valid=bool(fmin.is_valid),
            edm=float(fmin.edm),
            edm_goal=float(fmin.edm_goal),
            is_above_max_edm=bool(fmin.is_above_max_edm),
            has_reached_call_limit=bool(fmin.has_reached_call_limit),
            nfcn=int(minuit.nfcn),
            ngrad=int(minuit.ngrad),
            unique_expensive_evaluations=counters["executions"],
            memoized_bridge_hits=counters["hits"],
            analytic_gradient=True,
        )
    else:
        raise ValueError("optimizer must be 'lbfgsb' or 'migrad'")

    gradient = np.asarray(result.jac, dtype=np.float64)
    result.update(
        best_fit_parameters=np.asarray(result.x, dtype=np.float64),
        final_nll=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        iterations=(int(result.nit) if result.nit is not None else None),
        function_evaluations=int(result.nfev),
        gradient_evaluations=(
            int(result.njev) if getattr(result, "njev", None) is not None else None
        ),
        final_gradient=gradient,
        gradient_norm=float(np.linalg.norm(gradient)),
        optimizer_converged=bool(result.success),
        numerically_finite=bool(
            np.isfinite(result.fun)
            and np.all(np.isfinite(result.x))
            and np.all(np.isfinite(gradient))
        ),
        fit_quality_available=False,
        residual_diagnostics=None,
        identifiability_status="not_evaluated",
        identifiability_diagnostics=None,
        warnings=[],
        timing={"optimizer_wall_seconds": optimizer_wall},
    )
    if checkpoint_path is not None:
        checkpoint = dict(checkpoint_metadata or {})
        checkpoint.update(
            theta=result.best_fit_parameters,
            nll=result.final_nll,
            gradient=result.final_gradient,
            gradient_norm=result.gradient_norm,
            optimizer_name=result.optimizer_name,
            optimizer=result.optimizer,
            algorithm=result.algorithm,
            optimizer_success=result.optimizer_success,
            optimizer_message=result.optimizer_message,
            optimizer_status=int(result.status),
            nit=result.iterations,
            nfev=result.function_evaluations,
            njev=result.gradient_evaluations,
            optimizer_settings=result.optimizer_settings,
            optimizer_wall_seconds=optimizer_wall,
            unique_expensive_evaluations=result.unique_expensive_evaluations,
            memoized_bridge_hits=result.memoized_bridge_hits,
            analytic_gradient=result.analytic_gradient,
            timestamp=datetime.now(UTC).isoformat(),
        )
        for name in (
            "m_valid",
            "fmin_is_valid",
            "edm",
            "edm_goal",
            "is_above_max_edm",
            "has_reached_call_limit",
            "nfcn",
            "ngrad",
        ):
            if name in result:
                checkpoint[name] = result[name]
        result.checkpoint = write_json_atomic(checkpoint_path, checkpoint)
        result.checkpoint_path = str(Path(checkpoint_path))
    return result


def _local_identifiability_diagnostics(weighted_jacobian, parameter_names=None):
    """Return the maintained local Gauss-Newton identifiability calculation."""

    jacobian = np.asarray(weighted_jacobian, dtype=np.float64)
    if not np.all(np.isfinite(jacobian)):
        return (
            "non_finite",
            None,
            False,
            ["weighted Jacobian contains non-finite values"],
        )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = (
        max(jacobian.shape) * np.finfo(np.float64).eps * singular_values[0]
        if singular_values.size
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    nparameters = jacobian.shape[1]
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == nparameters and singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    gauss_newton = jacobian.T @ jacobian
    covariance = np.linalg.pinv(
        gauss_newton,
        rcond=max(jacobian.shape) * np.finfo(np.float64).eps,
        hermitian=True,
    )
    scale = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(scale, scale)
    correlation = np.full(covariance.shape, np.nan)
    np.divide(covariance, denominator, out=correlation, where=denominator > 0.0)
    names = (
        list(parameter_names)
        if parameter_names is not None
        else [f"parameter_{index}" for index in range(nparameters)]
    )
    if len(names) != nparameters:
        raise ValueError("parameter names must match the Jacobian columns")
    pairs = sorted(
        (
            {
                "parameters": [names[left], names[right]],
                "correlation": float(correlation[left, right]),
            }
            for left in range(nparameters)
            for right in range(left + 1, nparameters)
            if np.isfinite(correlation[left, right])
        ),
        key=lambda pair: abs(pair["correlation"]),
        reverse=True,
    )
    diagnostics = {
        "covariance_label": (
            "local Gauss-Newton diagnostic; not profile-likelihood uncertainty"
        ),
        "jacobian_rank": rank,
        "singular_values": singular_values,
        "condition_number": condition,
        "gauss_newton_matrix": gauss_newton,
        "local_gauss_newton_covariance": covariance,
        "normalized_parameter_correlation_matrix": correlation,
        "strongest_absolute_parameter_correlations": pairs[:5],
    }
    if rank < nparameters:
        return (
            "rank_deficient",
            diagnostics,
            True,
            [f"weighted Jacobian rank deficient: {rank}/{nparameters}"],
        )
    if not np.isfinite(condition):
        return (
            "non_finite",
            diagnostics,
            False,
            ["Jacobian condition number is non-finite"],
        )
    if condition > 1.0e12:
        return (
            "extremely_ill_conditioned",
            diagnostics,
            True,
            [f"weighted Jacobian is extremely ill-conditioned: {condition:.3e}"],
        )
    return "locally_identifiable", diagnostics, True, []


def add_postfit_diagnostics(
    result,
    dataset,
    model_sed,
    *,
    model_quantity=None,
    weighted_jacobian=None,
    parameter_names=None,
    timing=None,
):
    """Add lightweight residual and local identifiability diagnostics in place.

    ``weighted_jacobian`` is the Jacobian of selected log-SED model values
    divided by ``sigma_log_sed``. Its covariance is only a local Gauss-Newton
    diagnostic, not a final profile-likelihood uncertainty.
    """

    if not isinstance(dataset, LogSEDDataset):
        raise TypeError(
            "add_postfit_diagnostics supports Gaussian-only LogSEDDataset inputs"
        )
    _validate_log_sed_quantity(dataset, model_quantity)
    warnings = list(result.get("warnings", []))
    model = np.asarray(model_sed, dtype=np.float64)
    if model.shape != dataset.observed_sed.shape:
        raise ValueError("model SED must have the dataset shape")
    selected = dataset.mask
    valid_model = np.all(np.isfinite(model[selected])) and np.all(model[selected] > 0.0)
    if valid_model:
        residual = (
            np.log(model[selected]) - np.log(dataset.observed_sed[selected])
        ) / dataset.sigma_log_sed[selected]
        chi_square = float(residual @ residual)
        ndata = dataset.selected_count
        nfree = int(np.asarray(result.x).size)
        dof = ndata - nfree
        result.residual_diagnostics = {
            "chi_square_residual": chi_square,
            "number_of_data_points": ndata,
            "number_of_free_parameters": nfree,
            "degrees_of_freedom": dof,
            "reduced_chi_square": chi_square / dof if dof > 0 else None,
            "normalized_residual_rms": float(np.sqrt(np.mean(residual**2))),
            "normalized_residual_median_absolute": float(np.median(np.abs(residual))),
            "normalized_residual_maximum_absolute": float(np.max(np.abs(residual))),
        }
        result.fit_quality_available = True
        if dof <= 0:
            warnings.append("reduced chi-square is undefined because dof <= 0")
    else:
        result.residual_diagnostics = None
        result.fit_quality_available = False
        result.numerically_finite = False
        warnings.append("selected model SED values are not positive and finite")

    if weighted_jacobian is not None:
        jacobian = np.asarray(weighted_jacobian, dtype=np.float64)
        expected_shape = (dataset.selected_count, np.asarray(result.x).size)
        if jacobian.shape != expected_shape:
            raise ValueError(f"weighted Jacobian must have shape {expected_shape}")
        status, diagnostics, finite, diagnostic_warnings = (
            _local_identifiability_diagnostics(jacobian, parameter_names)
        )
        result.identifiability_status = status
        result.identifiability_diagnostics = diagnostics
        if not finite:
            result.numerically_finite = False
        warnings.extend(diagnostic_warnings)
    result.warnings = warnings
    if timing:
        result.timing.update(timing)
    return result


def json_safe(value):
    """Return a JSON-safe value and paths where non-finite values became null."""

    warnings = []

    def convert(item, path):
        if isinstance(item, Mapping):
            return {
                str(key): convert(child, f"{path}.{key}") for key, child in item.items()
            }
        if isinstance(item, (np.ndarray, jax.Array)):
            return convert(np.asarray(item).tolist(), path)
        if isinstance(item, np.generic):
            return convert(item.item(), path)
        if isinstance(item, (list, tuple)):
            return [
                convert(child, f"{path}[{index}]") for index, child in enumerate(item)
            ]
        if isinstance(item, Path):
            return str(item)
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not np.isfinite(item):
                warnings.append(path)
                return None
            return item
        raise TypeError(f"unsupported JSON value at {path}: {type(item).__name__}")

    return convert(value, "$"), warnings


def write_json_atomic(path, value):
    """Atomically write strict JSON and reopen it to verify parsing."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload, warnings = json_safe(value)
    if not isinstance(payload, dict):
        raise TypeError("top-level JSON value must be a mapping")
    payload["serialization_warnings"] = list(
        dict.fromkeys(payload.get("serialization_warnings", []) + warnings)
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, allow_nan=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        with destination.open(encoding="utf-8") as saved:
            return json.load(saved)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
