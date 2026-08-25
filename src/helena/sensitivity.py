"""Public Earth-photon log-SED sensitivities."""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from helena.fitting.public import _earth_photon_log_sed_target
from helena.parameters import DIFFERENTIABLE_PARAMETER_NAMES

__all__ = ["SensitivityResult", "jacobian", "jvp"]

_TARGET = "ln[(E F_E^Earth) / (1 erg cm^-2 s^-1)]"


def _immutable_array(value, *, dtype):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """One derivative of the fixed Earth-photon natural-log SED target."""

    energy_erg: np.ndarray
    values: np.ndarray
    parameters: tuple[str, ...]
    support_indices: np.ndarray
    direction: tuple[float, ...] | None = None

    def __post_init__(self):
        energy = _immutable_array(self.energy_erg, dtype=np.float64)
        values = _immutable_array(self.values, dtype=np.float64)
        support = _immutable_array(self.support_indices, dtype=np.int64)
        parameters = tuple(self.parameters)
        direction = None if self.direction is None else tuple(self.direction)
        expected = (
            (energy.size, len(parameters)) if direction is None else (energy.size,)
        )
        if energy.ndim != 1 or energy.size == 0 or values.shape != expected:
            raise ValueError(
                "sensitivity values must match the energy and parameter axes"
            )
        if support.shape != energy.shape:
            raise ValueError("support indices must match the energy axis")
        if (
            not np.all(np.isfinite(energy))
            or np.any(energy <= 0.0)
            or np.any(np.diff(energy) <= 0.0)
        ):
            raise ValueError("energy must be positive, finite, and increasing")
        if direction is not None and (
            len(direction) != len(parameters) or not np.all(np.isfinite(direction))
        ):
            raise ValueError("direction must be finite and match parameters")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("sensitivity values must be finite")
        object.__setattr__(self, "energy_erg", energy)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "support_indices", support)
        object.__setattr__(self, "direction", direction)

    @property
    def target(self):
        """Fixed scientific target differentiated by this result."""

        return _TARGET

    @property
    def orientation(self):
        """Meaning of the ``values`` axes."""

        return "energy" if self.direction is not None else "energy, parameter"


def _parameter_names(parameters):
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
    if any(not isinstance(name, str) for name in names):
        raise TypeError("parameter names must be strings")
    unknown = set(names) - set(DIFFERENTIABLE_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown parameter names: {', '.join(sorted(unknown))}")
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    return names


def _warn_if_large_cpu(names):
    if jax.default_backend() == "cpu" and len(names) > 2:
        warnings.warn(
            "Jacobians with more than two parameters are supported on CPU but "
            "may be slow or memory intensive; GPU execution is recommended",
            RuntimeWarning,
            stacklevel=3,
        )


def jacobian(model, *, parameters):
    """Return ``d ln(E F_E^Earth) / d theta`` in energy-by-parameter order.

    ``theta`` uses the shared reference-relative registry coordinates. The six
    logarithmic coordinates give dimensionless derivatives; ``p_e`` and
    ``p_p`` derivatives are per unit spectral-index change.
    """

    names = _parameter_names(parameters)
    _warn_if_large_cpu(names)
    energy, support, coordinates, target = _earth_photon_log_sed_target(model, names)
    values = jax.jit(jax.jacfwd(target))(coordinates)
    values.block_until_ready()
    return SensitivityResult(energy, values, names, support)


def jvp(model, *, parameters, direction):
    """Return the directional derivative of the fixed Earth-photon log SED.

    ``direction`` follows ``parameters`` order in the shared registry
    coordinates; values are ``d ln(E F_E^Earth) / d epsilon``.
    """

    names = _parameter_names(parameters)
    try:
        vector = np.asarray(direction, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "direction must be a finite vector matching parameters"
        ) from error
    if vector.shape != (len(names),) or not np.all(np.isfinite(vector)):
        raise ValueError("direction must be a finite vector matching parameters")
    _warn_if_large_cpu(names)
    energy, support, coordinates, target = _earth_photon_log_sed_target(model, names)

    def directional_derivative(values):
        return jax.jvp(target, (values,), (jnp.asarray(vector),))[1]

    values = jax.jit(directional_derivative)(coordinates)
    values.block_until_ready()
    return SensitivityResult(
        energy,
        values,
        names,
        support,
        tuple(float(value) for value in vector),
    )
