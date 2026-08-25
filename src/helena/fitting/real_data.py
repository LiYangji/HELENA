"""Public-SED parsing and mixed detection/upper-limit likelihoods."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr

EV_TO_ERG = 1.602176634e-12
UPPER_LIMIT_CONFIDENCE = 0.95
UPPER_LIMIT_NORMAL_QUANTILE = NormalDist().inv_cdf(UPPER_LIMIT_CONFIDENCE)

_STATISTIC_TYPES = frozenset(("gaussian", "asymmetric_gaussian", "upper_limit"))
_SCIENTIFIC_USES = frozenset(("fit", "constraint", "context", "exclude"))
_ACTIVE_USES = frozenset(("fit", "constraint"))
_BJET_HEADER = (
    "E(eV)",
    "F(ergcm-2s-1)",
    "delta_E(-)",
    "delta_E(+)",
    "delta_F(-)",
    "delta_F(+)",
    "instrument",
)


def _immutable_array(value, *, dtype):
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SEDData:
    """Earth-observed SED measurements with statistic and use labels.

    Energies are observer-frame values in erg and SED fluxes are ``E F_E`` in
    erg cm^-2 s^-1 at Earth. statistic_type describes the published measurement,
    while scientific_use is an explicit caller decision and defaults to no
    likelihood use in load_bjet_mcmc_sed.
    """

    energy_erg: np.ndarray
    sed_flux_erg_cm2_s: np.ndarray
    energy_error_lower_erg: np.ndarray
    energy_error_upper_erg: np.ndarray
    sed_flux_error_lower_erg_cm2_s: np.ndarray
    sed_flux_error_upper_erg_cm2_s: np.ndarray
    instrument: np.ndarray
    statistic_type: np.ndarray
    scientific_use: np.ndarray
    raw_energy_ev: np.ndarray | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        energy = np.asarray(self.energy_erg, dtype=np.float64)
        if energy.ndim != 1 or energy.size == 0:
            raise ValueError("SED fields must share a non-empty one-dimensional shape")
        shape = energy.shape
        numeric = {
            "energy_erg": energy,
            "sed_flux_erg_cm2_s": np.asarray(self.sed_flux_erg_cm2_s, dtype=np.float64),
            "energy_error_lower_erg": np.asarray(
                self.energy_error_lower_erg, dtype=np.float64
            ),
            "energy_error_upper_erg": np.asarray(
                self.energy_error_upper_erg, dtype=np.float64
            ),
            "sed_flux_error_lower_erg_cm2_s": np.asarray(
                self.sed_flux_error_lower_erg_cm2_s, dtype=np.float64
            ),
            "sed_flux_error_upper_erg_cm2_s": np.asarray(
                self.sed_flux_error_upper_erg_cm2_s, dtype=np.float64
            ),
        }
        if any(value.shape != shape for value in numeric.values()):
            raise ValueError("SED fields must share a non-empty one-dimensional shape")
        if any(not np.all(np.isfinite(value)) for value in numeric.values()):
            raise ValueError("SED numeric fields must be finite")
        if np.any(energy <= 0.0) or np.any(numeric["sed_flux_erg_cm2_s"] <= 0.0):
            raise ValueError("SED energies and fluxes must be positive")
        error_names = tuple(name for name in numeric if "error" in name)
        if any(np.any(numeric[name] < 0.0) for name in error_names):
            raise ValueError("SED errors must be non-negative")
        if np.any(numeric["energy_error_lower_erg"] >= energy):
            raise ValueError("lower energy errors must remain below the energy")

        text = {
            "instrument": np.asarray(self.instrument, dtype=str),
            "statistic_type": np.asarray(self.statistic_type, dtype="<U19"),
            "scientific_use": np.asarray(self.scientific_use, dtype="<U10"),
        }
        if any(value.shape != shape for value in text.values()):
            raise ValueError("SED fields must share a non-empty one-dimensional shape")
        if np.any(text["instrument"] == ""):
            raise ValueError("instrument labels must not be empty")
        unknown_statistics = set(text["statistic_type"]) - _STATISTIC_TYPES
        if unknown_statistics:
            raise ValueError(
                f"unsupported statistic types: {sorted(unknown_statistics)}"
            )
        unknown_uses = set(text["scientific_use"]) - _SCIENTIFIC_USES
        if unknown_uses:
            raise ValueError(f"unsupported scientific uses: {sorted(unknown_uses)}")

        flux = numeric["sed_flux_erg_cm2_s"]
        lower = numeric["sed_flux_error_lower_erg_cm2_s"]
        upper = numeric["sed_flux_error_upper_erg_cm2_s"]
        upper_limit = text["statistic_type"] == "upper_limit"
        gaussian = text["statistic_type"] == "gaussian"
        detection = ~upper_limit
        if np.any((lower[upper_limit] != 0.0) | (upper[upper_limit] != 0.0)):
            raise ValueError("upper limits must have zero published flux errors")
        if np.any((lower[detection] <= 0.0) | (upper[detection] <= 0.0)):
            raise ValueError("detections must have positive published flux errors")
        if np.any(lower[detection] >= flux[detection]):
            raise ValueError("lower flux errors must remain below the detection flux")
        if np.any(lower[gaussian] != upper[gaussian]):
            raise ValueError("gaussian detections require symmetric published errors")

        raw_energy = (
            energy / EV_TO_ERG
            if self.raw_energy_ev is None
            else np.asarray(self.raw_energy_ev, dtype=np.float64)
        )
        if raw_energy.shape != shape or not np.all(np.isfinite(raw_energy)):
            raise ValueError("raw_energy_ev must be finite and have the dataset shape")

        for name, value in numeric.items():
            object.__setattr__(self, name, _immutable_array(value, dtype=np.float64))
        for name, value in text.items():
            object.__setattr__(self, name, _immutable_array(value, dtype=str))
        object.__setattr__(
            self, "raw_energy_ev", _immutable_array(raw_energy, dtype=np.float64)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def likelihood_mask(self):
        """Return the fixed caller-selected fit/constraint mask."""

        return self.mask_for_scientific_use(*_ACTIVE_USES)

    @property
    def selected_count(self):
        return int(np.count_nonzero(self.likelihood_mask))

    def mask_for_scientific_use(self, *uses):
        """Return a boolean mask for one or more explicit scientific uses."""

        unknown = set(uses) - _SCIENTIFIC_USES
        if unknown:
            raise ValueError(f"unsupported scientific uses: {sorted(unknown)}")
        return np.isin(self.scientific_use, uses)

    def mask_for_statistic_type(self, *types):
        """Return a boolean mask for one or more published statistic types."""

        unknown = set(types) - _STATISTIC_TYPES
        if unknown:
            raise ValueError(f"unsupported statistic types: {sorted(unknown)}")
        return np.isin(self.statistic_type, types)

    def with_scientific_use(self, mask, use):
        """Return a copy with one explicit, fixed scientific-use selection."""

        if use not in _SCIENTIFIC_USES:
            raise ValueError(f"unsupported scientific use: {use!r}")
        mask = np.asarray(mask)
        if mask.shape != self.energy_erg.shape or mask.dtype != np.bool_:
            raise ValueError("selection mask must be boolean with the dataset shape")
        updated = np.asarray(self.scientific_use).copy()
        updated[mask] = use
        return replace(self, scientific_use=updated)

    def audit(self):
        """Return compact immutable-data counts and observer-energy coverage."""

        def counts(values):
            names, totals = np.unique(values, return_counts=True)
            return dict(zip(names.tolist(), totals.tolist(), strict=True))

        return {
            "rows": int(self.energy_erg.size),
            "instruments": counts(self.instrument),
            "statistic_types": counts(self.statistic_type),
            "scientific_uses": counts(self.scientific_use),
            "energy_min_erg": float(np.min(self.energy_erg)),
            "energy_max_erg": float(np.max(self.energy_erg)),
        }


def load_bjet_mcmc_sed(path):
    """Parse the public seven-column Bjet_MCMC ASCII SED format.

    Zero lower and upper flux errors are the upstream 95% upper-limit marker.
    All rows default to scientific_use="context"; the loader makes no
    instrument attribution or fit-mask decision.
    """

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].lstrip("!").split()) != _BJET_HEADER:
        raise ValueError("unexpected Bjet_MCMC SED header")
    rows = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != 7:
            raise ValueError(f"line {line_number} must contain seven columns")
        try:
            rows.append((*map(float, columns[:6]), columns[6]))
        except ValueError as error:
            raise ValueError(
                f"line {line_number} contains invalid numeric data"
            ) from error
    if not rows:
        raise ValueError("Bjet_MCMC SED file contains no data rows")

    values = np.asarray([row[:6] for row in rows], dtype=np.float64)
    lower_flux, upper_flux = values[:, 4], values[:, 5]
    one_zero = (lower_flux == 0.0) ^ (upper_flux == 0.0)
    if np.any(one_zero):
        raise ValueError("upper-limit rows require two zero flux errors")
    upper_limit = (lower_flux == 0.0) & (upper_flux == 0.0)
    statistic = np.full(len(rows), "asymmetric_gaussian", dtype="<U19")
    statistic[(~upper_limit) & (lower_flux == upper_flux)] = "gaussian"
    statistic[upper_limit] = "upper_limit"
    return SEDData(
        energy_erg=values[:, 0] * EV_TO_ERG,
        sed_flux_erg_cm2_s=values[:, 1],
        energy_error_lower_erg=values[:, 2] * EV_TO_ERG,
        energy_error_upper_erg=values[:, 3] * EV_TO_ERG,
        sed_flux_error_lower_erg_cm2_s=lower_flux,
        sed_flux_error_upper_erg_cm2_s=upper_flux,
        instrument=[row[6] for row in rows],
        statistic_type=statistic,
        scientific_use=np.full(len(rows), "context", dtype="<U10"),
        raw_energy_ev=values[:, 0],
        metadata={
            "source_path": str(path),
            "raw_energy_unit": "eV",
            "energy_unit": "erg",
            "sed_flux_unit": "erg cm^-2 s^-1",
            "upper_limit_convention": "95% confidence",
        },
    )


def interpolate_sed_to_observations(
    model_energy_erg, model_sed_erg_cm2_s, observation_energy_erg
):
    """Interpolate Earth-observed ``E F_E`` to observed energies.

    ``model_sed_erg_cm2_s`` is energy flux in erg cm^-2 s^-1, never source or
    blob ``E L_E`` luminosity. Any required EBL attenuation belongs upstream.
    """

    traced = any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(model_sed_erg_cm2_s)
    )
    model_energy = np.asarray(model_energy_erg, dtype=np.float64)
    observation_energy = np.asarray(observation_energy_erg, dtype=np.float64)
    if model_energy.ndim != 1 or model_energy.size < 2 or observation_energy.ndim != 1:
        raise ValueError("model and observation energies must be one-dimensional")
    if (
        not np.all(np.isfinite(model_energy))
        or np.any(model_energy <= 0.0)
        or np.any(np.diff(model_energy) <= 0.0)
    ):
        raise ValueError("model energy must be positive, finite, and increasing")
    if (
        not np.all(np.isfinite(observation_energy))
        or np.any(observation_energy <= 0.0)
        or np.any(observation_energy < model_energy[0])
        or np.any(observation_energy > model_energy[-1])
    ):
        raise ValueError("observation energy must lie inside the model energy domain")
    model_sed = jnp.asarray(model_sed_erg_cm2_s, dtype=jnp.float64)
    if model_sed.shape != model_energy.shape:
        raise ValueError("model SED must have the model-energy shape")
    if not traced:
        host_model = np.asarray(model_sed)
        if not np.all(np.isfinite(host_model)) or np.any(host_model <= 0.0):
            raise ValueError("model SED must contain positive finite values")
    return jnp.exp(
        jnp.interp(
            jnp.log(jnp.asarray(observation_energy)),
            jnp.log(jnp.asarray(model_energy)),
            jnp.log(model_sed),
        )
    )


def mixed_sed_nll_terms(model_energy_erg, model_sed_erg_cm2_s, dataset):
    """Return per-row dimensionless log-flux/upper-limit NLL contributions.

    The model and :class:`SEDData` are Earth-observed ``E F_E`` in
    erg cm^-2 s^-1; source/comoving ``E L_E`` or pre-EBL observer flux is not
    accepted by contract.

    Symmetric detections use a Gaussian in natural-log flux with the usual
    first-order sigma_log = sigma_F / F conversion. Asymmetric detections
    use exact lower/upper log scales and a continuous split-normal density.
    A published 95% upper limit is approximated by a null-centered one-sided
    Gaussian with sigma_eff = UL / Phi^-1(0.95); this is intentionally an
    approximate constraint, not the original instrument likelihood.
    """

    if not isinstance(dataset, SEDData):
        raise TypeError("dataset must be SEDData")
    selected_indices = np.flatnonzero(dataset.likelihood_mask)
    if selected_indices.size == 0:
        return jnp.zeros(dataset.energy_erg.shape, dtype=jnp.float64)
    model = interpolate_sed_to_observations(
        model_energy_erg,
        model_sed_erg_cm2_s,
        dataset.energy_erg[selected_indices],
    )
    observed = jnp.asarray(dataset.sed_flux_erg_cm2_s[selected_indices])
    lower_error = jnp.asarray(dataset.sed_flux_error_lower_erg_cm2_s[selected_indices])
    upper_error = jnp.asarray(dataset.sed_flux_error_upper_erg_cm2_s[selected_indices])
    upper_limit = jnp.asarray(dataset.statistic_type[selected_indices] == "upper_limit")
    gaussian = jnp.asarray(dataset.statistic_type[selected_indices] == "gaussian")

    safe_lower_error = jnp.where(upper_limit, 0.5 * observed, lower_error)
    safe_upper_error = jnp.where(upper_limit, 0.5 * observed, upper_error)
    symmetric_sigma = safe_lower_error / observed
    asymmetric_lower_sigma = jnp.log(observed) - jnp.log(observed - safe_lower_error)
    asymmetric_upper_sigma = jnp.log(observed + safe_upper_error) - jnp.log(observed)
    lower_sigma = jnp.where(gaussian, symmetric_sigma, asymmetric_lower_sigma)
    upper_sigma = jnp.where(gaussian, symmetric_sigma, asymmetric_upper_sigma)
    log_residual = jnp.log(model) - jnp.log(observed)
    side_sigma = jnp.where(log_residual < 0.0, lower_sigma, upper_sigma)
    detection_nll = (
        0.5 * (log_residual / side_sigma) ** 2
        + jnp.log(lower_sigma + upper_sigma)
        + 0.5 * jnp.log(jnp.pi / 2.0)
    )

    sigma_eff = observed / UPPER_LIMIT_NORMAL_QUANTILE
    limit_nll = -log_ndtr((observed - model) / sigma_eff)
    terms = jnp.where(upper_limit, limit_nll, detection_nll)
    return (
        jnp.zeros(dataset.energy_erg.shape, dtype=jnp.float64)
        .at[selected_indices]
        .set(terms)
    )


def mixed_sed_nll(model_energy_erg, model_sed_erg_cm2_s, dataset):
    """Assemble one scalar dimensionless NLL for fixed selected SED rows."""

    return jnp.sum(mixed_sed_nll_terms(model_energy_erg, model_sed_erg_cm2_s, dataset))


def mixed_sed_diagnostics(model_energy_erg, model_sed_erg_cm2_s, dataset):
    """Report detection residuals and upper-limit status without mixing them."""

    selected_indices = np.flatnonzero(dataset.likelihood_mask)
    if selected_indices.size == 0:
        return {
            "detections": {
                "indices": np.asarray([], dtype=int),
                "log_flux_residual": np.asarray([]),
                "nll_contribution": np.asarray([]),
            },
            "upper_limits": {
                "indices": np.asarray([], dtype=int),
                "constraint_status": np.asarray([], dtype=str),
                "nll_contribution": np.asarray([]),
            },
        }
    model = np.asarray(
        interpolate_sed_to_observations(
            model_energy_erg,
            model_sed_erg_cm2_s,
            dataset.energy_erg[selected_indices],
        )
    )
    terms = np.asarray(
        mixed_sed_nll_terms(model_energy_erg, model_sed_erg_cm2_s, dataset)
    )
    selected_type = dataset.statistic_type[selected_indices]
    upper_limit = selected_type == "upper_limit"
    detections = ~upper_limit
    return {
        "detections": {
            "indices": selected_indices[detections],
            "log_flux_residual": np.log(model[detections])
            - np.log(dataset.sed_flux_erg_cm2_s[selected_indices[detections]]),
            "nll_contribution": terms[selected_indices[detections]],
        },
        "upper_limits": {
            "indices": selected_indices[upper_limit],
            "constraint_status": np.where(
                model[upper_limit]
                <= dataset.sed_flux_erg_cm2_s[selected_indices[upper_limit]],
                "satisfied",
                "exceeded",
            ),
            "nll_contribution": terms[selected_indices[upper_limit]],
        },
    }
