"""Post-source gamma-ray and neutrino propagation to Earth."""

from collections.abc import Mapping
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from helena.core.observer import (
    _real_1d,
    _real_scalar,
    sed_luminosity_to_energy_flux,
    transform_comoving_sed,
)

__all__ = [
    "DEFAULT_PMNS",
    "DOMINGUEZ_2011_EBL",
    "NUFIT_6_0_NORMAL_ORDERING",
    "apply_ebl_attenuation",
    "prepare_ebl_transmission",
    "propagate_neutrino_flavors",
    "with_observer_propagation",
]

DOMINGUEZ_2011_EBL = {
    "model": "dominguez2011",
    "ebltable_model": "dominguez",
    "reference": "Dominguez et al., MNRAS 410 (2011) 2556",
    "doi": "10.1111/j.1365-2966.2010.17631.x",
}

NUFIT_6_0_NORMAL_ORDERING = {
    "release": "NuFIT 6.0 (2024)",
    "ordering": "normal",
    "data_variant": "IC24 with SK atmospheric data",
    "sin2_theta12": 0.308,
    "sin2_theta23": 0.470,
    "sin2_theta13": 0.02215,
    "delta_cp_degrees": 212.0,
    "source": "https://www.nu-fit.org/sites/default/files/v60.tbl-parameters.pdf",
}

_ERG_PER_TEV = 1.602176634
_SOURCE_FLAVORS = ("nu_e", "nu_mu", "nu_tau")
_SOURCE_ANTIFLAVORS = ("anti_nu_e", "anti_nu_mu", "anti_nu_tau")


def _averaged_pmns_probability():
    values = NUFIT_6_0_NORMAL_ORDERING
    s12, s23, s13 = np.sqrt(
        [values["sin2_theta12"], values["sin2_theta23"], values["sin2_theta13"]]
    )
    c12, c23, c13 = np.sqrt(1.0 - np.asarray([s12**2, s23**2, s13**2]))
    phase = np.exp(1j * np.deg2rad(values["delta_cp_degrees"]))
    matrix = np.asarray(
        [
            [c12 * c13, s12 * c13, s13 / phase],
            [
                -s12 * c23 - c12 * s23 * s13 * phase,
                c12 * c23 - s12 * s23 * s13 * phase,
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * phase,
                -c12 * s23 - s12 * c23 * s13 * phase,
                c23 * c13,
            ],
        ]
    )
    probability = np.abs(matrix) ** 2 @ (np.abs(matrix) ** 2).T
    probability.setflags(write=False)
    return probability


DEFAULT_PMNS = _averaged_pmns_probability()


@lru_cache(maxsize=1)
def _dominguez_optical_depth():
    try:
        from ebltable.tau_from_model import OptDepth
    except ImportError as error:
        raise ImportError(
            "Dominguez-2011 EBL propagation requires the optional 'ebl' extra: "
            "python -m pip install 'helena[ebl]'"
        ) from error
    return OptDepth.readmodel(model=DOMINGUEZ_2011_EBL["ebltable_model"])


def prepare_ebl_transmission(
    observer_energy_erg,
    redshift,
    *,
    model="dominguez2011",
):
    """Prepare fixed ``exp(-tau)`` values on an observer-frame energy grid.

    The Dominguez et al. (2011) table is evaluated by ``ebltable`` on the host,
    outside JAX tracing. Its closed energy/redshift domain is enforced before
    interpolation; no extrapolation or clipping to a table edge is performed.
    """
    if model != DOMINGUEZ_2011_EBL["model"]:
        raise ValueError("the only supported EBL model is 'dominguez2011'")
    observer_energy_erg = _real_1d(
        "observer_energy_erg", observer_energy_erg, positive=True
    )
    redshift = _real_scalar("redshift", redshift, positive=False)
    table = _dominguez_optical_depth()
    energy_tev = observer_energy_erg / _ERG_PER_TEV
    energy_domain_tev = np.power(10.0, [table.x[0], table.x[-1]]) / 1.0e3
    redshift_domain = np.asarray([table.y[0], table.y[-1]])
    if np.any(energy_tev < energy_domain_tev[0]) or np.any(
        energy_tev > energy_domain_tev[1]
    ):
        raise ValueError(
            "observer_energy_erg is outside the Dominguez-2011 ebltable domain "
            f"[{energy_domain_tev[0]:g}, {energy_domain_tev[1]:g}] TeV"
        )
    if redshift < redshift_domain[0] or redshift > redshift_domain[1]:
        raise ValueError(
            "redshift is outside the Dominguez-2011 ebltable domain "
            f"[{redshift_domain[0]:g}, {redshift_domain[1]:g}]"
        )
    if redshift == 0.0:
        return np.ones_like(observer_energy_erg)

    optical_depth = np.atleast_1d(
        np.asarray(table.opt_depth(redshift, energy_tev), dtype=np.float64)
    )
    if optical_depth.shape != observer_energy_erg.shape or not np.all(
        np.isfinite(optical_depth)
    ):
        raise FloatingPointError("EBL interpolation returned invalid optical depths")
    if np.any(optical_depth < -1.0e-12):
        raise FloatingPointError("EBL interpolation returned negative optical depth")
    transmission = np.exp(-np.maximum(optical_depth, 0.0))
    if not np.all(np.isfinite(transmission)) or np.any(
        (transmission < 0.0) | (transmission > 1.0)
    ):
        raise FloatingPointError("EBL interpolation returned invalid transmission")
    return transmission


def _prepare_ebl_supported_transmission(observer_energy_erg, redshift):
    """Return the non-extrapolated Dominguez support and transmission.

    Energies below the table are identity. Energies above it are omitted.
    """
    energy = _real_1d("observer_energy_erg", observer_energy_erg, positive=True)
    redshift = _real_scalar("redshift", redshift, positive=False)
    table = _dominguez_optical_depth()
    domain_tev = np.power(10.0, [table.x[0], table.x[-1]]) / 1.0e3
    redshift_domain = np.asarray([table.y[0], table.y[-1]])
    if redshift < redshift_domain[0] or redshift > redshift_domain[1]:
        raise ValueError(
            "redshift is outside the Dominguez-2011 ebltable domain "
            f"[{redshift_domain[0]:g}, {redshift_domain[1]:g}]"
        )
    energy_tev = energy / _ERG_PER_TEV
    indices = np.flatnonzero(energy_tev <= domain_tev[1])
    if indices.size == 0:
        raise ValueError("observer photon grid has no Dominguez-2011 support")
    selected = energy[indices]
    transmission = np.ones_like(selected)
    inside = selected / _ERG_PER_TEV >= domain_tev[0]
    if np.any(inside):
        transmission[inside] = prepare_ebl_transmission(
            selected[inside], redshift
        )
    return indices, transmission


def _jax_spectrum(name, values, *, shape=None):
    traced = any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(values)
    )
    if not traced:
        array = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be real-valued")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    array = jnp.asarray(values, dtype=jnp.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or (shape is not None and array.shape != shape)
    ):
        requirement = "" if shape is None else f" and have shape {shape}"
        raise ValueError(
            f"{name} must be a non-empty one-dimensional array{requirement}"
        )
    return array


def apply_ebl_attenuation(intrinsic_observer_spectrum, transmission):
    """Multiply an observer-frame intrinsic spectrum by fixed EBL transmission."""
    intrinsic = _jax_spectrum(
        "intrinsic_observer_spectrum", intrinsic_observer_spectrum
    )
    transmission = _jax_spectrum("transmission", transmission, shape=intrinsic.shape)
    if not any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(transmission)
    ) and np.any(np.asarray(transmission) > 1.0):
        raise ValueError("transmission must not exceed one")
    return intrinsic * jax.lax.stop_gradient(transmission)


def propagate_neutrino_flavors(source_flavors, *, mixing=DEFAULT_PMNS):
    """Apply averaged three-flavor vacuum mixing to nu and anti-nu spectra.

    ``source_flavors`` maps ``nu_e``, ``nu_mu``, ``anti_nu_e``, and
    ``anti_nu_mu`` to spectra on one energy grid. Missing ``nu_tau`` and
    ``anti_nu_tau`` entries are exact zero source injection. The same averaged
    probability matrix is applied separately to neutrinos and antineutrinos.
    """
    if not isinstance(source_flavors, Mapping):
        raise TypeError("source_flavors must be a mapping")
    missing = {
        "nu_e",
        "nu_mu",
        "anti_nu_e",
        "anti_nu_mu",
    } - source_flavors.keys()
    if missing:
        raise ValueError(f"source_flavors is missing {sorted(missing)!r}")

    nu_e = _jax_spectrum("nu_e", source_flavors["nu_e"])
    shape = nu_e.shape
    source = {
        "nu_e": nu_e,
        "nu_mu": _jax_spectrum("nu_mu", source_flavors["nu_mu"], shape=shape),
        "nu_tau": _jax_spectrum(
            "nu_tau", source_flavors.get("nu_tau", jnp.zeros_like(nu_e)), shape=shape
        ),
        "anti_nu_e": _jax_spectrum(
            "anti_nu_e", source_flavors["anti_nu_e"], shape=shape
        ),
        "anti_nu_mu": _jax_spectrum(
            "anti_nu_mu", source_flavors["anti_nu_mu"], shape=shape
        ),
        "anti_nu_tau": _jax_spectrum(
            "anti_nu_tau",
            source_flavors.get("anti_nu_tau", jnp.zeros_like(nu_e)),
            shape=shape,
        ),
    }
    probability = jnp.asarray(mixing, dtype=jnp.float64)
    if probability.shape != (3, 3):
        raise ValueError("mixing must have shape (3, 3)")
    if not any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(probability)
    ):
        probability_array = np.asarray(probability)
        if (
            not np.all(np.isfinite(probability_array))
            or np.any(probability_array < 0.0)
            or not np.allclose(probability_array.sum(axis=0), 1.0, atol=1.0e-12)
            or not np.allclose(probability_array.sum(axis=1), 1.0, atol=1.0e-12)
        ):
            raise ValueError("mixing must be a finite non-negative stochastic matrix")
    probability = jax.lax.stop_gradient(probability)
    neutrino = probability.T @ jnp.stack([source[name] for name in _SOURCE_FLAVORS])
    antineutrino = probability.T @ jnp.stack(
        [source[name] for name in _SOURCE_ANTIFLAVORS]
    )
    result = {
        name: values for name, values in zip(_SOURCE_FLAVORS, neutrino, strict=True)
    }
    result.update(
        {
            name: values
            for name, values in zip(_SOURCE_ANTIFLAVORS, antineutrino, strict=True)
        }
    )
    result["all_flavor"] = neutrino.sum(axis=0) + antineutrino.sum(axis=0)
    return result


def with_observer_propagation(
    full_forward_result,
    production_configuration,
    *,
    doppler_factor,
    redshift,
    luminosity_distance_cm,
    ebl_transmission,
    gamma_indices=None,
):
    """Return a shallow copy of a blob full-forward result with Earth outputs.

    ``gamma_indices`` selects the observer-energy nodes on which the caller
    prepared ``ebl_transmission``. This makes partial table-supported grids
    explicit and never invents EBL values outside the reference table.

    The source gamma input is blob-comoving ``E L_E`` in erg s^-1. Gamma
    outputs explicitly distinguish observer-frame intrinsic ``E F_E`` from
    Earth ``E F_E`` after EBL attenuation. Neutrino source injections are
    blob-comoving ``dn/(dE dt)``; observer unmixed and Earth mixed outputs are
    ``E F_E`` in erg cm^-2 s^-1. No host/AGN-frame target transformation is
    performed here.

    The four legacy gamma keys ``source_indices``, ``energy_erg``,
    ``intrinsic_energy_flux_erg_cm2_s``, and
    ``earth_energy_flux_erg_cm2_s`` remain thin aliases only for the frozen
    science-baseline runner. New callers must use the frame-explicit keys.
    """
    if not isinstance(full_forward_result, Mapping) or not full_forward_result.get(
        "iterations"
    ):
        raise ValueError("full_forward_result must contain at least one iteration")
    if "observer_propagation" in full_forward_result:
        raise ValueError("full_forward_result already contains observer propagation")
    final = full_forward_result["iterations"][-1]
    photon_energy = _real_1d(
        "gamma_ray_energy_erg",
        production_configuration["gamma_ray_energy_erg"],
        positive=True,
    )
    if gamma_indices is None:
        gamma_indices = np.arange(photon_energy.size)
    else:
        gamma_indices = np.asarray(gamma_indices)
        if (
            gamma_indices.ndim != 1
            or gamma_indices.size == 0
            or gamma_indices.dtype.kind not in "iu"
            or np.any(gamma_indices < 0)
            or np.any(gamma_indices >= photon_energy.size)
            or np.unique(gamma_indices).size != gamma_indices.size
        ):
            raise ValueError("gamma_indices must contain unique in-range integers")
    escaped = _real_1d(
        "escaped_gamma_sed_erg_s",
        final["inner_electromagnetic_cascade"]["escaped_gamma_sed_erg_s"],
        positive=False,
        shape=photon_energy.shape,
    )[gamma_indices]
    observer_gamma_energy, intrinsic_gamma_luminosity = transform_comoving_sed(
        photon_energy[gamma_indices],
        escaped,
        doppler_factor=doppler_factor,
        redshift=redshift,
    )
    intrinsic_gamma_flux = sed_luminosity_to_energy_flux(
        intrinsic_gamma_luminosity,
        luminosity_distance_cm=luminosity_distance_cm,
    )
    transmission = _real_1d(
        "ebl_transmission",
        ebl_transmission,
        positive=False,
        shape=intrinsic_gamma_flux.shape,
    )
    if np.any(transmission > 1.0):
        raise ValueError("ebl_transmission must not exceed one")

    neutrino_energy = _real_1d(
        "neutrino_energy_erg",
        production_configuration["neutrino_energy_erg"],
        positive=True,
    )
    radius_cm = _real_scalar(
        "radius_cm", production_configuration["radius_cm"], positive=True
    )
    volume_cm3 = 4.0 * np.pi * radius_cm**3 / 3.0
    chain = final["photomeson_chain"]
    source_injection = {
        "nu_e": chain["muon_decay_nu_e"],
        "anti_nu_e": chain["muon_decay_anti_nu_e"],
        "nu_mu": chain["total_nu_mu"],
        "anti_nu_mu": chain["total_anti_nu_mu"],
    }
    source_flux = {}
    validated_source_injection = {}
    observer_neutrino_energy = None
    for name in (*_SOURCE_FLAVORS, *_SOURCE_ANTIFLAVORS):
        injection = _real_1d(
            name,
            source_injection.get(name, np.zeros_like(neutrino_energy)),
            positive=False,
            shape=neutrino_energy.shape,
        )
        validated_source_injection[name] = injection
        energy, luminosity = transform_comoving_sed(
            neutrino_energy,
            volume_cm3 * neutrino_energy**2 * injection,
            doppler_factor=doppler_factor,
            redshift=redshift,
        )
        if observer_neutrino_energy is None:
            observer_neutrino_energy = energy
        source_flux[name] = sed_luminosity_to_energy_flux(
            luminosity, luminosity_distance_cm=luminosity_distance_cm
        )
    earth_flux = {
        name: np.asarray(value)
        for name, value in propagate_neutrino_flavors(source_flux).items()
    }
    source_flux["all_flavor"] = sum(source_flux.values())
    earth_gamma_flux = np.asarray(
        apply_ebl_attenuation(intrinsic_gamma_flux, transmission)
    )

    result = dict(full_forward_result)
    result["observer_propagation"] = {
        "doppler_factor": float(doppler_factor),
        "redshift": float(redshift),
        "luminosity_distance_cm": float(luminosity_distance_cm),
        "gamma": {
            "comoving_source_indices": gamma_indices.copy(),
            "observer_energy_erg": observer_gamma_energy,
            "observer_intrinsic_energy_flux_erg_cm2_s": intrinsic_gamma_flux,
            "earth_ebl_attenuated_energy_flux_erg_cm2_s": earth_gamma_flux,
            # Compatibility aliases required by the immutable science baseline.
            "source_indices": gamma_indices.copy(),
            "energy_erg": observer_gamma_energy,
            "intrinsic_energy_flux_erg_cm2_s": intrinsic_gamma_flux,
            "ebl_transmission": transmission.copy(),
            "earth_energy_flux_erg_cm2_s": earth_gamma_flux,
        },
        "neutrino": {
            "source_energy_blob_erg": neutrino_energy,
            "source_injection_blob_denergy_cm3_s": validated_source_injection,
            "observer_energy_erg": observer_neutrino_energy,
            "observer_unmixed_flavor_energy_flux_erg_cm2_s": source_flux,
            "earth_mixed_flavor_energy_flux_erg_cm2_s": earth_flux,
        },
    }
    return result
