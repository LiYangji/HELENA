"""Explicit migration errors for the retired independent internal photon field."""

__all__ = ["gamma_gamma_internal_photon_field_chain_jax"]


def gamma_gamma_internal_photon_field_chain_jax(*args, **kwargs):
    """Reject independent gamma/target populations; use the unified EM state.

    A produced gamma SED and a separate target density do not identify a unique
    internal photon population. They cannot be silently converted to total n.
    Source-independent sphere transfer primitives remain available in radiation.
    """
    raise ValueError(
        "Independent internal gamma-gamma fields are retired; use the unified EM photon state"
    )


def _gamma_gamma_internal_photon_field_with_absorption_coefficient_jax(*args, **kwargs):
    return gamma_gamma_internal_photon_field_chain_jax(*args, **kwargs)
