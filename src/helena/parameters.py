"""Validated differentiable parameter metadata shared by fitting and AD."""

from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "DIFFERENTIABLE_PARAMETERS",
    "DIFFERENTIABLE_PARAMETER_NAMES",
    "PARAMETERS_BY_NAME",
    "ParameterSpec",
]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Stable metadata for one dimensionless model-reference coordinate."""

    name: str
    quantity: str
    coordinate: str
    unit: str
    differentiable: bool
    validated: bool


DIFFERENTIABLE_PARAMETERS = (
    ParameterSpec(
        "log_Le",
        "electron kinetic injection luminosity",
        "natural-log reference offset: value = ln(L_e,kin / L_e,kin,ref)",
        "erg s^-1",
        True,
        True,
    ),
    ParameterSpec(
        "log_Lp",
        "proton kinetic injection luminosity",
        "natural-log reference offset: value = ln(L_p,kin / L_p,kin,ref)",
        "erg s^-1",
        True,
        True,
    ),
    ParameterSpec(
        "log_B",
        "blob-comoving magnetic-field strength",
        "natural-log reference offset: value = ln(B / B_ref)",
        "G",
        True,
        True,
    ),
    ParameterSpec(
        "log_R",
        "blob-comoving radius",
        "natural-log reference offset: value = ln(R / R_ref)",
        "cm",
        True,
        True,
    ),
    ParameterSpec(
        "p_e",
        "electron injection spectral index",
        "additive reference offset: value = p_e - p_e,ref",
        "dimensionless",
        True,
        True,
    ),
    ParameterSpec(
        "p_p",
        "proton injection spectral index",
        "additive reference offset: value = p_p - p_p,ref",
        "dimensionless",
        True,
        True,
    ),
    ParameterSpec(
        "log_gamma_e_max",
        "electron exponential-cutoff Lorentz factor",
        "natural-log reference offset: value = ln(gamma_e,max / gamma_e,max,ref)",
        "dimensionless",
        True,
        True,
    ),
    ParameterSpec(
        "log_gamma_p_max",
        "proton exponential-cutoff Lorentz factor",
        "natural-log reference offset: value = ln(gamma_p,max / gamma_p,max,ref)",
        "dimensionless",
        True,
        True,
    ),
)
DIFFERENTIABLE_PARAMETER_NAMES = tuple(
    parameter.name for parameter in DIFFERENTIABLE_PARAMETERS
)
PARAMETERS_BY_NAME = MappingProxyType(
    {parameter.name: parameter for parameter in DIFFERENTIABLE_PARAMETERS}
)
