"""Backend-neutral fixed arrays shared by NumPy and JAX physics kernels."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from helena.core.constants import GEV_TO_ERG, PROTON_REST_ENERGY_CGS


_GAMMA_GAMMA_ANGLE_NODE, GAMMA_GAMMA_ANGLE_WEIGHT = leggauss(128)
GAMMA_GAMMA_ANGLE_T = 0.5 * (_GAMMA_GAMMA_ANGLE_NODE + 1.0)
GAMMA_GAMMA_ANGLE_WEIGHT = 0.5 * GAMMA_GAMMA_ANGLE_WEIGHT

PHOTOMESON_RESONANCE_INELASTICITY = np.array([0.22, 0.39])
PHOTOMESON_DIRECT_INELASTICITY = np.array([0.13, 0.05, 0.001, 0.28, 0.22, 0.201])

# Species order: pi0, pi+, pi-.
PION_REST_ENERGIES_CGS = (
    np.array([0.1349768, 0.13957039, 0.13957039]) * GEV_TO_ERG
)
PION_TO_PROTON_MASS_RATIOS = PION_REST_ENERGIES_CGS / PROTON_REST_ENERGY_CGS

# Component order: LR, HR; T1L/M/H, T2aL/M/H, T2b; M1L/H through M7L/H.
# Table 4 has no initial-proton LR pi- component, so its unused chi is NaN.
_PHOTOMESON_DIRECT_COMPONENT_CHI = np.array(
    [0.13, 0.05, 0.001, 0.08, 0.02, 0.001, 0.2]
)
_PHOTOMESON_MULTIPION_COMPONENT_CHI = np.array(
    [0.1, 0.4, 0.15, 0.35, 0.15, 0.35, 0.07, 0.35, 0.02, 0.5, 0.007, 0.5, 0.002, 0.6]
)
PHOTOMESON_PION_COMPONENT_CHI = np.hstack(
    [
        np.array([[0.22, 0.26], [0.22, 0.25], [np.nan, 0.22]]),
        np.tile(_PHOTOMESON_DIRECT_COMPONENT_CHI, (3, 1)),
        np.tile(_PHOTOMESON_MULTIPION_COMPONENT_CHI, (3, 1)),
    ]
)
PHOTOMESON_PION_COMPONENT_MULTIPLICITY = np.hstack(
    [
        np.array([[2.0 / 3.0, 0.47], [1.0 / 3.0, 0.77], [0.0, 0.34]]),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 6.0],
                [1.0, 1.0, 1.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 3.0 / 4.0],
                [0.0, 0.0, 0.0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0, 1.0 / 12.0],
            ]
        ),
        np.array(
            [
                [0.32, 0.17, 0.42, 0.19, 0.59, 0.16, 1.38, 0.16],
                [0.34, 0.29, 0.31, 0.35, 0.57, 0.21, 1.37, 0.25],
                [0.04, 0.05, 0.07, 0.08, 0.30, 0.13, 1.11, 0.23],
            ]
        ),
        np.array(
            [
                [3.01, 0.20, 5.13, 0.27, 7.59, 0.26],
                [2.86, 0.21, 4.68, 0.29, 6.80, 0.27],
                [2.64, 0.14, 4.57, 0.12, 6.65, 0.13],
            ]
        ),
    ]
)
PHOTOMESON_MULTIPION_COMPONENT_MINIMUM_GEV = np.repeat(
    [0.5, 0.9, 1.5, 5.0, 50.0, 500.0, 5000.0], 2
)
PHOTOMESON_MULTIPION_COMPONENT_MAXIMUM_GEV = np.repeat(
    [0.9, 1.5, 5.0, 50.0, 500.0, 5000.0, np.inf], 2
)
PHOTOMESON_MULTIPION_COMPONENT_CROSS_SECTION_MICROBARN = np.repeat(
    [60.0, 85.0, 120.0, 120.0, 120.0, 120.0, 120.0], 2
)
