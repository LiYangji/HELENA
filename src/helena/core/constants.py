"""Backend-neutral scalar constants used by independent physics implementations."""

import numpy as np
from astropy.constants import c, e, h, m_e, m_p, sigma_T
from scipy.special import gamma as gamma_function

SPEED_OF_LIGHT_CGS = float(c.cgs.value)
PLANCK_CONSTANT_CGS = float(h.cgs.value)
ELEMENTARY_CHARGE_ESU = float(e.esu.value)
ELECTRON_MASS_CGS = float(m_e.cgs.value)
PROTON_MASS_CGS = float(m_p.cgs.value)
ELECTRON_REST_ENERGY_CGS = float((m_e * c**2).cgs.value)
PROTON_REST_ENERGY_CGS = float((m_p * c**2).cgs.value)
THOMSON_CROSS_SECTION_CGS = float(sigma_T.cgs.value)

GEV_TO_ERG = 1.602176634e-3
MICROBARN_TO_CM2 = 1.0e-30

# Fixed PDG values already used by the NumPy reference modules.
MUON_REST_ENERGY_CGS = 0.1056583755 * GEV_TO_ERG
MUON_MASS_CGS = MUON_REST_ENERGY_CGS / SPEED_OF_LIGHT_CGS**2
MUON_REST_LIFETIME_S = 2.1969811e-6
ELECTRON_TO_MUON_MASS_RATIO = 0.51099895069 / 105.6583755

NEUTRAL_PION_REST_ENERGY_CGS = 0.1349768 * GEV_TO_ERG
CHARGED_PION_REST_ENERGY_CGS = 0.13957039 * GEV_TO_ERG
CHARGED_PION_MASS_CGS = CHARGED_PION_REST_ENERGY_CGS / SPEED_OF_LIGHT_CGS**2
CHARGED_PION_REST_LIFETIME_S = 2.6033e-8
MUON_TO_PION_MASS_RATIO = 105.6583755 / 139.57039
MUON_TO_PION_MASS_RATIO_SQUARED = MUON_TO_PION_MASS_RATIO**2
NEUTRAL_PION_TWO_PHOTON_BRANCHING_RATIO = 0.98823
CHARGED_PION_MUONIC_BRANCHING_RATIO = 0.9998770

SYNCHROTRON_LOG_COOLING_COEFFICIENT = (
    np.log(9.0 / 4.0) + 5.0 * np.log(SPEED_OF_LIGHT_CGS)
)
# Switch errors are checked against the defining integrals in the physics tests.
SYNCHROTRON_SMALL_X = 1.0e-2
SYNCHROTRON_LARGE_X = 120.0
SYNCHROTRON_SMALL_X_COEFFICIENT = (
    2.0 ** (1.0 / 3.0) * gamma_function(1.0 / 3.0) ** 2 / 5.0
)
INVERSE_COMPTON_SMALL_X_THRESHOLD = 1.0e-2
SSA_NEGATIVE_ALPHA_ROUNDOFF_FACTOR = 64.0 * np.finfo(float).eps
PHOTOMESON_MULTIPION_INELASTICITY = 0.6
