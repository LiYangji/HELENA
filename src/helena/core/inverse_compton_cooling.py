"""Isotropic inverse-Compton continuous cooling in the blob frame."""

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import spence

from helena.core.constants import (
    ELECTRON_REST_ENERGY_CGS as _MEC2_CGS,
)
from helena.core.constants import (
    INVERSE_COMPTON_SMALL_X_THRESHOLD as _SMALL_X_THRESHOLD,
)
from helena.core.constants import (
    SPEED_OF_LIGHT_CGS as _C_CGS,
)
from helena.core.constants import (
    THOMSON_CROSS_SECTION_CGS as _SIGMA_T_CGS,
)
from helena.core.contracts import real_1d as _real_1d

__all__ = ["electron_inverse_compton_loss_rate_dgamma_s"]


def _klein_nishina_loss_suppression(x):
    """Return the analytic ultrarelativistic isotropic KN loss suppression."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("x must contain only finite non-negative values")

    result = np.empty_like(x)
    small = x <= _SMALL_X_THRESHOLD
    small_x = x[small]
    result[small] = (
        1.0
        - 63.0 * small_x / 40.0
        + 441.0 * small_x**2 / 200.0
        - 101.0 * small_x**3 / 35.0
    )

    full_x = x[~small]
    with np.errstate(over="ignore", invalid="ignore"):
        g = (
            (full_x / 2.0 + 6.0 + 6.0 / full_x) * np.log1p(full_x)
            - (11.0 * full_x**3 / 12.0 + 6.0 * full_x**2 + 9.0 * full_x + 4.0)
            / (1.0 + full_x) ** 2
            - 2.0
            + 2.0 * spence(1.0 + full_x)
        )
        result[~small] = 9.0 * g / full_x**3

    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError("Klein--Nishina loss function exceeds float64 range")
    return result


def electron_inverse_compton_loss_rate_dgamma_s(
    electron_gamma,
    target_photon_energy_erg,
    dn_target_photon_denergy_cm3_erg,
):
    """Return blob-frame isotropic electron inverse-Compton cooling ``-dγ/dt``.

    Parameters
    ----------
    electron_gamma : array_like, shape (n_electron,)
        One or more finite dimensionless electron Lorentz factors greater than
        one. Input order is preserved.
    target_photon_energy_erg : array_like, shape (n_photon,)
        At least two finite, positive, strictly increasing blob-comoving photon
        energies in erg.
    dn_target_photon_denergy_cm3_erg : array_like, shape (n_photon,)
        Finite non-negative isotropic photon density ``dn_ph/dE`` in
        cm^-3 erg^-1 on the supplied energy grid.

    Returns
    -------
    numpy.ndarray
        Positive continuous loss ``b_IC = -d gamma_e/dt`` in s^-1, as a
        float64 array matching ``electron_gamma``.

    Notes
    -----
    The analytic Moderski et al. (2005), Appendix A, KN suppression assumes
    ultrarelativistic inverse-Compton upscattering. It is not a signed
    arbitrary-energy Compton heating/cooling operator; gamma > 1 validation
    alone does not establish this approximation's physical accuracy.
    Integration uses dln(E) with E^2 n(E), consistent with the radiation
    quadrature on logarithmic photon grids; no interpolation is introduced. It performs no interpolation or resampling. The target field must
    already be isotropic and transformed to the blob frame. No anisotropy,
    electron redistribution, radiation spectrum, transport, or feedback is
    included.
    """
    gamma = _real_1d("electron_gamma", electron_gamma, minimum_size=1, positive=True)
    if np.any(gamma <= 1.0):
        raise ValueError("electron_gamma must contain only values > 1")
    photon_energy = _real_1d(
        "target_photon_energy_erg",
        target_photon_energy_erg,
        minimum_size=2,
        positive=True,
    )
    if np.any(np.diff(photon_energy) <= 0.0):
        raise ValueError("target_photon_energy_erg must be strictly increasing")
    photon_density = _real_1d(
        "dn_target_photon_denergy_cm3_erg",
        dn_target_photon_denergy_cm3_erg,
        minimum_size=2,
        positive=False,
        shape=photon_energy.shape,
    )

    if not np.any(photon_density):
        return np.zeros(gamma.shape, dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        x = 4.0 * gamma[:, None] * photon_energy[None, :] / _MEC2_CGS
        gamma_squared = gamma**2
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(gamma_squared)):
        raise FloatingPointError("inverse-Compton inputs exceed float64 range")

    suppression = _klein_nishina_loss_suppression(x)
    with np.errstate(over="ignore", invalid="ignore"):
        energy_density = trapezoid(
            photon_energy[None, :] ** 2 * photon_density[None, :] * suppression,
            x=np.log(photon_energy),
            axis=1,
        )
        result = (
            4.0
            * _SIGMA_T_CGS
            * _C_CGS
            / (3.0 * _MEC2_CGS)
            * gamma_squared
            * energy_density
        )
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("inverse-Compton cooling exceeds float64 range")
    return result
