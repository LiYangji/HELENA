# Units and Conventions

This document records the unit system and notation conventions used in HELENA.

## Unit System

HELENA uses cgs units internally unless otherwise specified.

Common units:

| Quantity | Unit |
|---|---|
| Energy | erg |
| Frequency | Hz |
| Magnetic field | Gauss |
| Length | cm |
| Time | s |
| Luminosity | erg s^-1 |
| Flux | erg cm^-2 s^-1 |
| Particle number density | cm^-3 |
| Differential particle density | cm^-3 erg^-1 or cm^-3 per Lorentz factor |
| Photon number density | cm^-3 |
| Differential photon density | cm^-3 erg^-1 |
| Cross section | cm^2 |
| Lorentz factor | dimensionless |

eV, keV, MeV, and GeV may be used for input/output convenience and plotting labels, but they should not be the main internal unit system.

## Particle Variables

Particle distributions should preferably use Lorentz factor grids:

- gamma_e for electrons;
- gamma_p for protons.

If particle energy is needed, it should be converted internally using cgs units:

```text
E_e = gamma_e m_e c^2
E_p = gamma_p m_p c^2
```

## Synchrotron Self-Absorption

The SSA implementation uses these comoving quantities:

- ``photon_energy_erg = h nu`` in erg;
- ``gamma`` as the dimensionless particle Lorentz factor;
- ``dn_dgamma_cm3 = dn/dgamma`` in cm^-3;
- ``alpha_nu`` as a signed absorption coefficient in cm^-1;
- ``radius_cm = R`` as the homogeneous sphere radius in cm;
- ``tau_d = 2 alpha_nu R`` as the center-to-center diameter optical depth;
- emergent ``E L_E = nu L_nu`` in erg s^-1.

The Dermer homogeneous-sphere attenuation is not the slab factor. Clearly
negative absorption is returned by the coefficient function but is not
supported by the transfer function. The observer-frame transformation is
applied to the emergent self-absorbed SED.

## Isotropic Inverse Compton and SSC

The inverse-Compton implementation uses comoving photon energies ``E`` and
``E_s`` in erg, ``dn_e/dgamma`` in cm^-3, and the angle-averaged isotropic seed
field ``dn_ph/dE`` in cm^-3 erg^-1. The homogeneous-sphere radius is in cm, and
the returned comoving ``E_s L_Es`` is in erg s^-1.

For SSC, the seed helper accepts the optically thin comoving ``E L_E`` in
erg s^-1 and non-negative ``alpha_nu`` in cm^-1. Its volume-and-direction
average uses ``tau_d = 2 alpha_nu R`` and has the exact thin limit
``dn_syn/dE = 9 S_thin / (16 pi R^2 c E^2)``. Doppler-boosted or otherwise
observer-transformed SEDs are not valid seed-helper inputs.

An already blob-comoving angle-averaged external field uses the low-level IC
interface directly. For an isotropic host-frame field, the external-field
helper accepts host ``E`` in erg and angle-integrated ``dn_ph/dE`` in
cm^-3 erg^-1, plus a caller-supplied comoving ``E'`` grid and bulk Lorentz
factor ``Gamma >= 1``. It returns angle-averaged comoving ``dn'_ph/dE'`` in the
same units using
``n'_E'(E') = (1/2) integral n_E(E'/[Gamma(1-beta_Gamma mu)]) dmu``.
``Gamma`` is not the observer Doppler factor ``delta``.

## Isotropic Gamma-Gamma Absorption

Gamma-ray energy ``E_gamma``, target energy ``E``, and the angle-integrated
isotropic target ``dn_ph/dE`` are all blob-comoving. Energies are in erg, the
target spectrum is in cm^-3 erg^-1, and the returned local absorption
coefficient ``alpha_gamma_gamma`` is in cm^-1. HELENA defines

```text
s = E_gamma E (1 - mu) / [2 (m_e c^2)^2]
y = E_gamma E / (m_e c^2)^2
```

so the reaction threshold is ``s >= 1``, with zero cross section through
threshold and non-zero cross section only for ``s > 1``. The isotropic average
vanishes for ``y <= 1``. ``radius_cm`` remains the homogeneous-sphere radius;
the reported ``tau_d = 2 R alpha_gamma_gamma`` is the center-to-center diameter
optical depth. An absorbed SED remains comoving ``E L_E`` in erg s^-1 and uses
mixed homogeneous-sphere attenuation. It is transformed to observer quantities
only after local absorption.

For differential pair injection, the primary gamma-ray and target photon
spectra are separate actual blob-comoving number densities ``dn/dE`` in
cm^-3 erg^-1. ``electron_gamma`` is dimensionless. The fixed return order is
``(Q_e+, Q_e-)`` and each spectrum is in cm^-3 s^-1 per unit electron Lorentz
factor. The public integral uses ``dE = E dln(E)`` on both caller grids and
retains the exact incident energy ``E_gamma + epsilon`` in its energy ledger.
No radius, volume, escape time, or conversion from photon injection to density
is implicit in this interface.

## Isotropic Bethe--Heitler Pair Production

``electron_gamma`` and ``proton_gamma`` are dimensionless blob-comoving
Lorentz factors. ``dn_p/dgamma_p`` is in cm^-3 per proton Lorentz factor;
target ``E_ph`` and ``dn_ph/dE_ph`` are in erg and cm^-3 erg^-1. Internally,
``epsilon = E_ph/(m_e c^2)`` and ``u = gamma_p epsilon``. Caller proton and
photon grids are integrated with ``d gamma_p = gamma_p dln(gamma_p)`` and
``dE_ph = E_ph dln(E_ph)`` without interpolation or extrapolation.

The differential KA/Blumenthal response is a single-charge cross response in
cm^2 per daughter Lorentz factor and excludes ``c``. The public fixed return
order is ``(Q_e+, Q_e-)``; each spectrum is in cm^-3 s^-1 per daughter Lorentz
factor and the two arrays are equal. Combined pair number and energy moments
sum both charge channels exactly once.

The proton API returns the Chodorowski fractional continuous loss
``t_BH,loss^-1 = -dot(gamma_p)/gamma_p`` in s^-1. A future transport wrapper
must multiply this by ``gamma_p`` to form ``b_BH = -dot(gamma_p)``; the value
is neither an interaction rate nor a local particle sink. All quantities are
local and blob-comoving, with no implicit radius, volume, observer transform,
or target-photon depletion.

## Isotropic Photomeson Rates

``proton_gamma`` is dimensionless, while target photon energy ``E`` and
``dn_ph/dE`` are blob-comoving and supplied in erg and cm^-3 erg^-1. The
Hümmer Sim-B response variable is ``y = proton_gamma E``; it is converted to
GeV only while evaluating the published analytic response functions. Cross
sections returned by those functions are converted with
``1 microbarn = 1e-30 cm^2`` before integration.

The interaction rate and fractional energy-loss rate are in s^-1. The latter
is ``-E_p^-1 dE_p/dt`` and includes the event inelasticity; the cooling time is
its reciprocal in seconds and is positive infinity for a zero loss rate. The
ultrarelativistic approximation ``beta_p ~= 1`` is used. Multiple independent
photon-field rates add linearly, but host-frame fields must first be transformed
to the blob frame.

For charge-resolved pion injection, ``pion_gamma`` is dimensionless and
``dn_p/dgamma_p`` is in cm^-3 per unit proton Lorentz factor. The fixed return
order is ``Q_pi0``, ``Q_pi+``, ``Q_pi-``; every array is in cm^-3 s^-1 per unit
pion Lorentz factor. For component ``i``, the mapped proton coordinate and
Jacobian are

```text
gamma_p* = [m_pi / (chi_i m_p)] gamma_pi
d gamma_p* / d gamma_pi = m_pi / (chi_i m_p).
```

Neutral and charged rest energies are respectively ``134.9768 MeV`` and
``139.57039 MeV`` (PDG 2024). Proton density is interpolated linearly in
``ln(gamma_p)`` without extrapolation, while the density rather than its
logarithm is interpolated. All injection inputs and outputs are local to the
blob comoving frame; no volume, radius, luminosity, or observer transform is
implicit.

## Neutral-Pion Two-Photon Decay

``Q_pi0 = dn_pi0/(d gamma_pi dt)`` is supplied in cm^-3 s^-1 per unit pion
Lorentz factor on a strictly increasing ``gamma_pi > 1`` grid. Photon energy
``E_gamma`` is blob-comoving and in erg. The returned
``Q_gamma = dn_gamma/(dE_gamma dt)`` is in cm^-3 s^-1 erg^-1 and follows the
caller's photon-energy order.

The calculation uses ``m_pi0 c^2 = 134.9768 MeV`` and
``B(pi0 -> gamma gamma) = 0.98823`` (PDG 2024). It is a local injection rate;
there is no implicit volume, escape time, absorption, luminosity, or observer
conversion.

## Charged-Pion Two-Body Decay

``Q_pi+/-_dec = dn_pi+/-_dec/(d gamma_pi dt)`` is the actual blob-comoving
decay-event rate spectrum in cm^-3 s^-1 per unit pion Lorentz factor. It is
not a pion density and is not automatically derived from photomeson injection.
The caller may supply prompt pion injection only after an external decision,
or may later supply ``N_pi+/- / t_dec`` from transport.

Muon outputs use dimensionless ``gamma_mu`` and are in cm^-3 s^-1 per unit
muon Lorentz factor. Their fixed order is ``mu+_L``, ``mu+_R``, ``mu-_L``,
``mu-_R``, where ``L`` is ``h=-1`` and ``R`` is ``h=+1``. Direct ``nu_mu`` and
``anti_nu_mu`` outputs use blob-comoving neutrino energy in erg and are in
cm^-3 s^-1 erg^-1.

The charged channel uses ``m_pi+/- c^2 = 139.57039 MeV``,
``m_mu c^2 = 105.6583755 MeV``, and ``B(pi -> mu nu) = 0.9998770`` (PDG 2024),
with ``r_pi = (m_mu/m_pi)^2``. Its scaling functions omit the branching ratio;
the production result applies it once. Unlike the exact finite-beta neutral
kernel, this charged-pion calculation uses the ultrarelativistic scaling
approximation.

## Charged-Pion Decay and Synchrotron Timescales

``pion_gamma`` is a dimensionless one-dimensional array with values strictly
greater than one, and ``magnetic_field_G`` is a positive scalar blob-comoving
magnetic-field magnitude in Gauss. Both returned timescales are blob-comoving
and in seconds. The dimensionless rate ratio is

```text
eta_syn/dec = t_syn^-1 / t_dec^-1 = t_dec / t_syn.
```

The module uses ``m_pi+/- c^2 = 139.57039 MeV`` and
``tau_pi+/- = 2.6033e-8 s`` (PDG 2024), converting the pion mass to grams for
the cgs synchrotron helper. ``eta_syn/dec = 1`` is an equal-rate diagnostic,
not an automatic prompt-decay boundary.

## Muon Decay and Synchrotron Timescales

``muon_gamma`` is a dimensionless one-dimensional array with values strictly
greater than one, and ``magnetic_field_G`` is a positive scalar blob-comoving
magnetic-field magnitude in Gauss. Both returned timescales are blob-comoving
and in seconds. The dimensionless rate ratio is

```text
eta_mu = t_syn^-1 / t_dec^-1 = t_dec / t_syn.
```

The module uses ``m_mu c^2 = 105.6583755 MeV`` and
``tau_mu = 2.1969811e-6 s`` (PDG 2026), converting the mass to grams before
calling the existing cgs synchrotron helper. Charge and helicity are not input
parameters. ``eta_mu = 1`` is an equal-rate diagnostic, not an automatic
prompt-decay boundary.

## Helicity-Dependent Muon Decay

The four parent inputs are actual blob-comoving decay-event rate spectra
``dn_mu_h+/-_dec/(d gamma_mu dt)`` in cm^-3 s^-1 per unit muon Lorentz factor,
ordered as ``mu+_L``, ``mu+_R``, ``mu-_L``, ``mu-_R`` with ``L: h=-1`` and
``R: h=+1``. They are not muon densities and are not automatically derived
from pion-decay injection or timescales.

Electron/positron outputs use dimensionless ``gamma_e`` and are returned as
``(Q_e+, Q_e-)`` in cm^-3 s^-1 per unit electron Lorentz factor. Neutrino
energies are blob-comoving and in erg; outputs are returned as
``(Q_nu_e, Q_anti_nu_e, Q_nu_mu, Q_anti_nu_mu)`` in cm^-3 s^-1 erg^-1.

The leading-order ultrarelativistic Michel calculation uses
``m_mu c^2 = 105.6583755 MeV`` (PDG 2026) and
``m_e c^2 = 0.51099895069 MeV`` (CODATA 2022). The charged-lepton Jacobian is
``m_e/m_mu`` and the neutrino Jacobian is ``1/(m_mu c^2)``. Electron mass is
otherwise omitted from the scaling kinematics. No additional branching ratio
is applied, and negative-muon scaling follows ``h -> -h``.

## Generic Steady-State Transport

The generic transport grid ``particle_gamma`` is dimensionless and strictly
increasing. The blob-comoving quantities are

```text
Q = dn/(d gamma dt)                 [cm^-3 s^-1]
b = -d gamma/dt > 0                 [s^-1]
lambda_total                        [s^-1]
N = dn/d gamma                      [cm^-3]
```

The solver applies ``-d(b N)/d gamma + lambda_total N = Q`` and the
high-energy boundary ``b(gamma_max) N(gamma_max) = 0``. Here escape is a local
sink rather than part of ``b``:

```text
lambda_escape = t_escape^-1
lambda_total = lambda_decay + lambda_escape + lambda_catastrophic
```

Each physical output retains its own rate, such as
``Q_decay = lambda_decay N`` and ``Q_escape = lambda_escape N``. The generic
solver neither decomposes ``lambda_total`` nor assigns all removed particles
to decay.

A possible future long-term one-zone reference prescription is

```text
t_escape = eta_escape R/c,  eta_escape >= 1.
```

Energy-dependent diffusive escape may instead scale as
``t_diff ~ R^2/D(gamma)``. The radius, diffusion coefficient, and any geometry
factor remain caller-side physics and are not part of the generic solver.

## Charged-Pion Steady-State Transport

The blob-comoving ``pi+`` and ``pi-`` injection spectra
``Q_pi,inj = dn_pi,inj/(d gamma_pi dt)`` are in cm^-3 s^-1 per unit pion
Lorentz factor. The transported densities ``N_pi = dn_pi/d gamma_pi`` are in
cm^-3 per unit Lorentz factor. Their continuous loss and local sink are

```text
b_pi,syn = gamma_pi/t_pi,syn          [s^-1]
lambda_pi,dec = 1/t_pi,dec            [s^-1]
```

The actual decay-event rate is
``Q_pi,dec = N_pi/t_pi,dec`` in cm^-3 s^-1 per unit Lorentz factor. The fixed
return order is ``(N_pi+, N_pi-, Q_pi+,dec, Q_pi-,dec)``. No decay branching
ratio is included in transport; the existing daughter-decay kernels apply
their own branching convention.

## Helicity-Resolved Muon Steady-State Transport

The blob-comoving injection spectra are ordered as
``mu+_L, mu+_R, mu-_L, mu-_R`` and use
``Q_mu,h,inj = dn_mu,h,inj/(d gamma_mu dt)`` in cm^-3 s^-1 per unit muon
Lorentz factor. The transported densities ``N_mu,h = dn_mu,h/d gamma_mu`` are
in cm^-3 per unit Lorentz factor. Their common continuous loss and local sink
are

```text
b_mu,syn = gamma_mu/t_mu,syn          [s^-1]
lambda_mu,dec = 1/t_mu,dec            [s^-1]
```

The actual decay-event rate is
``Q_mu,h,dec = N_mu,h/t_mu,dec`` in cm^-3 s^-1 per unit Lorentz factor. The
fixed return order is the four densities followed by the four decay-event
rates, each internally ordered ``mu+_L, mu+_R, mu-_L, mu-_R``. No decay
branching ratio is included in transport; the muon-decay kernels consume these
decay-event rates directly.

## Observer-Frame SED Transformation

The observer transformation distinguishes three frames:

- prime quantities such as ``E'`` and ``E' L'_E'`` are in the blob comoving
  frame;
- ``E_em`` and ``L_E,em,iso`` are in the host/emission frame and are
  line-of-sight isotropic-equivalent quantities;
- ``E_obs`` and ``F_E,obs`` are measured in the Earth observer frame.

For one discrete moving blob with isotropic comoving emergent radiation, at
corresponding energies,

```text
E_obs = delta E' / (1 + z)
L_E,em,iso(E_em) = delta^3 L'_E'(E')
E_em L_E,em,iso = delta^4 E' L'_E'
```

The source-side isotropic-equivalent SED luminosity is in erg s^-1. It is not
flux or the true luminosity integrated over every direction. With a standard
caller-supplied luminosity distance ``d_L`` in cm,

```text
E_obs F_E,obs = (E_em L_E,em,iso) / (4 pi d_L^2)
```

The observed energy flux is in erg cm^-2 s^-1. Its per-log-energy vertical
coordinate has no additional explicit ``1+z`` factor; redshift enters the
energy mapping and the definition of luminosity distance.
