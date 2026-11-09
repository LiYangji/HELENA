# Physics Assumptions

This document records the physical assumptions used in HELENA.

## Current Model Scope

The first version of HELENA is designed as a one-zone steady-state lepto-hadronic radiation code for AGN jets.

The emission region is assumed to be:

- homogeneous;
- isotropic in the comoving frame;
- characterized by a single magnetic field strength;
- characterized by a single size scale;
- described by steady-state particle distributions.

The physical model is not intended to be time-dependent or spatially resolved.
Optimization and likelihood evaluation are separate interfaces and do not
change these physical assumptions.

## Current Physical Assumptions

The physical model assumes:

- cgs units internally;
- isotropic particle distributions in the comoving frame;
- uniform magnetic field inside the emission region;
- steady-state particle distributions;
- no spatially resolved jet structure;
- no time-dependent particle evolution;

## Internal Variables

The internal physical representation should prioritize physical clarity and unit consistency.

Recommended internal variables:

- particle distributions should preferably use Lorentz factor grids, such as gamma_e and gamma_p;
- photon spectra should preferably use photon energy grids in cgs units, E_photon_erg;
- photon frequency should be derived through E = h nu;
- plotting interfaces should output nu_Hz and nuLnu_erg_s.

The code should not use eV, keV, MeV, or GeV as the main internal unit system. These units may be used for user-facing input, output, labels, or convenience conversion, but the internal calculation should remain cgs.

## Initial Physical Modules

The first planned physical modules are:

1. Existing leptonic reference backend.
2. Proton synchrotron emission.
3. Gamma-gamma absorption.
4. p-gamma interaction and neutrino production.
5. Steady-state pair cascade.

## Implemented Charged-Particle Synchrotron Scope

The current reference implementation provides classical, optically thin
charged-particle synchrotron emission in the comoving frame. It uses the exact
isotropic pitch-angle average and accepts an arbitrary tabulated
``dn/dgamma`` particle number density on the caller-supplied gamma grid.

The emitting region is a homogeneous comoving sphere. The population result is
returned as ``E L_E = nu L_nu`` in erg s^-1. Thin electron and proton wrappers
fix the corresponding Astropy mass and elementary-charge constants, while an
independent power-law helper provides a convenient inclusive-cutoff
``dn/dgamma`` distribution.

The exact isotropic normalization also provides the single-particle total
synchrotron power, the ultrarelativistic synchrotron cooling time, and the
cooling Lorentz factor for a supplied comoving timescale. Cooling uses particle
energy ``gamma m c^2``. Thin electron and proton cooling wrappers fix the same
Astropy constants. At ``B=0``, total power is zero while cooling time and the
cooling Lorentz factor are positive infinity.

Synchrotron self-absorption accepts the same arbitrary tabulated ``dn/dgamma``
on the caller-supplied grid. It evaluates the signed comoving absorption
coefficient from a numerical derivative in ``ln(gamma)``. The emergent SED is
the existing thin SED multiplied by the Dermer homogeneous-sphere attenuation
factor. ``radius_cm`` is the sphere radius, and the factor uses the full center
diameter optical depth ``tau_d = 2 alpha_nu radius_cm`` rather than a slab
factor. Electron and proton wrappers only fix mass and charge.

The SSA result remains comoving ``E L_E = nu L_nu``. The observer-frame
transformation acts on this emergent spectrum after self-absorption.
The coefficient function preserves negative absorption, but the sphere-transfer
function rejects a clearly negative coefficient because maser transfer is not
implemented. Numerical derivatives make SSA sensitive to particle-grid
resolution, noise, and sharp distribution cutoffs.

## Implemented Isotropic Inverse Compton Scope

The inverse-Compton implementation assumes angle-averaged isotropic electrons
and photons in the homogeneous blob comoving frame. Internal and external
fields use the same Jones/BG ultrarelativistic upscattering kernel including
Klein--Nishina recoil, not arbitrary-energy redistribution. The low-level kernel
continues to accept only blob-comoving tabulated ``dn_ph/dE`` in
cm^-3 erg^-1.

A separate helper transforms a caller-supplied isotropic, angle-integrated
host-frame ``dn_ph/dE`` into an angle-averaged blob-frame spectrum using the
bulk Lorentz factor ``Gamma`` and the exact directional Doppler factor before
calling the same IC kernel. This is an angle-averaged external-field IC
approximation, not anisotropic EC. ``Gamma`` transforms the external field;
the observer Doppler factor ``delta`` remains a separate observer quantity.
The reference transform uses fixed 128-point Gauss--Legendre angular
quadrature and piecewise-linear ``dn_ph/dE`` interpolation in ``ln(E)``;
shifted energies outside the caller's host grid contribute zero.

The SSC helper obtains the internal synchrotron photon density from the
optically thin comoving SED and the existing absorption coefficient using the
Dermer homogeneous-sphere volume-and-direction average. It is not an emergent
SSA luminosity multiplied by a fixed escape time. The IC result is comoving
``E_s L_Es`` and may then use the existing isotropic observer transformation.

Dedicated BLR, torus, disk, and blackbody fields, anisotropic host fields,
directional EC emission, and full anisotropic EC are not implemented.

## Implemented Isotropic Gamma-Gamma Absorption Scope

Local photon-photon pair-production absorption accepts a caller-supplied,
angle-integrated isotropic target ``dn_ph/dE`` in the blob comoving frame. It
uses the exact Breit--Wheeler cross section with
``s = E_gamma E (1-mu) / [2 (m_e c^2)^2]``. The reaction threshold is
``s >= 1``; the cross section is zero through threshold and non-zero only for
``s > 1``. The exact
isotropic angular average is
``bar_sigma(y) = 2 y^-2 integral_1^y s sigma(s) ds`` for
``y = E_gamma E/(m_e c^2)^2 > 1`` and is evaluated with fixed 128-point
Gauss--Legendre quadrature. The production implementation does not use the
common Coppi & Blandford (1990) approximation.

The local coefficient is integrated over the caller's target-energy grid.
For the homogeneous sphere, ``tau_d = 2 R alpha_gamma_gamma`` is explicitly a
diameter optical depth. Uniform intrinsic emissivity is attenuated with the
same validated homogeneous-sphere factor used by SSA, rather than a foreground
screen. Internal synchrotron fields may be supplied directly; host-frame
external fields must first use the external-field Lorentz transform. Observer
transformation follows absorption.

This absorption API is local, angle-averaged, and comoving. It contains no
anisotropic geometry, BLR/torus/disk path integration, EBL attenuation,
cascade, target depletion, or kinetic coupling. The cross-section and
isotropic-average conventions follow Breit & Wheeler (1934) and Gould &
Schréder (1967).

## Implemented Isotropic Gamma-Gamma Pair-Injection Scope

``gamma_gamma_pair_injection.py`` is the independent NumPy differential
reference for two labelled isotropic blob-comoving photon populations. It uses
the full unpolarized Boettcher--Schlickeiser differential QED kernel and the
same fixed 128-point angular convention as gamma-gamma absorption. It does not
use equal energy sharing, a delta function, or threshold smoothing. The fixed
collision kernel is active only for ``s > 1`` and is exactly zero outside the
finite daughter-energy support.

Both photon inputs are actual ``dn/dE`` number densities rather than injection
rates or SEDs. Charge symmetry gives identical positron and electron spectra,
returned separately. The two incident photon populations are labelled as
distinct; applying the interface to one self-interacting bath requires the
caller to supply the appropriate symmetry convention. No photon-density
closure, pair transport, radiation, cascade, target depletion, anisotropic
geometry, observer transform, or JAX pair-injection kernel is included.

## Cascade photon population architecture

Production now uses the unified internal photon state documented under
"Unified photon production state (M05 round 2)" below. Its source populations
and reaction kinematics determine collision coverage. The former asymmetric
primary/target assignment and 1 MeV source masks have been removed; independent
labelled-density kernels remain useful primitives, with explicit event counting.

## Implemented Isotropic Bethe--Heitler Reference Scope

The NumPy Bethe--Heitler reference evaluates the single-charge differential
pair spectrum of Kelner & Aharonian (2008), Equation (62), using the corrected
Born double-differential cross section of Blumenthal (1970), Equation (10).
It is exactly zero for ``u = gamma_p E_ph/(m_e c^2) <= 1`` and outside the
finite daughter support. Born charge symmetry gives separate, identical
positron and electron outputs; a single-charge spectrum is not multiplied by
two unless a combined pair number or energy is explicitly formed.

The independent proton result is the Chodorowski, Zdziarski & Sikora (1992)
Equation (3.11) continuous fractional loss ``-dot(gamma_p)/gamma_p`` with
their published ``phi`` fits. It is not an event rate, proton destruction, or
escape sink. The KA/Blumenthal pair spectrum and Chodorowski loss are distinct
but physically compatible recoil-neglected reference expressions. Their pair
first-moment energy ledger, after subtracting the incident target-photon
energy, is therefore a diagnostic rather than a machine-precision identity;
neither spectrum is renormalized to force closure.

Both calculations assume ``gamma_p >> 1``, soft isotropic photons, an
unpolarized Born interaction, a recoil-neglected heavy target, and
``u << m_p/m_e``. The module includes no proton or electron transport, target
depletion, cascade, JAX implementation, Karavola empirical surrogate, Monte
Carlo event generation, lookup table, interpolation, or observer transform.

## Implemented Isotropic Photomeson Rate and Pion-Injection Scope

The photomeson reference implementation accepts an arbitrary tabulated,
angle-integrated isotropic or already angle-averaged photon ``dn_ph/dE`` in the
blob comoving frame. It implements the factorized Hümmer et al. (2010) Sim-B
responses for lower and higher resonances, direct one- and two-pion production,
and the total-cross-section multipion contribution, assuming
``beta_p ~= 1``. Host-frame isotropic fields must first use
``transform_isotropic_external_photon_field``.

The numerical API accepts ``proton_gamma >= 1``, but the ultra-relativistic
``beta_p ~= 1`` response is physically intended for ``proton_gamma >> 1``. No
finite-``beta_p`` correction is included.

The interaction rate counts physical proton-photon events, not produced pion
components. In particular, direct T2a and T2b describe the same two-pion event
and are counted once, while T2a's inelasticity includes both pions. The
multipion interaction and energy-loss rates use the total-cross-section
response of Equation (38), not the separate Table 6 low/high pion components.
The fractional energy-loss rate uses the Sim-B event inelasticities and the
Appendix B value ``K_Multi-pi ~= 0.6``.

Charge-resolved local pion injection is implemented for initial protons only.
It returns ``Q_pi0``, ``Q_pi+``, and ``Q_pi-`` per unit pion Lorentz factor.
The injection follows the factorized response of Equations (27)--(28), using
the neutral and charged pion masses separately and every pion component in
Tables 4--6. T2b is included as the second pion produced by the same direct
two-pion event, and both L/H components of each Table 6 multipion sample are
included as different pion-energy components of the same event. These
components remain excluded from event counting as additional interactions.

The input ``dn_p/dgamma_p`` is interpolated linearly in ``ln(gamma_p)`` while
interpolating the density itself, not its logarithm. Mapped proton Lorentz
factors outside the supplied strictly increasing proton grid have exactly zero
density; there is no extrapolation or automatic sorting. All proton, pion, and
photon quantities are local blob-comoving quantities, and no volume or radius
factor is applied.

This energy-loss rate is the total fractional energy transferred from the
initial high-energy nucleon to secondaries. It does not distinguish whether
the leading final-state nucleon is a proton or neutron and is not a
species-resolved proton transport cooling term. Proton-to-neutron escape and
neutron reinjection remain unimplemented.

Consequently, pion-only energy injection is not required to equal this total
loss power for multipion-dominated fields: the public loss rate retains the
Appendix B total response with ``K_Multi-pi ~= 0.6``, while charge-resolved
pion injection uses the Table 6 pion components and their pion energy shares.
Energy closure against the public loss rate is therefore a controlled
resonance/direct diagnostic, not a general identity for all target spectra.

The module includes no pion evolution, cooling, synchrotron radiation or
decay, secondary gamma ray, muon, neutrino or pair production, neutron-initiated
photomeson process, proton/neutron transport, neutron escape or reinjection,
Bethe--Heitler or pp interaction, cascade, kinetic solver, anisotropic or
spatial treatment, JAX, or fitting. Rates from independent photon fields may
be evaluated on their own grids and added linearly.

## Implemented Neutral-Pion Two-Photon Decay Scope

The neutral-pion decay module converts a caller-supplied local blob-comoving
``Q_pi0(gamma_pi)`` into ``Q_gamma(E_gamma)`` for ``pi0 -> gamma + gamma``.
It uses exact relativistic two-body kinematics, the separate neutral-pion rest
energy ``134.9768 MeV``, and the physical branching ratio ``0.98823``. The
calculation retains finite pion beta and does not use ``beta_pi = 1``.

The pion injection is integrated only on the caller's strictly increasing
``gamma_pi > 1`` grid. A cumulative trapezoidal tail is linearly interpolated
without extrapolating the pion injection. ``gamma_pi = 1`` is excluded because
the two photons from a stationary pion form a delta-function line that this
continuous output interface cannot represent exactly.

This is local photon injection only. It includes no Dalitz channel, pion
population evolution, pion cooling or synchrotron radiation, charged-pion or
muon decay, neutrinos, secondary pairs, gamma-gamma absorption, cascade,
observer transformation, JAX, or fitting.

## Implemented Charged-Pion Two-Body Decay Scope

The pion-decay module also maps caller-supplied blob-comoving ``pi+`` and
``pi-`` decay-event rate spectra to the direct daughters of
``pi+ -> mu+ + nu_mu`` and ``pi- -> mu- + anti_nu_mu``. These inputs are
actual decays per unit pion Lorentz factor and time, not pion densities. Only
after making an external timescale or transport decision may a caller use
prompt photomeson pion injection or ``N_pi / t_dec`` as that rate.

Charged-pion decay uses the ultrarelativistic Hümmer/Lipari scaling convention,
physically intended for ``gamma_pi >> 1``. The direct-neutrino scaling is
constant on ``0 < x < 1-r_pi``. The two muon-helicity scalings are supported on
``r_pi < x < 1`` and sum to ``1/(1-r_pi)``. The fixed outputs keep
``mu+_L``, ``mu+_R``, ``mu-_L``, and ``mu-_R`` separate, with ``L`` denoting
``h=-1`` and ``R`` denoting ``h=+1``. Charge conjugation exchanges the two
helicity shapes.

The PDG 2024 principal muonic branching ratio is applied once outside the
individually normalized scaling functions. The existing neutral-pion path
remains the separate exact finite-beta two-photon calculation. Radiative and
electronic pion channels, pion transport or cooling evolution, muon decay or
cooling, secondary electrons, additional neutrino flavors, oscillation,
cascade, observer transformation, JAX, and fitting are not included in this
decay kernel.

## Implemented Charged-Pion Timescale Scope

The charged-pion timescale module is an independent blob-comoving diagnostic,
not part of photomeson production or pion-decay kinematics. It uses the PDG
2024 charged-pion rest energy ``139.57039 MeV`` and lifetime ``2.6033e-8 s``
to evaluate

```text
t_dec = gamma_pi tau_pi.
```

For a unit-charge pion it reuses the existing classical isotropic synchrotron
cooling helper, equivalently

```text
t_syn = 6 pi m_pi^3 c / (sigma_T m_e^2 B^2 gamma_pi).
```

The synchrotron expression is ultrarelativistic, assumes ``gamma_pi >> 1`` and
``chi << 1``, and deliberately includes no finite-beta correction. The
reported continuous comparison is
``eta = t_syn^-1 / t_dec^-1 = t_dec / t_syn``: values above one mean that
synchrotron cooling is faster than decay. The module does not impose a prompt-
decay switch; the caller must choose any approximation according to its energy
range and required accuracy.

This diagnostic module does not itself evolve pions or calculate their
synchrotron spectrum, decay injection, muons, neutrinos, secondary electrons or
positrons, escape, cascade, observer transformation, JAX, or fitting. The
charged-pion transport wrapper reuses these timescale interfaces without making
transport part of the diagnostic module. The decay kernels do not call these
diagnostics or select a prompt-decay approximation.

## Implemented Muon Timescale Scope

``muon_timescales.py`` is an independent blob-comoving diagnostic, separate
from pion-decay kinematics and any future muon-decay kinematics. It uses the
PDG 2026 muon rest energy ``105.6583755 MeV`` and lifetime
``2.1969811e-6 s`` to evaluate

```text
t_dec = gamma_mu tau_mu.
```

For a unit-charge muon it reuses the existing classical isotropic synchrotron
cooling helper, equivalently

```text
t_syn = 6 pi m_mu^3 c / (sigma_T m_e^2 B^2 gamma_mu).
```

The synchrotron expression assumes ``gamma_mu >> 1``, ``chi_mu << 1``, and
``beta_mu^2 ~= 1``. Muon charge and helicity change neither this classical
cooling time nor the rest-frame lifetime. The continuous comparison
``eta = t_syn^-1 / t_dec^-1 = t_dec / t_syn`` is not an automatic decision:
the caller must assess prompt decay over its valid muon-energy range and
required accuracy.

The helicity-resolved muon transport wrapper reuses these interfaces without
making transport part of the diagnostic module. The diagnostic itself includes
no muon injection conversion, decay products, cooling evolution, steady-state
or time-dependent transport, synchrotron SED, escape, cascade, observer
transformation, JAX, or fitting.

## Charged-Meson Synchrotron Photon Feedback Scope

Charged-pion and muon synchrotron cooling were already included in their
steady-state transport. The outer hadronic cascade now optionally evaluates
the corresponding photon emission from the current transported densities with
the existing arbitrary-mass synchrotron kernel. It retains separate
``pi+``, ``pi-``, ``mu+``, and ``mu-`` photon sources; the two muon helicity
densities are summed only within each charge because the classical emission is
helicity independent.

Each enabled meson source is evaluated once on the complete common photon
grid. It enters the same SSA/gamma-gamma transfer as every internal photon
source, and the accepted total density feeds the next photomeson/BH step.
There is no low/high meson partition or spectral renormalization. Dynamic
mode updates the target; frozen-target diagnostic mode remains explicitly
nonconverged with respect to the outer fixed point. Pion/muon switches remain
independent; the standard C513 configuration enables both and has zero external
photon density. Quantitative validation scope is recorded per repair round,
not inferred merely from the existence of this feedback path.

## Implemented Helicity-Dependent Muon Decay Scope

``muon_decay.py`` is an independent blob-comoving decay-kinematics module. It
maps actual decay-event rate spectra for ``mu+_L``, ``mu+_R``, ``mu-_L``, and
``mu-_R`` to ``e+``, ``e-``, ``nu_e``, ``anti_nu_e``, ``nu_mu``, and
``anti_nu_mu`` injection. These parent inputs are not muon densities. Only
after an external timescale or transport decision may a caller use pion-decay
muon injection or ``N_mu / t_dec`` as the decay-event rate.

The calculation uses leading-order Standard Model Michel scaling functions in
the ultrarelativistic ``gamma_mu >> 1`` limit. Positive-muon spectra use
``A+hB`` for the charged lepton and muon-flavor antineutrino and ``C+hD`` for
the electron-flavor neutrino. Negative-muon spectra use CP conjugation
``h -> -h``. Every scaling has unit number normalization, and its analytic
first moment closes the three daughter energies to the parent energy.

Neutrinos are massless. The electron mass is neglected in the scaling
kinematics and used only for the ``E_e <-> gamma_e`` mapping and Jacobian. The
effective three-body channel has unit branching ratio. QED radiative
corrections, explicit radiative muon decay, exact electron-mass kinematics,
cooling evolution, transport, secondary radiation, neutrino oscillation,
tau neutrinos, cascade, observer transformation, JAX, and fitting are not
included. The separate ``muon_timescales.py`` diagnostic is neither imported
nor called and does not select prompt decay.

## Implemented Generic Steady-State Transport Scope

``steady_state_transport.py`` solves the species-independent blob-comoving
equation

```text
-d[b(gamma) N(gamma)]/d gamma + lambda_total(gamma) N(gamma) = Q(gamma)
```

with ``b = -d gamma/dt > 0`` and the high-energy boundary
``b(gamma_max) N(gamma_max) = 0``. The NumPy reference uses midpoint
coefficients and an O(n_gamma) high-to-low recurrence. It returns only the
steady-state differential density; it does not construct any particle-specific
injection, loss, decay, or escape rate. The reverse recurrence is isolated so a
future backend can replace the loop with ``jax.lax.scan`` while retaining the
NumPy reference; JAX is not included here.

Escape is a local sink ``lambda_escape N``, not a continuous energy loss.
The supplied total sink may be
``lambda_decay + lambda_escape + lambda_catastrophic``, but callers must keep
the components separate and form, for example, ``Q_decay = lambda_decay N``
and ``Q_escape = lambda_escape N``. The total sink must not be passed wholesale
to a decay kernel.

An initial pion or muon transport wrapper may set ``lambda_escape = 0`` only
after checking ``t_decay << t_escape`` over its effective energy range. A
stable proton population cannot generally neglect escape in a long-lived
steady state. Low-energy electron cooling may likewise be too slow to establish
the assumed steady state without escape or a finite source age. With sustained
injection and no effective loss, escape, or finite age, a finite proton or
electron steady state need not exist.

Apart from the charged-pion and helicity-resolved muon wrappers described
below, no species wrapper, escape prescription, geometry, finite-age boundary,
time dependence, energy diffusion, stochastic acceleration, cascade, JAX
backend, or fitting interface is included.

## Implemented Charged-Pion Steady-State Transport Scope

``pion_transport.py`` applies the generic steady-state solver separately to
``pi+`` and ``pi-`` injection. Both charges use continuous synchrotron loss
``b_pi,syn = gamma_pi/t_pi,syn`` and the local decay sink
``lambda_pi,dec = 1/t_pi,dec``. The returned actual decay-event rate is
``Q_pi,dec = N_pi/t_pi,dec``; it is generally not identical to injection and
contains no decay branching ratio.

This first wrapper has no escape, so ``lambda_total = lambda_pi,dec``. That
approximation is valid only after checking ``t_dec << t_escape`` over the
relevant pion energies. The wrapper does not switch automatically to prompt
decay and does not call the pion-decay daughter kernels. Neutral pions are not
transported here. Muon transport, daughter production, escape, finite source
age, diffusion, cascade, observer transformation, JAX, and fitting remain
outside this module.

## Implemented Helicity-Resolved Muon Steady-State Transport Scope

``muon_transport.py`` applies the generic steady-state solver independently to
``mu+_L``, ``mu+_R``, ``mu-_L``, and ``mu-_R`` injection. Every channel uses
continuous muon synchrotron loss ``b_mu,syn = gamma_mu/t_mu,syn`` and the local
decay sink ``lambda_mu,dec = 1/t_mu,dec``. The actual decay-event rate returned
for each channel is ``Q_mu,h,dec = N_mu,h/t_mu,dec`` and is not assumed equal to
the input injection.

Charge and helicity do not alter the classical cooling or decay rates in this
approximation. Channels are never mixed or renormalized; helicity flip, spin
precession, and depolarization are absent. The wrapper includes no escape, so
the no-escape approximation requires decay or cooling to dominate escape over
the modeled range. It applies no automatic prompt-decay switch or branching
ratio and does not call the muon-decay daughter kernels. Daughter transport,
finite source age, cascade, observer transformation, JAX, and fitting remain
outside this module.

## Implemented Observer Transformation

The observer transformation applies only to one discrete moving blob whose
emergent radiation is isotropic in the blob comoving frame. Prime quantities
refer to that frame, emission quantities refer to the host/source frame, and
observer quantities are measured at Earth.

At corresponding energies, ``E_em = delta E'`` and
``L_E,em,iso = delta^3 L'_E'``. Therefore the line-of-sight
isotropic-equivalent SED luminosity is ``delta^4 E' L'_E'``, while
``E_obs = delta E'/(1+z)``. Dividing by ``4 pi d_L^2`` gives the observed
``E_obs F_E,obs``. There is no extra explicit ``1+z`` on this vertical SED
coordinate because the spectral-density bandwidth factor cancels the energy
redshift factor and the standard luminosity distance already contains the
cosmological dimming.

This transformation is not a continuous-jet beaming law and does not apply to
anisotropic external-Compton emission or ordered-field directional radiation.
It includes no polarization, attenuation, cosmology calculation, band
integration, interpolation, light curve, or time transformation.

The classical continuum treatment assumes ``gamma >> 1`` and quantum parameter
``chi << 1``. It includes no ``beta^2``, cyclotron/gyrosynchrotron, or quantum
synchrotron correction. Species-specific cooling evolution, acceleration,
dynamical-time diagnostics, escape models, maser transfer, and energy-budget
diagnostics are not implemented.

## Cold-Target Proton--Proton Interactions

The optional pp process uses a scalar blob-comoving density of cold,
non-relativistic target protons. It is disabled by default and is independent
of photomeson production, so pp and p-gamma interactions may operate
simultaneously. Their fractional proton losses add once in proton transport;
their charge-resolved pion sources add once before the existing pion/muon
transport and decay chain.

For proton kinetic energy below 100 GeV, pion production uses the AM3-like
delta approximation. Its per-charge inelasticity is the energy moment of the
Kelner/SIBYLL pion redistribution at 100 GeV. From 100 to 200 GeV a fixed
smoothstep changes to the high-energy Kelner/SIBYLL redistribution; the latter
is used only for ``x = E_pi/E_p >= 1e-3``. The inelastic cross section is the
threshold-corrected Kelner et al. (2006) expression. Fixed proton and pion
grids define a cached dense response containing quadrature weights, cross
section, redistribution, mass Jacobian, and kinematic support. Proton spectra
and cold-target density are dynamic and are never cached.

The pp proton loss is the energy moment of the same hybrid prescription, not
a fixed ``kappa = 0.5`` term. Finite pion-grid support can therefore leave a
small production/loss energy residual; spectra are not renormalized to remove
it. pp creates no direct gamma-ray, electron/positron, or neutrino source.
Those products come only from the existing pion/muon chain. Neutrons, nuclei,
kaons, charm, target evolution, and thermal-plasma dynamics are not included.

## Deferred Physics

The following physical extensions remain deferred:

- time-dependent solvers;
- spatially resolved jet models;
- magnetically dominated jet dynamics coupling;

## C513 Benchmark Configuration

The version-controlled C513 benchmark follows Cerruti et al. (2024), Table 3
and Section 4: a homogeneous blob with ``R = 1e16 cm``, ``B = 0.1 G``, and
``delta = 30``; no external photon field; electron and proton indices of 2;
cutoff Lorentz factors ``3e5`` and ``1e7``; injection luminosities
``3.7e40`` and ``2.8e46 erg s^-1``; and escape time ``R/c``. Primary
positron injection and electron/proton adiabatic losses are zero. The Doppler
factor is used only by the separate observer transformation.

The particle injection is ``Q(gamma) = A gamma^-2 exp(-gamma/gamma_max)``.
Its normalization integral uses kinetic energy ``(gamma - 1) m c^2`` from
``gamma = 1`` through the tabulated ``gamma_max``; the exponential tail is
then evaluated on the full numerical grid without a sharp cutoff or a second
normalization. C513 is a nested HELENA preflight grid
(``C513 = C2049[::4] = F4097[::8]``), not a universal grid prescribed by the
comparison paper.

The preflight enables synchrotron, inverse-Compton/SSC, SSA, proton
synchrotron, dynamic photomeson production, pion/muon transport and decay,
meson synchrotron photons, pi0 gamma rays, secondary leptons and neutrinos,
gamma-gamma transfer and pairs, dynamic continuous Bethe--Heitler proton
loss, and internal photon feedback. The Bethe--Heitler pair provider remains
the Born-domain ``omega < 600`` implementation; its domain is not identical
to the existing continuous-loss provider. pp is explicitly disabled in this
legacy regression; neutron, kaon/charm, general
finite-recoil Bethe--Heitler, EC, and external photons are excluded.

## Physics-First Fit-Friendly Design

HELENA should be fit-friendly, but not fitting-driven.

The physical model must be defined first, including formulae, assumptions, units, reference frames, thresholds, and valid parameter ranges. Fit-friendly implementation should only improve the numerical and software representation of the physically valid model.

Development priority:

1. Physical correctness.
2. Numerical stability and explicit boundary handling.
3. Fit-friendly implementation.
4. Performance optimization and JAX acceleration.

Fitting convenience must not be used to:

- smooth out real physical thresholds without justification;
- allow physically invalid parameter regions;
- mix quantities from different reference frames;
- hide unit conversions;
- modify conservation requirements;
- reinterpret real-source fitting as validation.

Future fitting wrappers may use log-space parameters, penalties, masks, clipping, or safe numerical functions. However, the underlying physics kernels should remain physically transparent and should accept physical parameters in explicit physical units.

## Scientific Caution

Real-source SED fitting is not considered proof that a physical module is correct.

Each module must be validated through analytic limits, sanity checks, energy-budget checks, numerical stability tests, or comparison with published benchmarks.


## Internal total-field gamma-gamma operator

`core/jax_backend/chains/gamma_gamma_pair_source.py` also provides
`prepare_gamma_gamma_total_field_response_jax(energy, daughter, ...)` and
`gamma_gamma_total_field_jax(density, response)`. This additive, frozen-field
operator is used by the unified electromagnetic fixed point described below.
The former one-way internal chain now raises an explicit migration error.

The input is one isotropic blob-comoving internal `dn/dE` in cm^-3 erg^-1,
with each physical photon represented once on the prepared energy grid (erg).
Do not form it by adding the overlapping legacy S, L and H fields. All allowed
photon pairs participate; no 1 MeV collision gate or hard/soft approximation is
introduced. Grids are fixed, strictly increasing and float64, with daughter
gamma > 1. Preparation is a host operation; dynamic density arithmetic is JAX.

The existing Breit-Wheeler opacity and full Boettcher-Schlickeiser differential
pair kernels are retained (BS97, https://arxiv.org/pdf/astro-ph/9703069,
Eqs. 8, 15-19, 24-25). The prepared pair response already contains c, angular
averaging and both energy integration weights. It has no same-population
counting factor: each charge's source is therefore `0.5 * R[n,n]`. Photon loss
is `c * alpha[n] * n`, without that half. Both density factors are differentiated.
Returned rates and powers are per volume, not whole-source luminosities; pair
power includes rest mass. External prescribed fields are outside this operator.

The result reports alpha, differential photon loss, each charge's source,
event rate, integrated number/energy budgets and conservative outside-daughter
bounds. Moment discrepancies include finite quadrature AND lost support; they
are never removed by overall renormalization. Outside bounds may be loose and
are not estimates of the missing spectrum. Caller-selected tolerances must
check them separately; `input_valid` flags invalid densities without clipping.
An optional reused response tensor must match the exact grids and kernel
configuration, not just its shape. No dynamic photon density is cached.

Minimal use (inspect the returned budgets before interpreting a spectrum):

```python
import jax
import jax.numpy as jnp
from helena.core.constants import ELECTRON_REST_ENERGY_CGS as mec2
from helena.core.jax_backend.chains.gamma_gamma_pair_source import (
    prepare_gamma_gamma_total_field_response_jax,
    gamma_gamma_total_field_jax,
)

jax.config.update("jax_enable_x64", True)
energy = mec2 * jnp.array([0.4, 1.0, 1.8, 3.0])
daughter = jnp.linspace(1.000001, 6.0, 513)
response = prepare_gamma_gamma_total_field_response_jax(energy, daughter)
collision = jax.jit(lambda n: gamma_gamma_total_field_jax(n, response))
result = collision(jnp.array([1.0, 2.0, 3.0, 0.5]) / mec2)
jax.block_until_ready(result)
```

Frozen-field operator tests and density derivatives do not by themselves
validate the coupled solver. Production closure is described below and must
pass its own convergence and ledger checks. Finite daughter support must not
be inferred from the historical `maximum photon energy + 1 MeV` endpoint.


## Unified photon production state

Production uses one complete photon grid (the union of the historical sampling
nodes), not separate low/high physical populations. A 1 MeV sampling node has
no collision, source-selection, or transfer meaning. The same seven populations
(electron/positron synchrotron, proton synchrotron, charged-pion synchrotron,
muon synchrotron, internal IC, EC, pi0 decay) supply one accepted internal dn/dE.
Full-grid internal IC is produced in every electromagnetic iteration. Optional meson switches
still set their corresponding sources to zero; optional pp products are already
included in the photomeson/pp aggregate sources passed by the hadronic owner.

All populations experience the same electron SSA plus internal/external gamma-
gamma opacity, using the existing homogeneous-sphere path/escape primitives.
SSA absorption is a property of the medium, not of which particle emitted a
photon. This removes the old arbitrary high-energy and pi0 SSA exclusions;
it does not extend the underlying classical, ultrarelativistic synchrotron/SSA
formulae beyond their stated validity. No new proton/meson SSA kernel is added.
Internal IC emission and cooling both read that accepted
TOTAL internal density, including IC, EC-produced and pi0 photons. Fixed
external seed density remains a separate EC reservoir and is never added to
internal n. No SSA factor is applied again at the seed read. Higher scattering
orders arise through the existing EM iteration, not separate populations.
The Jones/BG and Moderski physical approximations remain; their scope and the
necessary cooling quadrature correction are stated below.
BH and photomeson loss and secondary production read the same total internal
field, plus the separately prescribed external field. This broadens their old
source selection (including the formerly omitted low pi0 tail). These are real
physical prescription changes in addition to the gamma-gamma repair; they can
change results even with internal gamma-gamma switched off.

The EM scan carries photon components and electron/positron pair sources.
Its stopping criterion includes true unrelaxed component-wise photon number
and energy map defects, pair-source defects and particle/cooling changes.
Accepted n drives opacity, pair production, internal IC, and the next hadronic step.
Source production and escape belong to that same iterate; their mapped sphere
density differs from accepted n by the explicitly reported convergence defect.
Do not call this finite-tolerance state an exactly solved algebraic fixed point.
An exhausted EM solve is marked unconverged; the public outer driver stops with
failure, and the bounded fitting scan uses the same numerical-strategy outer limits
and shared photon/proton spectral and EM-ledger checks. It returns NaN after
failure or an unconverged iteration limit. Gradients remain those of this
finite bounded scan with convergence freezing, not an implicit steady solution.

The two historical EM positional photon-grid arguments must now be identical
complete grids. Old split sources cannot be silently reconstructed. The old
internal one-way gamma-gamma and independent photon-density orchestrators are
migration-error entries, with their former physics bodies removed. Standalone
radiation/absorption kernels remain available and are not production closures.
Old split keywords and low/high density fields are removed. Scalar initial SSC
cooling guesses are retired because cooling is derived from accepted photons.
The public gamma SED aliases now live on the same full photon grid; canonical
component output reads the accepted component arrays without retransferring.

A fixed external field is a separate reservoir: internal x external collisions
have no event half, remove only internal photons from the solved state, and
report the external energy contribution separately. Cached external responses
must include exact target-energy metadata in the prepared response. General
hard external fields need their own daughter support check; the internal
support bound is not a certificate for them. Internal daughter upper support
is set from twice the maximum internal photon energy, replacing '+1 MeV'.
Both BH and gamma-gamma response identities change with the actual grids;
old same-shape cache data are not reused. EBL remains source-external propagation.

Convergence and budget diagnostics are per-state requirements. Sharing a photon
state is not evidence that a physical approximation or every parameter region
has been validated.


## Full internal IC seed state

The historical `enable_ssc` / public `enable_synchrotron_self_compton` switch
now enables the whole internal IC process. This is an explicit compatibility
name, not a switch between old and new physical modes. The additive source
bucket is `internal_ic`; `internal_inverse_compton_loss_rate_dgamma_s` and
`internal_ic_seed_photon_density_cm3_erg` describe its cooling and total seed.
The retained `ssc_intrinsic_sed_erg_s`, SSC cooling and primary-electron SSC
outputs are diagnostics computed from the four synchrotron seeds only. Do not
add them to internal IC or to the additive source decomposition. Internally
created EC photons can scatter again; the prescribed external photons enter
EC emission/cooling only once.

The emission kernel is Jones/Blumenthal--Gould isotropic ultrarelativistic
upscattering, with recoil but not arbitrary-energy Compton redistribution.
The analytic Moderski et al. (2005), Appendix A, cooling is positive and cannot
represent cold-electron Compton heating by hard photons. No source or fixed
photon-energy mask is introduced to hide that domain limit. Quantitative
frozen and final-state ERF Klein--Nishina diagnostics must establish whether
the omitted heating/downscattering matters in a given calculation; a small
total output difference is not a universal validity certificate.

The cooling integral now uses `E^2 n(E) f_KN dln(E)` rather than coarse
`E n(E) f_KN dE`. These are the same continuous formula. Independent refinement
of the actual logarithmic source grids showed the old direct-energy trapezoid
biased the power by about 2%; the log measure converged to the same limit as
refined direct integration. The physical suppression and emitted spectrum
kernel are unchanged. This numerical repair affects even synchrotron-only
cooling, so comparisons with synchrotron-seeded cooling must separate it from added IC seeds.
Particle-energy integration, finite scattered-photon support and remaining
quadrature errors are still audited separately; there is no spectrum-wide
renormalization to force power equality.

There is no IC seed removal term in the production photon density mapping.
It is retained as an optically thin scattering approximation only where
independent scattering-rate/residence checks pass the stated error budget.
Production photon losses continue to mean SSA and gamma-gamma losses; seed
removal is not silently included in those rates. A general dense-scattering
or hard-photon dominated source is not certified by the two paper examples.
All dynamic internal seed components remain on both JAX feedback paths;
derivatives are of the existing bounded iteration with convergence freezing.


## v0.1.1 numerical accuracy and validation scope

The maintained production configuration uses 249 photon integration nodes,
537 pair-daughter nodes and 513 electron/positron transport nodes. The daughter
grid is locally refined near gamma = 1--100; it is not the BH transport grid.
Frozen-state refinement of the 249-point photon integral found about 2.0--2.5%
L1 differences in the pair spectrum in the two checked states. This is a known
quadrature limitation, not a 1% accuracy claim or a bound on the final SED.
Number/energy moments, interpolation and daughter support are separate checks.
No overall normalization is applied to hide discrepancies.

The checked TXS and controlled states converge with the common photon field.
The eight registered parameter directions passed multiple-step AD/FD checks
at the tested TXS point. Default/tighter/deeper iteration comparisons and a
CPU/GPU comparison support that specific calculation. These derivatives are
of the finite, bounded iteration with convergence freezing; they do not prove
strict implicit steady-state derivatives or validity throughout parameter space.

Convergence is not guaranteed for every source configuration. A former strongly
hadron-loaded GPU Quickstart initial point remains unconverged even with 80
inner and outer iterations: its fifth outer step loses convergence progress
and fails photon and pair budgets. It is not a valid steady state; replacing
the example starting point does not establish a converged solution at that
failed point. Inspect both `converged` and `conservation_checks_passed`; finite,
nonnegative spectra alone are insufficient. The v0.1.1 release does not broaden
the IC kernel to arbitrary-energy Compton redistribution, add an IC seed-removal
operator, or establish time-dependent stability.
