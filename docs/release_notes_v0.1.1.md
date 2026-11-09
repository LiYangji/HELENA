# HELENA v0.1.1

v0.1.1 updates the coupled internal photon state, photon-photon pair production,
internal inverse Compton emission/cooling, response resources and maintained
examples. These scientific fixes can change spectra relative to v0.1.0; this
is not a numerically interchangeable repackaging of that release.

## Scientific and numerical changes

- One internal photon state contains each enabled source once: lepton, proton,
  pion and muon synchrotron, internal IC, internally produced EC photons, and
  neutral-pion decay. Production, internal density, opacity, escape and the
  fields returned to BH/photomeson calculations use the common state.
- Internal gamma-gamma collisions use the same internal density on both sides,
  without an artificial 1 MeV eligibility boundary. The operator accounts for
  both photon losses and one electron/positron pair per event, including
  equal-source and comparable-energy collisions. Fixed external targets remain
  separate reservoirs with separate collision and energy accounting.
- The homogeneous-sphere transfer applies the shared electron SSA and
  gamma-gamma opacity to enabled photon sources. Former low/high source and
  transfer exclusions are removed; existing SSA kernel assumptions remain.
- Internal IC emission and electron/positron IC cooling read the complete
  internal photon density from the same iterate. IC and internally produced EC
  photons feed subsequent iterations; fixed external seeds enter EC separately.
  Cooling quadrature uses the logarithmic photon-energy measure. This remains
  isotropic ultrarelativistic upscattering, not arbitrary-energy Compton
  redistribution or a general photon/thermal-electron scattering solver.
- The pair-daughter grid has 537 nodes, including extra low-energy resolution;
  the photon integration grid remains 249 nodes and lepton transport/BH uses
  513 nodes. There is no whole-spectrum normalization to force energy closure.
- Photon mapping, particles and pair-source convergence are checked together.
  An unconverged or failed-budget return is not an acceptable steady spectrum.

## Output and resource compatibility

`internal_ic` is the complete additive internal-IC component. Retained SSC
fields describe its synchrotron-seeded subset and must not be added to it.
The historical `enable_ssc` spelling controls the complete internal IC process.
The normal forward, fitting predictor and maintained examples share these
semantics. Former low/high physical output fields and retired internal
one-way orchestration are not production alternatives.

BH and gamma-gamma internal/external responses are distributed as v0.1.1
Release assets, outside the source Git tree and wheel. `helena cache fetch`
downloads the version-bound resources with SHA256 and strict identity checks;
`status` validates them, `path` reports the selected directory, and `build` is
an explicit local preparation alternative. A normal model call does not
implicitly download or build responses. See the [cache instructions](../README.md#response-cache-lifecycle).

The v0.1.0 BH archive is incompatible with the current grids even where tensor
shapes coincide. The old tag, release, resource archive and checksums remain
unchanged; `fetch --legacy` and `status --legacy` retain that historical path.
Do not mix notebook, wheel and cache versions. Use the v0.1.1 source archive
for the CPU/GPU Quickstarts and TXS canonical example, together with the
v0.1.1 wheel or tagged source install.

The GPU Quickstart uses the LEHA parameter state as its fit start, retains its
physical switches and full teaching workflow, and bounds the real optimization
work. It reuses the identical LEHA model and prepared responses, releases
unused PS responses, and clears completed JAX compilation caches between
forward, optimization and Jacobian stages; these resource-management steps do
not change physical parameters, solver tolerances or process switches.
A previous extreme initial point is not used as the default example.
The TXS scientific reference parameters are not refitted by this release task;
historical v0.1.0 numerical fit results are not acceptance targets for v0.1.1.

## Validation scope and remaining limits

The M05 validation baseline includes converged TXS and controlled states,
forward/predictor agreement, and eight public parameter directions checked
with multiple-step AD/FD at the tested TXS point. Default, tighter and deeper
iteration comparisons support that calculation. These are derivatives of the
finite iteration with convergence freezing, not a proof of implicit
steady-state gradients or global parameter-space validity.

On the checked frozen states, refining the 249-node photon integral changes
the pair spectrum by about 2.0--2.5% in L1. The release retains that measured
limitation; it does not claim 1% pair-spectrum accuracy or identify that number
with an error bound on final SEDs. Daughter support, moments and interpolation
remain distinct diagnostics.

Convergence is not guaranteed for all physical inputs. The former strongly
hadron-loaded GPU initial point still fails at a later outer step with 80/80
limits, with photon and pair-budget failures despite finite output arrays.
Its replacement in the Quickstart does not establish a solution at that point.
Always inspect the returned convergence and conservation flags.

The retained IC kernel does not describe cold-electron heating or general
hard-photon downscattering. IC seed depletion is still neglected in production;
the validated optically thin examples do not justify that approximation for
all sources. No time-dependent evolution, new physical process, or global
parameter fit is introduced.

## Installation and citation

Install `helena_astro-0.1.1-py3-none-any.whl` from the
[v0.1.1 Release](https://github.com/LiYangji/HELENA/releases/tag/v0.1.1), or use
the matching tag as described in the [README](../README.md). Verify release
assets against the accompanying checksum list. JAX CPU/GPU setup remains an
explicit environment choice and scientific execution requires x64.

Cite the software version used. The existing Zenodo DOI belongs to v0.1.0;
v0.1.1 does not reuse that version-specific DOI and assigns no new DOI here.
