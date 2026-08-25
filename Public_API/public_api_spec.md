# Minimal Public API Specification — v0.1 Frozen

Status: P1--P7B PASS. The frozen disposition is recorded in the P7B summary
and freeze contract.

## P2 implemented foundations

```python
from helena import configure

runtime = configure(device="auto")  # also "cpu" or "gpu"
```

`configure` is an early-process call made before JAX backend initialization. `auto` accepts JAX's available/default platform; explicit `cpu` or `gpu` must resolve exactly as requested or fail clearly, with no backend reset and no silent fallback. Its read-only result contains `requested_device`, `resolved_platform`, `resolved_device`, and `x64`. Scientific production always enables and verifies x64; P2 exposes no precision mode.

CPU and GPU run the same model, objective, parameter coordinates, fitting semantics, AD semantics, and result schema. CPU use routinely favors about 1–2 simultaneously free parameters; GPU use favors 3–8 and multi-parameter Jacobians. This is performance guidance only. P2 implements no CPU limiter, staging, coordinate descent, or fitting warning.

```python
from helena.parameters import (
    DIFFERENTIABLE_PARAMETER_NAMES,
    DIFFERENTIABLE_PARAMETERS,
    PARAMETERS_BY_NAME,
    ParameterSpec,
)
```

`DIFFERENTIABLE_PARAMETERS` is the sole canonical descriptor tuple. The names tuple and read-only lookup mapping are derived from it; the existing fitting name export is a compatibility alias to the same tuple. Every coordinate is dimensionless and reference-relative. No source-independent fit bounds are invented in P2.

## P3 implemented model boundary

```python
from helena import ForwardResult, ModelConfig, OneZoneModel

model = OneZoneModel(ModelConfig(...))
result = model.forward()
```

`ModelConfig` is a frozen physical-value config. `OneZoneModel` maps its eight
continuous source values through the unchanged validated C513 mappers, hides
strict response loading/preparation, and calls the one canonical
`run_c513_forward`. `ForwardResult` exposes only read-only photon, component,
particle, neutrino, and compact diagnostic quantities. The public boundary has
no C513 name, raw PyTree, response tensor, solver state, or `physics_only` path.
Its component mapping includes a post-convergence source-resolved lepton
diagnostic with frozen final transport/target/transfer state; it does not alter
the canonical forward or expose a new root symbol.

`enable_internal_gamma_gamma=True` is the default production composition.
Setting it to `False` disables internal gamma-gamma absorption and the
corresponding pair injection only; it does not remove the synchrotron target,
photomeson interactions, or the independent external gamma-gamma semantics.

The current public model requires explicit redshift and luminosity distance,
uses the fixed no-external-field setup, and always performs the existing
neutrino propagation. Photon propagation is identity by default; the one
additive `enable_ebl` switch selects the validated Dominguez-2011 owner with
no edge extrapolation. Full field and result details are frozen in
`P3_model_forward_contract.md`.

## P4 implemented dataset and fitting boundary

```python
from helena import FitResult, SEDDataset, fit

result = fit(
    model,
    SEDDataset(...),
    parameters=["log_B", "log_Le"],
    optimizer="lbfgsb",  # or "migrad"
)
```

`SEDDataset` is the single root facade over the maintained `SEDData` real-data
contract. It fits Earth/post-propagation photon `E F_E` and preserves separate
statistic (`gaussian`, `asymmetric_gaussian`, `upper_limit`) and scientific-use
(`fit`, `constraint`, `context`, `exclude`) labels. Only fit/constraint rows
enter the existing `mixed_sed_nll`.

`parameters` is any unique ordered 1--8 registry subset. The input physical
`ModelConfig` is the initial point; omitted registry values remain fixed at
that exact config. Existing reference coordinates, mappers, fixed-five fitting
graph, likelihood, and optimizer driver are reused unchanged. L-BFGS-B and
MIGRAD return the same frozen/slotted `FitResult` schema with physical best and
fixed values plus a new `best_config`. No custom bounds are exposed in P4.

CPU and GPU use the identical call and objective. CPU requests above two free
parameters emit only a `RuntimeWarning`; no selection or algorithm changes.
The complete contract is `P4_dataset_fit_contract.md`.

## P5 implemented standalone L2 boundary

```python
import helena.physics as physics
import helena.transport as transport
import helena.cascade as cascade
```

`physics` exposes 21 scientific radiation, photon-field, interaction,
production, and decay functions. `transport.solve_steady_state` exposes the
one generic conservative steady equation in terms of `Q(gamma)`, continuous
loss, and local sink. `cascade.solve_em` hides source-provenance buffers,
prepared pair responses, Picard state, residual histories, and ledgers behind
one compact named result.

Every P5 quantity is explicitly blob-comoving and cgs: particle gamma is total
energy/rest energy, `N(gamma)` is cm^-3 per gamma, `Q(gamma)` is cm^-3 s^-1 per
gamma, photon targets are `dn_ph/dE` in cm^-3 erg^-1, and photon SEDs are
`E L_E` in erg s^-1. The facades call only production JAX owners; CPU/GPU
selection changes the device, not names or science. Details and deferrals are
frozen in `P5_standalone_physics_contract.md`.

## P6A maintained differentiable target

P6A freezes one private target on P4's existing model-bound fixed-five graph:

```text
physical output:    Earth photon E F_E [erg cm^-2 s^-1], linear
sensitivity target: ln[(Earth photon E F_E)/(1 erg cm^-2 s^-1)]
```

The target uses P3/P4 observer energies and a support mask computed once from
the finite, strictly positive baseline model prediction outside tracing. The
mask remains fixed. There is no clipping, floor, dynamic selection, copied
physics, or alternate parameter mapper. Any unique ordered 1--8 registry
subset is valid and its order is the Jacobian column order.

The canonical base model has all 113 bins in support, matches the public
`ForwardResult` Earth photon SED at machine precision, and has a frozen GPU
`(113, 8)` reference plus fresh low-dimensional CPU/GPU parity evidence. The
old nonzero-external-field Regime-B photon-plus-neutrino collector remains an
independent L4 target and is not numerically equated with this reference.
Details are in `P6A_derivative_target_contract.md`; it remains the unchanged
freeze provenance behind the implemented P6 contract.

## P6 implemented sensitivity and diagnostics boundary

```python
import helena.diagnostics as diagnostics
import helena.sensitivity as sensitivity

J = sensitivity.jacobian(model, parameters=["log_B", "log_Le"])
v = sensitivity.jvp(
    model,
    parameters=["log_B", "log_Le"],
    direction=[0.3, -0.2],
)
local = diagnostics.gauss_newton(J, dataset)
```

Both sensitivity functions reuse the P6A Earth-photon natural-log target,
fixed baseline-positive support, and shared registry coordinates. They accept
any caller-ordered unique 1--8 subset. `SensitivityResult` carries read-only
energy, support, parameter order, target, orientation, values, and an optional
JVP direction. Jacobians are `(energy, parameter)`; on the canonical model the
full result is `(113, 8)`. JVP values are `(energy,)` and equal the Jacobian
contracted with the direction.

CPU and GPU use the same API/result. CPU above two parameters emits only a
`RuntimeWarning`; GPU is recommended for larger Jacobians. There is no 8D
special function, scalar gradient, public AD-method selector, component,
neutrino, or particle sensitivity.

`diagnostics.gauss_newton` accepts only active independent symmetric Gaussian
observations under P4 masks. With `sigma_log = sigma_F/F`, it returns the
maintained `J_w = J/sigma_log`, `H_GN = J.T @ W @ J = J_w.T @ J_w` (ordinary
transpose), pseudo-inverse local covariance, correlation, singular values,
rank, condition number, and status. Active asymmetric detections and upper
limits fail; no silent approximation or regularization is added. The
covariance is local, not posterior/profile/global, and `FitResult` is unchanged.

## 1. Hard constraints

1. CPU and GPU use the same model, configuration, fit, parameter space, sensitivity API, result objects, and scientific semantics. `auto`, `cpu`, and `gpu` are execution choices only.
2. CPU guidance may warn that routine fits are usually practical with about 1–2 free parameters; GPU guidance may recommend 3–8 parameter joint fits and high-dimensional Jacobians. Neither path may change the requested free set, stage the fit, use coordinate descent, or alter the objective/optimizer.
3. The validated differentiable registry contains exactly `log_Le`, `log_Lp`, `log_B`, `log_R`, `p_e`, `p_p`, `log_gamma_e_max`, and `log_gamma_p_max`. A call selects any unique ordered subset of 1–8 names.
4. L1 answers model/forward/fitting workflows. L2 exposes standalone scientific operations in P5 and model-bound sensitivity/local curvature in P6. Neither boundary exposes JAX implementation mechanics.
5. Discrete process switches, grids, EBL/PMNS tables, solver iteration controls, and prepared-response tensors are not differentiable parameters: configurable does not mean differentiable.
6. L3 and L4 deep imports are not documented as stable and carry no backward-compatibility promise.

The current eight coordinates remain the validated natural-log offsets for kinetic `Le`, kinetic `Lp`, `B`, `R`, `gamma_e,max`, and `gamma_p,max`, plus additive offsets for `p_e` and `p_p`. Fixed grids, fixed observer/EBL/PMNS metadata, default-off pp, and finite-unroll cascade assumptions remain part of the current validation boundary. P3 adds a deterministic absolute-value model configuration without changing these fitting/AD coordinates; P4 converts fitted coordinates back to physical values and a new `best_config`.

## 2. Current ownership confirmed from definitions and callers

| Current symbol/feature | Confirmed role | Public policy |
|---|---|---|
| `helena.models.c513.run_c513_forward` | unique `CANONICAL_FULL_FORWARD` | Remains the internal owner behind `OneZoneModel.forward`; `c513` is not a public concept. |
| `solve_steady_state_hadronic_cascade` | `SUPPORTED_LOW_LEVEL_INTERNAL` | L3; raw Picard loop and state are not a result contract. |
| `outer_hadronic_picard_step` | `AD_PRIMITIVE` | L3. |
| `physics_only=True` | `AD_PRIMITIVE` | L3 parameter of the Picard step; never public workflow. |
| `_earth_photon_log_sed_target` | maintained model-bound Earth photon log-SED target | L3 private owner behind `helena.sensitivity`; never a public callable. |
| `steady_state_electromagnetic_cascade_jax` | `SUPPORTED_LOW_LEVEL_INTERNAL` | Raw owner remains L3; P5 exposes only the smaller backend-neutral `cascade.solve_em` result. |
| `proton_secondary_lepton_injection_jax` | `DEVELOPMENT_ONLY` | L3. |
| `proton_secondary_lepton_transport_chain_jax` | `TEST_ONLY` | L3; no production caller was found. |
| `tests.science_baseline.run_science_baseline.collect_autodiff` | official frozen 8D collector | L4 regression evidence, not the L2 Jacobian API. |
| `tests.science_baseline.run_model_bound_derivative_reference.collect` | frozen model-bound 8D reference owner | L4 P6A evidence, not the L2 Jacobian API. |

No competing full-forward or Public dataset owner exists. `SEDDataset` is the
single Public dataset contract; internal dataset classes and NumPy/JAX
function pairs remain implementation details rather than competing APIs.

## 3. L1 namespace

P2--P4 implement all seven L1 names:

| Symbol | Minimal responsibility |
|---|---|
| `helena.configure` | Select `device="auto"|"cpu"|"gpu"` before model construction and enforce scientific float64/x64. No physics settings. |
| `helena.ModelConfig` | Validated physical, process, observer, and propagation inputs with explicit units/frames; no device, optimizer, grids, or raw responses. |
| `helena.OneZoneModel` | Own one `ModelConfig`, reuse strict prepared responses, and perform the canonical forward. |
| `helena.ForwardResult` | Stable scientific outputs described in section 6. |
| `helena.SEDDataset` | One immutable observed-SED contract covering detections, asymmetric errors, upper limits, masks/use labels, frame, and measure. |
| `helena.fit` | Fit one model/dataset with a named 1–8 parameter subset and `lbfgsb` or `migrad`. |
| `helena.FitResult` | Stable statistical/runtime result described in section 7. |

Do not add `fit_cpu`, `fit_gpu`, `CPUModel`, `GPUModel`, `fit_2params`, `fit_4params`, `fit_8params`, or `full_8d_fit`. Do not put `device` on `fit`; the model uses the process/runtime choice established before construction.

### Minimal L1 workflow

```python
import helena as hj

hj.configure(device="auto")
model = hj.OneZoneModel(hj.ModelConfig(...))
forward = model.forward()
```

P4 extends this workflow with `SEDDataset`, top-level `fit`, and `FitResult`.

`parameters` is the ordered free set. Every registry name not listed is fixed at the value represented by the model configuration; redundant public `fixed=`, `initial=`, and custom `bounds=` arguments are not exposed in v0.1. Unknown, duplicate, empty, or more-than-eight selections fail at the boundary.

P4 emits a standard performance warning for a CPU request above the routine recommendation while still executing exactly the requested joint operation. `joint_dimension == len(parameters)` is statistical metadata, never a device type.

## 4. L2 namespace

P5 implements three flat modules:

| Namespace/operation | Implemented scope |
|---|---|
| `helena.physics` | Blob-frame synchrotron, SSA coefficient, escaped electron SSA SED, internal photon density, inverse Compton, gamma-gamma transfer, photomeson, BH loss, pp production/loss, and pion/muon daughter production. |
| `helena.transport` | One generic steady-state `Q + continuous loss + local sink -> N` operation for any species. |
| `helena.cascade` | One standalone steady EM cascade with scientific inputs and a compact final result. |

SSC and EC are not duplicate kernels: `physics.inverse_compton` receives a
caller-selected target. External targets must already be blob-comoving. Raw
gamma-gamma/BH pair response tensors prevent a safe traced pair-injection
facade today, so those two operations remain deferred rather than weakening
the JAX contract.

P6 adds two flat L2 modules:

| Namespace/operation | Implemented scope |
|---|---|
| `helena.sensitivity.jvp` / `.jacobian` | P6A Earth-photon natural-log target for an ordered unique 1--8 registry subset; one immutable metadata-bearing result. |
| `helena.diagnostics.gauss_newton` | Local curvature/identifiability for active independent symmetric Gaussian SED observations only. |

The sensitivity facade preserves the frozen target, fixed support, and caller
order without exposing JAX transforms, checkpointing, or raw fixed-point
machinery. Only the named scientific diagnostic subset is documented from the
pre-existing diagnostics module; conservation ledgers remain L4.

## 5. L3/L4 policy

L3 includes raw `core.jax_backend` names (even when wrapped by P5), species
composition chains, input-contract helpers, fixed-volume coefficients, `c513`
builders/mappers, response tensors/cache I/O, Picard state, `physics_only`,
`legacy_fixed_loss`, checkpoint/fixed-point AD machinery, and full internal
PyTrees. Deep imports may continue to serve production/tests but carry no
public compatibility promise.

L4 includes FAST/science fingerprints, the old Regime-B 8D collector, the P6A
model-bound 8D reference, benchmark/profile CLIs, cross-code comparisons,
conservation ledgers, fixtures, and frozen regression assets. L4 is evidence
about L1/L2 behavior, not user functionality.

## 6. Minimal `ForwardResult` scientific contract

The public result is a small scientific view over the canonical forward, never the complete internal PyTree.

### `result.sed`

Mandatory photon fields, matching the current `canonical_quantities` handoff:

- `energy_blob_erg`: blob-comoving photon energy;
- `escaped_energy_luminosity_blob_erg_s`: escaped blob `E L_E`;
- `energy_observer_erg`: observer-frame energy;
- `intrinsic_energy_flux_observer_erg_cm2_s`: pre-EBL observer `E F_E`;
- `ebl_attenuated_energy_flux_earth_erg_cm2_s`: Earth `E F_E` after EBL.

The existing five fields above retain their high-energy-grid semantics. The
following additive fields expose the canonical complete escaped photon SED:

- `broadband_energy_blob_erg`: strictly increasing blob-comoving energy;
- `broadband_escaped_energy_luminosity_blob_erg_s`: escaped blob `E L_E`;
- `broadband_energy_observer_erg`: observer energy `delta E_blob/(1+z)`;
- `broadband_intrinsic_energy_flux_observer_erg_cm2_s`: pre-EBL observer
  `E F_E = delta^4 (E L_E)_blob/(4 pi d_L^2)`.
- `broadband_earth_source_indices`: indices into the intrinsic broadband grid
  retained by the configured Earth-propagation support;
- `broadband_energy_earth_erg`: observer energy on that Earth support;
- `broadband_ebl_transmission`: identity or Dominguez-2011 transmission;
- `broadband_ebl_attenuated_energy_flux_earth_erg_cm2_s`: Earth-observed
  escaped `E F_E` used by fitting and sensitivity.

`enable_ebl=False` preserves identity propagation. When enabled, values below
the validated EBL energy domain have transmission one, table-supported
positive-transmission nodes use the existing Dominguez-2011 owner, and nodes
above the table are not exposed or extrapolated. The low-energy grid owns
nodes through and including the fixed 1 MeV split;
the high-energy grid owns nodes strictly above it. Concatenation therefore has
no duplicated overlap node, interpolation, smoothing, or re-normalization.
The low segment is total electron/positron plus meson synchrotron after the
existing homogeneous-sphere SSA transfer. The high segment is the existing
total escaped gamma-ray SED after internal gamma-gamma transfer. No broadband
smoothing or re-normalization is applied.

### `result.components`

A read-only mapping containing `energy_blob_erg` and these intrinsic
high-energy blob `E L_E` arrays:

- `electron_synchrotron_energy_luminosity_blob_erg_s`;
- `ssc_energy_luminosity_blob_erg_s`;
- `external_compton_energy_luminosity_blob_erg_s`;
- `proton_synchrotron_energy_luminosity_blob_erg_s`;
- `neutral_pion_decay_energy_luminosity_blob_erg_s`;
- `pion_synchrotron_energy_luminosity_blob_erg_s`;
- `muon_synchrotron_energy_luminosity_blob_erg_s`;
- `total_intrinsic_energy_luminosity_blob_erg_s`.

Scientifically defined disabled components are exact-zero arrays. External
Compton is zero while the P3 model has no public external field. Raw iteration
components are excluded.

The mapping also contains canonical broadband energy grids and the retained
aggregate process groups: transported-lepton synchrotron,
transported-lepton SSC, proton synchrotron, neutral-pion decay,
charged-pion synchrotron, and muon synchrotron. Each group is available as
blob `E L_E`, intrinsic-observer `E F_E`, and EBL-attenuated Earth `E F_E`.

Nine additive source-resolved groups close to the escaped total:
primary-electron synchrotron, primary-electron inverse Compton (with the
frozen compatibility key `primary_electron_ssc`), Bethe--Heitler secondary
radiation, photomeson secondary radiation, aggregate gamma-gamma reprocessed
radiation, proton synchrotron, neutral-pion decay, charged-pion synchrotron,
and muon synchrotron. The first five freeze the final transport, target, SSA,
and gamma-gamma transfer state rather than rerunning the cascade.

Their exact base keys are `primary_electron_synchrotron`,
`primary_electron_ssc`, `bethe_heitler_secondary_radiation`,
`photomeson_secondary_radiation`, `gamma_gamma_reprocessed_radiation`,
`proton_synchrotron`, `neutral_pion_decay`, `charged_pion_synchrotron`, and
`muon_synchrotron`. Each base receives blob-luminosity, intrinsic-observer
flux, and EBL-attenuated Earth-flux suffixes.

Mechanism-resolved aliases separately expose primary-electron inverse Compton
and synchrotron/inverse-Compton pairs for Bethe--Heitler, photomeson, and
gamma-gamma secondary populations. New documentation and plots use
"Primary-electron inverse Compton" rather than calling that field pure SSC,
because its converged seed field contains every enabled synchrotron source.
The transported-lepton aggregate groups remain compatibility views and are
not additional terms in the nine-group source-resolved closure.

The additional mechanism base keys are
`primary_electron_inverse_compton`, `bethe_heitler_pair_synchrotron`,
`bethe_heitler_pair_inverse_compton`,
`photomeson_secondary_pair_synchrotron`,
`photomeson_secondary_pair_inverse_compton`,
`gamma_gamma_reprocessed_pair_synchrotron`, and
`gamma_gamma_reprocessed_pair_inverse_compton`.

### `result.particles`

At minimum electron, positron, and proton entries expose:

- `gamma_total_energy_over_rest_energy`;
- `primary_injection_blob_dgamma_cm3_s` (`Q(gamma)`, source per unit gamma);
- `density_blob_dgamma_cm3` (`N(gamma)`, steady density per unit gamma).

Sources and densities are never interchangeable. Pion/muon work distributions
remain internal because the current stable handoff does not give them the same
compact, consistent species contract.

### `result.neutrinos`

Mandatory fields are blob energy, source flavor injection per energy, observer energy, observer unmixed flavor `E F_E`, and Earth mixed flavor `E F_E`, including all-flavor sums. Source and Earth products remain separately named.

### `result.diagnostics`

The exact fields are `converged`, `outer_iterations`, `inner_converged`,
`inner_iterations`, `maximum_outer_relative_residual`, `finite_nonnegative`,
and `conservation_checks_passed`. Runtime/device facts come from `configure`.
Raw fixed-point state, every iteration, responses, fingerprints, and full
development ledgers are not stable fields.

## 7. Minimal `FitResult` contract

| Field | Contract |
|---|---|
| `best_parameters` | Ordered mapping from each free registry name to its physical best-fit value. |
| `best_config` | New immutable physical `ModelConfig`; the input model/config is unchanged. |
| `free_parameters` | Ordered tuple of the requested 1–8 names. |
| `fixed_parameters` | Mapping of all remaining registry names to their fixed registry values. |
| `joint_dimension` | `len(free_parameters)`; statistical dimension only. |
| `nll` | Final finite scalar negative log likelihood. |
| `optimizer` | Backend-independent name plus settings: `lbfgsb` or `migrad`. |
| `success`, `status` | Boolean success and normalized status/message without hiding native optimizer diagnostics. |
| `iterations`, `evaluations` | Optional iteration count and explicit function/gradient/unique-expensive evaluation counts. |
| `prediction` | Selected dataset indices, observer energy, and aligned Earth photon `E F_E`; not a full forward result. |
| `residuals` | Statistic-aware detection residuals and upper-limit status/NLL contributions; never fabricated Gaussian residuals for limits. |
| `uncertainties` | `None` unless a named, scientifically meaningful method produced them. |
| `covariance` | `None` unless a named method produced it; a local Gauss–Newton matrix/covariance is explicitly labelled diagnostic, not a profile-likelihood interval. |
| `runtime_seconds`, `backend`, `device`, `x64` | Actual execution facts, not requested labels alone. |

L-BFGS-B and MIGRAD must populate the same public fields. Optimizer-specific EDM/call-limit details may live in a namespaced diagnostics mapping without changing `FitResult` semantics.

## 8. Direct answers to the required P1 questions

1. The implemented minimal L1 workflow is configure → construct config/model → forward → stable result; dataset/fit follows in P4.
2. The implemented L2 workflow composes P5 standalone physics/transport/EM cascade with P6 model-bound JVP/Jacobian and optional local Gaussian curvature diagnostics.
3. Device selection belongs at the process/runtime boundary before model construction, not in physics config or optimizer calls.
4. Two CPU/GPU APIs would permit divergent workflows and semantics; backend selection is only execution policy for one JAX program.
5. The eight validated names form one ordered registry with coordinate, units/meaning, and validation metadata implemented once in P2; model/caller-specific fit bounds are not invented here.
6. L1 `parameters` lists free names; omitted registry names stay fixed at model-config values.
7. P6 sensitivity accepts the identical registry names and preserves caller ordering.
8. Discrete switches change graph/composition and have no continuous derivative, so they live in model/process configuration.
9. Stable photon, component, particle, neutrino, and compact diagnostic outputs are defined in section 6.
10. Stable statistical, prediction, residual, runtime, and provenance fields are defined in section 7.
11. P5 promotes the safe validated standalone concepts, pp, generic transport, and EM cascade through backend-neutral JAX-backed facades with explicit units/frames/measures.
12. Raw kernels, Picard/fixed-point states, `physics_only`, prepared tensors, grid remapping, C513 mappers, and AD machinery remain L3.
13. Frozen fingerprints/collectors, benchmarks, profiles, cross-code evidence, fixtures, and scientific ledgers remain L4.
14. All current `c513` names remain internal; public names describe a configurable one-zone model, not a validation grid/scenario.
15. Ordinary L1/L2 users do not need to know prepared responses. Any future control must be opaque and model-bound; raw tensors never become contract fields.
16. Runtime/registry are complete in P2, model/config/forward in P3, dataset/fit/result in P4, standalone physics/transport/EM cascade in P5, the derivative target in P6A, sensitivity/local curvature in P6, and contract freeze in P7.

## 9. Explicitly deferred outside v0.1

P5 does not implement public pair-response preparation, gamma-gamma/BH pair
injection, general cooling/timescale convenience families, custom physical fit
bounds, selectable EBL models or host-to-blob external-field transforms,
component/neutrino/particle derivatives, scalar public gradient, advanced AD
strategy controls, automatic `FitResult` covariance, non-Gaussian curvature
approximations, or full hadronic cascade entrypoints. Quickstarts, examples,
packaging, CI, and release engineering follow the freeze but do not extend the
v0.1 scientific surface. Detailed open decisions remain in the developer
archive.
