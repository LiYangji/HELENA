# P5 Standalone Physics Contract

Status: PUBLIC API P5 PASS. Acceptance evidence remains in the preserved
developer archive and is not distributed in the public release tree.

## Boundary

P5 adds three flat L2 modules:

```python
import helena.physics as physics
import helena.transport as transport
import helena.cascade as cascade
```

They call the maintained float64 JAX production owners on either CPU or GPU.
No NumPy/reference twin, `_jax` name, C513 configuration, prepared-response
tensor, finite-volume work array, Picard state, or validation object is public.
The seven root L1 names and their semantics are unchanged.

All particle gamma values are total energy divided by rest energy. Particle
densities are `N(gamma)` in cm^-3 per gamma and particle sources are `Q(gamma)`
in cm^-3 s^-1 per gamma. Photon target densities are angle-integrated isotropic
`dn_ph/dE` in cm^-3 erg^-1. Unless a name explicitly says observer/Earth, every
P5 input and output is blob-comoving. P5 introduces no observer-frame output.

## Public surface

### `helena.physics`

| Public operation | Production owner | Return contract |
|---|---|---|
| `electron_synchrotron(E, gamma, N, *, magnetic_field_G, radius_cm)` | `electron_synchrotron_sed_jax` | Electron plus positron optically thin blob `E L_E` [erg s^-1]. |
| `proton_synchrotron(E, gamma, N, *, magnetic_field_G, radius_cm)` | `synchrotron_sed_jax`, fixed proton mass/charge | Proton optically thin blob `E L_E` [erg s^-1]. |
| `charged_pion_synchrotron(E, gamma, N, *, magnetic_field_G, radius_cm)` | `synchrotron_sed_jax`, fixed charged-pion mass/charge | Sum of pi+ and pi- blob `E L_E` [erg s^-1]. |
| `muon_synchrotron(E, gamma, N, *, magnetic_field_G, radius_cm)` | `synchrotron_sed_jax`, fixed muon mass/charge | Both-charge/helicity blob `E L_E` [erg s^-1]. |
| `inverse_compton(E_s, gamma_e, N_e, E_target, n_target, *, radius_cm)` | `inverse_compton_sed_jax` | Isotropic blob `E_s L_Es` [erg s^-1]. The supplied target selects SSC or EC. |
| `electron_synchrotron_absorption_coefficient(E, gamma_e, N_e, *, magnetic_field_G)` | `synchrotron_absorption_coefficient_jax`, fixed electron mass/charge | Signed `alpha_nu` [cm^-1]; negative population inversion is not hidden. |
| `electron_synchrotron_self_absorbed_sed(E, gamma_e, N_e, *, magnetic_field_G, radius_cm)` | `electron_synchrotron_self_absorbed_sed_jax` | Escaped/emergent homogeneous-sphere electron SSA `E L_E` [erg s^-1]; population inversion is rejected by the owner. |
| `internal_photon_density(E, produced_sed, alpha, *, radius_cm)` | `produced_sed_to_internal_photon_density_jax` | Homogeneous-sphere internal `dn_ph/dE` [cm^-3 erg^-1] for non-negative local `alpha`. |
| `gamma_gamma_absorption_coefficient(E_gamma, E_target, n_target)` | `gamma_gamma_absorption_coefficient_jax` | Isotropic local coefficient [cm^-1]. |
| `gamma_gamma_optical_depth(E_gamma, E_target, n_target, *, radius_cm)` | `gamma_gamma_optical_depth_jax` | Dimensionless diameter depth `2 R alpha`; not another geometry's path depth. |
| `gamma_gamma_absorbed_sed(intrinsic_sed, alpha, *, radius_cm)` | `gamma_gamma_absorbed_sed_jax` | Escaped mixed-emitter sphere `E L_E` [erg s^-1], not a foreground screen. |
| `photomeson_interaction_rate(gamma_p, E_target, n_target)` | `photomeson_interaction_rate_jax` | Sim-B event rate [s^-1]. |
| `photomeson_energy_loss_rate(gamma_p, E_target, n_target)` | `photomeson_energy_loss_rate_jax` | Fractional initial-nucleon loss `-dot(gamma_p)/gamma_p` [s^-1]. |
| `photomeson_pion_injection(gamma_pi, gamma_p, N_p, E_target, n_target)` | `photomeson_pion_injection_jax` | Named pi0, pi+, pi- `Q(gamma_pi)` fields [cm^-3 s^-1 per gamma]. |
| `bethe_heitler_proton_energy_loss_rate(gamma_p, E_target, n_target)` | `bethe_heitler_proton_energy_loss_rate_jax` | Fractional proton loss [s^-1]; not pair injection or `legacy_fixed_loss`. |
| `proton_proton_pion_injection_and_loss(gamma_pi, gamma_p, N_p, n_cold)` | `proton_proton_pion_injection_and_loss_jax` | Named fractional proton loss plus pi0/pi+/pi- sources; fixed grids stay outside tracing. |
| `neutral_pion_decay_photon_injection(E_gamma, gamma_pi, Q_pi0_decay)` | `neutral_pion_decay_photon_injection_jax` | Photon `Q(E)` [cm^-3 s^-1 erg^-1]. |
| `charged_pion_decay_muon_injection(gamma_mu, gamma_pi, Q_pi+_decay, Q_pi-_decay)` | `charged_pion_decay_muon_injection_jax` | Named mu+L, mu+R, mu-L, mu-R sources [cm^-3 s^-1 per gamma]. |
| `charged_pion_decay_neutrino_injection(E_nu, gamma_pi, Q_pi+_decay, Q_pi-_decay)` | `charged_pion_decay_neutrino_injection_jax` | Named direct nu_mu and anti-nu_mu sources [cm^-3 s^-1 erg^-1]. |
| `muon_decay_electron_positron_injection(gamma_e, gamma_mu, four_helicity_decay_rates)` | `muon_decay_electron_positron_injection_jax` | Named positron and electron sources [cm^-3 s^-1 per gamma]. |
| `muon_decay_neutrino_injection(E_nu, gamma_mu, four_helicity_decay_rates)` | `muon_decay_neutrino_injection_jax` | Named nu_e, anti-nu_e, nu_mu, anti-nu_mu sources [cm^-3 s^-1 erg^-1]. |

`inverse_compton` is the only IC radiation implementation. SSC is constructed
by passing an internally produced synchrotron target; EC is constructed by
passing a target already transformed into the blob frame. P5 performs no
AGN-to-blob transform and creates no `ssc` or `ec` duplicate.

The six multi-output physics records are stdlib immutable `NamedTuple` PyTrees:
`PionInjection`, `ProtonProtonResult`, `ChargedPionMuonInjection`,
`PionDecayNeutrinoInjection`, `MuonDecayElectronPositronInjection`, and
`MuonDecayNeutrinoInjection`. There is no result base class or dictionary
framework.

### `helena.transport`

```python
transport.solve_steady_state(
    particle_gamma,
    injection_dgamma_cm3_s,
    continuous_loss_rate_dgamma_s,
    sink_rate_s_inv,
)
```

This one species-neutral equation covers electrons, positrons, protons,
charged pions, and muons when the caller supplies their scientific loss and
sink rates. It returns only steady `N(gamma)` [cm^-3 per gamma]. The high-energy
boundary has no inflow and the low-energy face has outflow. Face positions,
matrix coefficients, recurrence state, and ledger buffers remain internal.
Species-specific composition wrappers remain L3 because they add no transport
equation and return model-chain-specific escape/decay products.

### `helena.cascade`

`cascade.solve_em(...)` accepts identical full-support blob photon grids (the
two historical grid arguments must agree), total electron and
positron sources, a prescribed proton distribution, a prescribed gamma source,
one already-comoving external target, `B`, `R`, adiabatic/escape rates, and an
optional extra continuous lepton loss. The two electron source arrays replace
the internal provenance split into primary, muon-decay, and Bethe-Heitler
buffers; the owner sees the same additive sources. The retained
`enable_synchrotron_self_compton` input enables internal IC on the complete
internal photon density. External IC reads its separate fixed reservoir once.

The facade returns `EMCascadeResult`, an immutable named PyTree containing final
electron/positron densities, intrinsic and escaped total-photon `E L_E` on the
full grid (historical field names contain `gamma`), gamma-gamma
diameter depth, and compact convergence facts. It omits raw next/relaxed
iterates, residual histories, pair-remap fields, response tensors, and ledgers.
Fixed-grid pair responses are prepared automatically on the host and the
dynamic calculation runs on the configured JAX device. Response preparation
itself is intentionally not a JIT/autodiff operation.

The full hadronic cascade remains internal. `OneZoneModel.forward()` is still
the sole public complete lepto-hadronic source-to-Earth path.

## Candidate audit

The audit examined 43 existing candidate symbols. Classification applies to
the scientific operation: a `PUBLIC_P5` row is reached only through the clean
facade name; its raw production symbol remains an unsupported deep import.
Every production row has existing focused NumPy/JAX, analytic, conservation,
chain, or frozen-baseline evidence in `tests/` and is generic unless the table
says otherwise.

| # | Candidate production symbol | Signature / scientific I/O | Trace and dependency audit | Classification / reason |
|---:|---|---|---|---|
| 1 | `electron_synchrotron_sed_jax` | `(E, gamma_e, N_e; B, R) -> E L_E` | Fully traced; no preparation; radiation parity tests. | `PUBLIC_P5` as `electron_synchrotron`. |
| 2 | `synchrotron_sed_jax` | `(E, gamma, N; B, R, mass, charge) -> E L_E` | Fully traced; generic classical kernel. | `PUBLIC_P5` through fixed proton/pion/muon scientific wrappers; mass/charge are not recalculated. |
| 3 | `inverse_compton_sed_jax` | `(E_s, gamma_e, N_e, E_t, n_t; R) -> E_s L_Es` | Fully traced; no prepared response; Klein-Nishina parity tests. | `PUBLIC_P5` as one target-driven `inverse_compton`. |
| 4 | `electron_inverse_compton_loss_rate_dgamma_s_jax` | `(gamma_e, E_t, n_t) -> -dot(gamma)` [s^-1] | Fully traced and validated; chiefly a composition coefficient today. | `DEFER_FUTURE_PUBLIC` until a concrete standalone cooling workflow fixes naming alongside other loss providers. |
| 5 | `synchrotron_absorption_coefficient_jax` | `(E, gamma, N; B, mass, charge) -> signed alpha` [cm^-1] | Fully traced; derivative-on-grid semantics validated. | `PUBLIC_P5` with fixed electron constants and explicit population-inversion caveat. |
| 6 | `synchrotron_self_absorbed_sed_jax` | `(E, gamma, N; B, R, mass, charge) -> E L_E` | Fully traced for valid non-inverted inputs. | `KEEP_L3_INTERNAL`; generic caller-supplied mass/charge transfer remains a deep implementation operation. |
| 7 | `electron_synchrotron_self_absorbed_sed_jax` | Electron convenience form of row 6. | Fully traced; validated on complete SSC-TH/KN arrays with the physical maser guard intact. | `PUBLIC_P5` as `electron_synchrotron_self_absorbed_sed`; one direct escaped-SED facade. |
| 8 | `proton_synchrotron_self_absorbed_sed_jax` | Proton convenience form of row 6. | Fully traced; specialized convenience only. | `DEFER_FUTURE_PUBLIC`; no demonstrated L2 need. |
| 9 | `produced_sed_to_internal_photon_density_jax` | `(E, E L_E, alpha; R) -> dn_ph/dE` [cm^-3 erg^-1] | Fully traced; homogeneous-sphere geometry validated. | `PUBLIC_P5` as `internal_photon_density`; enables explicit SSC target construction. |
| 10 | `synchrotron_sed_to_internal_photon_density_jax` | Compatibility signature identical to row 9. | Thin alias of row 9. | `KEEP_L3_INTERNAL`; duplicate name adds no science. |
| 11 | `leptonic_radiation_chain_jax` | Electron synchrotron + SSA + SSC + EC mapping. | Traced composition mapping with internal component layout. | `KEEP_L3_INTERNAL`; users compose the public operations, not a raw chain dictionary. |
| 12 | `gamma_gamma_absorption_coefficient_jax` | `(E_gamma, E_t, n_t) -> alpha` [cm^-1] | Fully traced; no response tensor. | `PUBLIC_P5` as `gamma_gamma_absorption_coefficient`. |
| 13 | `gamma_gamma_optical_depth_jax` | `(E_gamma, E_t, n_t; R) -> 2 R alpha` | Fully traced; homogeneous diameter geometry. | `PUBLIC_P5` as `gamma_gamma_optical_depth`. |
| 14 | `gamma_gamma_absorbed_sed_jax` | `(E L_E, alpha; R) -> escaped E L_E` | Fully traced; mixed-sphere transfer. | `PUBLIC_P5` as `gamma_gamma_absorbed_sed`. |
| 15 | `gamma_gamma_pair_injection_jax` | Photon densities -> `(Q_e+, Q_e-)` per gamma. | Eager auto-prepares; traced calls require a raw fixed response. | `DEFER_FUTURE_PUBLIC`; hiding the response would silently remove the owner's traced contract. |
| 16 | `prepare_gamma_gamma_pair_response_jax` | Three fixed grids -> rank-3 response tensor. | Host content cache; concrete grids only; tensor layout is implementation. | `KEEP_L3_INTERNAL`. |
| 17 | `photomeson_interaction_rate_jax` | `(gamma_p, E_t, n_t) -> event rate` [s^-1] | Fully traced; Sim-B response. | `PUBLIC_P5` as `photomeson_interaction_rate`. |
| 18 | `photomeson_energy_loss_rate_jax` | Same inputs -> fractional loss [s^-1]. | Fully traced; validated energy-weighted response. | `PUBLIC_P5` as `photomeson_energy_loss_rate`. |
| 19 | `photomeson_cooling_time_jax` | Same inputs -> reciprocal loss [s]. | Fully traced; derived convenience only. | `KEEP_L3_INTERNAL`; users can take the reciprocal with their own zero policy. |
| 20 | `photomeson_pion_injection_jax` | `(gamma_pi, gamma_p, N_p, E_t, n_t) -> Q_pi0,Q_pi+,Q_pi-` | Fully traced; charge/multiplicity prescription validated. | `PUBLIC_P5` with a named result. |
| 21 | `bethe_heitler_proton_energy_loss_rate_jax` | `(gamma_p, E_t, n_t) -> fractional loss` [s^-1] | Fully traced; no response preparation. | `PUBLIC_P5`; `legacy_fixed_loss` remains absent. |
| 22 | `bethe_heitler_pair_injection_jax` | Proton/photon densities -> `(Q_e+, Q_e-)` per gamma. | Eager auto-prepares; tracing requires a raw rank-3 response. | `DEFER_FUTURE_PUBLIC` pending opaque preparation semantics. |
| 23 | `prepare_bethe_heitler_pair_response_jax` | Three fixed grids -> rank-3 pair response. | Expensive host cache; concrete grids only; fingerprint/layout internal. | `KEEP_L3_INTERNAL`. |
| 24 | `proton_proton_inelastic_cross_section_jax` | `gamma_p -> sigma_pi0,pi+,pi-` [cm^2] | Eager validated low-level prescription piece. | `KEEP_L3_INTERNAL`; public workflow needs production/loss, not a subkernel. |
| 25 | `proton_proton_pion_response_jax` | Fixed pion/proton grids -> rank-3 response. | Host-cached grid response; no dynamic density. | `KEEP_L3_INTERNAL`; raw response stays hidden. |
| 26 | `proton_proton_fractional_energy_loss_rate_jax` | `(gamma_pi_grid, gamma_p_grid, n_cold) -> loss` | Fixed grids outside tracing; loss depends on the same production response. | `KEEP_L3_INTERNAL`; the combined public operation preserves the coupled prescription. |
| 27 | `proton_proton_pion_injection_and_loss_jax` | `(gamma_pi, gamma_p, N_p, n_cold) -> loss,Q_pi0,Q_pi+,Q_pi-` | Fixed grids, traced dynamic spectrum/target; cached response hidden. | `PUBLIC_P5` with one named result. |
| 28 | `neutral_pion_decay_photon_injection_jax` | `(E_gamma, gamma_pi, Q_pi0_decay) -> Q_gamma(E)` | Fully traced; two-body decay parity/conservation tests. | `PUBLIC_P5`. |
| 29 | `charged_pion_decay_muon_injection_jax` | Parent decay rates -> four charge/helicity muon sources. | Fully traced; helicity is necessary physics, not scratch state. | `PUBLIC_P5` with named fields. |
| 30 | `charged_pion_decay_neutrino_injection_jax` | Parent decay rates -> nu_mu, anti-nu_mu per energy. | Fully traced; direct neutrino flavor semantics validated. | `PUBLIC_P5` with named fields. |
| 31 | `muon_decay_electron_positron_injection_jax` | Four helicity decay rates -> positron/electron `Q(gamma)`. | Fully traced; Michel scaling validated. | `PUBLIC_P5` with named charge outputs. |
| 32 | `muon_decay_neutrino_injection_jax` | Four helicity decay rates -> four flavor/antiflavor sources. | Fully traced; pre-oscillation source semantics. | `PUBLIC_P5` with named flavor outputs. |
| 33 | `charged_pion_decay_time_jax` | `gamma_pi -> t_decay` [s] | Fully traced; stable low-level timescale. | `DEFER_FUTURE_PUBLIC`; no standalone timescale workflow requested by the minimal surface. |
| 34 | `charged_pion_synchrotron_cooling_time_jax` | `(gamma_pi, B) -> t_syn` [s] | Fully traced. | `DEFER_FUTURE_PUBLIC`; coordinate with a future loss/timescale family. |
| 35 | `muon_decay_time_jax` | `gamma_mu -> t_decay` [s] | Fully traced. | `DEFER_FUTURE_PUBLIC`; same policy as row 33. |
| 36 | `muon_synchrotron_cooling_time_jax` | `(gamma_mu, B) -> t_syn` [s] | Fully traced. | `DEFER_FUTURE_PUBLIC`; same policy as row 34. |
| 37 | `steady_state_finite_volume_solution_jax` | `(gamma, Q, -dot(gamma), sink) -> N(gamma)` | Fully JIT/AD compatible with implicit transpose solve; no preparation. | `PUBLIC_P5` as `transport.solve_steady_state`. |
| 38 | `electron_positron_steady_state_transport_jax` | Two sources + built-in losses -> densities/escape sources. | Fully traced; composition-specific tuple. | `KEEP_L3_INTERNAL`; generic public equation covers both charges. |
| 39 | `proton_steady_state_transport_jax` | Proton source + named model losses -> density/escape. | Fully traced; composition-specific loss list. | `KEEP_L3_INTERNAL`; generic public equation avoids freezing one process list. |
| 40 | `charged_pion_steady_state_transport_jax` | Two charge sources + built-in synchrotron/decay -> densities/decays. | Fully traced; model-chain convenience. | `KEEP_L3_INTERNAL`; caller can compose public times/losses when a workflow requires it. |
| 41 | `helicity_resolved_muon_steady_state_transport_jax` | Four sources + built-in synchrotron/decay -> eight outputs. | Fully traced; model-chain convenience. | `KEEP_L3_INTERNAL`; avoids an eight-field public transport result. |
| 42 | `steady_state_electromagnetic_cascade_jax` | Fixed scientific sources/targets -> large final/iterative mapping. | Dynamic JAX solve after host response preparation; generic, not C513-specific. | `PUBLIC_P5` only through reduced `cascade.solve_em` and `EMCascadeResult`. |
| 43 | `solve_steady_state_hadronic_cascade` | Production configuration + prepared responses -> raw outer state. | C513/static-layout-facing Picard owner with responses and ledgers. | `KEEP_L3_INTERNAL`; complete public path remains `OneZoneModel.forward()`. |

Totals: 21 production candidate symbols map to 23 public P5 functions, 8 are
`DEFER_FUTURE_PUBLIC`, 14 remain `KEEP_L3_INTERNAL`, and none are deferred to
P6 or moved to L4. The 23 functions are 21 physics operations, one transport
operation, and one EM-cascade operation.

## JAX and device contract

`configure(device="cpu")` and `configure(device="gpu")` select execution for
the same imports, signatures, result schemas, equations, and float64 arrays.
P5 contains no `np.asarray`, Python scalar extraction, mutation, host callback,
backend subclass, or reference-backend dispatch in a dynamic wrapper. The only
host phase is existing fixed-grid preparation in the EM cascade and the two
deferred pair-injection owners. Formal gradient/JVP/Jacobian functions remain
P6 work; P5 only verifies wrapper transparency for representative radiation,
hadronic, and transport calls.

## Deliberate deferrals

- Gamma-gamma and Bethe-Heitler pair injection need an opaque preparation
  design before they can preserve traced execution without exposing tensors.
- Manual prepared-response/cache lifecycle control remains undecided.
- General cooling/timescale convenience names wait for one real workflow.
- Non-identity EBL and AGN/host-to-blob external-field transformations retain
  the existing public-policy deferral.
- No sensitivity/Jacobian API, full hadronic cascade facade, custom fit bounds,
  quickstart, benchmark, packaging, or release work is part of P5.

## Occam audit

Added public modules: `physics`, `transport`, `cascade`.

Added public functions: 23. Added public result dataclasses: none. Seven small
stdlib named results are necessary only where positional tuples would confuse
charge, helicity, flavor, loss, or cascade fields. No manager, registry,
factory, base process, dispatcher, plugin, CPU/GPU split, NumPy twin, cache
architecture, C513 input, response tensor, or raw iterative result was added.
