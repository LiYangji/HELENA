# P3 Model / Forward Contract

## Audited internal boundary

`helena.models.c513.run_c513_forward` remains the only complete physics
owner. The public wrapper can reuse it without changing physics, the two
validated offset mappers, the solver, or the AD graph.

| Current input/state | P3 treatment |
|---|---|
| Physical `B`, `R`, electron/proton kinetic luminosity, injection index, and exponential cutoff `gamma_max` | Public absolute values; deterministically converted through the existing two mappers. |
| Redshift, Doppler factor, luminosity distance | Required public observer values; no cosmology is inferred. |
| SSC, internal gamma-gamma, dynamic/off Bethe–Heitler, cold-target pp, pion/muon synchrotron photons | Existing scientifically defined L1 controls. |
| Particle/photon/neutrino grids, quadrature, split energy, escape-rate construction, solver strategy | Fixed C513 setup; internal. |
| Electron/proton `gamma_min` | Fixed at 1.0 by the validated injection normalization and fixed grids; not safely mapper-configurable. |
| External photon target, EC, external gamma-gamma | Current setup is a blob-frame exact-zero placeholder; no meaningful public target contract yet. |
| EBL table | P3 uses explicit identity transmission. Table/domain selection remains a later propagation contract. |
| Neutrino propagation | Always applies the current fixed flavor-propagation contract; no new switch. |
| BH/gamma-gamma responses and fingerprints | Prepared internally using strict local-cache validation; never result fields or implicit builds. |

The full internal result includes every Picard iteration, target components,
particle chains, response-dependent work arrays, metrics, ledgers, checks, and
observer propagation. Only the quantity-aware leaves listed below are stable
L1 results. Raw iteration state, `physics_only`, solver scratch state,
responses, fingerprints, and validation ledgers remain L3/L4.

## `ModelConfig`

`ModelConfig` is one frozen, slotted dataclass. All physical values are
absolute; units are encoded in field names.

| Field | Unit / meaning |
|---|---|
| `magnetic_field_G` | Blob magnetic field, G; positive. |
| `radius_cm` | Homogeneous blob radius, cm; positive. |
| `redshift` | Source redshift; finite and non-negative. |
| `doppler_factor` | Observer Doppler factor; positive. |
| `luminosity_distance_cm` | Explicit luminosity distance, cm; positive. |
| `electron_kinetic_injection_luminosity_erg_s` | Electron kinetic injection luminosity, erg s^-1; positive. |
| `electron_injection_index` | Electron power-law index; finite. |
| `electron_gamma_max` | Electron exponential-cutoff scale; greater than fixed `gamma_min=1`. |
| `proton_kinetic_injection_luminosity_erg_s` | Proton kinetic injection luminosity, erg s^-1; positive. |
| `proton_injection_index` | Proton power-law index; finite. |
| `proton_gamma_max` | Proton exponential-cutoff scale; greater than fixed `gamma_min=1`. |
| `enable_ssc` | Include SSC emission and cooling; default `True`. |
| `enable_internal_gamma_gamma` | Include internal gamma-gamma absorption and its pair injection; default `True`. `False` leaves the synchrotron target, p-gamma, and external gamma-gamma semantics unchanged. |
| `enable_bethe_heitler` | Select existing dynamic BH or off semantics; default `True`. |
| `enable_proton_proton` | Include existing cold-target pp chain; default `False`. |
| `cold_proton_density_cm3` | Cold-proton density, cm^-3; exactly zero when pp is off and positive when on. |
| `enable_pion_synchrotron_photons` | Include pion synchrotron photon source; default `True`. |
| `enable_muon_synchrotron_photons` | Include muon synchrotron photon source; default `True`. |
| `enable_ebl` | Apply the validated Dominguez-2011 EBL owner to Earth photon outputs; default `False`. Below its table energy transmission is one, inside it uses the table, and above it has no Public Earth support. |

No device, grid, solver, response, optimizer, fitting coordinate, external
field, or selectable EBL-table field is present. Photomeson production remains intrinsic
to the current hadronic model: P3 adds no new physics switch.

The first four absolute/reference ratios become the existing natural-log
offsets `(log_Le, log_Lp, log_B, log_R)`. Physical indices become the existing
additive offsets, and cutoff ratios become the existing natural-log offsets.
The public layer calls `c513_configuration_from_log_offsets` and
`c513_configuration_from_injection_shape_offsets` in canonical registry order.
At the exact reference point it preserves the original configuration arrays
bitwise instead of needlessly re-normalizing an identity injection. Process
booleans are then assigned to their existing configuration keys. The mapper
definitions and coordinate semantics are unchanged.

## `OneZoneModel`

`OneZoneModel(config, *, cache_dir=None)` stores the immutable config, builds
the existing setup, performs the mapping above, and exposes only `forward()`.
On the first forward it calls `prepare_c513_responses(..., allow_build=False)`;
`cache_dir=None` delegates first to `HELENA_CACHE_DIR` and then to the
platform-standard user cache directory.
Exact cache hits and the in-memory gamma-gamma response are reused by later
forwards. A miss is not rebuilt implicitly. No response tensor is public.

Every forward calls `run_c513_forward` with a fresh solver state and the
configured identity or fixed Dominguez-2011 EBL transmission, then extracts a
`ForwardResult`. Only prepared/static data
is cached, so repeated forwards cannot inherit physics state. CPU and GPU use
the identical object and method; `configure` remains the sole device policy.

## `ForwardResult`

`ForwardResult` is one frozen, slotted dataclass containing read-only mappings
and read-only host NumPy arrays.

- `sed`: `energy_blob_erg`,
  `escaped_energy_luminosity_blob_erg_s`, `energy_observer_erg`,
  `intrinsic_energy_flux_observer_erg_cm2_s`, and
  `ebl_attenuated_energy_flux_earth_erg_cm2_s` retain their high-energy-grid
  semantics. The additive `broadband_*` fields expose the complete canonical
  low-plus-high escaped photon SED in the blob, intrinsic observer, and
  EBL-attenuated Earth frames, including explicit Earth source indices and
  transmission.
- `components`: one shared `energy_blob_erg` plus legacy intrinsic high-energy
  blob-frame `E L_E` arrays for electron synchrotron, SSC, external Compton,
  proton synchrotron, neutral-pion decay, pion synchrotron, muon synchrotron,
  and total. Additive canonical broadband escaped process groups expose
  transported-lepton synchrotron, transported-lepton SSC, proton synchrotron,
  neutral-pion decay, charged-pion synchrotron, and muon synchrotron in blob,
  intrinsic-observer, and EBL-attenuated Earth quantities. Scientifically
  defined disabled components are exact-zero arrays; external Compton is zero
  while the public external field is absent. The additive post-convergence
  diagnostic mapping additionally separates primary-electron synchrotron,
  primary-electron inverse Compton (with `primary_electron_ssc` retained as a
  frozen compatibility key), Bethe--Heitler secondary-pair radiation,
  photomeson secondary-pair radiation, aggregate gamma-gamma reprocessed-pair
  radiation, proton synchrotron, neutral-pion decay, charged-pion synchrotron,
  and muon synchrotron. These nine source-resolved groups are additive. The
  mapping also exposes mechanism-resolved synchrotron and inverse-Compton
  aliases for the primary and secondary lepton populations. It freezes the
  converged cooling, escape, target, SSA, and gamma-gamma transfer state, so
  these arrays do not rerun or alter the self-consistent cascade. The legacy
  transported-lepton groups remain for compatibility but are not separate
  additive terms alongside the resolved groups.
- `particles`: `electron`, `positron`, and `proton`, each with
  `gamma_total_energy_over_rest_energy`,
  `primary_injection_blob_dgamma_cm3_s`, and
  `density_blob_dgamma_cm3`. These are blob-frame quantities per unit gamma.
- `neutrinos`: blob energy and flavor source injection per energy, observer
  energy and unmixed flavor `E F_E`, and Earth mixed flavor `E F_E`, including
  `all_flavor` sums.
- `diagnostics`: `converged`, `outer_iterations`, `inner_converged`,
  `inner_iterations`, `maximum_outer_relative_residual`,
  `finite_nonnegative`, and `conservation_checks_passed`, plus source-resolved
  particle, synchrotron, inverse-Compton, blob-escape, and Earth-escape closure
  errors when that diagnostic is available. Gamma-gamma parent ancestry is
  explicitly reported as aggregate rather than assigned to a false parent.

Blob photon/component spectra are `E L_E` in erg s^-1. Observer and Earth
spectra are `E F_E` in erg cm^-2 s^-1. With `enable_ebl=False`, Earth photon
values equal intrinsic observer values bitwise; with it enabled, Earth fields
use the shared non-extrapolated Dominguez-2011 support and attenuation.

The broadband grid owns all low-energy nodes through and including the fixed
1 MeV synchrotron split, followed by high-energy nodes strictly above the
split. It is strictly increasing and uses no interpolation. The low segment is
the existing total electron/positron plus meson synchrotron source after the
same homogeneous-sphere SSA transfer used by the internal photon field; the
high segment is the existing total escaped gamma-ray SED after internal
gamma-gamma transfer. The observer grid is `delta E_blob/(1+z)`, and its
pre-EBL energy flux is `delta^4 (E L_E)_blob/(4 pi d_L^2)`.

Pion/muon particle work states, low-energy target-band components, complete
ledgers, warnings/provenance expansion, manual response control, and external-
field configuration remain internal or deferred. Dataset/fit results,
standalone facades, and sensitivity/local-curvature operations are provided by
the later P4--P6 contracts.
