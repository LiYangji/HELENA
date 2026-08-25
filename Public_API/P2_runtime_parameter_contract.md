# P2 Runtime, Device, and Parameter Contract

Status: implemented and focused validation PASS on 2026-08-22 with Python 3.11.15, JAX 0.10.2, CPU, and NVIDIA RTX 4060/CUDA.

## 1. Runtime ownership audit

| Audit item | Pre-P2/current finding | P2 contract |
|---|---|---|
| JAX imports | JAX is used throughout `src/helena`, tests, and examples. Import alone is not a device contract. | Root import exposes only `configure`; runtime policy stays in one tiny module. |
| x64 | `models/c513.py` calls `jax.config.update("jax_enable_x64", True)` at import. Many focused tests/examples also enable x64 locally. | `configure` is the public owner and enables/verifies x64. The compatible C513 safeguard remains for deep callers. |
| Platform selection | Production code had no platform selector. The CPU notebook sets `JAX_PLATFORMS=cpu` before importing JAX; CPU tests otherwise rely on a CPU-selected process/host and assert the result. | `auto` leaves JAX platform priority alone; explicit `cpu` selects `jax_platforms=cpu`; explicit `gpu` selects the supported NVIDIA `jax_platforms=cuda` backend. |
| GPU selection | The WSL production environment previously relied on JAX's default discovery, which resolves the available CUDA device. | Explicit GPU must resolve JAX platform `gpu`; initialization failure or another resolved platform is an error, never CPU fallback. |
| Device queries | Existing tests/examples call public `jax.default_backend()` and `jax.devices()` for assertions/reporting. | `configure` uses those public queries only after applying early configuration and returns their actual result. |
| Backend initialization | Public backend/device queries and the first dispatched JAX computation initialize execution. A fresh probe confirmed `import helena.models` can still be followed by `configure(device="cpu")`; the import-time x64 update itself did not lock the GPU backend. | Call `configure` before any backend query or JAX computation. A late incompatible request fails; no private initialization inspection is used. |
| Environment assumptions | `JAX_PLATFORMS` and `CUDA_VISIBLE_DEVICES` may influence a process before import. | `auto` respects the resulting default. Explicit requests are verified against the resolved public platform; hidden/unavailable GPU produces a clear failure. |

Direct audit answers:

1. Production x64 was enabled by importing `models/c513.py`; tests/examples also contain local safeguards. P2 additionally enables it in the public runtime entry.
2. Multiple compatible x64 safeguards exist, but device policy has one public owner: `helena.configure`.
3. `jax.default_backend()`, `jax.devices()`, and first execution initialize a backend; ordinary HELENA/JAX import was not treated as proof of initialization.
4. CPU-only work previously depended on a CPU host/process or early `JAX_PLATFORMS=cpu`; the CPU notebook sets that variable before JAX import.
5. GPU production previously used JAX default CUDA discovery in the WSL environment.
6. x64 must be set before scientific arrays/tracing, and the C513 safeguard must remain until deep callers are migrated safely.
7. Yes. P2 uses only public `jax.config.update`, `jax.default_backend`, `jax.devices`, and `jax.config.jax_enable_x64`; it uses no private state, reset, or cache hack.

## 2. Public runtime contract

```python
from helena import configure

runtime = configure(device="auto")
```

- Valid values are `auto`, `cpu`, and `gpu`.
- `auto` uses JAX's available/default platform and records what resolved.
- `cpu` must resolve `cpu`.
- `gpu` requests the current supported CUDA GPU backend and must resolve JAX platform `gpu`.
- Explicit request mismatch or unavailability raises `RuntimeError`; no fallback/reset is attempted.
- Invalid values raise `ValueError`.
- Compatible repeated calls are deterministic.
- The read-only result contains `requested_device`, `resolved_platform`, `resolved_device`, and `x64`.
- x64 is always enabled and verified. FP32, mixed precision, and a public precision option are outside the production contract.
- Device changes only execution location/performance/memory. Model, objective, parameter space, fit, AD, outputs, and scientific meaning are identical.

CPU routinely favors 1–2 free parameters. GPU routinely favors joint 3–8 parameter work and multi-parameter Jacobians. Both support the same validated 1–8 space. This is documentation-only guidance; P2 adds no limiter, staging, coordinate descent, automatic reduction, or fitting warning.

## 3. Validated parameter audit

The active mappers are `c513_configuration_from_log_offsets` and `c513_configuration_from_injection_shape_offsets`. Both consume float64 JAX arrays in the following exact order. `jnp.exp` proves the six log coordinates are natural-log offsets, not log10 or absolute logarithms.

| Order/name | Scientific quantity and C513 reference | Actual validated coordinate | Physical unit |
|---:|---|---|---|
| 1 `log_Le` | electron kinetic injection luminosity; 3.7e40 erg s^-1 | `ln(L_e,kin / L_e,kin,ref)` | erg s^-1 |
| 2 `log_Lp` | proton kinetic injection luminosity; 2.8e46 erg s^-1 | `ln(L_p,kin / L_p,kin,ref)` | erg s^-1 |
| 3 `log_B` | blob-comoving magnetic field; 0.1 G | `ln(B / B_ref)` | G |
| 4 `log_R` | blob-comoving radius; 1e16 cm | `ln(R / R_ref)` | cm |
| 5 `p_e` | electron injection index; 2.0 | `p_e - p_e,ref` | dimensionless |
| 6 `p_p` | proton injection index; 2.0 | `p_p - p_p,ref` | dimensionless |
| 7 `log_gamma_e_max` | electron cutoff Lorentz factor; 3e5 | `ln(gamma_e,max / gamma_e,max,ref)` | dimensionless |
| 8 `log_gamma_p_max` | proton cutoff Lorentz factor; 1e7 | `ln(gamma_p,max / gamma_p,max,ref)` | dimensionless |

All registry coordinate values are dimensionless. The luminosity mapper scales injection density consistently with `R^-3`; the radius coordinate also scales escape rates with `R^-1`. The injection-shape mapper re-normalizes changed indices/cutoffs to preserve each nominal kinetic injection luminosity. P2 does not alter any of this math.

The mappers enforce 4-vector shape and rely on finite local coordinates; `ParameterSelection` enforces finite baseline values and validates caller-supplied optimizer bounds. No universal hard domain or source-independent fit bounds exist in the validated code, so the registry invents none. The separate one-dimensional escape-rate development offset is not part of this validated public eight.

The fitting order and frozen L4 eight-parameter collector both use this same ordering. The collector slices the first four values into the luminosity/field/radius mapper and the last four into the injection-shape mapper.

## 4. Single registry contract

`src/helena/parameters.py` owns one immutable `tuple[ParameterSpec, ...]`, `DIFFERENTIABLE_PARAMETERS`. `DIFFERENTIABLE_PARAMETER_NAMES` and the read-only `PARAMETERS_BY_NAME` are derived from it. `helena.fitting.CANONICAL_PARAMETER_NAMES` remains a compatibility alias to the same names tuple, not an independent list.

The registry contains exactly the eight rows above. `enable_pp`, `enable_ssc`, `enable_ec`, Bethe–Heitler mode, EBL model, gamma-gamma switches, grids, tolerances, response selection, and `physics_only` are excluded: configurable is not differentiable. Future L1 fit and L2 gradient/JVP/Jacobian must validate user-provided 1–8 name subsets through this registry and preserve the user's explicit subset order.

P2 keeps the existing reference-relative coordinates canonical. It does not add an absolute-to-offset mapper because P3 has not yet defined the public model reference/configuration. P4/P6 may derive absolute physical result views from that future configuration without changing the optimization/AD coordinates.
