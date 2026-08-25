# HELENA Public API v0.1 Freeze Contract

Status: **FROZEN — P7B PASS (2026-08-22)**. Acceptance evidence remains in the
preserved developer archive and is not distributed in the public release tree.

## Identity and namespace

- Software: **HELENA — High-Energy Lepto-hadronic Emission and Neutrino
  Analysis**.
- Python namespace: `helena`.
- Distribution: `helena-astro`.
- Frozen L1 symbols: `configure`, `ModelConfig`, `OneZoneModel`,
  `ForwardResult`, `SEDDataset`, `fit`, and `FitResult`.
- Frozen L2 namespaces: `helena.physics`, `helena.transport`,
  `helena.cascade`, `helena.sensitivity`, and `helena.diagnostics`.
- `OneZoneModel.forward()` is the only complete Public lepto-hadronic forward;
  `helena.models.c513.run_c513_forward` remains its internal canonical owner.

CPU and GPU use the same names, signatures, result schemas, parameter
coordinates, and scientific semantics. `configure(device="auto"|"cpu"|"gpu")`
selects execution only, explicit unavailable-device requests fail without
fallback, and scientific production requires JAX x64.

## Scientific result contracts

`ForwardResult` has exactly the top-level fields `sed`, `components`,
`particles`, `neutrinos`, and `diagnostics`. Blob photons use `E L_E` in
erg s^-1; observer/Earth photons use `E F_E` in erg cm^-2 s^-1. Particle
densities are `N(gamma)` in cm^-3 per gamma and injections are `Q(gamma)` in
cm^-3 s^-1 per gamma. Neutrino source and Earth flavor quantities remain
separately named.

`SEDDataset` keeps observation statistic separate from scientific-use
selection. `fit` accepts an ordered unique 1--8 registry subset and either
`lbfgsb` or `migrad`; omitted parameters retain the input `ModelConfig`
physical values and the input model is not mutated. `FitResult.uncertainties`
and `.covariance` remain `None` unless a future explicitly named method
produces them.

## Parameters, sensitivity, and diagnostics

`helena.parameters.DIFFERENTIABLE_PARAMETERS` is the sole ordered owner:

```text
log_Le, log_Lp, log_B, log_R, p_e, p_p,
log_gamma_e_max, log_gamma_p_max
```

The six `log_*` coordinates are natural-log reference offsets; `p_e` and
`p_p` are additive slope offsets. Fitting and sensitivity derive their names
and ordering from this registry. Discrete switches are excluded.

`helena.sensitivity.jvp` and `.jacobian` differentiate

```text
ln[(E F_E^Earth) / (1 erg cm^-2 s^-1)]
```

on fixed baseline-finite-positive support, with no clipping, floor, or dynamic
mask. Jacobians are `(energy, parameter)` and preserve caller parameter order;
the frozen canonical model has 113 positive bins.

The frozen `helena.diagnostics` scientific exports are `GaussNewtonResult`
and `gauss_newton`. They accept active independent symmetric Gaussian
detections only, with `sigma_log = sigma_F/F`, `W = diag(1/sigma_log^2)`, and
`H_GN = J^T W J`. Active asymmetric detections and upper limits fail clearly.
The pseudo-inverse covariance is a local Gauss--Newton approximation, not a
posterior, profile interval, or global uncertainty.

## Freeze gates and compatibility

The official v0.1 Ruff acceptance command is:

```bash
ruff check --select E4,E7,E9,F src/helena tests examples
```

`compileall`, imports, `git diff --check`, import-graph, production-to-tests,
notebook JSON, and package-discovery checks are also required. Findings from
`ruff check --select ALL` outside the command above are future style or
exception-contract cleanup unless they expose a correctness failure; v0.1
does not refactor scientific code merely to make that expanded audit zero.

Frozen baselines, historical Jacobians, open decisions, and detailed gate
evidence remain developer-archive material. From v0.1 onward, a breaking
rename, deletion, reparameterization, or result-semantic change requires
explicit API review and a documented deprecation or migration policy. Deep
L3/L4 imports remain unsupported implementation or validation details and
receive no compatibility promise.
