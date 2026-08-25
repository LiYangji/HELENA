# P6 Sensitivity / Identifiability Contract

Status: **PUBLIC API P6 PASS** on 2026-08-22.

P6A resolved the earlier ownership blocker by freezing one maintained,
model-bound derivative target and reference. P6 exposes that owner through two
thin scientific functions and reuses the existing local Gauss--Newton
diagnostic without changing its mathematics.

## 1. Maintained owner audit

| Concern | Maintained owner reused by P6 | Public policy |
|---|---|---|
| Target and parameter map | `helena.fitting.public._earth_photon_log_sed_target`, frozen in P6A | Private owner behind the facade; no second derivative graph. |
| JVP/Jacobian | JAX differentiation of that exact target at its baseline registry coordinates | Exposed only as `helena.sensitivity.jvp` and `.jacobian`; raw transforms and closures stay private. |
| Observation mapping | P4 log-energy/log-SED interpolation and `SEDDataset.likelihood_mask` semantics | Jacobian columns are linearly sampled in log energy at active observations; no nearest-bin mapping. |
| Local curvature | Existing `helena.fitting.core.add_postfit_diagnostics` SVD, rank, condition, `J_w.T @ J_w`, pseudo-inverse, and correlation calculation | One private shared calculation serves the existing fitting diagnostic and public `helena.diagnostics.gauss_newton`. |

The physical public `ForwardResult` remains linear Earth photon `E F_E`.
Sensitivity differentiates only
`ln[(E F_E^Earth)/(1 erg cm^-2 s^-1)]` on the P6A baseline-finite-positive
support. The support is selected once outside tracing and never clipped,
floored, or changed by a perturbation. The canonical validated model has all
113 bins in support.

## 2. Public sensitivity surface

```python
import helena.sensitivity as sensitivity

J = sensitivity.jacobian(model, parameters=["log_B", "log_Le"])
v = sensitivity.jvp(
    model,
    parameters=["log_B", "log_Le"],
    direction=[0.3, -0.2],
)
```

Both calls accept any unique ordered 1--8 subset of the single differentiable
registry. Caller order is the Jacobian column order and the JVP direction
order. Discrete switches, grids, response modes, and tolerances fail as
unknown parameters. There is no public gradient because the only maintained
scalar gradient is the mixed-likelihood implementation inside `fit`.

`SensitivityResult` is one frozen/slotted result for both calls. It contains
read-only `energy_erg`, `values`, ordered `parameters`, `support_indices`, and
the optional JVP `direction`, plus unambiguous `target` and `orientation`
properties. Jacobian values have orientation `(energy, parameter)`; JVP values
have orientation `(energy,)` and equal `J.values @ direction`.

The six logarithmic registry coordinates yield dimensionless derivatives.
`p_e` and `p_p` derivatives are per unit spectral-index change. The public
surface does not expose `physics_only`, the fixed-five map, Picard state,
`jax.jacfwd`, `jax.jvp`, tangent trees, raw responses, or AD strategy choices.

CPU and GPU use these identical calls and result schema. CPU requests above
two parameters receive a standard `RuntimeWarning` only; no hard limit,
parameter reduction, automatic device change, or derivative-method change is
performed. GPU is recommended for multi-parameter and full eight-parameter
Jacobians. Eight dimensions are a validated request, not a special API.

## 3. Public local Gauss--Newton diagnostic

```python
import helena.diagnostics as diagnostics

local = diagnostics.gauss_newton(J, dataset)
```

Only active (`fit` or `constraint`) independent symmetric Gaussian detections
are supported. Active `asymmetric_gaussian` or `upper_limit` rows fail
clearly; unsupported context/exclude rows remain inactive under the P4 mask.
No row is silently dropped, symmetrized, or converted to an effective
Gaussian.

For active observation `i` and parameter `j`,

```text
J_ij       = d ln(F_i) / d theta_j
sigma_log  = sigma_F / F
J_w        = J / sigma_log
W          = diag(1 / sigma_log^2)
H_GN       = J^T W J = J_w^T J_w
```

`T` is the ordinary transpose because the Jacobian is real-valued. The result
contains the active energies/indices, weighted Jacobian, curvature,
pseudo-inverse local covariance, normalized correlation, singular values,
rank, condition number, and an honest status. Rank-deficient matrices remain
rank deficient with infinite condition number; P6 adds no ridge, Tikhonov, or
epsilon regularization.

The covariance is explicitly a **local Gauss--Newton approximation**, not a
posterior covariance, credible interval, MINOS/profile interval, or global
uncertainty. P4 `FitResult.uncertainties` and `.covariance` remain unchanged at
`None` unless another explicitly named method produces them.

The pre-existing conservation-ledger functions in `helena.diagnostics`
remain L4/developer operations and are not promoted by this contract.

## 4. Validation and deferrals

P6 validation covers module import, 1D, noncanonical 2D order, another legal
three-parameter subset, invalid/duplicate/discrete names, immutability,
deterministic repeats, Public/P6A equivalence, one centered finite difference,
`JVP = J @ direction`, fresh low-dimensional CPU/GPU parity, strict diagnostic
domain failures, rank deficiency, and the canonical Public GPU `(113, 8)`
comparison to the frozen reference retained in the developer archive.

Component, neutrino, and particle-distribution sensitivity; a public scalar
gradient; advanced AD strategies; automatic fit covariance; and non-Gaussian
local-curvature approximations remain deferred. The old Regime-B
photon-plus-neutrino reference remains independent L4 evidence and is not
modified or equated with the model-bound target.
