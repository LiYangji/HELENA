# P6A Model-Bound Derivative Target Contract

Status: PASS on 2026-08-22. P6A freezes one maintained internal target and
one L4 validation reference. It adds no public callable, result type, or
namespace; the public sensitivity facade remains P6 work.

## Frozen scientific target

The physical `ForwardResult` photon output remains linear Earth-frame
`E F_E` in `erg cm^-2 s^-1`. Differentiation uses the dimensionless natural-log
representation

```text
y_i(theta) = ln[(E_i F_E,i^Earth(theta)) / (1 erg cm^-2 s^-1)].
```

The first target is the total Earth photon SED only. Component, neutrino, and
particle derivatives are deferred. Natural logarithm is the selected
sensitivity representation; no base-10 or linear sensitivity target competes
with it.

## Owner and graph

`helena.fitting.public._earth_photon_log_sed_target(model, parameters)` is
the private maintained owner. It reuses P4's `_model_prediction` closure and
therefore its existing fixed five `outer_hadronic_picard_step(...,
physics_only=True)` rounds, physical model mapping, prepared responses,
observer transform, and luminosity-distance scaling. P6A copies no physics,
solver, mapper, or P4 likelihood code.

The energy grid is P3/P4's observer/Earth photon grid. The helper returns the
selected read-only energy array, read-only original-grid support indices,
baseline parameter coordinates, and traced target closure. These are internal
implementation objects, not a compatibility promise.

## Coordinates and ordering

Names come only from the eight-entry registry:

```text
log_Le, log_Lp, log_B, log_R, p_e, p_p,
log_gamma_e_max, log_gamma_p_max
```

A request selects any unique ordered subset of one through eight names. The
Jacobian parameter axis preserves the request order. Logarithmic coordinates
remain natural-log reference offsets; `p_e` and `p_p` remain additive
reference offsets. Unselected physical values remain fixed at the supplied
`OneZoneModel.config`.

## Fixed support and failure policy

Support is computed once, outside the JAX trace, from the supplied model's
baseline P4 prediction:

```text
support = finite(baseline Earth E F_E) AND baseline Earth E F_E > 0
```

The same indices are then fixed for every evaluation and derivative. An empty
baseline support fails immediately. There is no clipping, floor, epsilon,
absolute value, dynamic mask, or support reselection. A perturbed value that
is nonpositive or nonfinite remains visible as a nonfinite log target rather
than being silently repaired.

The P4 prediction first uses the same configured Earth support as Public
`forward()`. With EBL enabled, fixed Dominguez-2011 transmission is one below
the table, evaluated by the validated owner inside it, and not extrapolated
above it; only positive-transmission Earth nodes enter the log target.

For the frozen canonical configuration the support is the full 113-bin grid.
Its float64 energy hash is
`4271529fd1369f21e4dab368be01bd2d4a25e237ac30a944050b96eab67f0cfa`;
the int64 support hash is
`9b9a1c5afe44a8aa500f6d265edc165bce9e38dbba93ed972ca51ffbdc8e8fae`.

## Frozen validation evidence

The canonical baseline target reproduces the public `ForwardResult` Earth
photon SED on all 113 bins to `rtol=1e-12`. Focused checks cover 1D JVP versus
Jacobian, centered finite difference, a noncanonical ordered 2D selection,
repeat determinism, and fresh-process CPU/GPU parity.

The preserved developer archive owns the L4 simultaneous eight-parameter
check and its frozen payload: output shape `(113, 8)`, orientation
output-energy by parameter, float64, finite, and every column nonzero. The
comparison tolerance is `rtol=5e-10, atol=5e-12`. The old Regime-B
photon-plus-neutrino reference remains separate and unchanged; the two targets
are not required to have equal Jacobians.

## Explicit deferrals

P6A does not add `helena.sensitivity`, public JVP/Jacobian methods,
Gauss--Newton or identifiability diagnostics, component/neutrino/particle
targets, full fits, or new AD engines. A future public P6 facade must wrap this
exact maintained target and registry semantics rather than creating another
scientific graph.
