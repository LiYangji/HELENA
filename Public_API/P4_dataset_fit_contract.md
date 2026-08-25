# P4 Dataset / Fit Contract

## Audited internal owners

`LogSEDDataset` and `SEDData` are not interchangeable despite their names.

| Existing object | Actual responsibility | P4 policy |
|---|---|---|
| `helena.fitting.LogSEDDataset` | Generic positive SED detections with independent symmetric Gaussian errors in natural-log space. It can label comoving `E L_E`, intrinsic observer `E F_E`, or Earth `E F_E`, and has one boolean likelihood mask. It does not support asymmetric errors or upper limits. | Remains a deep compatibility/validation object, not a root public dataset. |
| `helena.fitting.SEDData` | Maintained real-observation table and input to `mixed_sed_nll`. It stores Earth-observed energy and `E F_E`, symmetric/asymmetric detections, upper limits, instruments, and independent statistic/use labels. | Remains the internal owner behind the root `SEDDataset` facade. |

`SEDDataset` is a zero-policy subclass facade over `SEDData`; it inherits the
validated immutable constructor, selection helpers, and audit behavior rather
than copying data or likelihood logic. The root exports only `SEDDataset`.

## Dataset semantics

The public observable is photon `E F_E` at Earth in `erg cm^-2 s^-1` on
observer-frame energies in erg. It is the same canonical broadband Earth
prediction exposed by `OneZoneModel.forward()`: identity when
`ModelConfig.enable_ebl=False`, otherwise the validated non-extrapolated
Dominguez-2011 mapping. P4 adds no independent EBL choice, external field,
neutrino likelihood, units package, or multi-observable framework.

Observation and selection remain separate arrays:

- `statistic_type`: `gaussian`, `asymmetric_gaussian`, or `upper_limit`;
- `scientific_use`: `fit`, `constraint`, `context`, or `exclude`.

`fit` and `constraint` rows enter the likelihood. `context` and `exclude` rows
do not. Relabeling scientific use never changes the published statistic.

Validation requires a non-empty shared one-dimensional shape; positive finite
energy and flux; finite non-negative energy/flux errors; lower energy errors
below energy; positive detection errors; symmetric errors for `gaussian`;
lower detection errors below flux; and exactly zero published flux errors for
an upper limit. Instrument, statistic, and use labels must be non-empty/known.
Inputs are copied into read-only arrays; invalid values are never repaired.

The maintained scalar NLL is `mixed_sed_nll`:

- symmetric detections use a Gaussian in natural-log flux with
  `sigma_log = sigma_F/F`;
- asymmetric detections use exact lower/upper log scales and one continuous
  split-normal density;
- a published 95% upper limit uses the existing approximate null-centered
  one-sided Gaussian constraint with `sigma_eff = UL/Phi^-1(0.95)`;
- the model is log-log interpolated to selected observation energies;
- inactive rows contribute exact zero.

Energy uncertainties are preserved observational metadata but are not part of
the current likelihood. Covariance and systematics are not supported.

## One public fit

```python
from helena import ModelConfig, OneZoneModel, SEDDataset, fit

model = OneZoneModel(ModelConfig(...))
data = SEDDataset(...)
result = fit(
    model,
    data,
    parameters=["log_Le", "log_B"],
    optimizer="lbfgsb",
)
```

`parameters` is one unique ordered subset of 1--8 names from
`DIFFERENTIABLE_PARAMETER_NAMES`, which is derived from the sole
`DIFFERENTIABLE_PARAMETERS` registry. The user order is retained. Every
omitted registry parameter is fixed exactly at the input `ModelConfig`
physical value. Configurable switches and numerical controls are rejected as
fit parameters.

The input `ModelConfig` is the initial physical model. P4 deterministically
converts it to the existing six natural-log and two additive reference
coordinates. Users never supply C513 offsets. `ParameterSelection` packs the
requested free values and reconstructs the canonical eight-vector; the two
existing C513 mappers then apply changes to the input model configuration.
Neither mapper nor the AD graph was changed.

The objective uses the already validated fixed-five-outer-round,
`physics_only=True` fitting graph and simultaneous forward-mode scalar
gradient. Static prepared Bethe--Heitler/gamma-gamma responses are obtained
through the same model-owned lazy cache used by `forward()` and are reused for
every evaluation. The production observer functions prepare the fixed Earth
support, luminosity-to-flux scale, and stop-gradient EBL transmission outside
tracing.

`optimizer="lbfgsb"` and `optimizer="migrad"` both call the existing
`fit_maximum_likelihood` driver with the same initial vector, parameter
selection, prediction, `mixed_sed_nll`, and analytic gradient. There is no
optimizer-specific public result class and no CPU-specific fitting path.

The maintained optimizer currently supports unbounded coordinates when
`bounds=None`; P4 exposes no custom bounds argument and invents no
source-independent astrophysical defaults. A future physical-bounds contract
is left open.

On CPU, more than two free parameters emits one standard `RuntimeWarning`.
The warning is advisory only: selection, objective, optimizer, and execution
are unchanged. CPU and GPU accept the identical 1--8 parameter API.

## `FitResult`

`FitResult` is one frozen, slotted dataclass with these stable fields:

- `best_parameters`: ordered mapping for free registry names to physical
  best-fit values (erg/s, G, cm, dimensionless index, or gamma as applicable);
- `best_config`: new physical `ModelConfig`; the input model/config is not
  mutated;
- `free_parameters`, `fixed_parameters`, and
  `joint_dimension == len(free_parameters)`;
- `nll`, `optimizer`, `success`, integer `status`, `message`, optional
  `iterations`, and `evaluations`;
- `prediction`: selected dataset indices, observer energies, and aligned Earth
  photon `E F_E`, not a full `ForwardResult`;
- `residuals`: statistic-aware detection residual/NLL and upper-limit
  status/NLL mappings from the maintained diagnostics;
- `uncertainties` and `covariance`, currently `None` because P4 performs no
  HESSE, MINOS/profile, or labelled local Gauss--Newton calculation;
- `runtime_seconds`, actual `backend`, actual `device`, and `x64`.

`fixed_parameters` contains the seven, six, ..., or zero omitted registry
values in physical units, so a conditional 1D/2D fit cannot be mistaken for an
8D joint inference. A complete convergence-managed best-model forward remains
explicit: `OneZoneModel(result.best_config).forward()`.

## Scope audit

P4 adds one dataset facade, one fit function, and one result dataclass. It adds
no dataset manager, optimizer factory, inference engine, result hierarchy,
CPU/GPU fit class, response manager, callback framework, or plugin system. It
does not modify physics, the solver, the canonical full-forward owner,
likelihood mathematics, validated mappers, response kernels/cache contract,
or the frozen 8D AD collector.
