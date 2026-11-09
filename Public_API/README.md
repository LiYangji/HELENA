# HELENA v0.1.1 Public API

The current root namespace retains the seven public names introduced in v0.1:

```python
from helena import (
    FitResult,
    ForwardResult,
    ModelConfig,
    OneZoneModel,
    SEDDataset,
    configure,
    fit,
)
```

The flat advanced namespaces are `helena.physics`, `helena.transport`,
`helena.cascade`, `helena.sensitivity`, and `helena.diagnostics`. CPU and GPU
use these same names, signatures, parameter coordinates, result schemas, and
scientific semantics.

## Model and photon components

`ModelConfig` contains absolute physical values. `OneZoneModel` maps them
through the validated internal owner and exposes the sole complete public
forward. Prepared responses, raw solver state, validation collectors, and
scenario-specific internal names are not public.

`ForwardResult.components` exposes full-grid intrinsic and aggregate broadband
fields while adding nine additive source-resolved photon groups. It also
provides mechanism-resolved synchrotron and inverse-Compton fields for primary
and secondary leptons. New usage should say **Primary-electron inverse
Compton**. Its internal IC reads the complete accepted internal photon state;
`primary_electron_ssc` is a synchrotron-seeded subset diagnostic and must not
be added to total IC or the nine source-resolved groups. `enable_ssc` retains
its spelling but now enables the complete internal IC process.

## Fitting, sensitivity, and local diagnostics

`helena.parameters` is the single owner of the eight differentiable
coordinates. `fit`, `helena.sensitivity.jvp`, and
`helena.sensitivity.jacobian` accept any caller-ordered unique subset of one to
eight names. Omitted coordinates stay fixed at the input `ModelConfig`.

Sensitivity differentiates the natural logarithm of the Earth photon SED on
fixed baseline-positive support. Jacobians are oriented `(energy, parameter)`.
`helena.diagnostics.gauss_newton` returns `J.T @ W @ J` and local
pseudo-inverse diagnostics for active independent symmetric Gaussian
observations only; it is not a posterior or global uncertainty estimate.

## Cache boundary

BH and gamma-gamma response tensors are private resources, never result fields
or package data. v0.1.1 prepares its exact internal/external responses explicitly
with `helena cache fetch` or `helena cache build`. `helena cache path` and
`status` report the selected directory and validate grid, physics, method and
SHA256 identities; commands accept `--cache-dir`, and model calls accept
`cache_dir=` or `HELENA_CACHE_DIR`. Normal forwards and fitting only load
compatible local entries and never download or rebuild them. The unchanged
historical v0.1.0 BH resource is available through `fetch --legacy`, but is
incompatible with the current solver. See the root README for preparation.

## Contracts

The current boundary is in [`public_api_spec.md`](public_api_spec.md).
[`P7B_freeze_contract.md`](P7B_freeze_contract.md) records the historical v0.1
freeze and must not override current numerical semantics. The P2--P6 files record the
implemented contracts for runtime, model/forward, fitting, standalone physics,
and sensitivity. Phase-local deferral language describes the state at that
phase; the current specification takes precedence.
