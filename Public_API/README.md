# HELENA Public API

The frozen v0.1 root namespace contains exactly seven names:

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

`ForwardResult.components` retains legacy high-energy and aggregate broadband
fields while adding nine additive source-resolved photon groups. It also
provides mechanism-resolved synchrotron and inverse-Compton fields for primary
and secondary leptons. New usage should say **Primary-electron inverse
Compton**; `primary_electron_ssc` remains only as a frozen compatibility key.

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

BH response tensors are private resources, never result fields or package
data. The release lifecycle is `official validated cache -> helena cache fetch
-> platform-standard cache directory`, with optional `cache_dir=` and
`HELENA_CACHE_DIR` overrides. `helena cache path`, `status`, and `fetch` manage
the resource; all accept `--cache-dir`. Normal forwards only read compatible
local entries and never download or rebuild them. The CLI implementation and
validated file manifest are present; the official asset URL/archive checksum
binding remains unpublished in this release candidate.

## Contracts

The authoritative current boundary is in
[`public_api_spec.md`](public_api_spec.md) and
[`P7B_freeze_contract.md`](P7B_freeze_contract.md). The P2--P6 files record the
implemented contracts for runtime, model/forward, fitting, standalone physics,
and sensitivity. Phase-local deferral language describes the state at that
phase; later contracts and the frozen specification take precedence.
