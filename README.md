# HELENA

**HELENA — High-Energy Lepto-hadronic Emission and Neutrino Analysis**

HELENA is a differentiable one-zone lepto-hadronic framework for high-energy
emission, neutrino production, SED fitting, and parameter-sensitivity analysis.
It is research software: the validated configurations and diagnostics are
narrower than the full parameter space exposed by the kernels.

## Current status

The repository contains the NumPy/reference physics and a differentiable JAX
backend. Implemented paths include electron/positron, proton, pion, and muon
transport; synchrotron, inverse-Compton, and synchrotron-self-absorption
processes; photomeson, proton-proton, and Bethe-Heitler interactions;
gamma-gamma absorption and pair injection; and steady electromagnetic and
hadronic cascades. Persistent response caching, a likelihood/fitting
foundation, and experimental checkpointed and implicit autodiff strategies are
also present.

These components have focused numerical and physics checks, but this is not a
claim that every parameter combination or end-to-end scientific use case is
validated. Current production model execution lives mainly under
``src/helena/core/jax_backend``.

## Installation and device setup

HELENA requires Python 3.11 or newer. A CPU installation from this release
tree is:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

JAX is a runtime dependency, and the default package installation supports its
CPU backend. For accelerator support, install the matching JAX build for the
platform by following the official JAX installation guide before installing
HELENA. HELENA does not maintain a second GPU API: select execution before
model construction with `configure(device="auto"|"cpu"|"gpu")`.

The optional EBL provider and MIGRAD optimizer are installed independently:

```bash
python -m pip install ".[ebl]"
python -m pip install ".[migrad]"
```

Contributors can install the test and lint tools with:

```bash
python -m pip install -e ".[dev]"
```

Scientific execution requires JAX x64. `configure` enables and verifies this
contract and fails rather than silently falling back from an explicitly
requested unavailable device.

## Bethe--Heitler cache lifecycle

The release contract is:

```text
install HELENA -> helena cache fetch -> normal HELENA forward
```

The approximately 425 MiB validated cache is intentionally not bundled in the
Python package. Manage it with:

```bash
helena cache path
helena cache status
helena cache fetch
```

All three commands accept `--cache-dir PATH` for HPC, offline, or shared
storage. Resolution order is an explicit `cache_dir`/`--cache-dir`, then
`HELENA_CACHE_DIR`, then the platform default:

- Linux/WSL: `${XDG_CACHE_HOME:-~/.cache}/helena/bethe_heitler`;
- macOS: `~/Library/Caches/helena/bethe_heitler`;
- Windows: `%LOCALAPPDATA%\helena\Cache\bethe_heitler`.

`fetch` verifies the package-owned resource manifest and archive SHA-256 before
an atomic install. For an offline system, fetch on a connected system, copy the
complete cache directory, and select it with `HELENA_CACHE_DIR` or
`OneZoneModel(..., cache_dir=...)`. Normal model construction and forward calls
only discover and validate local compatible entries: they never download or
silently rebuild them, and a miss fails explicitly.

This release candidate contains the final validated file identities, but the
official release-asset URL and archive checksum are not yet published. Until
that single binding is filled, `helena cache fetch` fails with a clear resource
unavailable error; no URL is guessed.

## Official examples

The v0.1.0 public examples are exactly:

1. [`quickstart/helena_cpu_quickstart.ipynb`](quickstart/helena_cpu_quickstart.ipynb)
2. [`quickstart/helena_gpu_quickstart.ipynb`](quickstart/helena_gpu_quickstart.ipynb)
3. [`examples/txs0506_2017/txs0506_2017_canonical.ipynb`](examples/txs0506_2017/txs0506_2017_canonical.ipynb)

The CPU and GPU quickstarts use the same calculations and public names. The
TXS notebook is the sole public real-source workflow and requires the `ebl`
extra for its configured Earth-propagation step. Notebook software itself is a
user-selected environment dependency rather than a HELENA runtime dependency.

## Public API

The frozen v0.1 API has one CPU/GPU-neutral root workflow:

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
`helena.cascade`, `helena.sensitivity`, and `helena.diagnostics`.
`OneZoneModel.forward()` is the only complete Public lepto-hadronic forward.
Its C513 implementation, prepared responses, solver state, and validation
collectors remain internal. The exact frozen boundary is recorded in
[the P7B contract](Public_API/P7B_freeze_contract.md).

`ForwardResult.components` includes legacy high-energy fields, escaped
broadband process groups, additive source-resolved photon groups, and
mechanism-resolved synchrotron and inverse-Compton fields. New displays should
use **Primary-electron inverse Compton** and the
`primary_electron_inverse_compton` key; the compatibility SSC name is retained
only as a frozen result key.

## Repository structure

```text
src/helena/
├── core/               # reference kernels and shared physics
│   └── jax_backend/    # differentiable kernels and production chains
├── models/             # validated package-owned scenarios
├── fitting/            # datasets, likelihood, and optimizer boundary
├── autodiff/           # experimental differentiation strategies
├── cli.py              # cache resource-management commands
└── runtime_cache.py    # persistent response cache support
```

See [the example index](examples/README.md), [physics assumptions](docs/physics_assumptions.md),
and [units and conventions](docs/units_and_conventions.md).

## License

HELENA is distributed under the BSD-3-Clause license. See [LICENSE](LICENSE).

## Conventions and scientific caution

Internal quantities use cgs units unless a public function says otherwise.
Physics correctness, explicit frame conventions, conservation diagnostics, and
numerical stability take precedence over fitting convenience or performance.
Real-source fits do not by themselves validate a physical model; energy,
cooling, size, and dynamical constraints must also be checked.
