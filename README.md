# HELENA

HELENA (High-Energy Lepto-hadronic Emission and Neutrino Analysis) calculates
coupled particle transport and photon and neutrino emission in a homogeneous,
isotropic, one-zone steady-state jet model. Its maintained differentiable
inference path fits broadband photons; flavor-resolved neutrino spectra are
post-fit predictions from the same coupled source model.

## Installation and device setup

HELENA v0.1.1 requires Python 3.11 or newer. Install the tagged source:

```bash
python -m pip install "https://github.com/LiYangji/HELENA/archive/refs/tags/v0.1.1.zip"
```

Alternatively, download `helena_astro-0.1.1-py3-none-any.whl` from the
[v0.1.1 Release](https://github.com/LiYangji/HELENA/releases/tag/v0.1.1) and install it:

```bash
python -m pip install helena_astro-0.1.1-py3-none-any.whl
```

These instructions do not require a PyPI release. For development from a
matching checkout, use `python -m pip install -e .`. Read the
[v0.1.1 release notes](docs/release_notes_v0.1.1.md) for scientific changes,
cache incompatibility and validation limits.

The default JAX dependency supports CPU execution. For a GPU, first install
the platform-appropriate JAX build using the
[official instructions](https://docs.jax.dev/en/latest/installation.html).
CPU and GPU share one API: call `configure(device="cpu")`, `"gpu"`, or
`"auto"` before constructing a model. `configure` enables and checks JAX x64
and rejects an explicitly requested unavailable device. The recorded example
environment used Python 3.11.15 and JAX/JAXLIB 0.10.2; other combinations are
not covered by that execution evidence.

From the matching source tree, the existing optional EBL provider and
MIGRAD optimizer can be installed with:

```bash
python -m pip install ".[ebl]"
python -m pip install ".[migrad]"
```

The TXS example requires `ebl`. Notebook software is a user-selected
environment dependency, not a HELENA runtime dependency. Contributors can
install the existing test/lint tools with `python -m pip install -e ".[dev]"`.

## Quick start

Use the notebook materials from the same source version as the installed
solver; installing a wheel alone does not provide the example tree.
Keep the notebooks, helpers, CSS, reference files, and data in their original
relative layout, and select a Jupyter kernel from the installation environment.

- [CPU Quickstart](quickstart/helena_cpu_quickstart.ipynb)
- [GPU Quickstart](quickstart/helena_gpu_quickstart.ipynb)
- [TXS 0506+056 canonical example](examples/txs0506_2017/txs0506_2017_canonical.ipynb)

These maintained notebook sources have stale execution outputs cleared and
require rerunning with the current solver. Historical outputs remain in the
v0.1.0 release tag; they are not current numerical regression targets. The
[example index](examples/README.md) records data qualifications and reference
attribution. Before running BH-enabled sections, prepare or select the exact-grid
response cache as described below.

For example-data sources, processing notes, and attribution, see
[Data sources and attribution](examples/README.md#data-sources-and-attribution).

## Response cache lifecycle

v0.1.1 uses a unified 249-point internal photon grid, a 537-point gamma-gamma
daughter grid, and a 513-point particle-transport/BH daughter grid. Its BH and
gamma-gamma responses require the exact current axes, kernel method and float64
identity. The v0.1.0 BH resource is incompatible, including its same-shaped
64-point external response: equal tensor shapes are not equal grids.

Download and validate the v0.1.1 internal/external BH and gamma-gamma responses
explicitly before running the complete examples:

```bash
helena cache fetch
helena cache path
helena cache status
```

`fetch` obtains the version-bound assets from the
[v0.1.1 Release](https://github.com/LiYangji/HELENA/releases/tag/v0.1.1), checks
SHA256, and installs only compatible response files. `status` checks the exact
current grid, physics, method and response identities. If downloads are not
available, `helena cache build` explicitly prepares the current responses with
the same kernels and axes. A cold build is expensive; prefer the matching
release resources. No source model is run by these cache commands.

All commands accept `--cache-dir PATH`. Resolution order is an explicit
`cache_dir`/`--cache-dir`, then `HELENA_CACHE_DIR`, then the platform default:

- Linux/WSL: `${XDG_CACHE_HOME:-~/.cache}/helena/bethe_heitler`;
- macOS: `~/Library/Caches/helena/bethe_heitler`;
- Windows: `%LOCALAPPDATA%\helena\Cache\bethe_heitler`.

The historical directory name is retained; this directory now holds both BH
and gamma-gamma response data and metadata. For offline use, copy all matching
response files and their metadata, then run `helena cache status`. Select the
same directory for later runs using `HELENA_CACHE_DIR` or
`OneZoneModel(..., cache_dir=...)`; a one-time `--cache-dir` selection does not
configure later model calls. Normal forward and fitting calls only read the
compatible local responses: they never download or silently rebuild them.

The immutable [BH C513 v1 resource release](https://github.com/LiYangji/HELENA-resources/releases/tag/bh-c513-v1)
remains available only for historical v0.1.0 grids:

```bash
helena cache fetch --legacy --cache-dir /path/to/legacy-cache
helena cache status --legacy --cache-dir /path/to/legacy-cache
```

This retains the original archive SHA256
`a3abbcf77452bcc5015e848130a463e691d717954150611b47f8f19569bd88d4`
and all six original files. It does not prepare responses for v0.1.1. The
v0.1.0 tag, release, assets and historical manifests are unchanged.

## Scope, documentation, and citation

See the [current public API contract](Public_API/public_api_spec.md),
[physics assumptions](docs/physics_assumptions.md), and
[units and conventions](docs/units_and_conventions.md). Photon-component
decomposition and local photon-Jacobian/Gauss--Newton diagnostics do not
establish global identifiability or a unique physical interpretation. v0.1.1
does not provide event-level neutrino inference or time-dependent evolution.
Always check convergence and conservation diagnostics before using a forward
result; finite spectra alone do not certify a steady solution.

[RELEASE_MANIFEST_v0.1.0.json](RELEASE_MANIFEST_v0.1.0.json) identifies the
tagged v0.1.0 files, external response resource, and bounded validation coverage.
`FREEZE_MANIFEST.json` is preserved historical evidence: its old hashes and
pending-publication statements are not the current release identity.

For HELENA v0.1.0, cite the version-specific software DOI
[10.5281/zenodo.22499660](https://doi.org/10.5281/zenodo.22499660).
Software citation metadata are in [CITATION.cff](CITATION.cff). This
release does not reuse that DOI for v0.1.1 or change the `v0.1.0` tag or archive.
For v0.1.1, cite the version and its GitHub release; no new Zenodo DOI is assigned here.
HELENA software
uses [BSD-3-Clause](LICENSE); third-party data and benchmark products retain
their own terms, documented in the [example index](examples/README.md).
