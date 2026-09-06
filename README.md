# HELENA

HELENA (High-Energy Lepto-hadronic Emission and Neutrino Analysis) calculates
coupled particle transport and photon and neutrino emission in a homogeneous,
isotropic, one-zone steady-state jet model. Its maintained differentiable
inference path fits broadband photons; flavor-resolved neutrino spectra are
post-fit predictions from the same coupled source model.

## Installation and device setup

HELENA requires Python 3.11 or newer. Install v0.1.0 from the tagged source:

```bash
python -m pip install "https://github.com/LiYangji/HELENA/archive/refs/tags/v0.1.0.zip"
```

Alternatively, download the wheel from the
[v0.1.0 Release](https://github.com/LiYangji/HELENA/releases/tag/v0.1.0)
and install it with `python -m pip install helena_astro-0.1.0-py3-none-any.whl`.
These instructions do not require a PyPI release.

The default JAX dependency supports CPU execution. For a GPU, first install
the platform-appropriate JAX build using the
[official instructions](https://docs.jax.dev/en/latest/installation.html).
CPU and GPU share one API: call `configure(device="cpu")`, `"gpu"`, or
`"auto"` before constructing a model. `configure` enables and checks JAX x64
and rejects an explicitly requested unavailable device. The recorded example
environment used Python 3.11.15 and JAX/JAXLIB 0.10.2; other combinations are
not covered by that execution evidence.

From an extracted tagged source tree, the existing optional EBL provider and
MIGRAD optimizer can be installed with:

```bash
python -m pip install ".[ebl]"
python -m pip install ".[migrad]"
```

The TXS example requires `ebl`. Notebook software is a user-selected
environment dependency, not a HELENA runtime dependency. Contributors can
install the existing test/lint tools with `python -m pip install -e ".[dev]"`.

## Quick start

Download and extract the tagged source archive to obtain the notebook
materials; installing the wheel alone does not provide the example tree.
Keep the notebooks, helpers, CSS, reference files, and data in their original
relative layout, and select a Jupyter kernel from the installation environment.

- [CPU Quickstart](quickstart/helena_cpu_quickstart.ipynb)
- [GPU Quickstart](quickstart/helena_gpu_quickstart.ipynb)
- [TXS 0506+056 canonical example](examples/txs0506_2017/txs0506_2017_canonical.ipynb)

These three official notebooks retain genuine saved outputs. The
[example index](examples/README.md) records data qualifications and reference
attribution. Before running BH-enabled sections, acquire the separate
validated response resource as described below.

For example-data sources, processing notes, and attribution, see
[Data sources and attribution](examples/README.md#data-sources-and-attribution).

## Bethe--Heitler cache lifecycle

The approximately 445 MB (425 MiB) validated response archive is not bundled
in the software. After installing HELENA, run:

```bash
helena cache fetch
helena cache path
helena cache status
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
`OneZoneModel(..., cache_dir=...)`. A one-time `fetch --cache-dir PATH` does not
configure later model runs; select the same directory explicitly or through
`HELENA_CACHE_DIR`. Normal model construction and forward calls
only discover and validate local compatible entries: they never download or
silently rebuild them, and a miss fails explicitly.

The package binds the version-specific ZIP from the
[BH C513 v1 resource release](https://github.com/LiYangji/HELENA-resources/releases/tag/bh-c513-v1).
For manual installation, download its ZIP and `.sha256` attachment, and verify
the complete ZIP against SHA256
`a3abbcf77452bcc5015e848130a463e691d717954150611b47f8f19569bd88d4`
(for example, with `sha256sum -c` and the checksum filename). Extract all six
files directly into the directory reported by `helena cache path`, without an
extra enclosing folder, then run `helena cache status`. If using a custom
directory, pass the same `--cache-dir PATH` to both commands or set
`HELENA_CACHE_DIR`.

## Scope, documentation, and citation

See the [public API contract](Public_API/P7B_freeze_contract.md),
[physics assumptions](docs/physics_assumptions.md), and
[units and conventions](docs/units_and_conventions.md). Photon-component
decomposition and local photon-Jacobian/Gauss--Newton diagnostics do not
establish global identifiability or a unique physical interpretation. v0.1.0
does not provide event-level neutrino inference or time-dependent evolution.

[RELEASE_MANIFEST_v0.1.0.json](RELEASE_MANIFEST_v0.1.0.json) identifies the
tagged v0.1.0 files, external response resource, and bounded validation coverage.
`FREEZE_MANIFEST.json` is preserved historical evidence: its old hashes and
pending-publication statements are not the current release identity.

For HELENA v0.1.0, cite the version-specific software DOI
[10.5281/zenodo.22499660](https://doi.org/10.5281/zenodo.22499660).
Software citation metadata are in [CITATION.cff](CITATION.cff). This
citation-only update on `main` does not change the `v0.1.0` tag or archive.
HELENA software
uses [BSD-3-Clause](LICENSE); third-party data and benchmark products retain
their own terms, documented in the [example index](examples/README.md).
