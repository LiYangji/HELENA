# Official HELENA examples

The maintained v0.1.1 example set contains only:

- the [CPU quickstart](../quickstart/helena_cpu_quickstart.ipynb);
- the [GPU quickstart](../quickstart/helena_gpu_quickstart.ipynb); and
- the [TXS 0506+056 2017 canonical example](txs0506_2017/txs0506_2017_canonical.ipynb).

These source notebooks target the v0.1.1 solver and have stale outputs cleared. Historical v0.1.0 output is retained in that immutable tag.
CPU and GPU use the same public API and scientific semantics. The TXS notebook
is the only public real-source workflow. Legacy, deep-private, diagnostic, and
profiling notebooks remain in the preserved developer archive.

Complete model sections require the exact v0.1.1 BH and gamma-gamma responses.
Run `helena cache fetch` once and `helena cache status` to validate the release
assets, exact grids, physics and SHA256 identities. `helena cache build` is the
explicit, more expensive local alternative. Select the same directory with
`cache_dir=` or `HELENA_CACHE_DIR`; offline copies must include all response
data and metadata. Normal examples never download or rebuild responses.
The historical [BH C513 v1 resource](https://github.com/LiYangji/HELENA-resources/releases/tag/bh-c513-v1)
is for v0.1.0 grids and is incompatible with v0.1.1. `fetch --legacy` preserves
access to that unchanged archive; it is not preparation for current examples.
See the [root cache instructions](../README.md#response-cache-lifecycle) and
[v0.1.1 release notes](../docs/release_notes_v0.1.1.md).

## Data sources and attribution

HELENA's BSD-3-Clause software license grants no additional rights in
third-party material. Applicable source-specific notices remain in force;
inclusion does not imply endorsement by the data providers.

### Benchmark reference products

The five `quickstart/*_reference.npz` files contain numerical curves, not
bundled AM3 or LeHaMoC executables. The published reference material is
Cerruti et al., *A Comprehensive Hadronic Code Comparison
for Active Galactic Nuclei - Online Material*, v1 (2025),
[doi:10.5281/zenodo.17177642](https://doi.org/10.5281/zenodo.17177642),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The `pgamma`, `ps`, and `leha` references and the published LeHaMoC SSC curves
are extracted numerical products covered by that license. Their NPZ metadata
record the applied unit/frame conversions; the compact files are not unmodified copies of the
original archive. The AM3 SSC curves were locally regenerated, and the
SYN-cool reference records AM3 1.0.1 and its source commit. No independent
runtime installation of either comparison code is needed. Retain this
attribution, the source DOI, and the transformation metadata when reusing the
derived published curves. The licensing of locally generated numerical
references is not inferred from AM3's software license.

### TXS photon data

This example uses data and services from the Space Science Data Center
([SSDC](https://www.ssdc.asi.it/)), a facility of the Italian Space Agency
(ASI), including the SED Builder developed at ASDC
([Stratta et al. 2011](https://arxiv.org/abs/1103.0749)). The
[NuBlazar](https://www.ssdc.asi.it/nustarblaz/) spectral products are described
by [Middei et al. 2022](https://doi.org/10.1093/mnras/stac1185).
Swift and Fermi-LAT values also use
[Keivani et al. 2018](https://doi.org/10.3847/1538-4357/aad59a).
Campaign measurements and the H.E.S.S./VERITAS differential upper-limit
tables (Supplementary Tables S5 and S7) come from the 2018 multimessenger
campaign, [IceCube Collaboration et al.](https://doi.org/10.1126/science.aat1378).

The canonical example reads the packaged
[`txs0506_2017_curated.csv`](txs0506_2017/data/txs0506_2017_curated.csv):
49 detections and ten 95% upper limits enter the photon likelihood; 714 rows
are context only and seven are excluded. The local symmetric diagnostic uses
48 detections, excluding the asymmetric Swift-XRT point.

[`photon_data_provenance.json`](txs0506_2017/data/photon_data_provenance.json)
records source-group mappings, selection, observing windows, units,
transformations, and adopted errors without changing the loader's numerical
inputs. The preserved observations and documented uncertainty approximations
form a scope-limited application example, not a statistically calibrated data
product. Measurements are campaign-centered, not all simultaneous; unknown
source confidence conventions and incomplete historical retrieval records
remain identified in the provenance.

### IceCube effective area

The detector response comes from *IceCube data from 2008 to 2017 related to
analysis of TXS 0506+056*, [doi:10.21234/B4QG92](https://doi.org/10.21234/B4QG92).
The packaged
[`icecube_aeff_ic86c.csv`](txs0506_2017/data/icecube_aeff_ic86c.csv)
reformats the original `Aeff_IC86c.txt` table and header without changing
its effective-area numerical values. It is an IceCube response, not a
HELENA-generated measurement. The [data provenance](txs0506_2017/data/PROVENANCE.md)
records its assumptions, source notices, and the author's release decision;
attribution is not itself a license.
