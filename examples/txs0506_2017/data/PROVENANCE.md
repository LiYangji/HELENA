# TXS 0506+056 2017 data provenance

This directory freezes the public numerical data used by the canonical
example. The notebook is offline: `fetch_public_sed.py` is a reproducible
one-time extractor, not a HELENA runtime dependency.

## Scientific state and selection window

- The source redshift is `z = 0.3365 +/- 0.0010` (Paiano et al. 2018,
  [doi:10.3847/2041-8213/aaad5e](https://doi.org/10.3847/2041-8213/aaad5e)).
- IceCube Collaboration et al. (2018), Science 361, eaat1378
  ([doi:10.1126/science.aat1378](https://doi.org/10.1126/science.aat1378)),
  defines the IceCube-170922A campaign and explicitly notes that the SED
  measurements are not all simultaneous.
- The likelihood selection uses the published Keivani et al. (2018) Epoch 1,
  MJD 58003--58033 (15 days before through 15 days after the alert), not a
  newly chosen window. Numerical Swift tables are CDS catalogue
  [J/ApJ/864/84](https://cdsarc.cds.unistra.fr/ftp/J/ApJ/864/84/).
- The source-informed model follows Cerruti et al. (2019), MNRASL 483 L12
  ([doi:10.1093/mnrasl/sly210](https://doi.org/10.1093/mnrasl/sly210)), with
  the corrections in Cerruti et al. (2021), MNRASL 502 L21
  ([doi:10.1093/mnrasl/slaa188](https://doi.org/10.1093/mnrasl/slaa188)).

## Frozen files

### `txs0506_public_raw.ecsv`

The raw table contains all 729 points returned on 2026-08-25 by the official
ASI/SSDC SED Builder REST query at the resolved J2000 coordinate
`RA=77.358185156 deg, Dec=+5.693148283 deg`. Dataset identifiers, bands,
catalogue names, observation MJDs, bibliographic URLs and original numerical
columns are retained. These points are archival context unless explicitly
selected below; their presence never implies simultaneity.

### `txs0506_2017_curated.csv`

Every row has observer-frame `frequency_hz`, `energy_ev`, and
`sed_flux_erg_cm2_s = nu F_nu = E F_E`, lower/upper uncertainties, an
upper-limit flag, observing times, provenance, and one explicit role:

- `fit`: compatible campaign detections entering the photon likelihood;
- `constraint`: compatible contemporaneous 95% VHE upper limits;
- `context`: archival, non-simultaneous, extended-radio, or otherwise
  display-only data;
- `exclude`: provenance-preserved rows excluded from both display and
  likelihood. Six current rows have no machine-readable uncertainty and one
  has a lower error exceeding its positive flux, incompatible with the
  log-SED contract; no error is guessed or repaired.

The active likelihood rows are:

- 44 machine-readable NuSTAR spectral bins from SSDC/NuBlazar, all carrying
  the exact observing interval MJD 58025.094594907--58025.750798611;
- the extinction-corrected Swift-UVOT value at `1e15 Hz` and the
  quasi-simultaneous Swift-XRT MJD 58023.8535 power law from Keivani et al.;
  the XRT row is deterministically evaluated at 1 keV from the published
  0.3--10 keV flux and photon index. The published asymmetric two-sided 90%
  ranges are propagated by endpoint evaluation and divided by
  `Phi^-1(0.95) = 1.644853626951...` as an adopted normal-quantile error-scale
  approximation. This marginal-endpoint envelope is not a calibrated joint
  confidence interval and does not establish Gaussian one-sigma coverage;
- the published AGILE 200 MeV and Fermi-LAT Epoch-1 1.44 GeV pivots;
- the published observed MAGIC 130 GeV normalization.

The ten active constraints are the six H.E.S.S. and four VERITAS
differential 95% upper-limit bins in the official Science supplement
([PDF](https://icecube.wisc.edu/wp-content/uploads/2021/01/detection-flaring-blazar.pdf)).
Each is converted at the geometric bin energy with
`E^2 dN/dE`, using `1 TeV = 1.602176634 erg`. They enter the likelihood only
when the Public Earth-observed SED uses validated Dominguez-2011 EBL
attenuation. HAWC is not assigned an invented numerical row because the
campaign release supplies its quasi-differential limits only graphically.

Radio and the remaining SSDC/catalogue measurements stay context-only.
Dense archival coverage is shown but does not gain likelihood weight merely
by having many rows.

## `icecube_aeff_ic86c.csv`

This preserves the tabulated values from `Aeff_IC86c.txt`, reformatted as CSV,
in the official release
"IceCube data from 2008 to 2017 related to analysis of TXS 0506+056"
([doi:10.21234/B4QG92](https://doi.org/10.21234/B4QG92)). It is the
point-source-selection effective area at the source declination, for an
incident flux containing equal parts muon neutrinos and antineutrinos.
The example applies it to the model Earth-mixed `nu_mu + anti_nu_mu` number
flux for the declared six-month exposure.

This remains a **post-fit consistency check**, not an IceCube event
likelihood, Bayesian posterior, or joint photon-neutrino fit.

## Release disposition and reuse conditions

The preserved CSV/ECSV values, errors, roles, and saved notebook results form
a documented numerical example. NuSTAR spectral-bin, Swift-UVOT, Fermi-LAT,
and AGILE error-confidence conventions remain unverified. The XRT envelope
treatment and MAGIC statistical/systematic quadrature are adopted
approximations, not established joint confidence calibrations. MAGIC's
horizontal 15% value is not a verified Gaussian energy sigma and does not
enter the likelihood weights. No specific incorrect active-data conversion
has been established. H.E.S.S./VERITAS descriptive timing corrections do not
change the adopted selection; historical retrieval gaps remain documented.
On 2026-09-06, the author accepted the preserved real observations and these
documented error approximations as a scope-limited application example.
This does not verify the unknown source conventions or establish statistical
calibration. The author subsequently directed publication with the adopted
numerical files unchanged and accurate source attribution. This release
decision is not new evidence of third-party permission: no additional
individual permissions have been obtained, and the previously unverified
source-specific rights information below remains unverified.

The [machine-readable provenance](photon_data_provenance.json) retains
source-specific citations, transformations, and unknown quantities. HELENA's
BSD-3-Clause software license does not grant rights over third-party data.
The [SSDC notice](https://www.ssdc.asi.it/doc/SSDC_catalogs_help.pdf) requests
acknowledgment of SSDC/ASI data and services but does not specify a blanket
redistribution license. The
[VizieR usage rules](https://cds.unistra.fr/vizier-org/licences_vizier.html)
permit scientific use with original author, publication, and publisher
citations; catalogue-origin conditions remain applicable. This does not
relicense the complete mixed-source snapshot. The
[IceCube data release](https://icecube.wisc.edu/data-releases/2018/07/icecube-data-from-2008-to-2017-related-to-analysis-of-txs-0506056/)
and its DOI identify the effective-area source, but the reviewed record
provides no explicit redistribution license. A public data location or a
citation is not substituted for an applicable reuse basis.
