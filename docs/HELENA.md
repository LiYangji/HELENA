# HELENA

## Name

HELENA stands for:

**High-Energy Lepto-hadronic Emission and Neutrino Analysis**

## Scientific scope

- **High-Energy**: gamma-ray, nonthermal, and relativistic-source modeling.
- **Lepto-hadronic**: coupled electron/positron and proton physics, including
  secondary pion and muon populations.
- **Emission**: synchrotron, inverse-Compton, secondary and cascade radiation,
  and source-to-observer photon output.
- **Neutrino**: photohadronic and hadronic neutrino production with Earth-frame
  multi-flavor observables.
- **Analysis**: forward modeling, SED fitting, differentiable sensitivities,
  Jacobians, and local identifiability diagnostics.

## Modeling philosophy

The current core model is steady and one-zone, with an end-to-end
differentiable implementation and the same scientific API on CPU and GPU.
Physical outputs remain linear; sensitivity outputs use a natural-log relative
representation. Science contracts take precedence over implementation details.

## Formal identity

Software name: HELENA

Python namespace: `helena`

Distribution name: `helena-astro`
