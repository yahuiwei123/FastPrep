# Upstream source provenance

Both `bfc_gauss_mixture.py` and `bfc_rbf_scatter.py` come from the TissueGuidedBiasFieldCorrection repository, revision `f6983fc0308283b1c4574782613222bfa28cd084`, as retained during the 2026-11-18 comparison. `UPSTREAM_README.md` retains the accompanying upstream explanation and citation. Reliable-core fitting and application changes are implemented separately in `bin/tissue_bfc/`.

Local execution optimization, 2026-11-24: `bfc_rbf_scatter.py` now caches fixed tissue-membership sums, their class-activity decisions, and the point-spacing diagonal. The dynamic weighted-total check is retained for tiny classes; classes with fixed sum >= 1 cannot reach its 1e-3 threshold because adaptive weights are clamped to >= 0.2. Loss transfers to the host occur only for the original log entries and end-of-epoch convergence test. Objective terms, reduction order, optimizer settings, iteration limits and convergence decisions are preserved. The Gaussian source remains unchanged. These local edits are not an upstream release or endorsement.

Source repository: https://github.com/yahuiwei123/TissueGuidedBiasFieldCorrection

No separate license text was present in the retained upstream snapshot. This toolkit does not invent or grant an upstream license; preserve existing notices and resolve redistribution rights with the upstream owner before external redistribution. This is a local research integration, not a claim of upstream endorsement or scientific validation.
