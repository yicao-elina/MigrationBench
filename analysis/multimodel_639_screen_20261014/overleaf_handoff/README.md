# Overleaf handoff｜639-candidate 12-model diagnostic

This directory contains copy-ready manuscript text and figure metadata. It is a handoff package only; it has not been synchronized into Overleaf and therefore has no Overleaf commit/version yet.

## Copy to Overleaf

1. Copy `../figures/fig_multimodel_639_overview.pdf` into the manuscript figure directory.
2. Add `multimodel_639_results_discussion.tex` to the relevant Results/Discussion subsection.
3. Add `multimodel_639_si_methods.tex` to the SI Methods section.
4. Add `multimodel_639_caption.tex` as the figure caption.
5. Use the table sources listed in `../analysis/` to generate or attach SI tables; do not copy the large 7,668-row matrix into the manuscript.
6. After synchronization, record the Overleaf commit/version in `../PAPER_RELEASE_MANIFEST.json` and in the PR description.

## Publication wording boundary

The text distinguishes verified model-response/distribution-shift evidence from pending DFT correctness and causal attribution. Keep the labels “legacy compiled OMAT”, “descriptor-space diagnostic”, “stage-balanced failure-attribution sample” and “static MLIP evaluation” unchanged unless a new provenance audit supports a stronger claim.
