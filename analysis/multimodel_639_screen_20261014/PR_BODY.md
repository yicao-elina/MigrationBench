## What this PR adds

This PR packages the complete 639-candidate static MLIP comparison as a provenance-first evidence bundle.

![Four-panel 639-candidate, 12-model diagnostic](https://raw.githubusercontent.com/yicao-elina/MigrationBench/codex/multimodel-639-evidence-pr/analysis/multimodel_639_screen_20261014/figures/fig_multimodel_639_overview.png)

### Verified computation

- Same 639 periodic 2050-atom structures evaluated with 9 additional checkpoints:
  - FT-600K: seeds 123, 234, 345;
  - FT-MultiT: seeds 123, 234, 345;
  - legacy compiled OMAT, MPA-0 medium and MP-0b3 medium.
- Existing 3 from-scratch results retained as the immutable comparison branch.
- Rockfish production array `31981441`: 32/32 tasks completed, exit code 0.
- Corrected collector `31986653`: completed with remote status `VERIFIED`.
- Combined matrix: 7,668 rows = 639 candidates × 12 models; 0 failed rows; 0 duplicate candidate–model pairs.

### Scientific result

- Candidates come from 3 source conditions and 9 trajectory/seed branches, not one trajectory.
- The selection is stage-balanced for failure attribution: 207 early-stable controls, 216 stable pre-onset, 208 precursors, 7 threshold/terminal structures and 1 stable 1000-ps target.
- It is representative of the audited failure-attribution frame, not a time-weighted MD population.
- Median 12-model maximum-force range increases from 3.07 eV Å⁻¹ in early-stable controls to 27.29 eV Å⁻¹ in precursors and 3,717.51 eV Å⁻¹ in threshold/terminal structures.
- Every candidate stage is outside the training SOAP q99 envelope; candidates are separated from both the training/validation/test region and the NEB reference region in the stored PCA diagnostic.

These results establish model divergence and descriptor-space distribution shift. They do not establish which model force is correct, DFT correctness, or causal attribution of the MD instability.

### Evidence and paper handoff

- `analysis/multimodel_639_screen_20261014/obsidian/` — Obsidian evidence package and navigation.
- `analysis/multimodel_639_screen_20261014/overleaf_handoff/` — copy-ready Nature-style Results/Discussion, SI Methods and figure caption TeX.
- `CLAIM_EVIDENCE_MATRIX.tsv` — claim-level mapping from manuscript language to data, scripts, jobs and status.
- `RESULTS_AND_DISCUSSION.md` — full interpretation and publication-boundary text.
- `PAPER_RELEASE_MANIFEST.json` — hashes, commit anchor and Overleaf handoff status.
- `analysis/combined_analysis_manifest.json` — machine-readable 12-model acceptance gate.

### Validation

- `git diff --check` passes for the committed evidence bundle.
- JSON manifests parse successfully.
- All analysis scripts pass Python syntax compilation.
- The PDF is one-page, unencrypted, and rendered by Plot Atlas; the PNG was visually inspected.

### Follow-up

DFT is intentionally not included in this PR. The next experiment is to construct and smoke-test approximately 120-atom PBC-preserving fragments selected from the high-disagreement structures, then submit a gated DFT batch.
