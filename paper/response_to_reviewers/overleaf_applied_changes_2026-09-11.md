# Overleaf Applied Changes - 2026-09-11

Overleaf project: https://www.overleaf.com/project/693b515e50f0071cb03b505c

Git commits pushed to Overleaf: `d81611c890bb66e8475b0de4203a8c258d1e6e9c`, `7a303bd`, `29310f4`

## Files Edited

- `sn-article-final.tex`
- `sn-article-SI.tex`
- `sn-bibliography.bib`
- `Revision1/response_to_reviewers/response_letter.md` local response draft rewritten against the latest source state
- `Revision1/response_to_reviewers/reviewer_marker_index.md` local marker-to-line index added

## Reviewer Challenges Addressed In Source

- `REV-A1`: Clarified that Fig. 3d and Fig. S3 are different NEB operators.
  - Fig. 3d: fixed-geometry scoring on DFT-relaxed NEB images.
  - Fig. S3: self-consistent MLFF NEB path generation.
  - Current fixed-path DFT candidate for path 1--4: `0.336050 eV`.
  - Foundation fixed-path barrier/error: `1.024296 eV`, `+0.688246 eV`.
  - SI Fig. S3 value: `0.41 eV`, now explicitly described as a model-reported MLFF barrier, not a DFT reference.
- `REV-A2 / REV-C10`: Added grouped-split and nearest-neighbor SOAP/RMSD audit placeholders for train/test leakage control.
- `REV-A3`: Softened deep-penetration interpretation; final model ranking waits for converged QE NEB and multi-seed checks.
- `REV-A4`: Scoped the contribution from "generalizable standard" to a candidate framework demonstrated on one representative system.
- `REV-A5`: Added Scratch-5% control placeholder to isolate foundation-prior effects from data-volume effects.
- `REV-A6 / REV-C11 / REV-C12 / REV-C13`: Softened latent-space and SHAP claims.
  - t-SNE/PHATE are qualitative diagnostics.
  - Original-space silhouette checks are now reported.
  - SHAP explains a surrogate error model, not MACE internals directly.
  - Corrected the SHAP top-feature text from the previously drafted `Cr-Cr_n13_l3` to the audited `Cr-Sb_n43_l0`.
  - Added an SI marker explaining removal of the always-zero force-sensitivity placeholder channel.
- `REV-C8`: Added dataset provenance / manifest placeholder.
- `REV-C9`: Marked long-run transport values as pending corrected 100000-step MD workflow.
- `REV-C13`: Added a code-availability marker requiring the working SHAP pipeline and cleanup of empty repository stubs before final submission.

## Validation

- `sn-article-final.tex` compiled successfully with `latexmk -pdf`.
- `sn-article-SI.tex` compiled successfully with `latexmk -pdf`.
- No fatal LaTeX errors remained after escaping the marker text `5\%`.
- Remaining warnings are template/layout/citation-cycle style warnings, not introduced fatal errors.

## Still Pending Before Final Submission

- Final QE NEB convergence table and replacement of any candidate barriers.
- DFT-on-MLFF-path or QE restart results if we decide to quantitatively compare the Foundation self-consistent path against DFT.
- Train/test proximity audit: nearest-neighbor SOAP distance and RMSD for benchmark NEB images.
- Scratch-5% control model results.
- Corrected long-run MD transport results.
- Final SHAP code-release provenance and optional Cr-Sb local-environment perturbation test.
- Final MigrationBench Hugging Face dataset commit/DOI and checksums.
