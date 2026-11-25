# Reviewer Response Readiness Audit

Last updated: 2026-10-18
Overleaf source commit: `29310f4`

## Bottom Line

The response package is ready for coauthor review, but not ready for final submission. The reason is not
the writing structure: every reviewer point now has a response, a manuscript location, and a status. The
remaining blockers are numerical and provenance placeholders that must be cleared by the Rockfish/local
audit workflow before the response can be sent.

## Point-by-Point Status

| Reviewer point | Current status | Blocks final submission? | What is still needed |
|---|---|---:|---|
| A1 Fig. 3d vs Fig. S3 inconsistency | Text-ready, data-partial | Yes | Final QE NEB convergence metadata for path 1--4 and accepted DFT barrier |
| A2 FT-600K train/test overlap | Grouped audit and retraining complete for declared FT-600K/Scratch/Scratch-5% files; DFT anchor provisional | Yes | Final QE reference closure; FT-Multi$_T$ was not part of this grouped round |
| A3 kinetic uncertainty / “chance” claim | Grouped three-seed in-gap and path-resolved candidate statistics complete; hard-OOD scope remains conditional | Yes | Converged higher-distortion DFT references, final path classification, and same-system self-relaxed DFT rescoring |
| A4 generalizability scope | Complete | No | Optional second-system result only if we choose that route |
| A5 Scratch-5% control | Grouped 89-frame control complete for the controlled in-gap scope; exact approximately 1000-frame ablation remains open | Yes | Decide whether to run the exact group-preserving ablation; retain current limitation explicitly |
| A6 t-SNE/PHATE and SHAP claims | Mostly complete | Partial | SHAP code-release provenance; optional full-set rerun / perturbation test if retained |
| A7 related work and delta | Complete | No | None |
| C8 zero-force exporters | Response drafted, evidence pending | Yes | Cluster training-log evidence and patched exporter commit |
| C9 asymmetric MD protocol | Values quarantined, rerun pending | Yes | Corrected FT-600K 100000-step trajectory and all-model transport table |
| C10 ungrouped split | Method written, retraining pending | Yes | Grouped split manifest and retrained Table S1 values |
| C11 silhouette on t-SNE | Complete locally | No | Only rerun if the source dataset changes |
| C12 always-zero feature stub | Manuscript correction complete | Partial | Code-release commit showing removal/deletion |
| C13 SHAP code absent / empty stubs | Manuscript correction complete, repo pending | Yes | MigrationBench commit with working SHAP pipeline and no empty advertised stubs |

## Current Manuscript Marker Coverage

All reviewer points A1--A7 and C8--C13 have corresponding source markers. The exact line index is:

`Revision1/response_to_reviewers/reviewer_marker_index.md`

The markers are invisible in the compiled PDF because the source defines:

```tex
\providecommand{\reviewermap}[3]{}
```

## Current Submission Risk

Do not submit while the manuscript still contains `[PLACEHOLDER: ...]`. These placeholders are useful for
coauthor review because they show exactly what value is missing, but they must be replaced or removed in
the final Overleaf source.

The highest-risk unresolved items are:

1. DFT NEB convergence status and final barrier for path 1--4.
2. FT-600K train/test proximity audit.
3. Three-seed kinetic uncertainty.
4. Scratch-5% control.
5. Corrected FT-600K MD transport.
6. Zero-force training-data provenance evidence.
7. MigrationBench code/data release commit and Hugging Face dataset manifest.

