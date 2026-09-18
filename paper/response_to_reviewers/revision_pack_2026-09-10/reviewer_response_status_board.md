# Reviewer Response Status Board

This is the coauthor-facing status summary for Revision 1.

## High-level readiness

The response is structurally ready, but not numerically final. It covers every reviewer item and
uses placeholders only where a value still needs a cluster run, metadata audit, or final code-release
evidence.

## Item-by-item status

| Item | Status | Ready for coauthor text review? | Blocks final submission? | Main blocker |
|---|---|---:|---:|---|
| A1 Fig. 3d vs Fig. S3 | partial | yes | yes | final QE 1-4 convergence and SI Fig. S3 metadata audit |
| A2 train/test overlap | pending | yes | yes | SOAP/RMSD nearest-neighbor audit |
| A3 kinetic uncertainty | pending | yes | yes | 3-seed Scratch and FT-600K barrier errors |
| A4 scope/generalizability | done-local | yes | no | none |
| A5 Scratch-5% control | pending | yes | yes | Scratch-5% training/evaluation |
| A6 interpretability | partial | yes | partly | MACE perturbation test and final code-release evidence |
| A7 related work/delta | done-local | yes | no | bibliography integration check |
| C8 zero-force exporters | partial | yes | yes | final release hash/evidence |
| C9 MD protocol asymmetry | partial | yes | yes | corrected FT-600K MD rerun |
| C10 grouped split | partial | yes | yes | grouped-split retrained RMSE |
| C11 projected silhouette | done-local | yes | no | manuscript integration |
| C12 zero feature stub | done-local | yes | no | code-release integration |
| C13 SHAP code/stubs | partial | yes | yes | final code release and stub audit |

## Reviewer-facing tone decisions

- Acknowledge the reviewer is right where the manuscript or code was unclear.
- Do not over-defend the original wording.
- Do not call the Fig. S3 Foundation result "exceptional" in the revision.
- Do not call the 0.336050 eV DFT value fully converged unless the final QE restart confirms it.
- Do not claim SHAP explains MACE internals directly; it explains the surrogate error model.
- Keep broad generality as an outlook, not a completed result.

## Current numbers safe to cite in the working draft

| Quantity | Value | Caveat |
|---|---:|---|
| DFT path 1-4 reference candidate | 0.336050 eV | current audit says unconverged but numerically stationary |
| Foundation fixed-path barrier on 1-4 | 1.024296 eV | update error if final DFT reference changes |
| Foundation fixed-path signed error on 1-4 | +0.688246 eV | computed against 0.336050 eV |
| Scratch fixed-path barrier on 1-4 | 4.537677 eV | update error if final DFT reference changes |
| Scratch fixed-path signed error on 1-4 | +4.201627 eV | computed against 0.336050 eV |
| FT-600K fixed-path barrier on 1-4 | 0.496792 eV | update error if final DFT reference changes |
| FT-600K fixed-path signed error on 1-4 | +0.160742 eV | computed against 0.336050 eV |
| FT-MultiT fixed-path barrier on 1-4 | 0.823040 eV | update error if final DFT reference changes |
| FT-MultiT fixed-path signed error on 1-4 | +0.486990 eV | computed against 0.336050 eV |
| Foundation SI Fig. S3 self-MLFF NEB barrier | 0.410000 eV | metadata audit pending; not a fixed-path DFT error |
| Original 6191-d euclidean silhouette | 0.333 | replacement for projected-space claim |
| t-SNE projection sensitivity silhouette | 0.397 +/- 0.027 | projection-only diagnostic |
| PHATE 2D silhouette | 0.473 +/- 0.000 | projection-only diagnostic |
| SHAP surrogate CV R2, FT-600K | 0.9819 | surrogate, not direct MACE explanation |
| SHAP surrogate CV R2, FT-MultiT | 0.8756 | surrogate, not direct MACE explanation |
| SHAP surrogate CV R2, Scratch | 0.9730 | surrogate, not direct MACE explanation |

## Files to send to coauthors

- `response_to_reviewers_revised_full.md`
- `pending_values_register.csv`
- `overleaf_search_replace_checklist.md`
- `reviewer_markers_for_overleaf.tex`
- `reviewer_response_status_board.md`
- A1 supplement: `../reviewer_a1_fig3_figs3_update/response_letter_A1_replacement.md`
