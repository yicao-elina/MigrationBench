# Reviewer Marker Index

Last updated: 2026-09-11

This file maps each reviewer concern to the exact manuscript locations marked with
`\reviewermap{...}{...}{...}` in the Overleaf source. The macro is defined as a no-op, so these
markers do not print in the PDF; they are only for coauthor review inside the `.tex` files.

Current Overleaf commit containing these markers: `29310f4`

## Marker Status Key

- `done-local`: text or local reanalysis is complete enough for coauthor review.
- `partial`: the main clarification is written, but final cluster numbers or convergence metadata still need to replace placeholders.
- `pending`: local/cluster values still need to be filled before final submission.
- `pending-local`: local code/data packaging or manifest work remains.
- `pending-cluster`: Rockfish calculation or re-run remains.

## Main Text: `sn-article-final.tex`

| Reviewer point | Line | Status | What changed / what to review |
|---|---:|---|---|
| REV-A4 | 155 | done-local | Abstract claim changed from “generalizable standard” to a candidate framework demonstrated on Cr-doped Sb2Te3. |
| REV-A7 | 194 | done-local | Added delta relative to the earlier workshop/arXiv report and positioned the work against related migration/fine-tuning literature. |
| REV-A4 | 200 | done-local | Research question reframed as an open generalizability question rather than a completed claim. |
| REV-C9 | 253 | pending-cluster | Transport conclusions marked preliminary; final values require corrected 100000-step MD workflow. |
| REV-A1 | 279 | partial | Fig. 3d clarified as fixed-geometry DFT-path scoring. Current 1--4 DFT candidate: 0.336050 eV; Foundation fixed-path error: +0.688246 eV. |
| REV-A3 | 303 | pending-cluster | Deep-penetration ranking softened; final claim waits for multi-seed and restarted QE NEB checks. |
| REV-A5 | 307 | pending-cluster | Scratch-5% control added as required future/result placeholder. |
| REV-A1 | 317 | partial | Fig. S3 clarified as a separate self-consistent MLFF-NEB operator, not the same quantity as Fig. 3d. |
| REV-A6 / REV-C11 / REV-C12 | 349 | done-local | Latent-space claim softened from direct mechanism to consistency with task-level behavior; zero-stub issue noted as removed. |
| REV-C11 | 359 | done-local | Fig. 5 caption now says original descriptor-space checks are primary; projected embeddings are visualization diagnostics. |
| REV-A6 / REV-C13 | 372 | partial | SHAP text now describes a surrogate error model and reports current 5-fold CV R2 values. Final release hash still pending. |
| REV-A4 | 402 | done-local | Conclusion reframed as case-study-supported hypotheses. |
| REV-A4 | 405 | done-local | Generality held as an outlook, not a fully established result. |
| REV-C13 | 431 | pending-local | Code availability marked for SHAP pipeline release and cleanup of prior empty stubs. |

## Supplementary Information: `sn-article-SI.tex`

| Reviewer point | Line | Status | What changed / what to review |
|---|---:|---|---|
| REV-A1 | 124 | partial | DFT NEB references require per-path convergence status; paths not formally converged are marked candidate references. |
| REV-C8 | 153 | pending-local | Dataset provenance manifest added: extraction scripts, key mapping, checksums, source trajectory. |
| REV-A2 / REV-C10 | 163 | pending | Grouped split and train/test overlap audit described; SOAP/RMSD values still pending. |
| REV-C9 | 307 | pending-cluster | FT-600K transport trajectory quarantined until corrected long-run protocol is complete. |
| REV-C12 | 375 | done-local | Always-zero force-sensitivity placeholder channel removed from feature matrix and described as not used. |
| REV-A1 | 416 | partial | Fig. S3 paragraph now says 0.41 eV is the Foundation model's self-consistent MLFF barrier. |
| REV-A1 | 422 | partial | Fig. S3 caption now distinguishes MLFF-NEB path generation from fixed-geometry DFT-path scoring. |
| REV-A6 / REV-C13 | 433 | partial | SHAP SI caption marked for surrogate CV R2 and final code-release provenance. |

## Still Missing Before Final Submission

- Replace all manuscript `[PLACEHOLDER: ...]` text with accepted values or remove the corresponding pending claim.
- Final QE NEB convergence table for all reference paths.
- Train/test proximity audit: nearest-neighbor SOAP distance and RMSD for benchmark NEB images.
- Multi-seed Scratch and FT-600K kinetic statistics.
- Scratch-5% control.
- Corrected FT-600K long-run MD transport values.
- Full MigrationBench/Hugging Face manifest commit or DOI.
- Final SHAP code-release provenance and, if retained, Cr--Cr / Cr--Sb perturbation test.
