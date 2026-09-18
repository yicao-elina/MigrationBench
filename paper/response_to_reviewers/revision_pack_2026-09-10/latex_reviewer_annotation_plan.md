# LaTeX Reviewer Annotation Plan

The goal is to make Overleaf review easy for the advisor and coauthors. These marks are intended as
source comments, so they do not change the compiled PDF. If visible draft notes are desired, add a
separate `\revnote{}` macro later; for the manuscript source, comments are safer.

Recommended comment format:

```latex
% REVIEWER-MAP[REV-A1 | status=partial | response=response_to_reviewers_revised_full.md#A1]
% Issue: Fig. 3d fixed DFT path vs SI Fig. S3 self-consistent MLFF NEB were not clearly separated.
% Action: define the two operators, remove ambiguous "~0.3 eV", report side-by-side values.
% Pending: final-QE-1-4-barrier; SI-FigS3-audit; optional DFT-on-MLFF-path.
```

## Main manuscript: `sn-article-final.tex`

| Tag | Insert near | Reviewer item | Status | Short source comment |
|---|---|---|---|---|
| REV-A4 | Abstract closing sentence | A4 | done-local | Scope "generalizable standard" to candidate framework on one system. |
| REV-A7 | Introduction related-work/delta paragraph | A7 | done-local | Add nearest-neighbor citations and delta vs prior report. |
| REV-A2/REV-C10 | Methods data split paragraph | A2, C10 | pending values | Grouped split and SOAP/RMSD overlap audit. |
| REV-C8 | Methods training data provenance | C8 | pending release hash | Confirm no zero-force NEB exports entered reported training. |
| REV-C9 | Methods MD protocol | C9 | pending corrected FT-600K run | Mark old naive-FT MD as quarantined until rerun. |
| REV-A1 | Sec. 3.2 local migration / Fig. 3d | A1 | partial | Fixed-geometry DFT-path evaluation; current 1-4 reference 0.336050 eV with convergence caveat. |
| REV-A3 | Sec. 3.2 deep penetration paragraph | A3 | pending cluster | Replace "by chance" with seed-stability result. |
| REV-A5 | Sec. 3.2 FT-600K discussion | A5 | pending cluster | Add Scratch-5% same-data-volume control. |
| REV-A6/REV-C11/REV-C12 | Sec. 3.4 latent-space analysis and Fig. 5 caption | A6, C11, C12 | done-local | Use original-space metrics; label projected silhouette as baseline only; drop zero feature stub. |
| REV-A6/REV-C13 | Sec. 3.5 SHAP and Fig. 6 caption | A6, C13 | partial | Scope claims to surrogate; report 5-fold CV R2; final code release pending. |
| REV-A4 | Conclusion | A4 | done-local | Present generality as future cross-system validation. |
| REV-C13 | Code/Data availability | C13 | pending release | Point to actual non-stub scripts and release hash. |

## SI: `sn-article-SI.tex`

| Tag | Insert near | Reviewer item | Status | Short source comment |
|---|---|---|---|---|
| REV-A1 | SI "Nudged Elastic Band (NEB) Benchmark for Cr Migration" | A1 | partial | State Fig. S3 is self-consistent MLFF NEB, not fixed DFT-path scoring. |
| REV-A1 | SI Fig. S3 caption | A1 | partial | Say "model-reported MLFF barrier of 0.41 eV"; remove "exceptional" and "~0.3 eV" as reference wording. |
| REV-A2 | New SI overlap table/figure | A2 | pending | SOAP/RMSD nearest-neighbor audit. |
| REV-A3/REV-A5/REV-C10 | Table S1 | A3, A5, C10 | pending | Multi-seed barrier uncertainty, Scratch-5%, grouped-split RMSE. |
| REV-C9 | SI transport table | C9 | partial | FT-600K old protocol quarantined; corrected rerun pending. |
| REV-A6/REV-C11/REV-C12 | SI latent robustness figure/table | A6, C11, C12 | done-local | Report original-space and sensitivity metrics. |
| REV-A6/REV-C13 | SI SHAP surrogate table | A6, C13 | partial | Report CV R2 values and release-code provenance. |

## Exact comment snippets to paste

### Abstract

```latex
% REVIEWER-MAP[REV-A4 | status=done-local | response=A4]
% Issue: Reviewer asked us to avoid claiming a general standard from one material and one architecture.
% Action: rewritten as a candidate framework demonstrated on Cr-doped Sb2Te3; broader generality moved to outlook.
```

### Methods: split/provenance

```latex
% REVIEWER-MAP[REV-A2/REV-C10 | status=pending | response=A2,C10]
% Issue: possible train/test overlap and frame-level random split.
% Action: use grouped splits by trajectory/pathway provenance; report SOAP/RMSD nearest-neighbor audit.
% Pending: SOAP-RMSD-overlap-audit; grouped-split-retrained-rmse.
```

```latex
% REVIEWER-MAP[REV-C8 | status=pending-release | response=C8]
% Issue: old NEB exporters could write fabricated all-zero forces.
% Action: confirm reported FT models used AIMD frames with real DFT forces; hard-disable fake force export.
% Pending: final-code-release-hash-for-zero-force-fix.
```

### Sec. 3.2 NEB

```latex
% REVIEWER-MAP[REV-A1 | status=partial | response=A1]
% Issue: Fig. 3d and SI Fig. S3 appeared inconsistent.
% Action: define Fig. 3d as fixed-geometry DFT-path evaluation and Fig. S3 as self-consistent MLFF NEB.
% Current values: DFT 1-4 reference candidate 0.336050 eV; Foundation fixed-path barrier 1.024296 eV
% (+0.688246 eV error); SI Fig. S3 Foundation self-MLFF barrier 0.41 eV.
% Pending: final-QE-1-4-barrier; SI-FigS3-audit; optional DFT-on-MLFF-path.
```

```latex
% REVIEWER-MAP[REV-A3/REV-A5 | status=pending | response=A3,A5]
% Issue: kinetic barrier uncertainty and missing Scratch-5% control.
% Action: add 3-seed Scratch/FT-600K errors and Scratch-5% same-data-volume control.
% Pending: multiseed-kinetic-barrier-errors; scratch-5pct-control-rmse-and-barriers.
```

### Sec. 3.4 latent space

```latex
% REVIEWER-MAP[REV-A6/REV-C11/REV-C12 | status=done-local | response=A6,C11,C12]
% Issue: projected-space silhouette and always-zero feature stub.
% Action: claims softened to "consistent with"; report original 6191-d silhouette and projection sensitivity;
% drop zero feature channel. Key values: original euclidean silhouette 0.333; t-SNE sensitivity 0.397 +/- 0.027;
% PHATE 0.473 +/- 0.000.
```

### Sec. 3.5 SHAP

```latex
% REVIEWER-MAP[REV-A6/REV-C13 | status=partial | response=A6,C13]
% Issue: SHAP claims were too direct and code release was incomplete.
% Action: state SHAP explains surrogate error model, not MACE directly; report 5-fold CV R2.
% Current values: FT-600K 0.9819, FT-MultiT 0.8756, Scratch 0.9730.
% Pending: MACE-CrCr-perturbation-test; final-SHAP-code-release-hash; final-empty-stub-audit.
```

### SI Fig. S3

```latex
% REVIEWER-MAP[REV-A1 | status=partial | response=A1]
% Issue: SI Fig. S3 wording made self-consistent MLFF NEB look directly comparable to fixed DFT-path scoring.
% Action: caption/text must say this is model-in-the-loop MLFF NEB and the 0.41 eV value is a model-reported
% MLFF barrier, not a fixed-path DFT-reference error.
% Pending: SI-FigS3-audit; optional DFT-on-MLFF-path.
```
