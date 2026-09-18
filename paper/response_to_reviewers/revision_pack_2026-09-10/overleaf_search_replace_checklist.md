# Overleaf Search/Replace Checklist for Reviewer-Linked Edits

This checklist is meant for manual Overleaf integration. Each item gives a stable text anchor,
the reviewer marker to paste nearby, and whether the text is ready or still waiting for a placeholder.

Use the no-op macro from `reviewer_markers_for_overleaf.tex`:

```latex
\providecommand{\reviewermap}[3]{}
```

Then paste the relevant `\reviewermap{tag}{status}{summary}` line immediately before the paragraph,
figure caption, or table it describes.

---

## 1. Abstract scope claim

Search anchor:

```text
This work establishes migration-based non-equilibrium probes as a data-efficient, generalizable standard
```

Paste marker before the sentence:

```latex
\reviewermap{REV-A4}{done-local}{Reviewer asked us to avoid claiming a general standard from one material and one architecture. Rewritten as a candidate framework demonstrated on Cr-doped Sb2Te3.}
```

Replace with:

```latex
This work demonstrates migration-based non-equilibrium probes as a data-efficient candidate framework for MLFF evaluation, developed and validated on one representative material system, and provides actionable guidance for robust MLFF development. Whether this framework generalizes across chemistries and architectures remains an open question that we identify as a priority for future cross-system validation.
```

Status: ready.

---

## 2. Introduction research question

Search anchor:

```text
Can non-equilibrium probes generated from methods, such as NEB, provide a generalizable and efficient benchmark
```

Paste marker before the item:

```latex
\reviewermap{REV-A4}{done-local}{Generalizability is now framed as a question and future validation target, not a completed claim.}
```

Replace with:

```latex
\item Can non-equilibrium probes generated from methods, such as NEB, provide an efficient benchmark for specialist versus generalist models, and to what extent does such a benchmark generalize beyond a single material system?
```

Status: ready.

---

## 3. Introduction delta paragraph and related work

Search anchor:

```text
diagnostic framework for designing more data-efficient learning loops
```

Paste marker after the paragraph:

```latex
\reviewermap{REV-A7}{done-local}{Reviewer asked for nearest-neighbor positioning and a clear delta relative to our prior report. Add requested citations and delta paragraph.}
```

Insert:

```latex
This article extends our preliminary workshop report~\citep{cao2025migration} in several substantive ways: (i) an extended molecular-dynamics transport analysis with uncertainty quantification (diffusivity and thermal conductivity with block-averaged error bars); (ii) a self-consistent NEB stability criterion, in which each model must optimize the full elastic band from the same endpoints, complementing fixed-geometry barrier evaluation; (iii) a feature-level interpretability framework based on SHAP surrogate models of per-configuration error; and (iv) an interlayer-sliding analysis probing robustness to collective, non-local displacements.
```

Status: ready, assuming citation keys are present in the bibliography.

---

## 4. Methods train/test split

Search anchor:

```text
Both FT-600K and FT-Multi_T models used identical train/validation/test splits
```

Paste marker before the paragraph:

```latex
\reviewermap{REV-A2/REV-C10}{pending}{Grouped split by trajectory/pathway provenance; pending SOAP-RMSD-overlap-audit and grouped-split-retrained-rmse.}
```

Replace or extend with:

```latex
Both FT-600K and FT-Multi\_T models used identical grouped train/validation/test splits (0.8:0.1:0.1), where all frames sharing the same trajectory or pathway provenance were assigned to the same partition. This grouped split prevents adjacent or near-duplicate configurations from the same trajectory from appearing in both training and test sets. For the revised barrier analysis, we additionally report a nearest-neighbor SOAP/RMSD audit between the in-gap NEB images and the FT-600K training set (Table~SX).
```

Status: pending final grouped-split retraining and SOAP/RMSD values.

---

## 5. Methods training data provenance

Search anchor:

```text
Energy and forces were read using the keys
```

Paste marker before or after the paragraph:

```latex
\reviewermap{REV-C8}{pending-release}{Old NEB exporters could write fabricated zero forces. The reported models are stated to use real-force AIMD frames only; pending final code-release hash for exporter fix.}
```

Add sentence:

```latex
The reported fine-tuned models were trained on AIMD frames with real DFT energies and forces; NEB-derived extxyz exports without real force labels were not used in the reported training runs.
```

Status: scientifically ready, pending final code release hash.

---

## 6. Main-text Fig. 3d / NEB local migration paragraph

Search anchor:

```text
Our DFT calculations serve as the ground-truth reference. This establishes
```

Paste marker before the paragraph:

```latex
\reviewermap{REV-A1}{partial}{Fig. 3d is fixed-geometry DFT-path scoring. Current 1-4 DFT reference candidate is 0.336050 eV; Foundation fixed-path barrier is 1.024296 eV, error +0.688246 eV. Pending final-QE-1-4-barrier and SI-FigS3-audit.}
```

Replace with:

```latex
Our DFT NEB calculation defines the fixed-path reference used for the in-gap pathway in Fig.~\ref{fig:barrier}d. For path 1-4, the current audited activation barrier is $E_{\mathrm{a}}=0.336050$~eV. Because the archived QE run is numerically stationary but not yet formally converged under the final force criterion, we report this value as the audited Revision-1 reference candidate and include the convergence status in Table~SX; the final restartable QE NEB run will either confirm this value or replace it in the final manuscript tables. All model errors in Fig.~\ref{fig:barrier}d are computed against this single pathway-specific value rather than against rounded or cross-pathway approximations.
```

Status: partial. Replace `0.336050` and signed errors if the final QE restart changes the reference.

---

## 7. Main-text fixed-vs-self NEB insertion

Search anchor:

```text
NEB Stability as an Additional Robustness Criterion
```

Paste marker before the new paragraph:

```latex
\reviewermap{REV-A1}{partial}{Reviewer challenged the apparent Fig. 3d/Fig. S3 inconsistency. This paragraph defines the two different NEB operators and reports them side-by-side.}
```

Insert the full paragraph from:

```text
outputs/reviewer_a1_fig3_figs3_update/proposed_text_edits_E3_E4_SI_replacement.md
```

Status: partial because final QE and SI Fig. S3 metadata audit remain pending.

---

## 8. Deep-penetration "by chance" sentence

Search anchor:

```text
not through physical insight but random chance
```

Paste marker before the paragraph:

```latex
\reviewermap{REV-A3}{pending-cluster}{The single-run "chance" interpretation must be replaced by seed-stability evidence over Scratch and FT-600K retraining seeds.}
```

Temporary replacement:

```latex
The scratch model's lower error on this OOD task is treated in the revision as a seed-stability question rather than as an assumed mechanistic explanation. We therefore report barrier-error mean and standard deviation over three independently trained Scratch and FT--600K models; if the low Scratch error is stable across seeds, we interpret it as a reproducible model behavior, whereas a large spread indicates seed-dependent behavior.
```

Status: pending `multiseed-kinetic-barrier-errors`.

---

## 9. FT-600K discussion / Scratch-5% control

Search anchor:

```text
The Critical Role of Task-Specific Fine-Tuning
```

Paste marker near this subsection:

```latex
\reviewermap{REV-A5}{pending-cluster}{Add Scratch-5% random-initialization control trained on the same approximately 1000 configurations as FT-600K.}
```

Insert once numbers are available:

```latex
To separate the effect of foundation pretraining from the effect of training-set size, we trained a Scratch-5\% control on the same grouped subset used for FT--600K. The Scratch-5\% model gives [PLACEHOLDER: scratch-5pct-control-rmse-and-barriers], compared with [PLACEHOLDER: FT600K-grouped-control-values] for FT--600K.
```

Status: pending.

---

## 10. Latent-space analysis

Search anchor:

```text
provides a direct mechanistic explanation
```

Paste marker before the paragraph:

```latex
\reviewermap{REV-A6/REV-C11/REV-C12}{done-local}{Claims softened to "consistent with"; original 6191-d euclidean silhouette 0.333; t-SNE sensitivity 0.397 +/- 0.027; PHATE 0.473 +/- 0.000; zero feature stub dropped.}
```

Replace "provides a direct mechanistic explanation" with:

```latex
is consistent with the models' differing performance on the diffusion task, though low-dimensional projections alone cannot establish a mechanistic link
```

Status: ready.

---

## 11. Fig. 5 caption

Search anchor:

```text
Average silhouette scores (from $t$-SNE)
```

Paste marker before the caption:

```latex
\reviewermap{REV-C11}{done-local}{Silhouette values are recomputed in original descriptor space; projected-space values retained only as a flagged baseline.}
```

Replace caption panel text with the E5c block in `proposed_text_edits.md` or the updated revision pack.

Status: ready.

---

## 12. SHAP text

Search anchor:

```text
This approach allows us to interpret the complex MACE potential
```

Paste marker before the SHAP paragraph:

```latex
\reviewermap{REV-A6/REV-C13}{partial}{SHAP explains surrogate error predictions, not MACE directly. Current 5-fold CV R2: FT-600K 0.9819, FT-MultiT 0.8756, Scratch 0.9730. Pending MACE-CrCr-perturbation-test and final code-release evidence.}
```

Use replacement from E6, with the real CV values:

```latex
This approach allows us to interpret a surrogate model of MACE's error behavior. The surrogate's 5-fold cross-validated $R^2$ values are 0.9819 for FT--600K, 0.8756 for FT--MultiT, and 0.9730 for Scratch. The SHAP values below therefore describe which structural descriptors are most important for the surrogate's error predictions; they should not be read as direct access to the internal decision process of MACE itself.
```

Status: partial. MACE perturbation and release-code evidence remain pending.

---

## 13. Conclusion

Search anchor:

```text
Our findings highlight several broader principles
```

Paste marker before the paragraph:

```latex
\reviewermap{REV-A4}{done-local}{Broader principles are now framed as hypotheses supported by this case study and requiring cross-system validation.}
```

Replace with E7 conclusion text from `proposed_text_edits.md`.

Status: ready.

---

## 14. SI Fig. S3 paragraph

Search anchor:

```text
In contrast, the MACE foundation model, without any system-specific fine-tuning, successfully converged the NEB calculation
```

Paste marker before the paragraph:

```latex
\reviewermap{REV-A1}{partial}{SI Fig. S3 is self-consistent MLFF NEB. The 0.41 eV value is a model-reported MLFF barrier, not a fixed-path DFT-reference error.}
```

Replace with E4b from:

```text
outputs/reviewer_a1_fig3_figs3_update/proposed_text_edits_E3_E4_SI_replacement.md
```

Status: partial pending SI Fig. S3 metadata audit.

---

## 15. SI Fig. S3 caption

Search anchor:

```text
yielding a migration barrier of 0.41 eV
```

Paste marker before the figure caption or figure environment:

```latex
\reviewermap{REV-A1}{partial}{Caption must say model-reported MLFF barrier and distinguish this self-consistent MLFF NEB from fixed-geometry DFT-path scoring.}
```

Replace caption with E4c from:

```text
outputs/reviewer_a1_fig3_figs3_update/proposed_text_edits_E3_E4_SI_replacement.md
```

Status: partial pending SI Fig. S3 metadata audit.
