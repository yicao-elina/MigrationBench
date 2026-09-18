# Replacement Draft for `proposed_text_edits.md` - E3/E4 and SI Fig. S3 Text

Use these blocks to replace the current E3/E4 text in
`Revision1/response_to_reviewers/proposed_text_edits.md`, and to add a new SI edit section.

---

## E3. Sec. 3.2 reference-value sentence - single canonical DFT reference and convergence status

**Location**: Sec. 3.2 "Local Migration Events", L272.

**OLD**:
```latex
Our DFT calculations serve as the ground-truth reference. This establishes a (reasonably low) migration energy barrier ($E_{\mathrm{a}}$) of 0.34~eV for this process. The results, presented in Fig.~\ref{fig:barrier}d, reveal a clear hierarchy in performance and highlight the considerable impact of training strategy on the prediction of kinetic barriers.
```

**NEW**:
```latex
Our DFT NEB calculation defines the fixed-path reference used for the in-gap pathway in Fig.~\ref{fig:barrier}d. For path 1-4, the current audited activation barrier is $E_{\mathrm{a}}=0.336050$~eV. Because the archived QE run is numerically stationary but not yet formally converged under the final force criterion, we report this value as the audited Revision-1 reference candidate and include the convergence status in Table~SX; the final restartable QE NEB run will either confirm this value or replace it in the final manuscript tables. All model errors in Fig.~\ref{fig:barrier}d are computed against this single pathway-specific value rather than against rounded or cross-pathway approximations.
```

**If final QE restart converges before submission, replace the second sentence with**:
```latex
For path 1-4, the converged activation barrier is $E_{\mathrm{a}}=[PLACEHOLDER: final-QE-1-4-barrier]$~eV, obtained from the restartable QE NEB calculation summarized in Table~SX.
```

`[PLACEHOLDER: final-QE-1-4-barrier]` is computed by parsing the final QE `neb.out` with
`scripts/migrationbench/parse_qe_neb_output.py` and regenerating
`tables/dft_neb_barriers.tex`.

---

## E4. Sec. 3.2 - fixed-geometry versus self-consistent MLFF NEB evaluation

**Location**: new paragraph inserted in Sec. 3.2 "NEB Stability as an Additional Robustness Criterion",
immediately after the Fig. S3 discussion paragraph.

**OLD**: no existing text - insertion.

**NEW**:
```latex
\paragraph{Fixed-geometry versus self-consistent NEB evaluation.}
We distinguish two NEB-based tests that were not clearly separated in the original manuscript.
In Fig.~\ref{fig:barrier}d, the DFT-relaxed NEB images are held fixed and each MLFF is evaluated by
single-point energies on the same geometries. This fixed-geometry protocol measures energetic
fidelity on a prescribed DFT path:
\[
\Delta E^{\mathrm{fixed}}_M(\Gamma_{\mathrm{DFT}})
=
\max_i E_M(X_i^{\mathrm{DFT}})-E_M(X_0^{\mathrm{DFT}}).
\]
In Fig.~S3, by contrast, each MLFF performs its own self-consistent NEB optimization from the same
endpoints, without access to the DFT-relaxed intermediate images:
\[
\Gamma_M^\star=\mathrm{NEB}_M(X_0,X_1),\qquad
\Delta E^{\mathrm{self}}_M
=
\max_i E_M(X_i^{M,\star})-E_M(X_0).
\]
These protocols therefore answer different questions: fixed-geometry evaluation isolates barrier
energy prediction on a DFT-defined path, whereas self-consistent MLFF NEB tests whether the model's
forces can generate a stable migration path. For the audited in-gap path 1-4, the current
fixed-geometry barriers are 1.024~eV for MACE Foundation, 4.538~eV for MACE Scratch, 0.497~eV for
FT--600K, and 0.823~eV for FT--MultiT, corresponding to signed errors of +0.688, +4.202, +0.161,
and +0.487~eV relative to the current DFT reference candidate of 0.336050~eV. The Foundation
self-consistent MLFF NEB barrier reported in Fig.~S3 is 0.41~eV; this value is a model-relaxed
MLFF barrier, not a fixed-path DFT-reference error. We therefore report the two protocols
side-by-side and avoid using the self-consistent MLFF barrier as evidence of fixed-path energetic
accuracy unless it is re-evaluated by DFT on the model-relaxed images.
```

**If final QE restart changes the DFT reference**, update the four signed errors using:
```text
signed_error = fixed_geometry_mlff_barrier_eV - final_QE_DFT_barrier_eV
```

---

## E4b. Revise the SI Fig. S3 paragraph

**Location**: `sn-article-SI.tex`, subsection "Nudged Elastic Band (NEB) Benchmark for Cr Migration",
paragraph currently beginning "In contrast, the MACE foundation model..."

**OLD**:
```latex
In contrast, the MACE foundation model, without any system-specific fine-tuning, successfully converged the NEB calculation. It predicts a migration barrier of 0.41 eV (Figure~\ref{fig:neb_benchmark}b), a value in good agreement with DFT calculations for similar in-gap diffusion pathways (~0.3 eV). This level of accuracy, close to the bounds of chemical accuracy, demonstrates the foundation model's exceptional capability to generalize to complex transition-state configurations. This benchmark underscores that evaluating performance on dynamic, physically relevant processes is a critical and necessary step for validating the true capabilities of MLFFs.
```

**NEW**:
```latex
In contrast, the MACE foundation model, without system-specific fine-tuning, completed the
self-consistent MLFF NEB optimization from the same endpoint structures. This calculation differs
from the fixed-geometry evaluation in the main text: here the model is not given the DFT-relaxed
intermediate images, but instead relaxes its own elastic band under its own predicted forces. The
resulting MLFF-relaxed pathway gives a model-reported barrier of 0.41~eV
(Figure~\ref{fig:neb_benchmark}b). We therefore interpret this result primarily as evidence of
stable model-in-the-loop path generation, rather than as a fixed-path DFT barrier error. The
corresponding fixed-geometry evaluation on the audited DFT path is reported separately in the main
text and Table~SX. A direct DFT validation of the model-relaxed path would require DFT single-point
energies or a QE NEB refinement initialized from the MLFF-relaxed images; this calculation is
included in the Revision-1 validation plan.
```

---

## E4c. Revise the SI Fig. S3 caption

**Location**: `sn-article-SI.tex`, caption for `\label{fig:neb_benchmark}`.

**OLD**:
```latex
\caption{Nudged Elastic Band (NEB) benchmark of MACE models for a Cr atom migration. (a) Visualization of the initial and final states of the diffusion pathway. (b) The converged Minimum Energy Pathway (MEP) calculated with the MACE foundation model, yielding a migration barrier of 0.41 eV. (c) The evolution of the relative system energy (top panel) and the maximum force ($f_{max}$, bottom panel) during the NEB optimization for each model. The foundation model (red) converges smoothly. In contrast, the scratch (orange), FT-600K (dark blue), and FT-MultiT (light blue) models all exhibit explosive behavior, where a rapid increase in $f_{max}$ indicates instability and leads to the termination of the calculation.}
```

**NEW**:
```latex
\caption{Self-consistent MLFF Nudged Elastic Band (NEB) benchmark of MACE models for a Cr atom
migration. (a) Visualization of the endpoint structures used to initialize the diffusion pathway.
(b) The model-relaxed pathway generated by the MACE Foundation model, yielding a model-reported
MLFF barrier of 0.41~eV. This self-consistent MLFF NEB protocol is distinct from the fixed-geometry
DFT-path energy evaluation shown in the main text. (c) The evolution of the relative system energy
(top panel) and maximum force ($f_{\max}$, bottom panel) during the MLFF NEB optimization for each
model. The Foundation model converges smoothly under this protocol, whereas Scratch, FT--600K, and
FT--MultiT exhibit unstable force growth leading to termination.}
```

---

## Required placeholders and how to compute them

`[PLACEHOLDER: SI-FigS3-audit]`

- Locate the raw Fig. S3 source trajectory/log for the Foundation self-consistent NEB.
- Record endpoint IDs/path label, number of images, model checkpoint, model version, seed,
  optimizer, spring constant, final \(f_{\max}\), and barrier extraction script.
- Output target: `data_processed/figS3_self_consistent_neb_audit.csv`.

`[PLACEHOLDER: DFT-on-MLFF-path]`

- Convert the Foundation self-consistent NEB images to QE explicit-image input.
- Run either DFT single-points on each image or a restartable QE NEB initialized from these images.
- Compute:
  \[
  \Delta E_{\mathrm{DFT}}(\Gamma_M^\star)
  =
  \max_i E_{\mathrm{DFT}}(X_i^{M,\star})-E_{\mathrm{DFT}}(X_0).
  \]
- Output target: `data_processed/dft_on_foundation_self_neb_path.csv`.

`[PLACEHOLDER: final-QE-1-4-barrier]`

- Continue/restart the path 1-4 QE NEB until it satisfies the final force criterion.
- Parse with `scripts/migrationbench/parse_qe_neb_output.py`.
- Regenerate `tables/dft_neb_barriers.tex` and recompute all signed errors in Fig. 3d.
