# Response to Reviewer 1 - Revised Working Draft

Draft date: 2026-09-10

This draft is written as a working response letter. Values that are already available from the
Revision 1 audit are written explicitly. Values that still require cluster runs or final integration
are kept as bracketed placeholders, with the calculation route listed in
`pending_values_register.csv`.

We thank the reviewer for the careful reading and for pointing out several places where the
manuscript and the released code did not make the provenance of the results clear enough. We have
treated the comments as a useful audit of the scientific claims. Below we describe the changes made
to the manuscript, the supporting calculations, and the remaining values that will be filled before
submission.

---

## A1. Fig. 3d vs Fig. S3 NEB barrier discrepancy

**Reviewer concern.** The reviewer asked us to reconcile the apparent inconsistency between
main-text Fig. 3d, where the Foundation model overestimates the in-gap barrier by about 0.7 eV, and
SI Fig. S3, where the Foundation model reports a self-run NEB barrier of 0.41 eV. The reviewer also
asked us to fix the mixed reference values of 0.34 eV and about 0.3 eV.

**Response.** We agree that the original wording was not clear enough. The two figures report two
different NEB-based measurements:

- Fixed-geometry DFT-path evaluation (Fig. 3d): the DFT-relaxed NEB images are held fixed, and each
  MLFF is asked to assign energies to those same geometries. This measures energetic accuracy on a
  DFT-defined path:
  \[
  \Delta E^{\mathrm{fixed}}_M(\Gamma_{\mathrm{DFT}})
  =
  \max_i E_M(X_i^{\mathrm{DFT}})-E_M(X_0^{\mathrm{DFT}}).
  \]

- Self-consistent MLFF NEB optimization (Fig. S3): the MLFF starts from endpoint structures and
  relaxes its own elastic band under its own forces:
  \[
  \Gamma_M^\star=\operatorname{NEB}_M(X_0,X_1), \qquad
  \Delta E^{\mathrm{self}}_M
  =
  \max_i E_M(X_i^{M,\star})-E_M(X_0).
  \]

These are not the same operator. Fig. 3d asks whether the model gives the right barrier on a fixed
DFT path, whereas Fig. S3 asks whether the model can generate a stable path by itself. In the
submitted SI, this distinction was not stated clearly, which made the two values look contradictory.

For the fixed-geometry in-gap path 1-4, the current audited DFT barrier from the raw QE `neb.out` is
0.336050 eV. The local audit currently labels this archived calculation as `UNCONVERGED` but
numerically stationary, so we will either keep it with an explicit convergence caveat or replace it
with `[PLACEHOLDER: final-QE-1-4-barrier]` after the restartable QE NEB closure run.

Using the current audited 1-4 reference, the fixed-geometry values are:

| Model | Fixed-geometry MLFF barrier (eV) | DFT reference used (eV) | Signed error (eV) |
|---|---:|---:|---:|
| MACE Foundation | 1.024296 | 0.336050 | +0.688246 |
| MACE Scratch | 4.537677 | 0.336050 | +4.201627 |
| MACE FT-600K | 0.496792 | 0.336050 | +0.160742 |
| MACE FT-MultiT | 0.823040 | 0.336050 | +0.486990 |

The 0.41 eV value in SI Fig. S3 is kept only as the Foundation model's self-consistent MLFF NEB
barrier, not as a fixed-path DFT-reference error. We have revised the response and proposed SI text
to remove language such as "exceptional" and to describe the result more carefully as stable
model-in-the-loop path generation. We will also add a side-by-side table specifying pathway,
endpoints, image source, relaxation engine, energy engine, barrier reported, DFT reference, and
convergence status.

**Manuscript/SI change.** Main-text Sec. 3.2 and SI Fig. S3 text/caption are marked with reviewer
tags `REV-A1` and `REV-C1`. The exact replacement text is in
`reviewer_a1_fig3_figs3_update/proposed_text_edits_E3_E4_SI_replacement.md`.

---

## A2. Train/test overlap for FT-600K

**Reviewer concern.** The reviewer asked us to rule out overlap between FT-600K training structures
and the in-gap NEB images.

**Response.** We agree this check is necessary. We have fixed the data-splitting logic so that
frames are split by trajectory/pathway provenance rather than by independent frame-level shuffle.
This avoids placing near-duplicate adjacent frames in different partitions.

The remaining quantitative check is:

`[PLACEHOLDER: SOAP-RMSD-overlap-audit]`

This will report the minimum SOAP distance between each in-gap NEB image and its nearest FT-600K
training frame, with Cartesian RMSD checks on the closest pairs. If any near-duplicate falls below
the pre-registered threshold, we will retrain FT-600K with the NEB-adjacent trajectory segment
excluded and report the new barrier error.

**Manuscript change.** Methods Sec. 2.1 will be marked with `REV-A2/REV-C10` near the train/valid/test
split description. The NEB result paragraph in Sec. 3.2 will point to the SI overlap audit table.

---

## A3. Uncertainty on kinetic results

**Reviewer concern.** The reviewer asked for at least three seeds for Scratch and FT-600K, especially
to test whether the Scratch model's deep-penetration result was a stable effect or a seed-dependent
accident.

**Response.** We agree and have converted this into a direct uncertainty test. The revised plan uses
seeds 123, 234, and 345 for Scratch and FT-600K on grouped splits. We will report mean +/- standard
deviation for barrier errors on the in-gap path and the deep-penetration paths.

`[PLACEHOLDER: multiseed-kinetic-barrier-errors]`

The current "by chance" wording will not remain as an unsupported statement. It will be replaced by
a seed-based conclusion: if the Scratch deep-penetration error is stable across seeds, we will say
so; if it varies strongly, we will state that the original single-run result was seed-dependent.

**Manuscript change.** Sec. 3.2 deep-penetration paragraph and Table S1 are marked with `REV-A3`.

---

## A4. Scope of generalizability claims

**Reviewer concern.** The reviewer asked us to soften claims that the work establishes a general
standard, since the current study uses one material system and one architecture family.

**Response.** We agree. The revised manuscript now describes the work as a candidate framework
demonstrated on Cr-doped Sb2Te3, rather than as a general standard. The abstract, introduction, and
conclusion are rewritten so that broader generality is presented as an outlook requiring validation
across additional chemistries and model architectures.

No new numerical placeholder is required for this response unless we decide to add an optional
second dopant case:

`[OPTIONAL PLACEHOLDER: second-dopant-cross-validation]`

**Manuscript change.** Abstract, introduction research questions, and conclusion are marked with
`REV-A4`.

---

## A5. Scratch-5% control

**Reviewer concern.** The reviewer asked for a Scratch-5% model trained on the same number of
configurations as FT-600K, to separate the effect of foundation pretraining from the effect of
training-set size.

**Response.** We agree that this is the cleanest control. The planned control uses the same grouped
FT-600K training subset, identical MACE hyperparameters, random initialization, and seeds 123, 234,
and 345. It will be evaluated on equilibrium RMSE and the same migration-barrier tasks.

`[PLACEHOLDER: scratch-5pct-control-rmse-and-barriers]`

**Manuscript change.** Sec. 3.2 and Table S1 are marked with `REV-A5`.

---

## A6. Interpretability claims: t-SNE/PHATE and SHAP

**Reviewer concern.** The reviewer asked us to soften the latent-space interpretation, compute
separation in the original descriptor space, report projection sensitivity, and report SHAP
surrogate fidelity.

**Response.** We agree. We no longer describe t-SNE/PHATE as a direct mechanistic explanation. The
revised wording says these analyses are consistent with the observed model behavior, and we report
quantitative checks outside the 2D projection.

The local re-analysis gives:

| Space | Overall silhouette | Notes |
|---|---:|---|
| old 2D t-SNE, p=30, seed 42 | 0.409 | retained only as the old baseline |
| 2D t-SNE, p in {5,30,50}, 3 seeds | 0.397 +/- 0.027 | projection sensitivity |
| PHATE 2D, 3 seeds | 0.473 +/- 0.000 | projection sensitivity |
| original 6191-d, euclidean | 0.333 | replacement metric |
| original 6191-d, cosine | 0.277 | replacement metric |
| original 6191-d, z-scored euclidean | -0.019 | shows scaling sensitivity |
| PCA 95%, z-scored euclidean | -0.026 | conservative high-dimensional check |

For SHAP, we have separated in-sample scores from cross-validated surrogate fidelity. The revised
5-fold CV R2 values are:

| Model | 5-fold CV R2 | 5-fold CV MAE |
|---|---:|---:|
| FT-600K | 0.9819 | 0.0380 |
| FT-MultiT | 0.8756 | 0.0269 |
| Scratch | 0.9730 | 0.0262 |

These values support using the surrogate as a model of error behavior, but the text will state that
SHAP explains the surrogate's error predictions, not MACE directly. A direct perturbation check on
MACE inputs remains pending:

`[PLACEHOLDER: MACE-CrCr-perturbation-test]`

**Manuscript change.** Sec. 3.4, Fig. 5 caption, Sec. 3.5, and Fig. 6 caption are marked with
`REV-A6/REV-C11/REV-C12/REV-C13`.

---

## A7. Related work and delta from our earlier report

**Reviewer concern.** The reviewer asked us to position the manuscript against close recent work and
to state clearly what is new relative to our earlier arXiv/workshop report.

**Response.** We agree and have added the requested citations and a delta paragraph. The revised
text positions this work relative to: MACE fine-tuning for Li-ion diffusion, foundational MLIP
migration-barrier evaluation, and systematic PES softening of universal MLIPs. We also state that
this Wiley version adds extended MD transport analysis, self-consistent NEB stability, SHAP-based
surrogate interpretability, and interlayer-sliding analysis beyond the prior report.

No numerical placeholder remains for this item.

**Manuscript change.** Introduction, Sec. 3.2, softening discussion, and bibliography are marked with
`REV-A7`.

---

## C8. Fabricated all-zero forces in NEB exporters

**Reviewer concern.** The reviewer identified scripts that paired real QE energies with fabricated
all-zero forces in training-format extxyz files.

**Response.** We agree that this was a serious code issue. Based on the current audit, there is no
evidence that the reported FT-600K or FT-MultiT models were trained on those zero-force NEB exports.
The affected exporters have been changed so that forces are written only when real force data are
available; otherwise force export is disabled or refused with an error.

The final response should cite the release commit or final code archive:

`[PLACEHOLDER: final-code-release-hash-for-zero-force-fix]`

**Manuscript/code change.** Methods data provenance and Code Availability are marked with `REV-C8`.

---

## C9. Asymmetric MD protocol for naive FT

**Reviewer concern.** The reviewer identified that the FT-600K/naive fine-tuning MD run used a much
shorter trajectory and denser sampling interval than the other models.

**Response.** We agree. The old FT-600K transport value is quarantined and should not be used as a
final comparison until the corrected run finishes. The three currently valid rows from the local
transport re-analysis are:

| Model | D (cm2/s) | kappa (W m-1 K-1) | Status |
|---|---:|---:|---|
| Foundation | (0.7 +/- 7.9) x 10^-9 | (2.4 +/- 8.6) x 10^-3 | valid local re-analysis |
| Scratch | (-3.4 +/- 3.2) x 10^-9 | (0.9 +/- 1.3) x 10^-2 | valid local re-analysis |
| FT-MultiT | (4.8 +/- 4.8) x 10^-8 | (2.0 +/- 1.7) x 10^-2 | valid local re-analysis |
| FT-600K | (3.9 +/- 4.0) x 10^-8 | (0.9 +/- 1.2) x 10^-2 | invalid old protocol; quarantined |

The corrected FT-600K run remains pending:

`[PLACEHOLDER: corrected-FT600K-MD-transport]`

**Manuscript/code change.** MD protocol, Fig. 2/transport figure, and SI transport table are marked
with `REV-C9`.

---

## C10. Ungrouped train/test split

**Reviewer concern.** The reviewer identified that the original split was performed at frame level,
which could allow nearby frames from the same trajectory to appear in both training and test sets.

**Response.** We agree. The split has been changed to group by trajectory/pathway provenance, with a
zero-overlap assertion across train, validation, and test partitions. The final retrained RMSE
values from grouped splits are still pending:

`[PLACEHOLDER: grouped-split-retrained-rmse]`

**Manuscript/code change.** Methods split description and Table S1 are marked with `REV-C10`.

---

## C11. Silhouette scores computed on embeddings

**Reviewer concern.** The reviewer noted that the reported silhouette scores were computed on
projected t-SNE/PHATE coordinates rather than the original feature space.

**Response.** We agree. The revised analysis reports original descriptor-space values and labels the
old embedding-space values as a projection-only baseline. The main replacement value is original
6191-d euclidean silhouette = 0.333, with the scaling-sensitive checks reported in the SI.

No numerical placeholder remains for the local analysis, although the manuscript integration remains
to be applied in Overleaf.

**Manuscript change.** Sec. 3.4 and Fig. 5 caption are marked with `REV-C11`.

---

## C12. Always-zero force-sensitivity feature stub

**Reviewer concern.** The reviewer identified an always-zero feature channel in the latent-probe
scripts.

**Response.** We confirm this issue. The dead feature channel is removed from the analysis inputs and
the revised latent-space metrics are computed without it. We do not introduce a new force-sensitivity
claim unless a genuine finite-difference feature is computed later.

No numerical placeholder remains.

**Manuscript/code change.** SI latent-analysis method and code release notes are marked with
`REV-C12`.

---

## C13. SHAP code and repository stubs

**Reviewer concern.** The reviewer noted that the released repository did not contain the working
SHAP pipeline and included empty stubs.

**Response.** We agree. The response will not claim reproducibility from files that were not actually
released. The working SHAP pipeline is being ported into the release, and the manuscript will describe
only the released pipeline. The current local SHAP surrogate audit gives the 5-fold CV R2 values
listed in A6, but the final response still needs the release-level evidence:

`[PLACEHOLDER: final-SHAP-code-release-hash]`

`[PLACEHOLDER: final-empty-stub-audit]`

**Manuscript/code change.** Sec. 3.5 and Code Availability are marked with `REV-C13`.

---

## Short status summary for coauthors

The response is structurally ready, but not numerically final. The strongest completed pieces are
the scope edits, literature positioning, latent-space re-analysis, SHAP surrogate CV R2 audit, and
the Fig. 3d vs Fig. S3 protocol clarification. The main open items are final QE NEB convergence,
SOAP/RMSD overlap audit, multi-seed retraining, Scratch-5% control, corrected FT-600K MD, and final
release-code evidence.
