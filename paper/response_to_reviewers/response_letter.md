# Response to Reviewer 1 — Draft Aligned with Revision 1

Last updated: 2026-09-11  
Overleaf source commit: `29310f4`

We thank the reviewer for the careful reading and for pointing out several places where our original
manuscript and repository did not make the evidence chain clear enough. We have tried to handle the
comments in two ways. First, wherever the issue was a wording, scope, or provenance problem, we have
revised the manuscript directly. Second, wherever the issue requires a new calculation or a cluster-side
audit, we mark the response with an explicit placeholder and do not treat the result as final until the
corresponding workflow has finished.

The reviewer-related manuscript edits are marked in the Overleaf source with no-op
`\reviewermap{...}{...}{...}` comments. These markers do not appear in the PDF; they are there so the
coauthors can see which reviewer point each edited paragraph addresses. A line-by-line index is in
`Revision1/response_to_reviewers/reviewer_marker_index.md`.

## Current Readiness

The response is structurally ready, but not numerically final. The text-only corrections and several local
audits are already incorporated. The remaining placeholders are mainly for Rockfish calculations:

| Area | Status |
|---|---|
| Fig. 3d vs Fig. S3 protocol distinction | Text updated in main text and SI; final QE convergence metadata still pending |
| Generalizability / scope language | Text-complete |
| Related-work positioning and delta from our earlier report | Text-complete |
| t-SNE/PHATE wording and original-space silhouette audit | Text-complete with current local values |
| SHAP surrogate framing and CV R2 | Text updated with current local CV values; final code-release provenance pending |
| Train/test overlap audit | Pending SOAP/RMSD values |
| Three-seed Scratch and FT-600K kinetic uncertainty | Pending cluster runs |
| Scratch-5% control | Pending cluster runs |
| Corrected FT-600K long-run MD transport | Pending cluster run |
| MigrationBench/Hugging Face manifest | Pending final dataset commit/DOI |

---

## A. Main Manuscript Comments

### A1. Foundation-model in-gap barrier discrepancy between Fig. 3d and Fig. S3

**Reviewer concern.** The reviewer asked whether Fig. 3d and Fig. S3 use the same pathway and DFT
reference, why the Foundation model appears to be poor in the main text but good in the SI, and why the
reference barrier is written as both 0.34 eV and approximately 0.3 eV.

**Response.** We agree that the original text made two different evaluations look like one inconsistent
evaluation. We have revised the main text and SI to define the two protocols separately.

In Fig. 3d, the DFT-relaxed NEB image sequence is fixed. Each MLFF is only asked to assign energies to
the same DFT geometries:

\[
\Delta E^{\mathrm{fixed}}_M(\Gamma_{\mathrm{DFT}})
=
\max_i E_M(X_i^{\mathrm{DFT}})-E_M(X_0^{\mathrm{DFT}}),
\qquad
\epsilon^{\mathrm{fixed}}_M
=
\Delta E^{\mathrm{fixed}}_M-\Delta E_{\mathrm{DFT}} .
\]

This is a fixed-path scoring test. It asks whether the model gives the right energy profile on a
DFT-supplied path.

In Fig. S3, the MLFF is not given the DFT-relaxed intermediate images. It starts from the endpoints and
relaxes its own elastic band:

\[
\Gamma_M^\star=\operatorname{NEB}_M(X_0,X_1),
\qquad
\Delta E^{\mathrm{self}}_M
=
\max_i E_M(X_i^{M,\star})-E_M(X_0).
\]

This is a self-consistent MLFF-NEB stability test. It asks whether the model can generate a stable path
under its own forces.

For the in-gap path labeled 1--4 in the Revision 1 path ledger, the current audited DFT barrier parsed
from the QE output is **0.336050 eV**. This value is currently treated as a candidate reference because
the audit marks the run as numerically stationary but not yet formally converged under the final force
criterion. We will either keep this value with the final convergence metadata, or replace it with the
value from the restarted QE closure run.

Using the current candidate reference, the fixed-path values are:

| Model | Fixed-path MLFF barrier (eV) | DFT candidate reference (eV) | Signed error (eV) |
|---|---:|---:|---:|
| MACE Foundation | 1.024296 | 0.336050 | +0.688246 |
| MACE Scratch | 4.537677 | 0.336050 | +4.201627 |
| MACE FT-600K | 0.496792 | 0.336050 | +0.160742 |
| MACE FT-MultiT | 0.823040 | 0.336050 | +0.486990 |

The 0.41 eV value in Fig. S3 is now described only as the Foundation model's self-consistent MLFF barrier
on its own optimized band. It is not a second DFT reference and not the same quantity as the fixed-path
error in Fig. 3d.

**Placeholder before final submission.**  
`[PLACEHOLDER: final QE NEB convergence table for path 1--4; final accepted DFT reference barrier; raw-log path and checksum]`

**Marked manuscript locations.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:279), [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:317)
- SI: [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:124), [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:416), [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:422)

---

### A2. Train/test overlap for FT-600K

**Reviewer concern.** The reviewer asked us to rule out the possibility that the strong FT-600K
in-gap result comes from NEB-adjacent structures appearing in the fine-tuning set.

**Response.** We agree this needs to be checked directly. The revised workflow uses grouped splits rather
than frame-level random splits. Frames that share the same trajectory or configuration-family provenance
are assigned to the same partition, so near-duplicate neighboring frames cannot be split across train and
test by construction.

We are also computing the nearest-neighbor distance from each in-gap NEB image to the FT-600K training
set using SOAP descriptors, with Cartesian RMSD checks on the closest pairs. If any NEB image is too close
to the training data, we will retrain FT-600K with the adjacent segment excluded and rerun the barrier
evaluation.

**Placeholder before final submission.**  
`[PLACEHOLDER: minimum SOAP distance for each 1--4 NEB image; closest-pair RMSD values; exclusion-retrain result if the threshold is triggered]`

**Marked manuscript locations.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:307)
- SI: [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:163)

---

### A3. Uncertainty on kinetic results and the deep-penetration interpretation

**Reviewer concern.** The reviewer asked for at least three seeds for Scratch and FT-600K, and noted that
the current “by chance” explanation for the scratch model's deep-penetration behavior is not falsifiable.

**Response.** We agree. We have removed the unsupported “by chance” wording from the manuscript. The
revised text now says that the deep-penetration result is treated cautiously until multi-seed statistics
and the restarted QE NEB references are complete.

The planned test is direct: retrain Scratch and FT-600K with three fixed seeds, evaluate all models on the
same in-gap and deep-penetration reference paths, and report mean and standard deviation of the barrier
errors. If the scratch model's lower deep-penetration error is seed-stable, we will say that. If it is
seed-dependent, we will say that instead.

**Placeholder before final submission.**  
`[PLACEHOLDER: Scratch and FT-600K three-seed in-gap barrier error mean ± std; deep-penetration barrier error mean ± std; seed-stability interpretation]`

**Marked manuscript location.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:303)

---

### A4. Generalizability claims

**Reviewer concern.** The reviewer asked us to avoid presenting one material and one model architecture
as a fully general standard.

**Response.** We agree and have softened the scope throughout the manuscript. The abstract now describes
the work as a candidate framework developed and tested on one representative material system. The
Introduction now frames generalizability as a question rather than a conclusion. The Conclusions now say
that the results support broader hypotheses, rather than proving universal principles.

We are not presenting an additional material system as complete evidence in the current text unless the
optional second-system calculation is finished and passes the same audit standard.

**Placeholder before final submission.**  
None if we keep the text-only route.  
`[OPTIONAL PLACEHOLDER: second dopant/system barrier table if the optional cross-validation job is accepted]`

**Marked manuscript locations.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:155), [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:200), [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:402), [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:405)

---

### A5. Scratch-5% control

**Reviewer concern.** The reviewer asked for a random-initialization model trained on the same
approximately 1000 configurations as FT-600K, so that we can separate foundation-pretraining effects from
data-volume effects.

**Response.** We agree this is the right control. We have added a placeholder for this comparison in the
main text and will report it only after the control model is trained and evaluated with the same grouped
split and seeds.

**Placeholder before final submission.**  
`[PLACEHOLDER: Scratch-5% equilibrium RMSE; Scratch-5% in-gap barrier error mean ± std; Scratch-5% deep-penetration barrier error mean ± std; matched FT-600K values for comparison]`

**Marked manuscript location.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:307)

---

### A6. Interpretability claims: t-SNE/PHATE and SHAP

**Reviewer concern.** The reviewer asked us to soften the t-SNE/PHATE claims, report projection
sensitivity, compute separation in the original feature space, report SHAP surrogate quality, and avoid
claiming that SHAP directly explains MACE internals.

**Response.** We agree. The revised text no longer says that the embedding provides a direct mechanistic
explanation. We now describe the latent-space analysis as a diagnostic that is consistent with the task
benchmarks and should be read together with NEB, transport, and stability results.

For separability, we now report both projected and original-space values. The current audit gives:

| Feature space | Metric | Overall silhouette |
|---|---|---:|
| 2D t-SNE, old setting | Euclidean | 0.409 |
| 2D t-SNE, 9 settings | Euclidean | 0.397 ± 0.027 |
| PHATE 2D, 3 seeds | Euclidean | 0.473 ± 0.000 |
| Original 6191-d | Euclidean | 0.333 |
| Original 6191-d | Cosine | 0.277 |
| Original 6191-d, z-scored | Euclidean | -0.019 |

The z-scored result is important: it shows that part of the apparent separation is scale-dependent. We
therefore present the embeddings as useful diagnostics, not as stand-alone proof of mechanism.

For SHAP, the manuscript now states that TreeSHAP explains a gradient-boosting surrogate trained to
predict MACE energy errors from SOAP descriptors. It does not directly explain the MACE message-passing
model. The corrected 5-fold CV values for the surrogate are:

| Model | 5-fold CV R2 | 5-fold CV MAE (eV) |
|---|---:|---:|
| FT-600K | 0.9819 | 0.03795 |
| FT-MultiT | 0.8756 | 0.02692 |
| Scratch | 0.9730 | 0.02618 |

The corrected FT-MultiT top surrogate feature is `Cr-Sb_n43_l0`, not the previously drafted
`Cr-Cr_n13_l3`. We have updated the source text and Fig. 6 caption accordingly.

**Placeholder before final submission.**  
`[PLACEHOLDER: final SHAP code-release hash; final full-set surrogate rerun if expanded beyond the 150-frame local audit; Cr-Sb local-environment perturbation test if retained]`

**Marked manuscript locations.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:349), [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:359), [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:372)
- SI: [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:433)

---

### A7. Positioning against related work and our earlier report

**Reviewer concern.** The reviewer asked us to cite recent nearby work and to state clearly what this
manuscript adds beyond our earlier report.

**Response.** We have added the relevant citations and a short delta paragraph. The revised Introduction
now positions our work relative to recent MACE fine-tuning work for ion diffusion, foundational MLIP
migration-barrier evaluation, and reported softening behavior in universal MLIPs. We also state that this
manuscript extends our earlier report by adding the fixed-path/self-consistent NEB distinction,
interlayer sliding, representation diagnostics, and a reviewer-audited data provenance workflow.

**Placeholder before final submission.**  
None.

**Marked manuscript location.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:194)

---

## B. Code and Repository Comments

### C8. Fabricated all-zero forces in NEB-to-training-data exporters

**Reviewer concern.** The reviewer found scripts that wrote real QE energies with fabricated all-zero
forces into training-format extxyz files. They asked us to confirm whether those frames were used in the
reported fine-tuning runs, and to fix the exporters.

**Response.** We agree this is a serious provenance issue. The author's current reconstruction is that
the affected zero-force NEB extxyz files were not used in the reported FT-600K or FT-MultiT training runs.
We are treating this as something that needs written evidence, not memory. The remaining step is to
archive the cluster-side training logs and file manifests showing that the reported models used AIMD
training frames with real DFT forces, not the zero-force NEB exports.

The exporter behavior itself is being corrected so that forces are exported only when real forces are
available. Otherwise the script must omit forces or stop with an explicit error; it must not write zeros
as if they were DFT labels.

**Placeholder before final submission.**  
`[PLACEHOLDER: zero-force contamination audit verdict; training-log paths; sampled extxyz force checksums; patched exporter commit]`

**Marked manuscript location.**

- SI: [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:153)

---

### C9. Asymmetric MD protocol for the naive FT-600K condition

**Reviewer concern.** The reviewer identified that the FT-600K MD script used a much shorter trajectory
and different sampling interval than the other model conditions, which can affect diffusion and thermal
transport comparisons.

**Response.** We agree. The FT-600K transport values from that run are quarantined and should not be used
as final. The manuscript now marks the long-timescale transport comparison as preliminary until the
corrected long-run FT-600K trajectory has been generated and processed with the same workflow as the other
models.

For the currently audited full-length conditions, the block-averaged values are:

| Model | Status | D (cm2/s) | MSD fit R2 | kappa (W m-1 K-1) |
|---|---|---:|---:|---:|
| Foundation | valid local audit | (0.7 ± 7.9) × 10^-9 | 0.011 | (2.4 ± 8.6) × 10^-3 |
| Scratch | valid local audit | (-3.4 ± 3.2) × 10^-9 | 0.001 | (0.9 ± 1.3) × 10^-2 |
| FT-MultiT | valid local audit | (4.8 ± 4.8) × 10^-8 | 0.861 | (2.0 ± 1.7) × 10^-2 |
| FT-600K | invalid current run | (3.9 ± 4.0) × 10^-8 | 0.903 | (0.9 ± 1.2) × 10^-2 |

The Foundation and Scratch diffusion fits have very low R2, so the honest statement is that those runs do
not show a reliable diffusive regime in the present window.

**Placeholder before final submission.**  
`[PLACEHOLDER: corrected FT-600K 100000-step trajectory; final all-model transport table regenerated from the same frame-count and timestep provenance]`

**Marked manuscript locations.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:253)
- SI: [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:307)

---

### C10. Ungrouped random train/test split

**Reviewer concern.** The reviewer noted that frame-level random splitting can put near-duplicate
neighboring frames into both training and test sets, inflating test RMSE.

**Response.** We agree. The revised workflow uses grouped splitting by trajectory or configuration-family
provenance. The split file must also preserve provenance fields so that downstream audits can exclude
NEB-adjacent or trajectory-adjacent frames when needed.

**Placeholder before final submission.**  
`[PLACEHOLDER: grouped split manifest; zero group-overlap assertion; retrained Table S1 energy/force RMSE mean ± std]`

**Marked manuscript location.**

- SI: [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:163)

---

### C11. Silhouette scores computed on t-SNE embeddings

**Reviewer concern.** The reviewer noted that silhouette scores computed on t-SNE embeddings are not a
reliable measure of original-space separability.

**Response.** We agree. The revised manuscript no longer treats t-SNE silhouette values as the primary
quantitative evidence. We now report original descriptor-space scores and retain projected scores only as
visualization diagnostics. The current original-space Euclidean silhouette is 0.333, cosine silhouette is
0.277, and z-scored Euclidean silhouette is -0.019.

**Placeholder before final submission.**  
None for the local audit values. If we regenerate the figure from a larger dataset, the same table will
be updated from the audited script.

**Marked manuscript locations.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:349), [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:359)

---

### C12. Always-zero force-sensitivity feature

**Reviewer concern.** The reviewer found an always-zero feature channel concatenated into the latent
analysis feature matrix.

**Response.** We agree this channel should not have been included. The SI now states explicitly that the
placeholder force-sensitivity channel contained no physical information, has been removed from the
revised analysis, and is not part of the feature dimension used for the original-space silhouette
calculation.

**Placeholder before final submission.**  
None for the manuscript correction.  
`[PLACEHOLDER: code-release commit showing the zero channel removed or the old stub deleted]`

**Marked manuscript locations.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:349)
- SI: [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:375)

---

### C13. SHAP code absent and empty repository stubs

**Reviewer concern.** The reviewer noted that the SHAP analysis code was not present in the released
repository and that several advertised files were empty stubs.

**Response.** We agree. The manuscript now scopes the SHAP result to a documented surrogate model, and
the code availability section is marked to require the working SHAP pipeline and cleanup of empty stubs
before final submission. The current local audit has already produced the corrected SHAP feature table and
5-fold CV surrogate scores listed in A6.

**Placeholder before final submission.**  
`[PLACEHOLDER: MigrationBench commit containing SHAP analysis script, required inputs, README, non-empty package files or removed stubs, and reproduced Fig. 6 artifact hashes]`

**Marked manuscript locations.**

- Main text: [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:372), [sn-article-final.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-final.tex:431)
- SI: [sn-article-SI.tex](/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/sn-article-SI.tex:433)

---

## Final Placeholders to Clear

1. `[PLACEHOLDER: final QE NEB convergence table for path 1--4; final accepted DFT reference barrier; raw-log path and checksum]`
2. `[PLACEHOLDER: minimum SOAP distance for each 1--4 NEB image; closest-pair RMSD values; exclusion-retrain result if the threshold is triggered]`
3. `[PLACEHOLDER: Scratch and FT-600K three-seed in-gap barrier error mean ± std; deep-penetration barrier error mean ± std; seed-stability interpretation]`
4. `[PLACEHOLDER: Scratch-5% equilibrium RMSE; Scratch-5% in-gap barrier error mean ± std; Scratch-5% deep-penetration barrier error mean ± std; matched FT-600K values for comparison]`
5. `[PLACEHOLDER: zero-force contamination audit verdict; training-log paths; sampled extxyz force checksums; patched exporter commit]`
6. `[PLACEHOLDER: corrected FT-600K 100000-step trajectory; final all-model transport table regenerated from the same frame-count and timestep provenance]`
7. `[PLACEHOLDER: grouped split manifest; zero group-overlap assertion; retrained Table S1 energy/force RMSE mean ± std]`
8. `[PLACEHOLDER: code-release commit showing the zero channel removed or the old stub deleted]`
9. `[PLACEHOLDER: final SHAP code-release hash; final full-set surrogate rerun if expanded beyond the 150-frame local audit; Cr-Sb local-environment perturbation test if retained]`
10. `[PLACEHOLDER: MigrationBench commit containing SHAP analysis script, required inputs, README, non-empty package files or removed stubs, and reproduced Fig. 6 artifact hashes]`
11. `[PLACEHOLDER: final dataset DOI or Hugging Face commit hash for MigrationBench]`
