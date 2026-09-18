# A1 Replacement Draft - Fig. 3d vs Fig. S3 NEB Protocol Clarification

Use this block to replace the current A1 section in
`Revision1/response_to_reviewers/response_letter.md`.

---

### A1. [Critical] Foundation-model in-gap barrier discrepancy (Fig. 3d vs Fig. S3, and 0.34 vs ~0.3 eV)

**Reviewer (in short)**: State explicitly whether Fig. 3d (fixed-geometry, ~0.7 eV overestimate,
"unacceptable") and Fig. S3 (self-run NEB, 0.41 eV, "exceptional") use the same pathway and DFT
reference; fix the reference value; present both evaluations side-by-side and explain why relaxation
changes the verdict.

**Response**: We agree that the original wording made two different NEB-based evaluations appear
internally inconsistent. They are not the same task. In the revised manuscript and SI, we now define
the two protocols explicitly and report them side-by-side:

1. **Fixed-geometry DFT-path evaluation** (main-text Fig. 3d):
   the DFT-relaxed NEB image sequence is held fixed, and each MLFF is asked only to assign energies
   to those same geometries. This measures energetic fidelity on a reference DFT path:
   \[
   \Delta E^{\mathrm{fixed}}_M(\Gamma_{\mathrm{DFT}})
   =
   \max_i E_M(X_i^{\mathrm{DFT}})-E_M(X_0^{\mathrm{DFT}}),
   \qquad
   \epsilon^{\mathrm{fixed}}_M
   =
   \Delta E^{\mathrm{fixed}}_M-\Delta E_{\mathrm{DFT}} .
   \]
   The model is given the DFT path geometries, so this protocol isolates barrier-energy prediction
   on a fixed, physically defined path.

2. **Self-consistent MLFF NEB optimization** (SI Fig. S3):
   each MLFF starts from the same endpoint structures but is not given the DFT-relaxed intermediate
   images. The model relaxes its own elastic band under its own forces:
   \[
   \Gamma_M^\star=\operatorname{NEB}_M(X_0,X_1),
   \qquad
   \Delta E^{\mathrm{self}}_M
   =
   \max_i E_M(X_i^{M,\star})-E_M(X_0).
   \]
   This protocol measures whether the model can generate a stable migration pathway, not only
   whether it can score a DFT-provided pathway. Therefore
   \(\Delta E^{\mathrm{fixed}}_M(\Gamma_{\mathrm{DFT}})\) and
   \(\Delta E^{\mathrm{self}}_M(\Gamma_M^\star)\) need not be identical, because the underlying
   image geometries can differ.

**Corrected values and status**: For the in-gap path used in the fixed-geometry Fig. 3d audit
(path 1-4 in our Revision 1 path ledger), the current audited DFT barrier from the raw QE `neb.out`
is **0.336050 eV**. This replaces the mixed use of "0.34 eV" and "~0.3 eV"; the latter will be
removed or rewritten as an approximate literature/context statement, not as the reference used for
error calculation. The current local audit labels this 1-4 DFT calculation as
`UNCONVERGED` but numerically stationary; therefore the revised response and manuscript will either
(i) keep the value as the audited candidate reference with an explicit convergence caveat, or
(ii) replace it with the final value from the restartable QE NEB closure run once completed.

Using the current audited 1-4 reference, the fixed-geometry barriers and signed errors are:

| Model | Fixed-geometry MLFF barrier (eV) | DFT reference used (eV) | Signed error (eV) | Status |
|---|---:|---:|---:|---|
| MACE Foundation | 1.024296 | 0.336050 | +0.688246 | audited current value |
| MACE Scratch | 4.537677 | 0.336050 | +4.201627 | audited current value |
| MACE FT-600K | 0.496792 | 0.336050 | +0.160742 | audited current value |
| MACE FT-MultiT | 0.823040 | 0.336050 | +0.486990 | audited current value |

For SI Fig. S3, the submitted SI reports that the MACE Foundation model converges in a
self-consistent MLFF NEB run with a barrier of **0.41 eV**, while Scratch, FT-600K, and FT-MultiT
show unstable force growth and were terminated. In the revision, we will not describe the 0.41 eV
value as a fixed-path barrier error or as a DFT-revalidated barrier. We will describe it as the
Foundation model's self-consistent MLFF NEB barrier under the SI protocol, pending the following
metadata audit:

`[PLACEHOLDER: SI-FigS3-audit]` Endpoint IDs/path label, number of images, model checkpoint hash,
random seed if applicable, final NEB force criterion, optimizer settings, and source trajectory file.
This is computed by parsing the SI Fig. S3 source directory and/or rerunning the self-consistent
MACE NEB through the unified `MigrationBench` pipeline, then writing the manifest to
`data_processed/fig3d_vs_figS3_unified.csv`.

`[PLACEHOLDER: DFT-on-MLFF-path]` If we want to compare the self-consistent MLFF path directly to
DFT, we will evaluate either DFT single-point energies on the Foundation-relaxed images or run a QE
NEB initialized from those images. This gives
\[
\Delta E_{\mathrm{DFT}}(\Gamma_M^\star)
=
\max_i E_{\mathrm{DFT}}(X_i^{M,\star})-E_{\mathrm{DFT}}(X_0),
\]
which is the appropriate quantity for asking whether the MLFF-generated path is also a valid DFT
path. It is distinct from both the Fig. 3d fixed-path score and the SI Fig. S3 MLFF-only barrier.

**Manuscript/SI changes**: We now revise the Fig. S3 caption and the surrounding SI text to say
"self-consistent MLFF NEB" and "MLFF barrier", and revise the main text to say "fixed-geometry
single-point evaluation along DFT-relaxed NEB images." We also add a side-by-side protocol table
with the columns: pathway, endpoints, image source, relaxation engine, energy-evaluation engine,
barrier reported, DFT reference used, convergence criterion, and interpretation. This directly
addresses the reviewer's concern: the original manuscript did not make the operator distinction
clear enough, so the revision makes the distinction explicit and prevents the two numbers from being
used as the same metric.

**Revised short outcome sentence for the response letter**:
> The apparent discrepancy arose because Fig. 3d and Fig. S3 report two different operators:
> fixed-geometry energy prediction on a DFT-relaxed path versus self-consistent MLFF NEB path
> generation. Under the fixed-geometry operator, the Foundation model predicts a 1.024 eV barrier
> for the audited 1-4 in-gap path, corresponding to a +0.688 eV error against the current DFT
> reference of 0.336050 eV. Under the self-consistent MLFF NEB operator in Fig. S3, the Foundation
> model relaxes its own path and reports a 0.41 eV MLFF barrier. These two values are therefore not
> contradictory; they answer different questions. We have revised the main text, SI, and captions to
> define both protocols explicitly, remove the ambiguous "~0.3 eV" reference wording, and report the
> values side-by-side with convergence metadata.
