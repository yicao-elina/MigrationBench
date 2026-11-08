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
internally inconsistent. The audit identified two separate issues. First, fixed-path scoring and
self-consistent NEB are different measurement operators. Second, the historical 0.41 eV record
reported in SI Fig. S3 is not the same physical path or system as main-text Fig. 3d, and its
calculator/serialization defects prevent it from serving as a quantitative barrier. We therefore
report the valid fixed-path results, retain the 0.41 eV record only as a provenance audit, and do
not use it in the model ranking:

1. **Fixed-geometry DFT-path evaluation** (main-text Fig. 3d):
   the DFT-relaxed NEB image sequence is held fixed, and each MLIP is asked only to assign energies
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

2. **Self-consistent MLIP NEB optimization** (historical SI Fig. S3 record):
   this was intended to start from endpoint structures and relax an elastic band under the model's
   own forces:
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

| Model | Fixed-geometry MLIP barrier (eV) | DFT reference used (eV) | Signed error (eV) | Status |
|---|---:|---:|---:|---|
| MACE Foundation | 1.024296 | 0.336050 | +0.688246 | audited current value |
| MACE Scratch | 4.537677 | 0.336050 | +4.201627 | audited current value |
| MACE FT-600K | 0.496792 | 0.336050 | +0.160742 | audited current value |
| MACE FT-MultiT | 0.823040 | 0.336050 | +0.486990 | audited current value |

The historical SI Fig. S3 record reports a Foundation self-consistent MLIP NEB value of **0.41 eV**.
However, the provenance audit shows that it uses a different system, Cr2Sb32Te48 (82 atoms, two Cr)
with twelve images, rather than the CrSb24Te36 (61 atoms, one Cr) five-image path used in Fig. 3d.
It also uses inconsistent endpoint and movable-image calculators, and the final XYZ serialization
assigns one shared energy to the band. Uniform rescoring of the preserved band gives a forward
endpoint-referenced value of 0.000 eV, a reverse value of 0.379 eV, and a whole-band span of
1.037 eV, not 0.410 eV. We therefore mark the historical 0.41 eV value
`EXCLUDE_FROM_PUBLICATION` and retain it only to explain the original discrepancy. The Scratch,
FT-600K, and FT-MultiT termination records are likewise not used as a same-system ranking result.

`[VALIDATION GAP: DFT-on-MLIP-path]` A valid same-system comparison would require evaluating either
DFT single-point energies on a correctly serialized Foundation-relaxed path or running a QE NEB
initialized from those images. This gives
\[
\Delta E_{\mathrm{DFT}}(\Gamma_M^\star)
=
\max_i E_{\mathrm{DFT}}(X_i^{M,\star})-E_{\mathrm{DFT}}(X_0),
\]
which is the appropriate quantity for asking whether the MLIP-generated path is also a valid DFT
path. It is distinct from both the Fig. 3d fixed-path score and the invalidated historical SI Fig. S3
record.

**Manuscript/SI changes**: We now revise the Fig. S3 caption and surrounding SI text to identify the
historical record as an audited, invalidated self-consistent MLIP-NEB record; it is no longer called
"exceptional" or used as a quantitative barrier. We revise the main text to say "fixed-geometry
single-point evaluation along DFT-relaxed NEB images." We also add a side-by-side protocol/provenance table
with the columns: pathway, endpoints, image source, relaxation engine, energy-evaluation engine,
barrier reported, DFT reference used, convergence criterion, and interpretation. This directly
addresses the reviewer's concern: the original manuscript did not make the operator distinction
clear enough, so the revision makes the distinction explicit and prevents the two numbers from being
used as the same metric.

**Revised short outcome sentence for the response letter**:
> The apparent discrepancy arose from both an operator mismatch and a provenance failure. Under the
> valid fixed-geometry operator, the Foundation model predicts a 1.024 eV barrier for the audited
> 61-atom path 1-4, corresponding to a +0.688 eV error against the current DFT reference candidate
> of 0.336050 eV. The historical 0.41 eV SI Fig. S3 record was intended to be a self-consistent
> MLIP-NEB result, but it belongs to a different 82-atom, two-Cr system and contains inconsistent
> calculators and corrupted final-band energy serialization. It is therefore excluded from model
> ranking and retained only as a provenance audit. A valid same-system self-consistent comparison
> remains an explicit validation gap; the revised main text, SI, and response figure now make this
> boundary clear.
