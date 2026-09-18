# Draft Response A1: Fixed-Path And Self-Consistent NEB

We thank the reviewer for identifying that the original presentation conflated
two different observables. The primary distinction is between fixed-path
scoring in Fig. 3d and self-consistent/self-relaxed NEB scoring in Fig. S3.
Those protocols can legitimately give different scalar barriers and different
model assessments because one evaluates a model on externally supplied
geometries and the other lets the model choose and relax its own band. Recovery
of the original Rockfish artifacts further suggests that the archived Fig. S3
run is not the same registered 1-4 system: Fig. 3d uses the 61-atom
`CrSb24Te36` 1-4 path, whereas the recovered Fig. S3 candidate uses an 82-atom
`Cr2Sb32Te48` path with two Cr atoms and 12 images. We are therefore treating
the protocol distinction as the explanation for why the two panels need not
match, while independently rechecking both numerical values.

For a fixed DFT image sequence
`Gamma_DFT = {R_0^DFT, ..., R_N^DFT}`, Fig. 3d evaluates

`E_a^fixed(M | Gamma_DFT) = max_i E_M(R_i^DFT) - E_M(R_0^DFT)`.

This asks each model to score the same externally supplied geometries. It does
not test whether the model can generate or maintain a physically meaningful
path. The archived 1-4 QE output gives a forward DFT candidate of
`0.336050 eV`. On those fixed images, the reported Foundation-model value is
`1.024296 eV`, or `+0.688246 eV` relative to that candidate.

By contrast, a self-consistent MLFF NEB first generates a model-dependent band,

`Gamma_M* = NEB(M; R_0, R_N)`,

and then evaluates

`E_a^self(M) = max_i E_M(R_i^{M,*}) - E_M(R_0^{M,*})`.

This combines optimizer stability, path selection, and energetic prediction.
It is not the same observable as fixed-path scoring, even when the endpoint
structures are identical. Relaxation may move the band into a different basin
or mechanism, so apparent agreement of scalar barriers alone does not establish
agreement with the DFT minimum-energy path.

Our audit also found that the stronger original interpretation of Fig. S3 is
not supportable. The historical optimizer assigned MACE calculators only to
the ten movable images; the endpoints retained stored DFT calculators. Its
trajectory consequently mixes endpoint energies near `-179665 eV` with internal
MACE energies near `-309 eV`. Because ASE NEB uses image energies to construct
the path tangent, this invalidates the historical optimization itself. The
final XYZ then serialized one shared calculator result for all 12 images.

Rockfish job `30851756` independently rescored every final geometry using the
exact historical Foundation checkpoint. It gives a forward endpoint-referenced
barrier of `0.000000 eV`, a reverse barrier of `0.379052 eV`, and a whole-path
max-min span of `1.037407 eV`; the highest-energy image is the frozen initial
endpoint. None reproduces `0.41 eV`. Until the `0.41 eV` value is reproduced
from a lossless log or replaced by a registered rerun, we will not use the
words "exceptional" or "close to chemical accuracy" as evidence. Fig. S3 must
be replaced by a correctly rerun, losslessly logged self-NEB comparison or
retained only as a clearly labeled protocol/audit panel.

The inconsistent `0.34 eV` and `~0.3 eV` references will likewise not be mixed.
The current audited 1-4 value is `0.336050 eV`, but its maximum movable-image
error is `0.482407 eV/A`; it remains a quarantined candidate rather than a final
reference. The final manuscript table will report one unrounded value only
after endpoint, NEB-force, barrier-drift, and calculator-identity gates pass.

The revised side-by-side table will distinguish:

| Quantity | Geometry provider | Energy provider | Current evidence | Final treatment |
|---|---|---|---|---|
| 1-4 DFT reference candidate | archived QE NEB | QE DFT | `0.336050 eV`; force gate failed | replace/confirm using accepted DFT result |
| Foundation fixed-path result | archived 1-4 images | Foundation MLFF | `1.024296 eV`; `+0.688246 eV` candidate error | retain only with path/checkpoint hashes |
| Foundation self-NEB claim | Foundation MLFF for internal images; DFT stored at endpoints during optimization | mixed/invalid in recovered artifact | uniform rescoring gives forward `0.000 eV`, reverse `0.379 eV`, span `1.037 eV`; not losslessly reproducing `0.41 eV` | reproduce from original provenance or rerun as a registered self-NEB path |
| Restartable DFT NEB | accepted endpoints | accepted QE identity | pending | sole final migration-barrier reference |

This revision changes the conclusion from a scalar model ranking to a
task-specific comparison: fixed-path energy error, self-consistent path
stability, transition-state localization, and DFT-refined barrier error are
reported separately and are not treated as interchangeable measures.
