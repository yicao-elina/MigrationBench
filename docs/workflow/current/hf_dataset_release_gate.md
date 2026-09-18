# MigrationBench Dataset Release Gate

## Current Staging

Four normalized, audit-only bundles are available:

- `qe_neb_history_v1`: 1,110 QE configuration/calculation/iteration records from
  the complete current `1-6` and `1-7` histories;
- `mlff_neb_history_smoke_v1`: 42 configuration/calculation/iteration records
  from two corrected MACE diagnostic trajectories.
- `mlff_historical_61_v1`: 8,390 configuration/calculation/iteration records
  reconstructed from four historical 61-atom MACE NEB trajectories;
- `mlff_historical_81_v1`: 24,030 configuration/calculation/iteration records
  reconstructed from five historical 81-atom MACE NEB trajectories.

All four pass primary-key, foreign-key, array-shape, force/negative-gradient,
iteration/calculation-force, endpoint NEB-force, grouped-split, and cross-split
duplicate checks. The paired MACE runs share one split group because they derive
from the same candidate path. The reconstructed historical bundles explicitly
retain their assumed legacy NEB protocol (`k=0.1`, no climbing image,
`improvedtangent`) and missing runtime provenance. They are audit evidence, not
production labels or manuscript-reference barriers.

## Checkpoint-Scoped Leakage Evidence

The current `1-6/1-7` QE histories contain 1,110 evaluation frames. Against
each declared fine-tuning split, the audit found zero exact whole-structure or
Cr-local fingerprint matches, zero SOAP cosine distances at or below `1e-4`,
and zero species-compatible Cr-local RMSDs at or below `0.05 A`. The minimum
SOAP distances are `2.79x`, `3.71x`, and `5.09x` the registered threshold for
train, validation, and test. The minimum comparable RMSDs are `23.89x`,
`12.52x`, and `11.82x` the threshold.

This passes the declared-checkpoint gate only. It does not establish anything
about unavailable foundation-model pretraining data. Any change to benchmark
frames, mechanism grouping, training files, or checkpoint invalidates the pass
and must trigger the same audit again.

## Fail-Closed Manuscript Gate

A barrier is manuscript eligible only when all of the following are true:

1. protocol is `dft_neb` and convergence status is exactly `converged`;
2. a hash-bound path artifact says `accepted_final_reference` for the same path;
3. a hash-bound endpoint artifact says both endpoints are accepted local minima;
4. a hash-bound calculator artifact accepts the exact calculator identity.
5. a hash-bound runtime provenance artifact records the exact scientific
   executables, software environment, and declared run inputs.

Missing evidence is a failure, not an unknown that silently passes. MLFF
barriers remain audit-only even when their optimizer converges.

## Force Semantics

`forces_eV_A` always means the true calculator force. `neb_force_eV_A` is the
projected/spring NEB force and is present only for movable internal images.
Frozen endpoints have no NEB force. Earlier MACE exports used ASE's zero-filled
endpoint `real_forces`; those files are superseded by the corrected v3 exports,
which call each endpoint calculator directly before NEB projection.

For a trust-region MACE run, the primary `energy_eV`, `forces_eV_A`, and
`path_energy_gradient_eV_A` fields come only from the unmodified base MACE
calculator. The restrained optimization energy, restrained atomic forces,
projected restrained NEB force, and regularization energy are stored in separate
fields. The validator requires

`optimization_energy_eV = energy_eV + regularization_energy_eV`

within `2e-6 eV`, complete atomic-array shapes, nonnegative regularization
energy, and explicit potential semantics. A tethered quantity can never become
a training label or manuscript barrier through an ambiguous field name.

This separation is covered by a full synthetic export test: base energies and
forces are intentionally different from restrained values, the resulting path
barrier must be computed from base MACE, and malformed energy decompositions
must fail validation. Real Rockfish Slurm job `30853107` completed with
`80 passed` on 2026-09-13.

## Publication State

Publication to `alinacao2000/MigrationBench` remains blocked. The schemas and
validators are ready, but the current QE paths are unconverged and their
endpoint/calculator acceptance artifacts are not final.

## Relaxation-Pair Evaluation Table

The endpoint-initialization experiments use a separate `relaxation_pairs`
table rather than overloading NEB image records. Each row preserves both input
hashes, transform objective and parameters, job and restart identities,
calculator identities, repeat-relax states, final-basin metrics, cumulative
ionic/SCF/walltime work, and the comparison artifact hash.

Prospective rows are marked `evaluation_only` and
`eligible_for_current_proposer_training=false`. A non-formal row is forbidden
from carrying any numerical speedup. Formal speedup fields are allowed only
after both repeat acceptances, same-basin classification, complete runtime
provenance, and calculator acceptance. The builder is
`scripts/migrationbench/build_relaxation_pair_dataset.py`; Rockfish job
`30862305` passed all `100` tests for the integrated updater and table gate.
