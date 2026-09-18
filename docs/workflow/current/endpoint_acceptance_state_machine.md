# Endpoint Acceptance State Machine

## Purpose

A frozen NEB endpoint, a completed SCF, or one successful geometry relaxation
is not by itself an accepted local minimum. Production nonlinear paths and MACE
preconditioning may consume an endpoint only after an independent repeat-relax
check passes under the same QE calculator identity.

## States

1. `candidate`: selected from a NEB endpoint, hidden basin, or site proposal.
2. `first_relax_running`: standalone fixed-cell QE BFGS relaxation is active.
3. `first_relax_accepted`: `JOB DONE`, BFGS converged, and final maximum atomic
   force passes the configured threshold.
4. `repeat_relax_running`: a new from-scratch QE relax starts from the final
   complete first-relax geometry under the same calculator identity.
5. `accepted`: repeat relaxation passes force, displacement, energy, geometry,
   provenance, and calculator-identity gates.
6. `rejected`: any scientific gate fails. The raw trajectory remains in the
   audit dataset, but the endpoint cannot seed production NEB.

Infrastructure failure and clean `max_seconds` termination are execution states,
not scientific rejection. They may be retried from preserved inputs/outputs.

## Canonical Gates

All values are read from `configs/representative_path_selection.json` and its
SHA-256 is recorded in every repeat manifest and pair acceptance artifact:

- final maximum atomic force: `<= 0.05 eV/A`;
- maximum periodic atom displacement in repeat relax: `<= 0.10 A`;
- absolute parent/repeat final-energy difference and repeat energy lowering:
  `<= 0.02 eV`;
- minimum pair distance: `>= 1.80 A`;
- complete, unchanged QE calculator identity;
- exact parent job, input, output, policy, and repeat-input hash bindings.

## Commands

After the first-relax status has accepted rows:

```bash
python scripts/migrationbench/prepare_qe_repeat_relaxations.py \
  --first-status state/qe_endpoint_relax_exact_direct_job_status.json \
  --out-dir data_processed/endpoint_repeat_relax/<batch> \
  --select '1-6:2' --select '1-6:5'

python scripts/migrationbench/submit_qe_endpoint_relax_batch.py \
  data_processed/endpoint_repeat_relax/<batch>/endpoint_relax_batch_manifest.json \
  --remote-code-root /scratch16/pclancy3/yi/revision1_migrationbench_pipeline
```

Monitor the submitted jobs with `monitor_qe_relax_jobs.py`. Then build a pair
artifact only for the two endpoint records intended to define one path:

```bash
python scripts/migrationbench/build_endpoint_pair_acceptance.py \
  --first-status state/qe_endpoint_relax_exact_direct_job_status.json \
  --repeat-status state/qe_endpoint_repeat_relax_status.json \
  --initial '1-6:2' --final '1-6:5' \
  --out-dir data_processed/endpoint_acceptance/1-6_img2_to_img5
```

The resulting `endpoint_pair_acceptance.json` supplies the exact accepted
structure paths and hashes expected by `generate_nonlinear_neb_candidates.py`.
No manual coordinate copying is permitted.

Accepted structures from all pair artifacts are then deduplicated with
`assign_endpoint_basins.py`. Basin equivalence requires all three configured
conditions: migrant minimum-image distance, host RMSD, and local-neighbor
distance-spectrum RMS. Clustering uses complete linkage, so every pair inside
one basin must satisfy the thresholds; a transitive A-near-B-near-C chain
cannot merge geometrically distant A and C endpoints. Basin identifiers hash
the system fingerprint and the representative's full structure hash.

For discovery batches where endpoints are not yet organized into physical
migration pairs, use `build_endpoint_acceptance_batch.py` after the same repeat
relax gate. It emits individual accepted/rejected endpoint records without
inventing an A-to-B path. The accepted records can be passed directly to
`assign_endpoint_basins.py`; only after basin assignment and physically valid
pair selection may they become NEB endpoints.

Both parent and repeat acceptance now require `runtime_provenance.json` to bind
the exact `relax.in`, Slurm job id, and a hash-recorded `pw.x` binary. A
structure cannot be accepted merely because its output text and force happen
to pass the geometric gates.
Rockfish CPU Slurm job `30856509` validates the complete current chain:
single-record acceptance, discovery-batch acceptance, and complete-linkage
basin assignment. It completed with exit `0:0` in three seconds. Both synthetic
records passed parent and repeat input-hash, Slurm-job-id, and `pw.x` runtime
identity checks; the batch contained two accepted records and basin assignment
kept them as two distinct basins. This is a software/provenance smoke only and
does not establish a physical stable site.

## Current Status

The state-machine implementation and synthetic end-to-end acceptance fixtures
are validated locally and by Rockfish CPU Slurm jobs `30850580`, `30850601`, and
`30856509` (all `COMPLETED`, exit `0:0`). The last job additionally exercises
`build_endpoint_acceptance_batch.py` before basin assignment. Two earlier attempts, `30850569` and
`30850575`, are retained as infrastructure failures caused by an incomplete
remote code-tree sync; no scientific checks ran in those attempts. No current
`1-6`/`1-7` endpoint is accepted yet because the real
first-relax jobs are still running. Therefore no repeat job has been submitted
and the production MACE/QE handoff remains closed.

Canonical evidence for the current strict smoke is archived at
`data_processed/cluster/mb_epaccept_sm_s42_30856509/`, including Slurm status,
the single-record result, batch result, accepted structures, and basin table.
