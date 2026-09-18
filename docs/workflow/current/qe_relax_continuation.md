# QE Relaxation Continuation

## Purpose

Long standalone QE relaxations may stop at the configured `max_seconds` or
maximum ionic-step count before BFGS convergence. A continuation must start from the latest complete printed ionic geometry,
not from the original geometry, and its work must remain attributable to the
same direct or transformed branch.

## Restartability Gate

`monitor_qe_relax_jobs.py` assigns `clean_max_seconds_restartable` or
`clean_max_ionic_steps_restartable` only when:

- the Slurm job is terminal;
- QE explicitly reports that its maximum CPU time or maximum ionic-step count was reached;
- at least one ionic step has complete coordinates, energy, and atomic forces;
- no SCF non-convergence was reported.

Scheduler timeout without QE's clean stop, incomplete final force data, SCF
failure, NaN geometry, and arbitrary cancellation remain manual-review states.

## Continuation Semantics

`continue_qe_relax.py` replaces `ATOMIC_POSITIONS` with the parent's latest
complete printed geometry and preserves the calculator identity. Energy and
force histories are attached to the geometry at which each SCF evaluation was
performed: the first evaluation uses `relax.in`, and every later evaluation
uses the preceding QE position update. A maximum-step stop can therefore have
a latest continuation geometry that has not yet been force-evaluated. The
manifest preserves separate hashes for the latest printed and last evaluated
geometries. It uses a new
prefix and `restart_mode='from_scratch'`: geometry is continued exactly, while
electronic charge/wavefunctions and the BFGS Hessian are deliberately not
claimed as portable restart state.

The continuation manifest binds the parent input, output, manifest, status
snapshot, final structure, energy, force, job ID, and resource policy by hash.
It also records cumulative ionic steps, SCF iterations, and segment count before
the new segment. Submission is lock-protected and idempotent; the active-job
registry replaces the terminal parent only after the new job ID is recovered or
  submitted.

Every continuation also writes an explicit QE `nstep` into `&CONTROL` and its
resource manifest. The default is `200`; it must be positive and may be
overridden with `--nstep`. This prevents a clean 24-hour continuation from
silently ending after QE's default 50 ionic steps. `max_seconds` remains below
the Slurm walltime so QE can write a restartable, provenance-complete stop.

## Example

```bash
python scripts/migrationbench/continue_qe_relax.py \
  --relax-status state/qe_geometry_transform_direct_status_s47.json \
  --active-jobs configs/qe_geometry_transform_relax_direct_jobs_s47.json \
  --parent-job-id JOB_ID --attempt 2 \
  --out-dir data_processed/qe_geometry_transform_relax_ab_s47/restarts/BRANCH_r2 \
  --remote-input-dir /scratch16/pclancy3/yi/revision1_migrationbench_inputs/qe_relax/BRANCH_r2 \
  --walltime 24:00:00 --max-seconds 84600 --nstep 200 \
  --ntasks 2 --memory 160G --submit
```

Final A/B efficiency uses lineage-cumulative ionic and SCF counts. A continued
branch is not compared using only its last segment.
