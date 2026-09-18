# Restartable QE NEB Chain Design

> Legacy note: submission examples below document the original r1-r3 chain. New
> continuations must use `../continue_qe_neb.py` with
> `../submit_slurm_qe_neb_restart_verified.sh`. The verified path enforces the
> monitor gate, parent hashes, full restart-state SHA-256 comparison, atomic
> registry replacement, and scratch-local Slurm logs.

The shell launchers in this directory now terminate with exit code 64 before
submission. They are retained only to explain historical r1-r3 provenance and
must not be re-enabled. Use `../continue_qe_neb.py` for every new continuation.

Use this design for long MigrationBench QE NEB jobs.

## Rule

For every Slurm walltime, set QE `max_seconds` slightly below the allocation:

- 24 h Slurm: `max_seconds = 84600` with a 30 min safety margin.
- 48 h Slurm: `max_seconds = 171000` with a 30 min safety margin.

This gives `neb.x` time to stop cleanly and write restart files instead of being killed by Slurm.

## First Long Run

Start from MACE-relaxed images and a generated explicit-image `neb.in`:

```bash
bash scripts/migrationbench/submit_qe_neb_restart_job.sh \
  mb_qe17_mace_r1_s42 \
  /scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe17_mace_r1_s42/neb.in \
  - \
  2 \
  24:00:00 \
  160G \
  1800 \
  from_scratch
```

From the local Codex working directory, the prepared one-command submission is:

```bash
bash work/revision1_neb_restart_design/submit_1_7_long_qe_restart_chain.sh 24:00:00
```

For a 48 h first segment:

```bash
bash work/revision1_neb_restart_design/submit_1_7_long_qe_restart_chain.sh 48:00:00
```

## Continuation Run

Continue from the previous run directory:

```bash
bash scripts/migrationbench/submit_qe_neb_restart_job.sh \
  mb_qe17_mace_r2_s42 \
  /scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe17_mace_r1_s42/neb.in \
  /scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_qe17_mace_r1_s42_<JOBID> \
  2 \
  24:00:00 \
  160G \
  1800 \
  restart
```

The Slurm wrapper copies the parent run directory into the new run directory,
excludes old logs and old `neb.in`, patches `restart_mode='restart'`, patches
`max_seconds`, and writes `qe_restart_manifest.json`.

## Gate

After each segment, parse `neb.out`.

Continue until all are true:

- QE writes `JOB DONE`; this alone is not a convergence claim.
- The final activation table is complete and has maximum movable-image (`frozen=F`) error `<= 0.03 eV/A`.
- Barrier drift over the last complete iterations is `<= 0.02 eV`.
- Frozen-endpoint errors are retained as `max_image_error_all_eV_A` diagnostics but excluded from the NEB force gate.
- A nonconverged continuation is allowed only when `monitor_qe_neb_jobs.py` reports `clean_max_seconds_restartable`: terminal job, `tcpu` near the configured `max_seconds`, and complete `.pathN` plus charge/XML state for every image.
- `terminal_needs_review` and `nonphysical_or_overflow` must not be restarted automatically.

If the job times out by Slurm, increase safety margin or reduce `max_seconds`;
the design goal is for QE to exit before Slurm timeout.
