# NEB Restart Chain Design

## Goal

Make every long QE NEB job restartable and auditable:

`MACE-relaxed images -> QE explicit-image input -> long QE segment -> parse -> restart or accept`

## Walltime Rule

Set QE `max_seconds` below Slurm walltime so QE exits cleanly before Slurm kills the job.

| Slurm walltime | Safety margin | QE `max_seconds` |
|---|---:|---:|
| 24:00:00 | 1800 s | 84600 s |
| 48:00:00 | 1800 s | 171000 s |

## First Segment

1. Use MACE-relaxed `mlff_neb_images.extxyz`.
2. Generate `neb.in` using `qe_neb_from_images.py`.
3. Set `restart_mode='from_scratch'`.
4. Submit on CPU with 2 MPI ranks, 160G memory, 24-48 h walltime.

## Continuation Segment

1. Use the previous run directory as `MIGRATIONBENCH_PARENT_RUN_DIR`.
2. Copy restart artifacts into a new run directory.
3. Patch `restart_mode='restart'`.
4. Patch `max_seconds` according to the new walltime.
5. Write `qe_restart_manifest.json`.
6. Submit the next Slurm job.

## Acceptance Gate

Continue until:

- Slurm state is `COMPLETED 0:0`.
- QE exits cleanly or writes a restart-compatible state.
- Last complete NEB iteration has max image error <= 0.03 eV/A.
- Barrier drift over recent complete iterations is < 0.02 eV.

If a path repeatedly times out but makes progress, continue. If it collapses chemically or the barrier becomes meaningless, quarantine it with the parent chain preserved.

## Current 1-7 Status

The first two short QE preflights proved the generated QE input parses and reaches SCF image 3. The OOM issue is fixed by using fewer MPI ranks and explicit memory. The remaining gap is long walltime, not input validity.

Recommended next job:

```bash
mb_qe17_mace_r1_s42
walltime: 24:00:00 or 48:00:00
ntasks: 2
mem: 160G
QE max_seconds: walltime - 1800 s
restart_mode: from_scratch
source images: MACE-relaxed 1-7 path
```
