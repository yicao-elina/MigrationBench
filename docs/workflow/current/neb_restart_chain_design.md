# NEB Restart Chain Design

## Goal

Make every long QE NEB job restartable and auditable:

`source images -> optional repair -> optional MACE relaxation -> QE explicit-image input -> long QE segment -> parse -> restart or accept`

## Walltime Rule

Set QE `max_seconds` below Slurm walltime so QE exits cleanly before Slurm kills the job.

| Slurm walltime | Safety margin | QE `max_seconds` |
|---|---:|---:|
| 24:00:00 | 1800 s | 84600 s |
| 48:00:00 | 1800 s | 171000 s |

## First Segment

1. Use the accepted source images for that branch. This may be historical DFT images, repaired images, or MACE-relaxed `mlff_neb_images.extxyz`.
2. Generate `neb.in` using `qe_neb_from_images.py`.
3. Set `restart_mode='from_scratch'`.
4. Submit on CPU with 2 MPI ranks, 160G memory, 24-48 h walltime.
5. Write a branch manifest with the source lineage and input checksum.

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

The MACE-derived `1-7` QE branch was quarantined after QE produced a nonphysical activation table. The current live recovery branch is the direct-historical QE preflight `30791508` (`mb_qe17_direct_pf_s42`), generated from audited historical images with sane path length and no overflow at startup.

If the direct-historical preflight produces a physical but unconverged activation table, the next job should be a 24 h or 48 h restart segment:

```yaml
source_parent_run_dir: /scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_qe17_direct_pf_s42_30791508
restart_mode: restart
walltime: 24:00:00 or 48:00:00
ntasks: 2
mem: 160G
qe_max_seconds: 84600 for 24 h, 171000 for 48 h
decision: restart_from_parent
```
