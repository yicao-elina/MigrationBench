# Rockfish Cr-NEB monitoring handoff | 2026-10-16

## Scope

This monitoring task is Rockfish-only. It does not access, submit to, cancel,
modify, or merge with Discovery. The authoritative Rockfish campaign root is:

`/scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013`

The local additive audit root is:

`analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013/`

## Existing jobs

| path | Rockfish job | current local status |
|---|---:|---|
| `2L-Cu__neb-QLs` | 31916908 | `COMPLETED_BUT_SCF_WARNING` |
| `2L-Cu__neb-gap2QL` | 31916909 | `COMPLETED_BUT_SCF_WARNING` |
| `2L-Cu__neb-gap-diffuse` | 31916910 | `COMPLETED_BUT_SCF_WARNING` |

The three jobs reported finite ten-image outputs and `JOB DONE`, but each has a
recorded SCF-convergence warning. Preserve that status. Do not promote these
values to publication-grade converged barriers without a clean rerun or an
explicit user decision accepting them as provisional diagnostics.

## Fixed scientific identity

- system: 2L-Sb2Te3, 81 atoms;
- chemistry: Cr substituted at the migrating Cu site;
- path classes: within QL, gap to QL, within gap;
- images: 10, fixed atom order/cell/PBC;
- QE/NEB: 70/280 Ry, cold smearing, degauss 0.005 Ry, DFT-D3,
  `1 1 1 1 1 1`, Broyden, `nstep_path=1000`, `path_thr=0.001`;
- directional reporting: always retain both `activation energy (->)` and
  `activation energy (<-)` from `neb.out`;
- iteration-one or warning-contaminated values are diagnostics, not final
  barriers.

## Monitoring goal

On Rockfish, verify the three terminal jobs and their raw artifacts, reconcile
the job/path/result mapping, check all ten images and directional barriers, and
prepare an auditable provisional merge with the existing Cu/Li/Na comparison.
If any artifact is missing or inconsistent, preserve the evidence and report a
blocker. Do not touch Discovery and do not create duplicate valid results.

## Acceptance gates

1. `sacct` terminal state and `ExitCode=0:0` for each job;
2. `neb.out` contains `JOB DONE` and explicit forward/reverse activation lines;
3. ten images, 81 atoms, correct Cr/Sb/Te identity, cell/PBC and path order;
4. finite energies and gradients/forces for every image;
5. no fatal QE/MPI/FIO/NaN/Inf marker;
6. input, pseudopotential, launcher, runtime and raw-output hashes mapped to job;
7. SCF-warning status recorded separately from all publication tables;
8. only after all gates are reviewed, render Rockfish-only and combined figures.

## Safe commands

```text
ssh rockfish
cd /scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013
sacct -j 31916908,31916909,31916910 --format=JobID,State,ExitCode,Elapsed,NodeList
squeue -j 31916908,31916909,31916910
```

Do not substitute Discovery paths or containers in this audit.

## Approved continuation submitted 2026-10-16

The user approved an additive restart of the three warning-contaminated
Rockfish paths. The original jobs and directories remain the authoritative
first-run evidence; no original `neb.out`, `out/`, or path file was modified.

Restart root:

`/scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013/restarts_20261016/`

| restart path | previous job | restart job | restart directory |
|---|---:|---:|---|
| within QLs | 31916908 | 32069316 | `01_within_QLs` |
| gap to QL | 31916909 | 32069317 | `02_gap_to_QL` |
| within gap | 31916910 | 32069318 | `03_within_gap` |

The continuation uses `restart_mode='restart'`, the latest surviving
`sb2te3.path`, a fresh `out/`, native Rockfish QE 7.2, and the original
physical DFT/NEB contract. The remote preparation and job-map scripts are
`prepare_and_submit_restart.sh` and `rockfish_cr_neb_restart.slurm`; local
copies and the submitted map are stored beside this file.

The new jobs are not accepted merely because they complete. They must pass the
restart gates in `RESTART_PLAN.md`, including no remaining SCF warning before
being promoted to a converged publication barrier.
