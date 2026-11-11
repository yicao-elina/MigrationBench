# Rockfish Cr-NEB continuation plan | 2026-10-16

## Scientific decision

The three original Rockfish Cr paths are retained as immutable first runs:

| path | previous job | previous last iteration | status before continuation |
|---|---:|---:|---|
| within QLs | 31916908 | 161 | `COMPLETED_BUT_SCF_WARNING` |
| gap to QL | 31916909 | 69 | `COMPLETED_BUT_SCF_WARNING` |
| within gap | 31916910 | 82 | `COMPLETED_BUT_SCF_WARNING` |

They have finite ten-image path tables and `JOB DONE`, but each ended with an
SCF-convergence warning. They remain diagnostic/provisional and are not
overwritten or promoted by this restart.

## Restart design

The continuation is Rockfish-only and uses a new root:

`/scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013/restarts_20261016/`

For each path, the preparation script copies the latest `sb2te3.path`, all
historical `sb2te3.path*` checkpoints, `sb2te3.broyden`, input files, and the
first-run `neb.out` as `neb_previous.out`. It deliberately does not copy the
old ~92 GB `out/` directory. The new job uses a fresh `./out/`, so the first
run remains intact and the continuation cannot mutate it.

All three continuation inputs use `restart_mode='restart'`, retain the
original QE/NEB physical contract (QE 7.2 module, 70/280 Ry, cold smearing,
degauss 0.005 Ry, DFT-D3, 1x1x1 k mesh, Broyden, 10 images,
`nstep_path=1000`, `path_thr=0.001`), and retain the original Rockfish Slurm
resource shape (two nodes for within-QL and gap-to-QL; one node for within-gap
in the original run). The submission driver overrides the wrapper's default
with those per-path node counts; this preserves the prior resource shape and
does not alter the DFT method.

## Acceptance gates

Each continuation is accepted only if all are true:

1. `sacct` is `COMPLETED` with `ExitCode=0:0`.
2. `neb.out` has `JOB DONE`, forward and reverse activation-energy lines, and
   a later iteration than the previous run.
3. Ten images are present in the final path; all energies and gradients/forces
   are finite.
4. No fatal QE, MPI, Hydra, FIO, NaN, Inf, or errno marker is present.
5. No `scf convergence NOT achieved` warning remains in the final continuation
   output. If it remains, the continuation is still `PROVENANCE_PARTIAL`.
6. Input, Cr/Sb/Te pseudopotential, native QE module, launcher, runtime, and
   restart-lineage hashes are recorded.
7. The old run remains available and its original hashes still match the
   pre-restart manifest.

Only paths passing all seven gates may be called converged DFT barriers. A
completed restart with a remaining SCF warning is still useful as a diagnostic
and must be reported as such, not silently merged into a clean barrier table.

## Monitoring and failure policy

The dedicated Rockfish monitoring window will inspect the three restart jobs
by job ID and path, then check scheduler state, output growth, `neb.err`, path
iteration, image completeness, finite values, warnings, and provenance. It will
stay quiet while unchanged and healthy.

If one path fails, only that path is diagnosed and restarted again in a new
versioned directory. No original output is deleted, no valid result is
duplicated, and no Discovery campaign is accessed.
