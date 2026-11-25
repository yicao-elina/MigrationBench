# QE mixing-history fresh-start PW recipe (reviewed, not run)

> Historical proposal only. The 12 source mixing files were subsequently
> removed under the guarded receipt; this recipe has not been run and is not
> an instruction to recreate or clean up those paths.

No Rockfish file was changed and no QE job was submitted while preparing this proposal. Do not overwrite the retained source inputs, scripts, or result directories.

This recipe rebuilds the QE `pw.x` stage from scratch. It does not claim that
the historical relaxations converged: all three `relax.out` files report
`Maximum CPU time exceeded` before `JOB DONE.`. Their scheduler records are
`COMPLETED 0:0`, but the launch script ends with `date`, so that status does not
establish convergence. Each existing `pp.out` separately fails with
`Error in routine chdens (1): wrong nx`; PP is preserved as failure evidence
and is not included in the clean PW rerun.

Removing these restart histories loses the ability to continue the exact
time-limited historical calculations. It is permitted here only under the
requested fresh-start policy: preserve their logs/XML and reproduce with a
new run directory. That produces a new trajectory, not an exact continuation.

Put fresh output on `/scratch16/pclancy3/yi` (or another writable scratch root)
instead of making a new output tree under `/data`.

For each run directory below, make a separate working directory under
`/scratch16/pclancy3/yi`, then copy the input and Slurm script there. Use a
unique new output directory so the old XML, logs, and QE save directory remain
untouched:

- `.../2.Cr-in-gap/spin` (job 7104884)
- `.../2.Cr-in-gap/spin/mag3` (job 7105613)
- `.../2.Cr-in-gap/charged_3/spin` (job 7167643)

Proposed copy names and edits:

1. Copy `relax.in` to `relax.fresh.in`. In the copied `&CONTROL` namelist, explicitly set `restart_mode='from_scratch'` and change `outdir` to a new `./out_fresh_<run-id>/`. For `charged3_spin`, this replaces `restart_mode='restart'`; for `base_spin` and `mag3`, it makes QE's default fresh-start choice explicit.
2. Copy `sbatch-relax.slurm` to `sbatch-relax-pw-fresh.slurm`; change the `pw.x` command to read `relax.fresh.in` and write `relax.fresh.out`, create the new outdir, and remove the historical `pp.x` stage from this PW-only rerun. Preserve the original script and logs. The source script selects QE 7.2 and launches `/data/apps/extern/qe/7.2/bin/pw.x` via `mpirun -np 4` under `intel` and `qe/7.2` modules.
3. Use the run-specific pseudopotential path from its input. `base_spin` and `mag3` point to `/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/`; `charged3_spin` points to `/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/`, which currently resolves under `/data/pclancy3/yi/cyqe/data/pseudos/ONCVPseudoPack/`. The corresponding Sb/Te/Cr files have matching hashes; see `qe_mix_retained_artifact_hashes_20261123.jsonl`.
4. Keep `pp.in` and `pp.out` linked to each historical run. Treat the old `pp.out` as a post-processing failure. If PP is attempted again, diagnose `wrong nx` separately and use a fresh copied input/outdir; do not mark the current PP result as reproduced.
5. Register the exact source input/script hashes, pseudopotential hashes, QE binary/module path, historical `relax.out`, failed `pp.out`, final XML, Slurm job/log, and per-file `.mix` SHA-256 values. The separate independent audit and retained-file hash ledger are `QE_MIX_RECIPE_INDEPENDENT_AUDIT_20261123.md` and `qe_mix_retained_artifact_hashes_20261123.jsonl`.
6. Before any future cleanup, verify current scheduler/workdir overlap for the exact target paths with an available Slurm client. The historical cleanup has already completed and has its own receipt.

The concrete fresh-run helper is
`prepare_qe_mix_fresh_start.py`. Copy it to Rockfish, then run
`python3 prepare_qe_mix_fresh_start.py base_spin` (or `mag3` or
`charged3_spin`). It verifies the captured input, original Slurm script,
`pp.in`, all three pseudopotential hashes, the QE 7.2 `pw.x` SHA-256, and that
`/scratch16/pclancy3/yi` is writable. It creates a new isolated directory with
fresh input/script copies and prints the manual `sbatch` command. It does not
edit originals, copy `.mix` files, run PP, or submit the job. Review the
generated files before submission.

This is a reviewed recipe only. The fresh directory and copied inputs/scripts
have not been created or submitted. Catalog status must remain
`DOCUMENTED`/`REVIEWED`, never `REPRODUCED`, until a clean-start PW job is
actually run and its output gates are inspected.
