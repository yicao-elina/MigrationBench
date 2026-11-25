# Independent QE `.mix*` reproducibility audit — 2026-11-23

> Historical pre-deletion review. The 12 mixing files were later deleted under
> the guarded cleanup receipt. The scientific status conclusions below remain
> applicable; the pre-deletion preservation recommendation is superseded.

## Scope and access

Read-only audit of the three QE `relax` calculations whose 12 `sb2te3.mix1`–`mix4` files were identified for cleanup. No Rockfish file, input, source script, scheduler state, or SQLite row was modified; no job was submitted. Exact retained-file hashes were streamed from Rockfish using `O_NOFOLLOW`, chunked SHA-256, and before/after `fstat` identity checks. The JSONL hash ledger is `qe_mix_retained_artifact_hashes_20261123.jsonl` (27 path observations; SHA-256 `5bb782b601203e2457a9f42d29caffdbfa4202b9481583efaeaaae317babe06a`).

## Findings

| Run | Historical job | Calculation input | QE log and XML | `.mix*` bytes | Assessment |
|---|---:|---|---|---:|---|
| `base_spin` | 7104884 | `.../2.Cr-in-gap/spin/relax.in` | `relax.out`; `out/sb2te3.save/data-file-schema.xml` | 1,649,140,224 | `relax.out` reports `Maximum CPU time exceeded` before `JOB DONE.`; this is a time-limited partial result, not convergence. No `restart_mode` line, so a fresh run uses the QE default. |
| `mag3` | 7105613 | `.../2.Cr-in-gap/spin/mag3/relax.in` | `relax.out`; `out/sb2te3.save/data-file-schema.xml` | 1,855,282,752 | `relax.out` reports `Maximum CPU time exceeded` before `JOB DONE.`; this is a time-limited partial result, not convergence. No `restart_mode` line, so a fresh run uses the QE default. |
| `charged3_spin` | 7167643 | `.../2.Cr-in-gap/charged_3/spin/relax.in` | `relax.out`; `out/sb2te3.save/data-file-schema.xml` | 1,848,265,920 | `relax.out` reports `Maximum CPU time exceeded` before `JOB DONE.`; this is a time-limited partial result, not convergence. Source input says `restart_mode='restart'`, so a fresh recipe must override it. |

The paths, per-file sizes/hashes, Slurm scripts, inputs, logs, scheduler outputs, and XML references for all 12 `.mix*` files are in `qe_mix_file_lineage_20261123.tsv`. Total `.mix*` size is 5,352,688,896 logical bytes / 5,352,980,480 allocated bytes. Their SHA-256 values were independently recorded in that lineage pass; all 12 are distinct. No WFC candidate was found in the scanned non-Conda roots.

All three calculation inputs use `calculation='relax'`, `prefix='sb2te3'`, `nat=81`, `ntyp=3`, and species Sb/Te/Cr. `base_spin` and `mag3` use `/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/`; `charged3_spin` uses `/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/`. The corresponding Sb/Te/Cr files have identical hashes across roots; see the retained-artifact ledger. The `/home` files show `nlink=3`, so record the canonical content/hash and hard-link condition instead of counting the paths as three independent copies.

Each retained Slurm script calls `module reset`, sources `/data/apps/go.sh`, loads `intel` and `qe/7.2`, runs `/data/apps/extern/qe/7.2/bin/pw.x` with `mpirun -np 4`, then runs `/data/apps/extern/qe/7.2/bin/pp.x` with `mpirun -np 1`. Both binary paths currently exist and are executable. The historical date-bounded `sacct` records say `COMPLETED`, exit `0:0`, but all three PW logs say `Maximum CPU time exceeded` before `JOB DONE.`, and all three Slurm outputs record `Abort(1) ... MPI_Abort` during the later PP step. This is not evidence of converged PW results or successful post-processing. Hashes for each input, script, `relax.out`, `pp.in`, `pp.out`, Slurm output, and final XML are recorded in the JSONL ledger. Exact QE binary identities are in `qe_binary_identity_20261123.txt`.

## Important post-processing limitation

The historical `pp.out` is **not a successful post-processing result** in any of these three folders. Each ends with `Error in routine chdens (1): wrong nx` followed by `stopping ...`. The batch script has no `set -e`, and its final `date` command can return success after `pp.x` fails. The PW `JOB DONE.` follows a configured CPU-time stop, so it records a clean QE stop, not convergence; the saved XML is partial state. The scheduler's `COMPLETED 0:0` likewise does not override the `MPI_Abort` in its stdout or validate `pp.out`, `pp.dat`, or a planar-potential result. Preserve the error logs as failure evidence. If a fresh post-processing output is required, diagnose that separately and do not label it reproduced from the current `pp.in`.

## Recommended reproducibility registration

Register three distinct workflow runs and one fresh-start recipe per run. Keep the historical runs and recipes explicitly separate from the proposed reruns. Bind the exact retained `relax.in`, `sbatch-relax.slurm`, `pp.in`, QE logs, Slurm logs, XML, QE 7.2 executable/module environment, and three pseudopotential hashes. Bind each `.mix*` path/hash as `REPRODUCIBLE_INTERMEDIATE`, `required_for_reproduction=0`; bind inputs and pseudopotentials as required inputs. Record the current historical `pp.out` as a failed derivative, not accepted output.

For a clean PW rerun, use copied files with a new out directory and new log names; never overwrite the original output tree:

1. Copy `relax.in` to `relax.fresh.in`; add `restart_mode='from_scratch'` under `&CONTROL` for all three copies (it explicitly replaces `restart_mode='restart'` for `charged3_spin`). Point `outdir` at `./out_fresh_20261123/`.
2. Copy the Slurm file to a fresh scheduler script. Run `mkdir -p out_fresh_20261123`, then run the same QE 7.2 `pw.x` executable with `relax.fresh.in` and direct output to `relax.fresh.out`. Keep the original Slurm/output files intact. A fresh output directory prevents the old XML/save data from being consumed or overwritten.
3. If testing post-processing independently, copy `pp.in` to `pp.fresh.in`, point it to the fresh outdir, and write `pp.fresh.out`; expect the existing `wrong nx` failure until diagnosed. Do not make PP success a gate for removing PW mixing histories when the preserved `relax.out` and XML are the retained scientific result.

The status should be `DOCUMENTED` or `REVIEWED` until this fresh-start PW recipe is actually run. This audit does not claim a new calculation was executed or bitwise parity with the historical run.

The official QE `pw.x` input description states `restart_mode` defaults to `from_scratch`, defines `from_scratch` as the normal PWscf calculation, and says `restart` is for continuing an interrupted calculation; it also documents `outdir` as the location for input/temporary/output files: [QE pw.x input description](https://www.quantum-espresso.org/Doc/INPUT_PW.html). The linked current manual identifies itself as QE 7.5; the Rockfish scripts and historical runs explicitly use QE 7.2.

## Decision

A fresh-start PW method can be documented from the retained input, script, exact QE binaries, and pseudopotentials, but it has not been rerun and does not preserve the historical restart state. All three historical PW outputs are partial because `Maximum CPU time exceeded` preceded `JOB DONE.`; the later PP step failed. The two default-fresh inputs need an explicit copied recipe for clarity; the charged run must replace its old restart mode and use a new outdir. The PP derivative remains separate failure evidence. The 12 files were later removed under the guarded cleanup receipt `qe_mix_guarded_cleanup_receipt_20261123.jsonl`; this pre-deletion review's preservation recommendation is superseded.


## Runtime and scheduler-time addendum

`qe_binary_identity_20261123.txt` records `readlink -e`, `stat`, and SHA-256 for the QE 7.2 executable paths. `pw.x` resolves to `/data/apps/extern/qe/7.2/bin/pw.x`, mode 0755, 32,383,312 bytes, SHA-256 `e893e4835199422ac09218709bab451b6d47620fdfaa873d135c582da028cb81`. `pp.x` resolves to `/data/apps/extern/qe/7.2/bin/pp.x`, mode 0755, 25,618,952 bytes, SHA-256 `ab617735f7d8ea62870a2886a8afcfab4fd342a6a56221dec63c8036dbc6c328`.

The original `qe_candidate_sacct_20261123.psv` contains no time-zone field or adjacent command receipt establishing `TZ`; treat its Start/End values as raw timestamps with unspecified zone. The parent-provided recapture `qe_candidate_sacct_utc_20261123.psv` was captured with `TZ=UTC`; every Start and End for all 12 matching job/step rows is exactly four hours later than the original PSV. The Slurm output files themselves label their July 2025 end dates `EDT`. A separate current Rockfish `date +%Z %z` read returned `EST -0500`; `TZ=UTC date` returned `UTC +0000`. That confirms the current system setting only; it does not establish the environment of the old PSV capture. A direct `sacct` command is not on the current remote login PATH, so this audit did not make a second Slurm query.
