# Rockfish Cr-NEB blueprint and monitoring goal

## One-sentence goal

Audit the three existing Rockfish Cr-substitution NEBs end to end, preserve
their warning-aware scientific status, and produce a provenance-complete
forward/reverse comparison against Cu/Li/Na plus fixed-path MLIP diagnostics;
Discovery is out of scope.

## Execution sequence

1. `ssh rockfish` and verify account, root, `sacct`, `squeue`, and job mapping.
2. Re-check each `neb.out`, `neb.err`, `sb2te3.path`, `sb2te3.xyz`, image count,
   directional barriers, force/gradient finiteness, and SCF warnings.
3. Preserve the three existing jobs as unique results. Do not resubmit merely
   because a result is warning-contaminated; request a decision before any
   clean rerun.
4. Rebuild the provisional QE table and model-path table with explicit status:
   `COMPLETED_BUT_SCF_WARNING`, `PROVISIONAL`, or `VERIFIED`.
5. Compare Cr with the original Cu/Li/Na paths by path class. For gap-to-QL,
   report both forward and reverse activation energies because the physically
   uphill direction may be reverse.
6. Keep Rockfish figures separate from the Li/Na-derived-Cr Discovery design.
   Only create a combined paper figure after the user accepts the warning
   policy or clean Rockfish reruns pass.

## Expected tables

- `rockfish_cr_qe_summary.csv`: job, path class, forward/reverse QE values,
  iteration count, SCF-warning count, and publication status;
- `rockfish_cr_foundation_image_profiles.csv`: all model/image energies and
  finite force diagnostics;
- `rockfish_cr_foundation_barrier_table.csv`: model-vs-QE directional errors,
  marked diagnostic while QE warnings remain;
- combined table: Cu/Li/Na and Cr by source geometry and path class;
- exclusion ledger: any timeout, nonconvergence, missing image, or hash mismatch.

## Expected figures

- Rockfish-only three-column bidirectional Cr barrier panel;
- source-template comparison: Cr←Cu versus Cr←Li versus Cr←Na, kept separate
  from the Rockfish-only release;
- updated Figure 1 v4 only after the corresponding source/path rows are
  scientifically qualified;
- per-image energy profiles with forward/reverse annotations and no hidden
  failed curves.

## Publication language boundary

The current Rockfish values are finite, completed path-table diagnostics with
SCF warnings. They can support a clearly labelled diagnostic comparison, but
not a claim of converged DFT barriers unless the warning is resolved or the
user explicitly accepts a provisional analysis. MLIP values are fixed-path
single-point evaluations and are not self-relaxed MLIP NEBs.

## No-go rules

- Never access or submit to Discovery from this task.
- Never overwrite raw Rockfish files.
- Never treat `COMPLETED` or `JOB DONE` alone as scientific success.
- Never merge warning-contaminated barriers into a clean publication table.
- Never create a duplicate valid result for an existing path.

## Continuation branch

The user-approved continuation is an additive Rockfish branch, not a rewrite
of the first run. Its root is:

`/scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013/restarts_20261016/`

Submitted jobs are 32069316 (within QLs), 32069317 (gap to QL), and 32069318
(within gap), continuing jobs 31916908, 31916909, and 31916910 respectively.
The original `out/` directories are intentionally not copied; each restart
has a fresh `out/` and retains the prior `neb.out` as `neb_previous.out`.
The dedicated monitoring window should audit these three jobs with the same
warning-aware gates and must not access Discovery.
