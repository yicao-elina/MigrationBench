# Restart coordinate audit — 2026-11-23

Read-only candidate scan: `2026-11-23T11:05:06Z`; final lstat refresh: `2026-11-23T11:29:30Z`; fresh all-user queue: `2026-11-23T11:45:31+00:00` (1366 job rows). No restart-file payloads were opened or hashed, and no deletion was executed.

Current exact candidates: **7,589 files / 1,225,323,826,928 logical bytes**. Coordinate and queue gates classify **7,581 files / 1,225,323,826,928 logical bytes** DELETE-ELIGIBLE and **8 files / 0 logical bytes** HOLD.

## Scope and file classes

| Scope/class | Eligible files | Eligible logical bytes | Held files | Held logical bytes |
|---|---:|---:|---:|---:|
| scratch16 / QE_WFC_N | 5,204 | 297,054,172,800 | 0 | 0 |
| scratch16 / QE_WFC_DAT | 201 | 906,736,303,792 | 0 | 0 |
| scratch16 / QE_MIX_N | 592 | 12,296,444,416 | 0 | 0 |
| data / QE_WFC_N | 0 | 0 | 0 | 0 |
| data / QE_WFC_DAT | 0 | 0 | 0 | 0 |
| data / QE_MIX_N | 1,584 | 9,236,905,920 | 8 | 0 |

## Coordinate and scheduler gate

All 7,589 current candidates have parseable same-run or same-prefix QE XML atomic positions with matching atom counts. All eligible NEB roots have a latest parseable complete `.path` with every declared image. The 1,344 data restart-mode candidates are in seven NEB roots; their latest `.path` files parse with 7/9/10-image coverage and are newer than every candidate in each root. The two from-scratch data NEBs add 192 eligible mixes; their paths cover all seven images. SCF nonconvergence warnings are preserved in the per-file reason and the runs are not promoted as converged results.

The 48 `raw_dicover_node/.../relax.cr/out/sb2te3.mixN` files are now eligible: all map to one parseable two-atom QE XML under the same output prefix, and its mtime is 28.7 ms later than the newest candidate mix. No NEB path is expected for this non-NEB relaxation. The `sacct` record for job 6181656 is COMPLETED/0:0 with submit line `sbatch sbatch-relax.slurm` and WorkDir `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/qe_inputs/2L_octo_Cr2/relax.cr`, while the preserved candidate/XML tree is under `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/raw_dicover_node/input/qe_inputs/2L_octo_Cr2/relax.cr`; this unresolved scheduler-path mismatch is carried as a provenance warning. The `relax.out` and `relax_old.out` SCF/error markers remain attached; do not treat the calculation as scientific ground truth.

The **8 held** files are zero-byte Cr-in-gap placeholders in two run roots, with no complete trajectory or resolved run/job chain. They offer zero storage return. There are 46 eligible run roots and 2 held run roots. The 11:29:07Z all-user queue snapshot evaluated all 46 eligible roots with zero WorkDir overlap.

## Existing SQL evidence

An exact `artifact_key IN (...)` lookup across the main provenance catalog found 0 matches for these 7,589 candidate paths, 0 `run_artifact` links, and 0 cleanup-item links. This audit was not written to SQL.

## Files

- Exact candidate dispositions: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/candidate_restart_disposition_20261123T1129Z.tsv`
- Per-run coordinate/scheduler crosswalk: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/run_coordinate_crosswalk_20261123T1129Z.tsv`
- Eligible-only exact-path manifest: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/eligible_restart_deletion_manifest_20261123T1129Z.json`
- Guarded executor (not run): `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/guarded_delete_exact_restart_manifest.py`
- Summary JSON: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/restart_coordinate_audit_summary_20261123T1129Z.json`
- Fresh queue snapshot: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/squeue_allusers_latest_20261123T1145Z.txt`
- Final lstat refresh: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/candidate_lstat_refresh_20261123T1129Z.jsonl`

The earlier 4,298-file removal was confined to `/scratch16/pclancy3/yi/revision1_migrationbench_runs` and is outside this 7,589-file candidate set.

Executor final source SHA-256: `a9e43aea3ba785f21fde00f6547ad39e315fe56794a35f01fa388a5635c5710d`. It records a per-root queue snapshot immediately before that root's unlink batch, after a separate per-root preflight, and aborts on any queue overlap or malformed/unknown WorkDir. The executor was not run.

A final all-user queue refresh at `2026-11-23T11:45:31+00:00` contained 1350 rows, no malformed/unknown WorkDir records, and zero candidate-root/job-ID overlaps across all 46 eligible roots. The executor will repeat its per-root queue checks immediately before each root batch.

The guarded executor emits a per-file deletion receipt with pre-delete identity, run root, queue-check timestamp, and verified-missing status, followed by the independent batch verification.
