# Restart coordinate audit — 2026-11-23

Read-only candidate snapshot: `2026-11-23T11:05:06Z`; fresh all-user queue: `2026-11-23T11:08:06+00:00` (1352 job rows). No restart file payloads were opened or hashed, and no deletion was executed.

Current exact candidates: **7,589 files / 1,225,323,826,928 logical bytes**. Coordinate and queue gates classify **7,533 files / 1,224,400,136,336 logical bytes** DELETE-ELIGIBLE and **56 files / 923,690,592 logical bytes** HOLD.

## Scope and classes

| Scope/class | Eligible files | Eligible logical bytes | Held files | Held logical bytes |
|---|---:|---:|---:|---:|
| scratch16 / QE_WFC_N | 5,204 | 297,054,172,800 | 0 | 0 |
| scratch16 / QE_WFC_DAT | 201 | 906,736,303,792 | 0 | 0 |
| scratch16 / QE_MIX_N | 592 | 12,296,444,416 | 0 | 0 |
| data / QE_WFC_N | 0 | 0 | 0 | 0 |
| data / QE_WFC_DAT | 0 | 0 | 0 | 0 |
| data / QE_MIX_N | 1,536 | 8,313,215,328 | 56 | 923,690,592 |

## Coordinate gate

All 7,589 current candidates were present at post-scan stat and mapped to parseable same-run QE XML atomic positions. All eligible NEB roots have a latest parseable complete `.path` with every declared image. The 1,344 data restart-mode candidates are in seven NEB roots; their latest path files parse with full 7/9/10-image coverage and are newer than every candidate in the respective root. The 192 data from-scratch candidates are in two roots with complete 7-image paths and exact WorkDir/job links. Some logs warn that image-4 SCF failed; that quality warning is carried into the per-file decision reason. Restart-mode flags did not override preserved coordinates.

## Holds

The 56 held candidates are 48 data mix files with explicit R04 `sacct WorkDir` mismatch and 8 zero-byte data mix placeholders without a complete path or resolved run/job chain. No scratch candidate is held. No active queue WorkDir overlaps any candidate run root.

## Files

- Exact candidate decisions: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/candidate_restart_disposition_20261123T1115Z.tsv`
- Per-run coordinate/scheduler crosswalk: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/run_coordinate_crosswalk_20261123T1115Z.tsv`
- Bounded, exact-path deletion preflight manifest: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/eligible_restart_deletion_manifest_20261123T1115Z.json` (draft only; not executed)
- Fresh queue snapshot: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/squeue_allusers_20261123T1108Z.txt`
- `sacct` crosswalk: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/sacct_restart_ids_with_submitline_20261123T1114Z.psv`
- Per-run scan JSONL: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/remote_coordinate_crosswalk.jsonl`
- Run input/manifest/runtime context: `/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/restart_coordinate_audit_20261123T1044Z/run_context_20261123T1112Z.jsonl`

The prior 4,298-file removal was only under `/scratch16/pclancy3/yi/revision1_migrationbench_runs` and is outside these 7,589 current candidates. Exact path lookup in the main provenance SQLite found 0 matching artifact keys, run-artifact links, or cleanup links; this audit was not written to SQL.
