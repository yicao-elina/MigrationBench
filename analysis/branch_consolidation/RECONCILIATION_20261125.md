# Canonical repository reconciliation — 2026-11-25

## MigrationBench source mapping

This candidate is based on the existing MigrationBench PR #20 head `4eeaaaa44bfdc06ae911d9d1bb4d7def674b3ca2` and reconciles the local unified evidence head `8fd83b65` (`/Users/alina/.codex/worktrees/migrationbench-unified-20261121`). Those histories diverged at `476ca161ac6235965f8062b83484d580c846fed7`; a normal merge produced 45 add/add conflicts. The candidate uses a squash of the local content onto PR #20 so its published history remains anchored to the current PR branch, while this record preserves the second source head for audit.

Conflict decisions were made by comparing the competing file contents and their provenance timestamps, not by accepting one side globally:

- Kept PR #20 content for 24 conflict paths with the later 2026-11-24 revision, including Canva export metadata, Cr sliding analysis, Fig. 6 artifacts, audit SQL/catalog material, SHAP updates, and Task 2 manifests/scripts.
- Kept the local unified content for 21 conflict paths where the local Phase 1/Phase 2 evidence revision dated 2026-10-03 superseded the PR branch's 2026-10-02 version.
- Four MACE checkpoint blobs were excluded from the Git tree after exact SHA-256 matches were confirmed in the organized Rockfish model archive. The repository model manifest remains the source of archive pointers and labels these weights as archive-only; the checkpoints were not renamed or relabeled.
- The 54.3 MB Canva P03 editable SVG is also archive-only to satisfy the repository's release size audit. Rockfish holds an exact SHA-256 copy at `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_git_large_assets_20261124/figures/CANVA-P03_parity_2x4_palette_updated.svg` (`6844ea40abfb75f8f1310b52b3e5633af2f024fd19329695332b815f8656371f`).

The resulting staged tree had no unresolved index entries at capture. The eventual commit SHA and CI run must be added here after the candidate is committed and pushed. This document does not claim that the PR is merged.

## Current storage scope and evidence

New inventory/cleanup work is restricted to `/data/pclancy3/yi` and `/scratch4/pclancy3/yi`. The 2026-11-25 metadata-only reports and compact evidence are in `phase2_data/data_yi_refresh_20261125/` and `phase2_data/scratch4_yi_refresh_20261125/`. Full inventories and hardlink path lists are intentionally retained in the Rockfish evidence archive rather than Git:

- `/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/data_pclancy3_yi_refresh_20261125/data_yi_refresh_20261125/`
- `/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/scratch4_yi_refresh_20261125/scratch4_yi_refresh_20261125/`

The two compressed transfer bundles have SHA-256 `43885965c9c95f84b2c4a9115175a3e3e28525e50125a138d6ce3ca55e24c7cd` (data) and `f1658eb2be2fad0a89b1b3c53add530cfe7b5c506ff1caef7e1f897f2ad940fc` (scratch4). Per-file archive checks passed. These walks made no changes and cleared no deletion: the live owner job/workdir check could not run because `squeue` is unavailable in the Rockfish login environment.

The 2026-11-25 checksums sidecars are `7bef4f5824d547dfeff4c6d9c105b2c801403c521cfca74246e3b9ff933d4787` (data inventory) and `a798fadd684bb3c25401aa6c796af913eae80e6cf9338d71b09b2a1489b384ae` (scratch4 inventory). The new SQL snapshot at `/scratch16/pclancy3/yi/MigrationBench_data_archive/provenance_catalog_snapshots/20261125/provenance_catalog_20261125.sqlite` is 4,032,221,184 bytes, mode `0600`, SHA-256 `00750bbc76123777294e59338c0ac823309ec56bff483a482d3a87958274ba7f`; `quick_check=ok`, foreign-key violations 0. It records the complete scan totals and the documented 317,898 `/data` plus 100 `/scratch4` row-level observations. Query examples and the importer are in `phase2_schema/scoped_inventory_queries_20261125.sql` and `phase2_schema/register_scoped_inventory_20261125.py`.

The candidate PR excludes the legacy mixed-owner/group scans, other `/data/pclancy3` roots, and raw all-user queue exports. Historical cleanup receipts remain only where their target files were inside the two authorized roots; reports are redacted to remove unrelated root inventories and group quota context.

No claim is made that the large storage inventory is itself a complete experiment provenance map. The top-200 catalog cross-reference is included in the compact evidence folder. Full run/figure/claim SQL traceability remains a separate acceptance criterion for publication evidence.

## Remaining canonicalization gates

- Rockfish login has no `git` executable. The two Scratch16 clones therefore have not yet been refreshed with a Git pull or verified clean against these PR heads.
- Both PRs remain open. Do not delete source branches, worktrees, or duplicate checkouts until the updated PRs are reviewed/merged and each canonical Scratch16 clone is verified at the merged commit.
- The DUAL-X crosswalk currently matches model identity by SHA-256 for 10 of 23 X-FORCE checkpoints. It does not yet map MigrationBench task-level performance, claim evidence, failure diagnoses, or active-learning decisions into DUAL-X. The 414-cell DUAL-X design matrix is not a 414-result dataset; the run database currently contains 80 model-run records.
- Publication audit findings remain intact, including QE NEB residual/convergence limits, the SI Fig. S2 caption/plot mismatch, and MD termination/result-lineage gaps. No scientific claims were rewritten by repository reconciliation.
