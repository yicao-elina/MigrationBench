#!/usr/bin/env python3
"""Register the human-readable audit entry points and the scripts that produce them."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
LOCAL_ROOT = "LOCAL-MAIN"

DELIVERABLES = [
    ("PHASE2_REPORT.md", "PHASE2_REPORT", "Phase-2 inventory, quota, cleanup and open-scope report", "PERMANENT_SOURCE"),
    ("README.md", "AUDIT_INDEX", "Audit folder index and current scope summary", "PERMANENT_SOURCE"),
    ("phase2_schema/README.md", "PROVENANCE_SCHEMA_GUIDE", "SQL catalog schema and maintenance guide", "PERMANENT_SOURCE"),
    ("phase2_schema/009_paper_figure_archive.sql", "SQL_SCHEMA_MIGRATION", "Typed paper-figure and source-claim observation migration", "PERMANENT_SOURCE"),
    ("phase2_schema/010_paper_claim_figure_links.sql", "SQL_SCHEMA_MIGRATION", "Explicit source-backed paper-claim to figure links", "PERMANENT_SOURCE"),
    ("phase2_schema/catalog_queries.sql", "SQL_QUERY_COOKBOOK", "Read-only project, run, claim, inventory, and cleanup provenance queries", "PERMANENT_SOURCE"),
    ("phase2_schema/revision1_catalog_queries.sql", "REVISION1_PAPER_EVIDENCE_QUERIES", "Read-only submitted Revision 1 figure, claim, restart, and MD archive retrieval queries", "PERMANENT_SOURCE"),
    ("phase2_schema/import_paper_figure_archive.py", "PROVENANCE_IMPORT_SCRIPT", "Idempotent figure, claim and evidence importer", "PERMANENT_SOURCE"),
    ("phase2_schema/register_phase2_deliverables.py", "PROVENANCE_REGISTRATION_SCRIPT", "Idempotent registration of audit report and schema deliverables", "PERMANENT_SOURCE"),
    ("phase2_schema/register_rockfish_status_refresh.py", "PROVENANCE_REGISTRATION_SCRIPT", "Idempotent registration of fresh quota, queue, and mount-access captures", "PERMANENT_SOURCE"),
    ("phase2_schema/import_quota_capture.py", "PROVENANCE_IMPORT_SCRIPT", "Idempotent import of timestamped GPFS quota captures into quota_observation", "PERMANENT_SOURCE"),
    ("phase2_schema/register_dualx_safe_core.py", "PROVENANCE_IMPORT_SCRIPT", "Idempotently registers DUAL-X Safe Core v6 repository, runs, verified artifacts, figures, claims, and derivation links", "PERMANENT_SOURCE"),
    ("phase2_schema/import_partial_refresh_inventories.py", "PROVENANCE_IMPORT_SCRIPT", "Additive importer for explicitly partial Rockfish metadata refreshes", "PERMANENT_SOURCE"),
    ("phase2_schema/import_mhp_defect_r04.py", "PROVENANCE_IMPORT_SCRIPT", "Idempotent importer for the reconciled current MHP_defect metadata inventory", "PERMANENT_SOURCE"),
    ("phase2_schema/import_accessible_data_mount_census_r05.py", "R05_DATA_MOUNT_CENSUS_IMPORTER", "Imports the bounded partial /data mount census with incomplete coverage explicitly preserved", "PERMANENT_SOURCE"),
    ("phase2_schema/register_scratch4_mount_census_r05.py", "R05_SCRATCH4_MOUNT_CENSUS_EVIDENCE_IMPORTER", "Registers aggregate /scratch4 outer-root evidence without misrepresenting it as a full per-path snapshot", "PERMANENT_SOURCE"),
    ("phase2_schema/import_full_git_reconciliation_r04.py", "R04_FULL_GIT_REF_IMPORTER", "Imports current cached/live ref and dirty-worktree observations for all 29 Rockfish project repositories", "PERMANENT_SOURCE"),
    ("phase2_data/refresh_mount_access_boundary_20261123.py", "MOUNT_ACCESS_BOUNDARY_SCRIPT", "Read-only current immediate-root ownership and traversal probe for /data and /scratch4", "PERMANENT_SOURCE"),
    ("phase2_data/MOUNT_ACCESS_BOUNDARY_REVIEW_20261123.md", "MOUNT_ACCESS_BOUNDARY_REVIEW", "Current account ownership and access boundary review for both Rockfish mounts", "PERMANENT_SOURCE"),
    ("phase2_data/rockfish_quota_refresh_20261123T0115Z.txt", "GPFS_QUOTA_CAPTURE", "Fresh GPFS quota observation for /data and /scratch4", "PERMANENT_SOURCE"),
    ("phase2_data/rockfish_quota_git_ref_preflight_20261123T023959Z.txt", "GPFS_QUOTA_CAPTURE", "Fresh group quota observation captured before Rockfish Git ref reconciliation", "PERMANENT_SOURCE"),
    ("phase2_data/git_ref_update_preflight_receipts_20261123.jsonl", "GIT_REF_UPDATE_PREFLIGHT_RECEIPT", "Local-only preflight and postcheck receipts; no refs changed and no GitHub push", "PERMANENT_SOURCE"),
    ("phase2_data/dualx_safe_core_git_snapshot_20261123.txt", "DUALX_LOCAL_GIT_SNAPSHOT", "Read-only DUAL-X local branch, head, tag, upstream, remote, and worktree status snapshot", "PERMANENT_SOURCE"),
    ("phase2_data/current_queue_all_users_refresh_20261123T0115Z.txt", "SLURM_QUEUE_ALL_USERS_CAPTURE", "Fresh all-user queue snapshot for cleanup overlap checks", "PERMANENT_SOURCE"),
    ("phase2_data/root_permission_refresh_20261123T0115Z.txt", "STORAGE_ROOT_ACCESS_BOUNDARY_CAPTURE", "Aggregate immediate-child traversal access at /data and /scratch4", "PERMANENT_SOURCE"),
    ("phase2_data/root_owner_boundary_refresh_20261123T0130Z.txt", "STORAGE_ROOT_OWNER_BOUNDARY_CAPTURE", "Aggregate immediate-child ownership boundary at /data and /scratch4", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/squeue_group_20261123.txt", "SLURM_QUEUE_PCLANCY3_GROUP_CAPTURE", "Fresh group-account queue snapshot for Scratch4 cleanup overlap", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_gt500_refresh_20261123.txt", "TARGETED_LARGE_FILE_THRESHOLD_RECHECK", "Fresh >500 MB count for the revision1 migrationbench run tree", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_restart_population_refresh_20261123.txt", "TARGETED_RESTART_FILENAME_RECHECK", "Fresh WFC and mixing filename population for revision1 runs", "PERMANENT_SOURCE"),
    ("phase2_data/GIT_RECONCILIATION_REVIEW_20261123.md", "GIT_MAINLINE_RECONCILIATION_REVIEW", "Read-only review of current canonical branches, dirty trees, and local merge candidates", "PERMANENT_SOURCE"),
    ("phase2_data/GIT_LIVE_MAINLINE_CANDIDATE_DECISIONS_20261123.md", "GIT_LIVE_MAINLINE_CANDIDATE_DECISIONS", "Live-ref-verified local mainline candidates; dirty Rockfish worktrees preserved and no push", "PERMANENT_SOURCE"),
    ("phase2_data/git_live_mainline_candidates_20261123/neuromorphic_filament_local_main_promotion_20261123.json", "GIT_LOCAL_MAINLINE_PROMOTION_RECEIPT", "Guarded fast-forward of main in the isolated neuromorphic-filament bare candidate clone only", "PERMANENT_SOURCE"),
    ("phase2_data/first_party_live_default_heads_20261123.tsv", "GIT_LIVE_DEFAULT_HEADS", "Fresh live remote default branch heads for user-owned Rockfish repositories", "PERMANENT_SOURCE"),
    ("phase2_data/first_party_live_ref_graph_deltas_20261123.tsv", "GIT_LIVE_REF_GRAPH_DELTAS", "Local versus live branch graph comparison for updated remotes", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123/DATA_REFRESH_PARTIAL_REPORT_20261123.md", "PARTIAL_DATA_INVENTORY_REPORT", "Coverage and failure report for the incomplete /data/pclancy3 refresh", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123/data_pclancy3_refresh_partial_summary_20261123.json", "PARTIAL_DATA_INVENTORY_SUMMARY", "Machine-readable partial /data/pclancy3 inventory summary", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123/data_pclancy3_refresh_20261123.jsonl.gz", "PARTIAL_DATA_INVENTORY_LEDGER", "Immutable raw partial /data/pclancy3 path metadata stream", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123/inventory_data_pclancy3_remote.py", "PARTIAL_DATA_INVENTORY_SCRIPT", "Metadata-only scanner used for the incomplete /data/pclancy3 refresh", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/README.md", "PARTIAL_SCRATCH4_INVENTORY_REPORT", "Coverage, candidate triage, queue, Git, and limitation report for partial /scratch4/pclancy3 refresh", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/inventory_summary_partial.json", "PARTIAL_SCRATCH4_INVENTORY_SUMMARY", "Machine-readable lower-bound summary of partial /scratch4/pclancy3 refresh", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/r01_vs_r02_partial_comparison.json", "PARTIAL_SCRATCH4_INVENTORY_COMPARISON", "Comparison of emitted R02 paths with the earlier R01 /scratch4/pclancy3 baseline", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/scratch4_pclancy3_metadata_20261123.jsonl.gz", "PARTIAL_SCRATCH4_INVENTORY_LEDGER", "Immutable raw partial /scratch4/pclancy3 path metadata stream", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/inventory_stderr.txt", "PARTIAL_SCRATCH4_SCAN_STDERR", "Remote metadata scanner stderr for the incomplete /scratch4/pclancy3 refresh", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/scan_invocation.txt", "PARTIAL_SCRATCH4_SCAN_INVOCATION", "Exact scanner launch command and remote target for the incomplete /scratch4 scan", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/scan_execution_record.json", "PARTIAL_SCRATCH4_SCAN_EXECUTION", "Local execution and disconnect record for the partial /scratch4 scan", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/artifact_hashes.sha256", "PARTIAL_SCRATCH4_ARTIFACT_HASH_MANIFEST", "SHA-256 manifest for the partial /scratch4 scan outputs", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/next_review_batches_20261123.tsv", "SCRATCH4_REVIEW_QUEUE", "Owner-scoped candidate batches and next evidence steps; not deletion authorization", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/large_files_over_100MB_partial.tsv", "PARTIAL_SCRATCH4_LARGE_FILE_REVIEW", "Large files above 100 MB emitted before partial scan ended", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/large_files_over_500MB_partial.tsv", "PARTIAL_SCRATCH4_LARGE_FILE_REVIEW", "Large files emitted before partial scan ended", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/wfc_mixing_candidates_partial.tsv", "PARTIAL_SCRATCH4_RESTART_REVIEW", "Filename-based WFC/mixing matches from partial scan, including false-positive source names", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/duplicate_indicators_large_exact_metadata.tsv", "PARTIAL_SCRATCH4_DUPLICATE_INDICATORS", "Metadata-only duplicate indicators requiring hash and consumer verification", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/cache_build_path_candidates_partial.tsv", "PARTIAL_SCRATCH4_CACHE_BUILD_REVIEW", "Cache/build path matches from partial scan; names alone do not prove disposability", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/temporary_suffix_candidates_partial.tsv", "PARTIAL_SCRATCH4_TEMPORARY_SUFFIX_REVIEW", "Temporary-suffix matches from partial scan; retain pending owner and consumer review", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/scripts_manifests_index_partial.tsv", "PARTIAL_SCRATCH4_REPRODUCIBILITY_INDEX", "Scripts and manifest path index emitted by partial scan", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/owner_project_summary_partial.tsv", "PARTIAL_SCRATCH4_OWNER_PROJECT_SUMMARY", "Owner and project path totals for emitted partial rows", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/owner_summary_partial.tsv", "PARTIAL_SCRATCH4_OWNER_SUMMARY", "Owner byte and file totals for emitted partial rows", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/extension_summary_partial.tsv", "PARTIAL_SCRATCH4_EXTENSION_SUMMARY", "File-extension summary for emitted partial rows", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/top_level_breakdown_partial.tsv", "PARTIAL_SCRATCH4_ROOT_SUMMARY", "Top-level subtree breakdown for emitted partial rows", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/symlinks_partial.tsv", "PARTIAL_SCRATCH4_SYMLINK_INDEX", "Symlink paths and targets observed in partial scan", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/traversal_errors_partial.tsv", "PARTIAL_SCRATCH4_TRAVERSAL_ERRORS", "Traversal errors observed in partial /scratch4/pclancy3 refresh", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_metadata_paths_partial.tsv", "PARTIAL_SCRATCH4_GIT_METADATA_INDEX", "Git metadata paths observed before partial scan stopped", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_refs_partial.jsonl", "PARTIAL_SCRATCH4_GIT_REF_CAPTURE", "Manually read Git HEAD and ref metadata for in-tree .git directories", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_repository_summary_partial.tsv", "PARTIAL_SCRATCH4_GIT_REF_SUMMARY", "Manually read Git HEAD/ref metadata for observed .git directories; worktree status unavailable", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_repository_status_partial.jsonl", "PARTIAL_SCRATCH4_GIT_STATUS_ATTEMPT", "Git status attempt output; repository cleanliness was unavailable", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_repository_status_partial.exit_status", "PARTIAL_SCRATCH4_GIT_STATUS_EXIT", "Exit status from partial Git status attempt", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_repository_status_partial.stderr", "PARTIAL_SCRATCH4_GIT_STATUS_STDERR", "Error output from partial Git status attempt", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_refs_partial.exit_status", "PARTIAL_SCRATCH4_GIT_REF_EXIT", "Exit status for partial Git ref metadata read", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_refs_partial.stderr", "PARTIAL_SCRATCH4_GIT_REF_STDERR", "Error output for partial Git ref metadata read", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/git_files_skipped_partial.tsv", "PARTIAL_SCRATCH4_GITDIR_FILE_LIMITS", "Git worktree .git files deliberately not followed during metadata scan", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/squeue_group_20261123T0210Z.txt", "SLURM_QUEUE_PCLANCY3_REFRESH_CAPTURE", "Fresh pclancy3-account queue capture at 02:10 UTC for active workdir exclusion", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123/squeue_group_20261123T0210Z.stderr", "SLURM_QUEUE_PCLANCY3_REFRESH_STDERR", "Stderr for the fresh pclancy3-account queue capture", "PERMANENT_SOURCE"),
    ("scoped_storage_inventory_20261122.tsv", "SCOPED_STORAGE_INVENTORY", "Path-level Sb2Te3 and large-file scope inventory", "PERMANENT_SOURCE"),
    ("phase2_scratch4_group/remaining_large_file_triage_20261122.md", "SCRATCH4_LARGE_FILE_TRIAGE", "Owner-scoped disposition of large shared-filesystem paths", "PERMANENT_SOURCE"),
    ("phase2_data/LOCAL_GIT_MERGE_CANDIDATES_REPORT.md", "GIT_MAINLINE_RECONCILIATION", "Local-only repository mainline and merge candidate summary", "PERMANENT_SOURCE"),
    ("phase2_data/LOCAL_GIT_ALIGNMENT_STATUS_20261123.md", "GIT_UNIFIED_MAINLINE_ALIGNMENT_REVIEW", "Ancestry and dirty-worktree review for unified main, isolated candidate clones, and blocked Rockfish refs", "PERMANENT_SOURCE"),
    ("phase2_schema/stage_qe_mix_cleanup.py", "QE_CLEANUP_SQL_STAGER", "Registers exact QE partial-run lineage, fresh-start recipe links, and guarded mix-file cleanup items", "PERMANENT_SOURCE"),
    ("phase2_schema/execute_qe_mix_guarded_cleanup.py", "QE_GUARDED_CLEANUP_EXECUTOR", "All-user queue and exact identity/hash guarded deletion executor with immediate pre-unlink rechecks", "PERMANENT_SOURCE"),
    ("phase2_schema/register_qe_mix_preflight_failure.py", "QE_PREFLIGHT_FAILURE_REGISTRAR", "Preserves the preflight-only Python 3.6 compatibility failure and exact first-attempt executor", "PERMANENT_SOURCE"),
    ("phase2_schema/register_qe_mix_cleanup_queue.py", "QE_QUEUE_EVIDENCE_LINKER", "Extracts the all-user queue snapshot from the guarded receipt and links it to cleanup_execution_receipt", "PERMANENT_SOURCE"),
    ("phase2_schema/record_unified_main_git_snapshot.py", "UNIFIED_GIT_SNAPSHOT_IMPORTER", "Records the unified local main, cached refs, clean-worktree state, and branch ancestry without fetching or pushing", "PERMANENT_SOURCE"),
    ("phase2_schema/record_cleanup_execution_receipt.py", "CLEANUP_RECEIPT_IMPORTER", "Reconciles a guarded per-file deletion receipt and before/after quota bracket into SQL", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/YI_BOUNDED_REFRESH_REPORT_20261123.md", "QE_BOUNDED_REFRESH_AND_CLEANUP_REPORT", "Bounded /data/pclancy3/yi candidate refresh, QE run status, and completed guarded cleanup report", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/rockfish_quota_before_qe_mix_cleanup_20261123.txt", "GPFS_QUOTA_BEFORE_QE_MIX_CLEANUP", "Raw GPFS quota capture immediately before the QE mixing cleanup", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/rockfish_quota_after_qe_mix_cleanup_20261123.txt", "GPFS_QUOTA_AFTER_QE_MIX_CLEANUP", "Raw GPFS quota capture immediately after the QE mixing cleanup", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/qe_mix_guarded_cleanup_receipt_20261123.jsonl", "QE_MIX_GUARDED_DELETION_RECEIPT", "Successful all-user queue, per-file hash/identity, verified-missing, and allocated-byte receipt", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/qe_mix_guarded_cleanup_preflight_failure_20261123.jsonl", "QE_MIX_PREFLIGHT_FAILURE_RECEIPT", "First attempt failed before queue parsing or any deletion due to remote Python 3.6 API compatibility", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/execute_qe_mix_guarded_cleanup_attempt1_20261123.py", "QE_MIX_INITIAL_EXECUTOR_VERSION", "Exact guarded cleanup executor source from the zero-deletion preflight failure", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/qe_mix_live_alluser_squeue_20261123.jsonl", "QE_MIX_ALL_USER_QUEUE_SNAPSHOT", "All-user queue snapshot directly linked to the QE cleanup execution receipt", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/qe_mix_cleanup_sql_verification_20261123.json", "QE_MIX_SQL_CLEANUP_VERIFICATION", "Read-only verification summary for the completed batch, run states, recipe statuses, queue link, and quota bracket", "PERMANENT_SOURCE"),
    ("phase2_schema/import_scoped_r03_inventories.py", "SCOPED_INVENTORY_IMPORT_SCRIPT", "Idempotent importer for complete isolated R03 data and scratch4 subtree snapshots; skips incomplete or failed roots", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123_r03/R03_CONDA_COVERAGE_REPORT_20261123.md", "R03_DATA_INVENTORY_REPORT", "Reconciled .conda coverage, removed-path receipts, and held large environment payload review", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123_r03/conda_coverage_reconciliation_r03.json", "R03_DATA_INVENTORY_RECONCILIATION", "Machine-readable reconciliation of complete R01 baseline, R02 partial stream, R03 roots, and exact cleanup receipts", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123_r03/conda_large_files_over_100MB_r03.tsv", "R03_DATA_LARGE_FILE_REVIEW", "Metadata-only list of reconciled .conda files over 100 MB; all remain held as installed environment or package-cache payloads", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123_r03/scan_conda_env_roots_r03.py", "R03_DATA_INVENTORY_SCANNER", "Metadata-only remote scanner for isolated Conda environment roots", "PERMANENT_SOURCE"),
    ("phase2_data/data_refresh_20261123_r03/analyze_conda_coverage_r03.py", "R03_DATA_INVENTORY_RECONCILER", "Reconciles exact .conda path coverage, duplicate observations, and cleanup receipt removals", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/R03_REPORT.md", "R03_SCRATCH4_INVENTORY_REPORT", "Selected subtree refresh, UID ownership, WFC/mixing candidate boundaries, and incomplete mount coverage", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/selected_root_refresh_summary_r03.json", "R03_SCRATCH4_INVENTORY_SUMMARY", "Machine-readable selected-root completion, ownership, candidate, and prune summary", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/qe_wfc_mix_restart_candidates_r03.tsv", "R03_SCRATCH4_RESTART_REVIEW", "Metadata-only WFC/mixing-name rows from selected roots; all observed matches are other-owner review rows", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/root_status_r03.jsonl", "R03_SCRATCH4_ROOT_STATUS", "Per-root completion and exact pruned-subtree statuses for selected scratch4 scans", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/scan_selected_roots_r03.py", "R03_SCRATCH4_INVENTORY_SCANNER", "Isolated no-follow metadata scanner for selected scratch4 subtrees", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/scan_root_skip_prefix_r03.py", "R03_SCRATCH4_BOUNDED_SCANNER", "Bounded metadata scanner that records one exact excluded subtree", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/summarize_selected_roots_r03.py", "R03_SCRATCH4_INVENTORY_SUMMARIZER", "Summarizes root ownership, candidate patterns, and path-set comparisons", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/compare_r01_r02_by_toproot.py", "R03_SCRATCH4_COVERAGE_COMPARATOR", "Compares top-root coverage across the prior R01/R02 inventory evidence", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r03/squeue_allusers_r03.txt", "R03_SLURM_QUEUE_ALL_USERS_CAPTURE", "Latest successful all-user queue snapshot for R03 inventory and cleanup overlap review", "PERMANENT_SOURCE"),
    ("phase2_data/rockfish_final_quota_20261123T0506Z/quota_capture.txt", "GPFS_QUOTA_CAPTURE", "Latest successful timestamped GPFS user and group quota capture for /data and /scratch4", "PERMANENT_SOURCE"),
    ("phase2_data/rockfish_final_quota_20261123T0506Z/execution_status.json", "GPFS_QUOTA_CAPTURE_STATUS", "Successful SSH command and byte-count receipt for the 05:02 UTC quota capture", "PERMANENT_SOURCE"),
    ("phase2_data/rockfish_final_refresh_20261123_attempt3/execution_status.json", "SLURM_REFRESH_FAILURE_STATUS", "Final unsuccessful Slurm refresh attempt; failed during cluster DNS SRV lookup", "PERMANENT_SOURCE"),
    ("phase2_data/rockfish_final_refresh_20261123_attempt3/quota_queue_stderr.txt", "SLURM_REFRESH_FAILURE_STDERR", "Slurm DNS SRV lookup error from the final refresh attempt", "PERMANENT_SOURCE"),
    ("phase2_data/rockfish_final_refresh_20261123_attempt3/quota_queue_stdout.txt", "SLURM_REFRESH_FAILURE_STDOUT", "Partial stdout preserved from the final failed Slurm refresh attempt", "PERMANENT_SOURCE"),
    ("phase2_data/former_collab_cleanup_execution_addendum_20261123.md", "FORMER_COLLAB_CLEANUP_EXECUTION_ADDENDUM", "Post-delete reconciliation for the same-pipeline Yuqi and Victor duplicate cleanup batches", "PERMANENT_SOURCE"),
    ("phase2_schema/reconcile_former_collab_duplicate_outcomes.py", "FORMER_COLLAB_CLEANUP_SQL_RECONCILER", "Guarded SQL-only reconciler that verifies receipts/current handoff paths and updates stale duplicate and cleanup notes", "PERMANENT_SOURCE"),
    ("phase2_data/victor_duplicate_delete_receipt_20261122.json", "FORMER_COLLAB_GUARDED_DELETE_RECEIPT", "Authoritative checked-queue Victor receipt; nine exact duplicate copies unlinked and verified missing", "PERMANENT_SOURCE"),
    ("phase2_data/yuqi_pseudopotential_delete_receipt_20261122.jsonl", "FORMER_COLLAB_GUARDED_DELETE_RECEIPT", "Zero-overlap Yuqi receipt; 65 exact duplicate UPF copies unlinked and verified missing", "PERMANENT_SOURCE"),
    ("phase2_data/yi_refresh_20261123/handoff_roots_20261123.jsonl", "FORMER_COLLAB_CURRENT_HANDOFF_INVENTORY", "Complete current metadata refresh confirming retained paths in for-victor and for-yuqi after cleanup", "PERMANENT_SOURCE"),
    ("phase2_schema/import_scoped_r04_scratch4.py", "R04_SCRATCH4_INVENTORY_IMPORTER", "Imports only personal /scratch4/pclancy3/yi metadata roots; skips sibling roots before resolving or reading ledgers", "PERMANENT_SOURCE"),
    ("phase2_schema/011_project_artifact_lookup_indexes.sql", "SQL_SCHEMA_MIGRATION", "Indexes for project-scoped artifact and reverse path lookup", "PERMANENT_SOURCE"),
    ("phase2_schema/import_r04_targeted_lstat.py", "R04_TARGETED_LSTAT_IMPORTER", "Imports one explicitly partial no-follow lstat sample under the historical R04 audit", "PERMANENT_SOURCE"),
    ("phase2_schema/import_r04_data_yi.py", "R04_DATA_YI_COMPOSITE_IMPORTER", "Imports exact /data/pclancy3/yi path coverage from R04 non-Conda metadata plus reconciled R03 .conda evidence", "PERMANENT_SOURCE"),
    ("phase2_schema/import_r04_collab_roots.py", "R04_COLLAB_ROOTS_IMPORTER", "Imports exact current metadata for ycao73_home, for-victor, and for-yuqi without claiming full /data coverage", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r04_remaining/R04_CGANLEY_USERS_COVERAGE_REPORT.md", "R04_HISTORICAL_COVERAGE_REPORT", "Historical metadata-only R04 coverage and exact comparison for the cganley archive subtree", "PERMANENT_SOURCE"),
    ("phase2_scratch4/scratch4_refresh_20261123_r04_remaining/R04_REMAINING_INACTIVE_ROOTS_REPORT.md", "R04_HISTORICAL_COVERAGE_REPORT", "Historical metadata-only R04 coverage and exact comparison for 16 inactive roots", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/scan_yi_full_nofollow_r04.py", "YI_METADATA_INVENTORY_SCANNER", "Metadata-only no-follow scanner retained for the interrupted full-root attempt", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/scan_yi_nonconda_nofollow_r04.py", "YI_NONCONDA_INVENTORY_SCANNER", "Metadata-only no-follow scanner restricted to /data/pclancy3/yi outside exact .conda", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/reconcile_yi_r04.py", "YI_R04_RECONCILIATION_SCRIPT", "Reconciles exact non-Conda R04 paths against the R01 baseline and joins R03 .conda coverage", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/finalize_yi_r04_report.py", "YI_R04_REPORT_SCRIPT", "Builds scoped /data/pclancy3/yi reports and metadata-only candidate lists", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/R04_REPORT.md", "YI_R04_INVENTORY_REPORT", "Complete path-coverage report for exact /data/pclancy3/yi scope, with .conda freshness limits", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_coverage_reconciliation_r04.json", "YI_R04_COVERAGE_RECONCILIATION", "Machine-readable complete union, exact path-set diffs, scan status, and storage totals", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_uid3901_current_entries_r04.tsv", "YI_R04_UID_SCOPED_PATHS", "Current R04 metadata rows owned by UID 3901 outside .conda", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_direct_child_storage_summary_r04.tsv", "YI_R04_DIRECT_CHILD_STORAGE_SUMMARY", "File counts and logical/allocated bytes by direct child of /data/pclancy3/yi", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_direct_child_large_and_restart_summary_r04.tsv", "YI_R04_DIRECT_CHILD_CANDIDATE_SUMMARY", "Large-file and WFC/mix filename candidate counts by direct child", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_large_files_over_100MB_current_r04.tsv", "YI_R04_LARGE_FILE_REVIEW", "Complete current metadata-only file list above 100 MB in /data/pclancy3/yi", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_git_related_large_files_over_100MB_r04.tsv", "YI_R04_GIT_LARGE_FILE_REVIEW", "Git metadata paths above 100 MB; size/name observations only, no Git object content read", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_wfc_mix_name_candidates_current_r04.tsv", "YI_R04_QE_RESTART_NAME_REVIEW", "Metadata-only WFC/mix basename matches with QE-style .mixN split; not deletion authorization", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_r01_to_r04_added_nonconda_paths.tsv", "YI_R04_ADDED_PATHS", "New paths outside .conda compared with R01; path difference is not cleanup authorization", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_r01_to_r04_missing_nonconda_paths.tsv", "YI_R04_MISSING_PATHS", "Paths absent outside .conda compared with R01; exact cleanup receipt reconciliation", "PERMANENT_SOURCE"),
    ("phase2_data/yi_full_inventory_20261123_r04/yi_r01_to_r04_metadata_changes_nonconda.jsonl", "YI_R04_METADATA_CHANGE_LEDGER", "Metadata changes for common non-Conda paths compared with R01; no deletion inference", "PERMANENT_SOURCE"),
]

# Register every immutable raw observation and execution receipt from these
# scoped refreshes. Globbing is constrained to the audit output directories;
# it never walks the remote storage roots.
_R04_SCRATCH_DIR = AUDIT / "phase2_scratch4/scratch4_refresh_20261123_r04_remaining"
for _path in sorted(_R04_SCRATCH_DIR.iterdir()):
    if _path.is_dir():
        continue
    _name = _path.name
    if _name.endswith(".jsonl.gz"):
        _kind, _role = "R04_HISTORICAL_RAW_METADATA_LEDGER", "Immutable historical metadata-only R04 observation stream"
    elif _name.endswith(".status.json") or _name.endswith("_summary.json") or _name.endswith("_coverage_summary.json"):
        _kind, _role = "R04_HISTORICAL_SCAN_STATUS", "Historical R04 scan status or coverage summary; not current cleanup authorization"
    elif _name.endswith(".stderr.txt"):
        _kind, _role = "R04_HISTORICAL_SCAN_STDERR", "Historical R04 remote metadata scanner stderr capture"
    elif _name.endswith(".jsonl"):
        _kind, _role = "R04_HISTORICAL_STATUS_LEDGER", "Historical append-only R04 scan status ledger"
    else:
        continue
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), _kind, _role, "PERMANENT_SOURCE"
    ))

_YI_SCOPE_DIR = AUDIT / "phase2_data/yi_only_scope_refresh_20261123T062955Z"
for _path in sorted(_YI_SCOPE_DIR.iterdir()):
    if _path.is_dir():
        continue
    _name = _path.name
    _prefix = "queue" if _name.startswith("queue.") else "user_quota"
    if _name.endswith(".status.json"):
        _kind = "YI_SCOPE_QUEUE_STATUS" if _prefix == "queue" else "YI_SCOPE_USER_QUOTA_STATUS"
    elif _name.endswith(".stderr.txt"):
        _kind = "YI_SCOPE_QUEUE_STDERR" if _prefix == "queue" else "YI_SCOPE_USER_QUOTA_STDERR"
    else:
        _kind = "YI_SCOPE_QUEUE_CAPTURE" if _prefix == "queue" else "YI_SCOPE_USER_QUOTA_CAPTURE"
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), _kind,
        "Immutable capture or execution receipt limited to the two yi roots and ycao73 quota rows",
        "PERMANENT_SOURCE",
    ))

_YI_R04_DIR = AUDIT / "phase2_data/yi_full_inventory_20261123_r04"
for _path in sorted(_YI_R04_DIR.iterdir()):
    if _path.is_dir():
        continue
    _name = _path.name
    if _name.endswith(".jsonl.gz"):
        _kind, _role = "YI_R04_RAW_METADATA_LEDGER", "Immutable metadata-only /data/pclancy3/yi observation stream; partial attempts remain explicitly partial"
    elif _name.endswith(".status.json"):
        _kind, _role = "YI_R04_SCAN_STATUS", "Execution and coverage status for the metadata-only /data/pclancy3/yi refresh"
    elif _name.endswith(".stderr.txt"):
        _kind, _role = "YI_R04_SCAN_STDERR", "Remote metadata-only /data/pclancy3/yi scanner stderr capture"
    else:
        continue
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), _kind, _role, "PERMANENT_SOURCE"
    ))

_COLLAB_R04_FILES = (
    "collab_home_path_inventory_20261123.jsonl.gz",
    "collab_home_summary_20261123.json",
    "collab_home_refresh_report_20261123.md",
    "collab_home_stderr_20261123.log",
    "refresh_collab_home_inventory_20261123.py",
    "collab_roots_current_metadata_r04.jsonl.gz",
)
for _name in _COLLAB_R04_FILES:
    DELIVERABLES.append((
        "phase2_data/" + _name, "COLLAB_ROOTS_R04_EVIDENCE",
        "Current metadata-only inventory or audit evidence for exactly three selected /data/pclancy3 roots",
        "PERMANENT_SOURCE",
    ))

_DATA_MOUNT_CENSUS_DIR = AUDIT / "phase2_data/data_mount_accessible_census_20261123"
for _path in sorted(_DATA_MOUNT_CENSUS_DIR.iterdir()):
    if _path.is_dir() or _path.name == "accessible_data_mount_r05_normalized_manifest_20261123.jsonl.gz":
        continue
    _name = _path.name
    if _name.endswith(".jsonl.gz"):
        _kind = "DATA_MOUNT_CENSUS_RAW_LEDGER"
    elif _name.endswith("_summary.json") or _name == "accessible_data_census_summary_20261123.json":
        _kind = "DATA_MOUNT_CENSUS_BATCH_SUMMARY"
    elif _name.endswith(".json"):
        _kind = "DATA_MOUNT_CENSUS_ROOT_ACCESS_CAPTURE"
    elif _name.endswith(".tsv"):
        _kind = "DATA_MOUNT_CENSUS_REVIEW_TABLE"
    elif _name.endswith(".md"):
        _kind = "DATA_MOUNT_CENSUS_REPORT"
    elif _name.endswith(".py"):
        _kind = "DATA_MOUNT_CENSUS_SCRIPT"
    else:
        continue
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), _kind,
        "Read-only metadata census evidence for accessible other-owner /data roots; partial scans are lower bounds and no content hashes were taken",
        "PERMANENT_SOURCE",
    ))

_GIT_LIVE_CANDIDATE_DIR = AUDIT / "phase2_data/git_live_mainline_candidates_20261123"
for _path in sorted(_GIT_LIVE_CANDIDATE_DIR.iterdir()):
    if not _path.is_file() or _path.suffix not in (".json", ".jsonl", ".txt"):
        continue
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), "GIT_LIVE_MAINLINE_CANDIDATE_EVIDENCE",
        "Live-ref, ancestry, or original-worktree evidence supporting isolated local mainline candidates; no push performed",
        "PERMANENT_SOURCE",
    ))

_FULL_GIT_RECON_DIR = AUDIT / "phase2_data/full_git_reconciliation_r04_20261123"
for _path in sorted(_FULL_GIT_RECON_DIR.iterdir()):
    if not _path.is_file():
        continue
    _kind = "FULL_GIT_RECONCILIATION_REPORT" if _path.suffix == ".md" else \
            "FULL_GIT_RECONCILIATION_MATRIX" if _path.suffix == ".tsv" else \
            "FULL_GIT_RECONCILIATION_EVIDENCE"
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), _kind,
        "Per-repository status, branch ancestry, live heads, missing objects, candidate verification, and safe-action review for the 29 targeted Git checkouts",
        "PERMANENT_SOURCE",
    ))

for _path in (
    AUDIT / "phase2_scratch4/SCRATCH4_MOUNT_ROOT_R05_REVIEW_20261123.md",
    AUDIT / "phase2_scratch4/SCRATCH4_PCLANCY3_SCOPE_RECONCILIATION_20261123.md",
    AUDIT / "phase2_scratch4/probe_scratch4_mount_root_20261123_r05.py",
    AUDIT / "phase2_scratch4/inventory_accessible_scratch4_roots_r05.py",
    AUDIT / "phase2_scratch4/orchestrate_accessible_scratch4_root_batches_r05.py",
    AUDIT / "phase2_scratch4/check_scratch4_uid3901_r06.py",
):
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), "SCRATCH4_MOUNT_R05_CENSUS_EVIDENCE",
        "Read-only mount-root coverage, bounded scan method, or explicit group-root reconciliation evidence",
        "PERMANENT_SOURCE",
    ))

for _directory, _prefix in (
    (AUDIT / "phase2_scratch4/scratch4_mount_root_20261123_r05", "SCRATCH4_MOUNT_ROOT_PROBE"),
    (AUDIT / "phase2_scratch4/scratch4_accessible_roots_r05_20261123", "SCRATCH4_ACCESSIBLE_ROOT_R05"),
    (AUDIT / "phase2_scratch4/scratch4_uid3901_outer_roots_r06_20261123", "SCRATCH4_UID3901_ROOT_R06"),
    (AUDIT / "phase2_scratch4/scratch4_accessible_roots_r05_r06_20261123", "SCRATCH4_OUTER_ROOT_SUMMARY_R05_R06"),
    (AUDIT / "phase2_data/full_git_reconciliation_r04_20261123", "FULL_GIT_RECONCILIATION_R04"),
):
    if not _directory.exists():
        continue
    for _path in sorted(_directory.rglob("*")):
        if not _path.is_file():
            continue
        if _path.suffix == ".jsonl":
            _kind = _prefix + "_RAW_METADATA"
        elif _path.name.endswith(".status.json") or _path.name == "receipt.json":
            _kind = _prefix + "_STATUS_RECEIPT"
        elif _path.suffix == ".json":
            _kind = _prefix + "_SUMMARY_OR_SELECTION"
        elif _path.suffix in (".txt", ".stderr.txt"):
            _kind = _prefix + "_OUTPUT_OR_STDERR"
        else:
            continue
        DELIVERABLES.append((
            _path.relative_to(AUDIT).as_posix(), _kind,
            "Bounded metadata-only per-root evidence; timeouts, permission errors, and missing output remain partial",
            "PERMANENT_SOURCE",
        ))

for _path in (
    AUDIT / "phase2_data/yi_git_repository_state_r04_20261123.tsv",
    AUDIT / "phase2_data/yi_git_repository_state_r04_20261123.json",
    AUDIT / "phase2_data/collab_home_git_repository_state_r04_20261123.tsv",
    AUDIT / "phase2_data/collab_home_git_repository_state_r04_20261123.json",
    AUDIT / "phase2_data/yi_git_branch_relationships_r04_20261123.tsv",
    AUDIT / "phase2_data/yi_git_branch_relationships_r04_20261123.json",
    AUDIT / "phase2_data/yi_git_temp_object_candidate_review_r04_20261123.tsv",
    AUDIT / "phase2_data/yi_git_temp_object_candidate_review_r04_20261123.json",
):
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), "R04_SCOPED_GIT_REVIEW",
        "Read-only Git worktree/ref/temporary-object evidence; cached upstream refs may be stale",
        "PERMANENT_SOURCE",
    ))

_MIX_REVIEW_DIR = AUDIT / "phase2_data/yi_mix_repro_review_20261123"
for _path in sorted(_MIX_REVIEW_DIR.iterdir()):
    if _path.is_dir() or _path.suffix in (".raw.txt", ".status.json"):
        continue
    if _path.name.endswith(".tsv"):
        _kind = "QE_MIX_REPRODUCIBILITY_MATRIX"
    elif _path.name.endswith(".psv"):
        _kind = "QE_MIX_SACCT_CAPTURE"
    elif _path.suffix == ".md":
        _kind = "QE_MIX_REPRODUCIBILITY_REVIEW"
    else:
        _kind = "QE_MIX_REPRODUCIBILITY_EVIDENCE"
    DELIVERABLES.append((
        _path.relative_to(AUDIT).as_posix(), _kind,
        "Hashed inputs/scripts/logs/pseudopotentials and scoped exact-path reproducibility review; mix payloads are not hashed",
        "PERMANENT_SOURCE",
    ))

for _capture_dir in sorted((AUDIT / "phase2_data").glob("selected_cleanup_queue_*")):
    if not _capture_dir.is_dir():
        continue
    for _path in sorted(_capture_dir.iterdir()):
        if _path.is_file():
            DELIVERABLES.append((
                _path.relative_to(AUDIT).as_posix(), "SELECTED_ROOTS_QUEUE_CAPTURE",
                "Timestamped all-user queue summary filtered remotely to exact selected cleanup roots",
                "PERMANENT_SOURCE",
            ))

DELIVERABLES.extend([
    ("AUDIT_REPORT.md", "MIGRATION_AUDIT_REPORT", "Current consolidated MigrationBench/Rockfish storage, cleanup, Git, and paper-evidence audit report", "PERMANENT_SOURCE"),
    ("phase2_schema/import_pr10_pr12_pr13_candidate_releases_20261124.py", "PAPER_CANDIDATE_PROVENANCE_IMPORTER", "Additive, idempotent PR #10/#12/#13 review-candidate evidence importer with live archive verification", "PERMANENT_SOURCE"),
    ("phase2_schema/import_revision1_git_large_asset_archive_20261124.py", "MIGRATIONBENCH_LARGE_ASSET_SQL_IMPORTER", "Idempotent importer for the verified 2026-11-24 model and vector archive, control files, and storage paths", "PERMANENT_SOURCE"),
    ("phase2_schema/import_revision1_md_fig6_archive_20261124.py", "MIGRATIONBENCH_MD_ARCHIVE_SQL_IMPORTER", "Idempotent importer for the hash-verified six-run Revision 1 MD evidence archive and paper item links", "PERMANENT_SOURCE"),
    ("phase2_schema/import_zero_byte_mix_cleanup_20261124.py", "QE_ZERO_BYTE_MIX_CLEANUP_IMPORTER", "Read-only preflight and idempotent import of the zero-byte QE mix receipt plus independent post-delete path check", "PERMANENT_SOURCE"),
    ("phase2_schema/import_catalog_snapshot_partial_cleanup_20261124.py", "SQLITE_SNAPSHOT_PARTIAL_CLEANUP_IMPORTER", "Read-only preflight and idempotent import of exact stat/hash, queue, archive, and post-delete evidence for two transfer partials", "PERMANENT_SOURCE"),
    ("phase2_data/capture_catalog_snapshot_partials_20261124.py", "SQLITE_SNAPSHOT_PARTIALS_CAPTURE_SCRIPT", "Read-only exact-path stat, queue, process, and SHA-256 capture for the two stale Rockfish catalog transfers", "PERMANENT_SOURCE"),
    ("phase2_data/archive_catalog_snapshot_20261124.py", "SQLITE_SNAPSHOT_ARCHIVE_SCRIPT", "Read-only preflight and consistent SQLite backup, remote integrity verification, and atomic archive replacement", "PERMANENT_SOURCE"),
    ("phase2_data/remove_catalog_snapshot_partials_20261124.py", "SQLITE_SNAPSHOT_PARTIALS_CLEANUP_SCRIPT", "Read-only preflight and guarded exact-hash removal of two obsolete transfer partials after a verified replacement", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_md_fig6_table_s8_20261124/ARCHIVE_MANIFEST.tsv", "REVISION1_MD_ARCHIVE_MANIFEST", "Exact source and archive paths, sizes, and SHA-256 for the selected Revision 1 MD subset", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_md_fig6_table_s8_20261124/RUN_SUMMARY.tsv", "REVISION1_MD_ARCHIVE_RUN_SUMMARY", "Per-run model/input identity, continuation stop outcome, and final checkpoint step", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_md_fig6_table_s8_20261124/VERIFICATION_REPORT.json", "REVISION1_MD_ARCHIVE_VERIFICATION", "Independent size/SHA-256 verification and scope summary for the 49-file MD bundle", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_md_fig6_table_s8_20261124/README.md", "REVISION1_MD_ARCHIVE_README", "Layout, limitations, and transfer instructions for the selected MD evidence bundle", "PERMANENT_SOURCE"),
    ("phase2_schema/pr_candidate_release_catalog_queries_20261124.sql", "PAPER_CANDIDATE_SQL_QUERY_COOKBOOK", "Read-only SQL queries for PR #10/#12/#13 claims, figures, source manifests, and Rockfish raw archive locations", "PERMANENT_SOURCE"),
    ("phase2_schema/import_xforce_historical_partial_sidecars_20261124.py", "XFORCE_PARTIAL_PROVENANCE_IMPORTER", "Read-only-preflight-first, idempotent import for historical DUAL-X X-FORCE sidecars; statuses remain PROVENANCE_PARTIAL", "PERMANENT_SOURCE"),
    ("phase2_data/PR13_GIT_TREE_RECONCILIATION_20261124.md", "PR13_GIT_TREE_RECONCILIATION_REPORT", "Read-only comparison of all PR #13 manifest paths with both pinned Git trees", "PERMANENT_SOURCE"),
    ("phase2_data/pr13_git_tree_reconciliation_20261124.json", "PR13_GIT_TREE_RECONCILIATION_DATA", "Machine-readable comparison of 1,384 PR #13 manifest rows with both pinned Git trees", "PERMANENT_SOURCE"),
    ("phase2_data/pr_release_candidates_20261124/README.md", "PAPER_CANDIDATE_CAPTURE_README", "Source capture index and review boundaries for the separate PR #10/#12/#13 candidates", "PERMANENT_SOURCE"),
    ("phase2_data/pr_release_candidates_20261124/CAPTURE_MANIFEST.tsv", "PAPER_CANDIDATE_CAPTURE_MANIFEST", "Hashes and pinned GitHub commit URLs for the 19 captured PR source files", "PERMANENT_SOURCE"),
    ("phase2_data/pr13_raw_archive_capture_20261124/ARCHIVE_MANIFEST.tsv", "PR13_ROCKFISH_ARCHIVE_MANIFEST", "File-by-file hashes and paths for 1,336 archived PR #13 raw-evidence files and the source tar", "PERMANENT_SOURCE"),
    ("phase2_data/pr13_raw_archive_capture_20261124/VERIFICATION_REPORT.json", "PR13_ROCKFISH_ARCHIVE_VERIFICATION", "Independent source-manifest and live Rockfish verification result for the PR #13 raw archive", "PERMANENT_SOURCE"),
    ("phase2_data/pr13_raw_archive_capture_20261124/README.md", "PR13_ROCKFISH_ARCHIVE_README", "Archive layout and integrity instructions for PR #13 raw evidence", "PERMANENT_SOURCE"),
    ("phase2_data/MANUSCRIPT_REVISION1_ALIGNMENT_AUDIT_20261123.md", "MANUSCRIPT_EVIDENCE_ALIGNMENT_AUDIT", "PDF, LaTeX, submitted Git release, figure and scientific-evidence alignment audit with limitations", "PERMANENT_SOURCE"),
    ("phase2_data/MIGRATIONBENCH_ROCKFISH_ARCHIVE_INDEX_20261124.md", "ROCKFISH_ARCHIVE_INDEX", "Verified MigrationBench and DUAL-X archive layout, source data boundaries, and transfer entry points", "PERMANENT_SOURCE"),
    ("phase2_data/MANUSCRIPT_RELEASE_REVIEW_20261123.md", "MANUSCRIPT_RELEASE_REVIEW", "Revision 1 release and reproducibility boundary review", "PERMANENT_SOURCE"),
    ("phase2_data/HISTORY_NORMALIZATION_NOTE_20261124.md", "GIT_HISTORY_NORMALIZATION_PLAN", "Evidence and procedure for canonical PR #20 tree reconstruction while preserving live branch-tip parents", "PERMANENT_SOURCE"),
    ("phase2_data/archive_additional_migrationbench_models_20261124.py", "ROCKFISH_MODEL_ARCHIVE_SCRIPT", "Idempotent verifier and copier for verified MigrationBench checkpoint assets", "PERMANENT_SOURCE"),
    ("phase2_data/externalize_migrationbench_model_assets_20261124.py", "MODEL_ARCHIVE_REFERENCE_UPDATE_SCRIPT", "Updates model loaders and figure/model manifests to resolve the Rockfish archive paths", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_git_large_assets_20261124/ARCHIVE_MANIFEST.tsv", "MIGRATIONBENCH_MODEL_FIGURE_ARCHIVE_MANIFEST", "Full-file SHA-256 and Git blob identities for archived checkpoints and CANVA-P03 SVG", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_git_large_assets_20261124/VERIFICATION_REPORT.json", "MIGRATIONBENCH_MODEL_FIGURE_ARCHIVE_VERIFICATION", "Independent Rockfish archive verification report for five model and figure payloads", "PERMANENT_SOURCE"),
    ("phase2_data/revision1_git_large_assets_20261124/README.md", "MIGRATIONBENCH_MODEL_FIGURE_ARCHIVE_README", "Archive layout, asset counts, and restoration pointer", "PERMANENT_SOURCE"),
    ("../task2_cross_element/models/README.md", "MIGRATIONBENCH_MODEL_ARCHIVE_POINTER", "Model checkpoint archive locations, SHA-256 hashes, and restore environment variables", "PERMANENT_SOURCE"),
    ("../task2_cross_element/manifests/model_manifest.json", "MIGRATIONBENCH_MODEL_MANIFEST", "Model identities, source paths, hashes, Git blob IDs, and central archive locations", "PERMANENT_SOURCE"),
    ("../task2_cross_element/scripts/model_smoke.py", "MODEL_ARCHIVE_AWARE_LOADER", "Diagnostic smoke loader resolving checkpoint files from MIGRATIONBENCH_MODEL_DIR or archive environment variable", "PERMANENT_SOURCE"),
    ("../task2_cross_element/scripts/run_provisional_mace_profiles.py", "MODEL_ARCHIVE_AWARE_PROFILE_SCRIPT", "Provisional profile runner using the centralized model archive and preserving DFT failure gates", "PERMANENT_SOURCE"),
    ("../task2_cross_element/scripts/package_gate_outputs.py", "MODEL_ARCHIVE_AWARE_PACKAGER", "Health-gate manifest generator using the centralized model archive paths", "PERMANENT_SOURCE"),
    ("../canva_palette_vector_bundle_20261105/README.md", "CANVA_P03_ARCHIVE_README", "Clarifies that the oversized editable P03 SVG is in the Rockfish archive while PDF and PNG remain in Git", "PERMANENT_SOURCE"),
    ("../canva_palette_vector_bundle_20261105/CANVA_PALETTE_VECTOR_MANIFEST.json", "CANVA_VECTOR_ARCHIVE_MANIFEST", "Canvas figure outputs with P03 SVG archive-only path, Git blob ID, and source commit", "PERMANENT_SOURCE"),
])

_LARGE_DATA_REFRESH_DIR = AUDIT / "phase2_data/large_file_disposition_refresh_20261124"
for _path in sorted(_LARGE_DATA_REFRESH_DIR.iterdir()):
    if not _path.is_file():
        continue
    if _path.suffix == ".tsv":
        _kind = "DATA_LARGE_FILE_DISPOSITION_TABLE"
    elif _path.suffix == ".json":
        _kind = "DATA_LARGE_FILE_DISPOSITION_CAPTURE"
    else:
        _kind = "DATA_LARGE_FILE_DISPOSITION_REPORT"
    DELIVERABLES.append((_path.relative_to(AUDIT).as_posix(), _kind,
                         "2026-11-24 exact-path read-only lstat and queue review; candidate set remains HOLD where cleanup evidence is incomplete",
                         "PERMANENT_SOURCE"))

_SCRATCH4_LARGE_REFRESH_DIR = AUDIT / "phase2_scratch4/large_file_disposition_refresh_20261124"
for _path in sorted(_SCRATCH4_LARGE_REFRESH_DIR.iterdir()):
    if not _path.is_file():
        continue
    if _path.suffix == ".tsv":
        _kind = "SCRATCH4_LARGE_FILE_DISPOSITION_TABLE"
    elif _path.suffix == ".json":
        _kind = "SCRATCH4_LARGE_FILE_DISPOSITION_CAPTURE"
    elif _path.name.startswith("SHA256SUMS"):
        _kind = "SCRATCH4_LARGE_FILE_DISPOSITION_CHECKSUMS"
    else:
        _kind = "SCRATCH4_LARGE_FILE_DISPOSITION_EVIDENCE"
    DELIVERABLES.append((_path.relative_to(AUDIT).as_posix(), _kind,
                         "2026-11-24 exact stat refresh of 19 historical >500 MiB paths and filtered queue; mount scope remains partial",
                         "PERMANENT_SOURCE"))

_CATALOG_SNAPSHOT_RECOVERY_DIR = AUDIT / "phase2_data/catalog_snapshot_recovery_20261124"
if _CATALOG_SNAPSHOT_RECOVERY_DIR.exists():
    for _path in sorted(_CATALOG_SNAPSHOT_RECOVERY_DIR.iterdir()):
        if not _path.is_file():
            continue
        _kind = "SQLITE_SNAPSHOT_RECOVERY_SCRIPT" if _path.suffix == ".py" else "SQLITE_SNAPSHOT_RECOVERY_EVIDENCE"
        DELIVERABLES.append((_path.relative_to(AUDIT).as_posix(), _kind,
                             "Exact verification and guarded replacement of incomplete Rockfish catalog snapshot transfers",
                             "PERMANENT_SOURCE"))

_GIT_MAINLINE_REFRESH_DIR = AUDIT / "phase2_data/git_mainline_refresh_20261124"
for _path in sorted(_GIT_MAINLINE_REFRESH_DIR.iterdir()):
    if not _path.is_file():
        continue
    if _path.suffix == ".py":
        _kind = "GIT_MAINLINE_REFRESH_CAPTURE_SCRIPT"
    elif _path.suffix == ".tsv":
        _kind = "GIT_MAINLINE_REFRESH_REVIEW_TABLE"
    elif _path.suffix == ".json":
        _kind = "GIT_MAINLINE_REFRESH_CAPTURE"
    else:
        _kind = "GIT_MAINLINE_REFRESH_REPORT"
    DELIVERABLES.append((_path.relative_to(AUDIT).as_posix(), _kind,
                         "2026-11-24 read-only live-ref and isolated-candidate refresh; no original worktree/ref update or push",
                         "PERMANENT_SOURCE"))

_WFC_RESIDUAL_DIR = AUDIT / "phase2_data/wfc_mix_residual_20261124"
for _path in sorted(_WFC_RESIDUAL_DIR.iterdir()):
    if not _path.is_file():
        continue
    if _path.suffix == ".jsonl":
        _kind, _role = "QE_RESTART_RESIDUAL_METADATA_INVENTORY", "Full metadata-only WFC/mix name-match inventory; no payload bytes read"
    elif _path.name == "delete_empty_mix_receipt.json":
        _kind, _role = "QE_RESTART_EMPTY_PLACEHOLDER_DELETE_RECEIPT", "Exact guarded deletion receipt for eight zero-byte QE mixing placeholders"
    elif _path.suffix == ".md":
        _kind, _role = "QE_RESTART_RESIDUAL_REVIEW", "Accessible-root exact QE restart results, retained false-positive name matches, and mount limits"
    else:
        _kind, _role = "QE_RESTART_RESIDUAL_REVIEW_EVIDENCE", "Exact-name candidate, summary, or post-delete verification evidence"
    DELIVERABLES.append((_path.relative_to(AUDIT).as_posix(), _kind, _role, "PERMANENT_SOURCE"))

_PR_CANDIDATE_DIR = AUDIT / "phase2_data/pr_release_candidates_20261124"
for _path in sorted(_PR_CANDIDATE_DIR.iterdir()):
    if not _path.is_file() or _path.name in {"CAPTURE_MANIFEST.tsv", "README.md", "IMPORT_RECEIPT.json", "DB_SNAPSHOT_MANIFEST.json"}:
        continue
    _kind = "PAPER_CANDIDATE_SOURCE_CAPTURE" if _path.suffix in {".json", ".tsv", ".md"} else "PAPER_CANDIDATE_SOURCE_CAPTURE"
    DELIVERABLES.append((_path.relative_to(AUDIT).as_posix(), _kind,
                         "Immutable source capture for a separate open PR review candidate; not the submitted release",
                         "PERMANENT_SOURCE"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=180)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT,)).fetchone()
        if not root or Path(root[0]).resolve() != REPO.resolve():
            raise RuntimeError("LOCAL-MAIN must be the unified worktree")
        source_ids = []
        for relative, kind, role, retention in DELIVERABLES:
            path = REPO / "analysis/rockfish_skipjack_migration_audit_20261121" / relative
            path = path.resolve(strict=True)
            relative_to_repo = path.relative_to(REPO).as_posix()
            sha = digest(path)
            size = path.stat().st_size
            suffix = sha[:16]
            source_key = f"audit_deliverable:{relative_to_repo}:{suffix}"
            artifact_key = f"AUDIT-DELIVERABLE:{relative_to_repo}:{suffix}"
            prior_source = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
            if prior_source and prior_source != (str(path), sha):
                raise RuntimeError(f"Source key changed identity: {source_key}")
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                    immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?, ?,?,?,1,?,?)""",
                (source_key, str(path), kind, sha, now, now,
                 "Captured local audit deliverable; hash identifies this exact revision.", role),
            )
            source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]
            source_ids.append(source_id)
            prior_artifact = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if prior_artifact and prior_artifact[1:] != (sha, size):
                raise RuntimeError(f"Artifact key changed identity: {artifact_key}")
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,
                    byte_size,scientific_status,retention_class,notes)
                   VALUES('ROCKFISH_STORAGE_AUDIT',?,?,?,?,?,?,'UNKNOWN',?,?)""",
                (artifact_key, kind, role, "USER_CONFIRMED", sha, size, retention,
                 "Local audit artifact; registry_source preserves the exact hash and capture time."),
            )
            artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()[0]
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
                        (LOCAL_ROOT, relative_to_repo, now))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
                                  (LOCAL_ROOT, relative_to_repo)).fetchone()[0]
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""",
                (artifact_id, path_id, now, now),
            )
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'deliverable_file',?,'ARTIFACT',?,'DOCUMENTS',?)""",
                (source_id, relative_to_repo, artifact_key, f"{kind}; exact revision sha256={sha}"),
            )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed while registering deliverables")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    with sqlite3.connect(DB) as check:
        print("registered_deliverables=" + str(len(DELIVERABLES)))
        print("deliverable_sources=" + str(len(source_ids)))
        print("foreign_key_violations=" + str(len(check.execute("PRAGMA foreign_key_check").fetchall())))


if __name__ == "__main__":
    main()
