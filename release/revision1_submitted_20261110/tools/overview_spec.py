"""Static descriptions and classification rules used by build_overview.py."""
from __future__ import annotations

import re

# Master files of the trace package: file -> (SQL table, one row is, purpose)
MASTER_FILES = [
    ("RESULTS_INDEX.tsv", "rev1_results_index", "one figure panel or table",
     "Maintenance list: final asset + sha256, generator, plotted data, upstream data, raw location, models, "
     "calculation type, status"),
    ("items/<section>/<item>/PROVENANCE.md", "rev1_items", "one figure or table",
     "Per-item page generated from RESULTS_INDEX; item-specific sources sit in the same folder"),
    ("PIPELINE.md", "rev1_pipeline_stages", "one pipeline stage",
     "DFT/AIMD -> split -> training -> evaluation -> features -> SHAP -> figures"),
    ("MODEL_REGISTRY.tsv", "rev1_model_registry", "one checkpoint file",
     "Every trained and foundation checkpoint (used and unused): sha256, config, training data, location"),
    ("STORAGE_LOCATIONS.tsv", "rev1_storage_locations", "one raw-data location",
     "Every location outside git (Rockfish, archive store, iCloud) with its migration target"),
    ("ROCKFISH_MIGRATION_PLAN.tsv", "rev1_rockfish_migration_plan", "one Rockfish directory",
     "Priority 1-3 plan for moving Rockfish data to Skipjack (not executed)"),
    ("KNOWN_ISSUES.md", "rev1_known_issues", "one issue",
     "Inconsistencies in the submitted version, to fix at the next revision"),
    ("imports/IMPORT_MANIFEST.tsv", "rev1_imported_files", "one imported file",
     "Files copied into the repo from outside, with source and sha256"),
    ("imports/EXTERNAL_STORE_MANIFEST.tsv", "rev1_external_store", "one heavy binary",
     "Imported heavy binaries kept in the archive store instead of git"),
    ("imports/ARCHIVED_UNTRACKED_RAW.tsv", "rev1_archived_untracked_raw", "one raw file",
     "Untracked raw NEB/sliding/audit outputs moved to the archive store"),
    ("imports/DOWNLOADS_ARCHIVE.tsv", "rev1_downloads_archive", "one file from ~/Downloads",
     "MigrationBench files moved from ~/Downloads to the archive store"),
    ("imports/ICLOUD_DATALESS.tsv", "rev1_icloud_dataless", "one iCloud-only file",
     "Workspace files still evicted to iCloud at import time (re-import later)"),
    ("REPRODUCIBILITY.tsv", "rev1_reproducibility", "one figure or table",
     "Reproduction script, evidence files and PASS/FAIL per item (tools/verify_reproducibility.py)"),
    ("tools/path_roots.py", "rev1_path_roots", "one path label",
     "Resolves portable paths such as archive:..., chatgpt:..., rockfish:..."),
]

OTHER_TABLES = {
    "rev1_import_plan": "outside source -> repo destination, with the reason (drives the importer)",
    "rev1_deferred_large_files": "large files left out of git at import time, with intended destination",
    "rev1_import_conflicts": "files whose repo copy differs from the workspace copy (repo kept)",
    "rev1_item_calc_layers": "curated L0-L3 computed-data layers per figure/table",
    "rev1_reproduce_exceptions": "items that cannot have a script, with reason and resolution",
    "rev1_known_discrepancies": "exact failing checks accepted because the submitted version is wrong",
    "v_rev1_item_status": "per item: statuses and number of rows not FULL",
    "v_rev1_models_used": "checkpoints used in the paper",
    "v_rev1_open_issues": "unresolved known issues",
    "v_rev1_rockfish_to_migrate": "Rockfish directories in migration priority order",
    "v_chart_catalog": "pre-submission chart catalogue", "v_model_task_coverage": "model x task coverage",
    "v_model_task_summary": "model x task summary",
}
LEGACY_TABLES = {
    "models": "model identities (2026-10-25 registry)", "model_families": "model taxonomy",
    "model_groups": "model groupings", "model_group_members": "model -> group",
    "model_aliases": "alternative model names", "model_name_map": "paper name <-> registry id",
    "tasks": "evaluation tasks/operators", "task_groups": "task groupings", "task_group_members": "task -> group",
    "task_aliases": "alternative task names", "task_group_aliases": "alternative task-group names",
    "systems": "material systems (Cr61, Cr81, ...)", "campaigns": "compute campaigns",
    "datasets": "datasets", "coverage": "model x task x system x campaign fact table",
    "coverage_resolution": "resolved coverage gaps", "assets": "data packages and locators",
    "sources": "where registry facts came from", "gaps": "open provenance gaps",
    "charts": "logical figures (pre-submission)", "chart_panels": "figure panels",
    "chart_links": "figure -> model/task/system/dataset", "chart_data_files": "figure data files",
    "chart_provenance": "figure manifests and hashes", "chart_sources": "figure sources (Canva migration 001)",
    "canva_designs": "Canva designs", "canva_pages": "Canva pages", "rockfish_job_status": "Rockfish job states",
    "metadata": "registry metadata", "audit_log": "registry checks and migrations",
}

# Layers of computed data, as columns of sources/item_calc_layers.tsv (curated 2026-11-10).
LAYERS = [("L0", "L0_reference", "reference calculations (QE DFT, AIMD)"),
          ("L1", "L1_split_models", "data split and trained / foundation models"),
          ("L2", "L2_mlip_sims", "MLIP simulations (single points, MD, NEB, sliding)"),
          ("L3", "L3_derived", "derived analysis (metrics, features, projections, SHAP)"),
          ("L4", "plotted_data", "plotted / tabulated data"),
          ("L5", "L5_kind", "generator and final asset")]
TIERS = [("git", re.compile(r"(?<![\w:])(analysis|data|paper|release|reproducibility|docs|src)/"
                            r"|\b(sn-article[\w-]*|response_to_reviewers)\.tex|\bin[- ]repo\b", re.I)),
         ("archive store", re.compile(r"\barchive:")),
         ("Rockfish", re.compile(r"\brockfish\b", re.I)),
         ("iCloud workspace", re.compile(r"\bchatgpt:")),
         ("Canva", re.compile(r"Canva", re.I))]
SAME_AS = re.compile(r"\b(?:see|as|same as)\s+((?:Table\s*|Fig\.?\s*)?[RS]?\d+)", re.I)


def tiers(text: str) -> str:
    found = ", ".join(name for name, pattern in TIERS if pattern.search(text))
    same = SAME_AS.search(text)
    return found or (f"as {same.group(1)}" if same else "—")
