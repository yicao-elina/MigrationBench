#!/usr/bin/env python3
"""Register the exact DUAL-X Safe Core v6 evidence and reproducibility links."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
REPO = Path("/Users/alina/Project/DUAL-X-safe-core").resolve()
BUNDLE_REL = Path("safe_core/evidence/20261023_safe_core/full_v3_scalar_controls")
BUNDLE = REPO / BUNDLE_REL
ROOT_ID = "LOCAL-DUALX-SAFECORE"
CAMPAIGN_ID = "DUALX-SAFECORE-M1-M6-V6"
RELEASE_ID = "DUALX-SAFECORE-V6-REVIEW"
RUNS = {
    "inference": "DUALX-SAFECORE-V6-INFERENCE",
    "analysis": "DUALX-SAFECORE-V6-ANALYSIS",
    "figures": "DUALX-SAFECORE-V6-FIGURE-RENDER",
    "validation": "DUALX-SAFECORE-V6-VALIDATION",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def capture_git() -> tuple[str, str, str, str, bool, str]:
    snapshot = (AUDIT / "phase2_data/dualx_safe_core_git_snapshot_20261123.txt").read_text()
    values = {}
    for line in snapshot.splitlines():
        if line.startswith("[branch]") or line.startswith("[head]") or line.startswith("[upstream]") or line.startswith("[safe_core_tag]"):
            key = line.split("]", 1)[0][1:]
            values[key] = snapshot.split(line, 1)[1].lstrip().splitlines()[0]
    status_block = snapshot.split("[status_porcelain_v2]", 1)[1].split("[remote]", 1)[0]
    dirty = any(line.startswith(("1 ", "2 ", "u ", "? ")) for line in status_block.splitlines())
    remote_block = snapshot.split("[remote]", 1)[1].split("[safe_core_tag]", 1)[0]
    remote = next((line.split()[1] for line in remote_block.splitlines() if line.startswith("origin\t")), "")
    captured = next(line.split("=", 1)[1] for line in snapshot.splitlines() if line.startswith("captured_utc="))
    return values["branch"], values["head"], values["upstream"], values["safe_core_tag"], dirty, remote + "\n" + captured


def main() -> None:
    if not REPO.is_dir() or not BUNDLE.is_dir():
        raise RuntimeError("Expected DUAL-X Safe Core repository and evidence bundle are unavailable")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    branch, head, upstream, tag_commit, dirty, remote_and_time = capture_git()
    remote, observed_utc = remote_and_time.split("\n", 1)

    release_path = BUNDLE / "RELEASE_MANIFEST.json"
    release = json.loads(release_path.read_text())
    if release["release_commit"] != tag_commit:
        raise RuntimeError("Local tag target does not match RELEASE_MANIFEST.json")
    if release["repo"] != remote:
        raise RuntimeError("Local origin does not match RELEASE_MANIFEST.json")
    figure_rows = read_tsv(BUNDLE / "FIGURE_TABLE_REGISTRY.tsv")
    claim_rows = read_tsv(BUNDLE / "CLAIM_EVIDENCE_MATRIX.tsv")
    input_rows = read_tsv(BUNDLE / "INPUT_MANIFEST.tsv")
    hash_rows = read_tsv(BUNDLE / "evidence_hash_manifest.tsv")
    analysis_manifest = json.loads((BUNDLE / "analysis_manifest.json").read_text())
    run_manifest = json.loads((BUNDLE / "run_manifest.json").read_text())
    validation = json.loads((BUNDLE / "clean_checkout_validation.json").read_text())

    # Verify every file listed in the release hash manifest before importing it.
    files: dict[str, tuple[Path, str, str]] = {}
    for row in hash_rows:
        rel = row["path"]
        path = BUNDLE / rel
        expected_sha = row["sha256"]
        expected_size = int(row["bytes"])
        if not path.is_file() or path.stat().st_size != expected_size or sha256(path) != expected_sha:
            raise RuntimeError(f"Evidence hash manifest mismatch: {rel}")
        files[(BUNDLE_REL / rel).as_posix()] = (path, expected_sha, "EVIDENCE")

    support_paths = {
        "RELEASE_MANIFEST.json": "DUALX_RELEASE_MANIFEST",
        "analysis_manifest.json": "DUALX_ANALYSIS_MANIFEST",
        "run_manifest.json": "DUALX_RUN_MANIFEST",
        "INPUT_MANIFEST.tsv": "DUALX_INPUT_MANIFEST",
        "MODEL_REGISTRY.tsv": "DUALX_MODEL_REGISTRY",
        "FIGURE_TABLE_REGISTRY.tsv": "DUALX_FIGURE_TABLE_REGISTRY",
        "CLAIM_EVIDENCE_MATRIX.tsv": "DUALX_CLAIM_EVIDENCE_MATRIX",
        "evidence_hash_manifest.tsv": "DUALX_EVIDENCE_HASH_MANIFEST",
        "clean_checkout_validation.json": "DUALX_CLEAN_CHECKOUT_VALIDATION",
        "FAILURE_LEDGER.tsv": "DUALX_FAILURE_LEDGER",
        "SCOPE_DECISION.md": "DUALX_SCOPE_DECISION",
        "HIDDEN_SCHEMA.tsv": "DUALX_HIDDEN_SCHEMA",
        "README.md": "DUALX_BUNDLE_README",
        "analysis_config_snapshot.json": "DUALX_ANALYSIS_CONFIG",
        "config_snapshot.json": "DUALX_CONFIG_SNAPSHOT",
        "manifest.yaml": "DUALX_BUNDLE_MANIFEST",
    }
    for name in support_paths:
        path = BUNDLE / name
        files[(BUNDLE_REL / name).as_posix()] = (path, sha256(path), "SUPPORT")

    lock_rel = "safe_core/requirements.lock"
    lock_path = REPO / lock_rel
    files[lock_rel] = (lock_path, sha256(lock_path), "ENVIRONMENT")

    script_hashes = release["source_script_hashes"]
    for rel, expected_sha in script_hashes.items():
        path = REPO / rel
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"Release script hash mismatch: {rel}")
        files[rel] = (path, expected_sha, "SCRIPT")

    # The source-data paths in INPUT_MANIFEST use a portable <repo-root> token.
    source_data: dict[tuple[str, str], str] = {}
    for row in input_rows:
        source_path = row["source_path"]
        expected_sha = row["source_sha256"]
        rel = source_path.replace("<repo-root>/", "")
        path = REPO / rel
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"Input manifest source hash mismatch: {source_path}")
        files[rel] = (path, expected_sha, "SOURCE_DATA")
        source_data[(source_path, expected_sha)] = rel

    model_rows = read_tsv(BUNDLE / "MODEL_REGISTRY.tsv")

    con = sqlite3.connect(DB, timeout=180)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            """INSERT OR IGNORE INTO storage_root
               (root_id,cluster_name,filesystem_name,mount_path,scoped_path,
                owner_user,scope_kind,project_id,status,notes)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (ROOT_ID, "LOCAL-MAC", "local", str(REPO), str(REPO), "alina",
             "PROJECT_SUBTREE", "DUALX", "READ_ONLY",
             "Local Safe Core evidence checkout; imported read-only; this is not a Rockfish storage root."),
        )
        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
        if not root or Path(root[0]).resolve() != REPO:
            raise RuntimeError("DUAL-X local storage root is already bound to another path")

        source_ids: dict[str, int] = {}

        def register_source(name: str, path: Path, kind: str, authority: str) -> int:
            digest = sha256(path)
            try:
                rel = path.relative_to(REPO).as_posix()
            except ValueError:
                rel = "AUDIT/" + path.relative_to(AUDIT).as_posix()
            key = f"dualx_safe_core:{rel}:{digest[:16]}"
            prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
            if prior and prior != (str(path), digest):
                raise RuntimeError(f"Source key collision: {key}")
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                    immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (key, str(path), kind, digest, observed_utc, observed_utc, authority,
                 f"DUAL-X Safe Core v6 source; repository branch={branch}; observed HEAD={head}."),
            )
            source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0]
            source_ids[name] = source_id
            return source_id

        source_ids["release"] = register_source(
            "release", release_path, "DUALX_RELEASE_MANIFEST",
            "Source release manifest records tag dualx-safe-core-m1-m6-v6 and release commit; source status remains READY_FOR_INDEPENDENT_REVIEW.",
        )
        source_ids["figure_registry"] = register_source(
            "figure_registry", BUNDLE / "FIGURE_TABLE_REGISTRY.tsv", "DUALX_FIGURE_TABLE_REGISTRY",
            "Exact figure/table outputs, data paths, scripts, configurations, commands, and hashes from the source registry.",
        )
        source_ids["claims"] = register_source(
            "claims", BUNDLE / "CLAIM_EVIDENCE_MATRIX.tsv", "DUALX_CLAIM_EVIDENCE_MATRIX",
            "Source claim statuses and exact evidence/panel references; catalog status is not promoted from source labels.",
        )
        source_ids["evidence_hashes"] = register_source(
            "evidence_hashes", BUNDLE / "evidence_hash_manifest.tsv", "DUALX_EVIDENCE_HASH_MANIFEST",
            "119-file SHA-256 and byte-size manifest; all listed files verified before import.",
        )
        source_ids["validation"] = register_source(
            "validation", BUNDLE / "clean_checkout_validation.json", "DUALX_CLEAN_CHECKOUT_VALIDATION",
            "Recorded clean-checkout commands and outcomes; no command was run by this importer.",
        )
        source_ids["input_manifest"] = register_source(
            "input_manifest", BUNDLE / "INPUT_MANIFEST.tsv", "DUALX_INPUT_MANIFEST",
            "Frame-level source paths and hashes; source data files were verified against exact hashes.",
        )
        source_ids["git_snapshot"] = register_source(
            "git_snapshot", AUDIT / "phase2_data/dualx_safe_core_git_snapshot_20261123.txt",
            "LOCAL_GIT_WORKTREE_SNAPSHOT", "Local-only Git snapshot; no refs or project files were changed.",
        )

        def classify(rel: str, category: str) -> tuple[str, str, str]:
            logical_rel = rel
            bundle_prefix = BUNDLE_REL.as_posix() + "/"
            if rel.startswith(bundle_prefix):
                logical_rel = rel[len(bundle_prefix):]
            if category == "SCRIPT":
                return "SOURCE_CODE", "REBUILD_SCRIPT", "PERMANENT_SOURCE"
            if category == "ENVIRONMENT":
                return "ENVIRONMENT_LOCK", "REPRODUCIBLE_INPUT", "PERMANENT_SOURCE"
            if category == "SOURCE_DATA":
                return "SOURCE_DATASET", "REPRODUCIBLE_INPUT", "REPRODUCIBLE_INPUT"
            if logical_rel.endswith("FIGURE_TABLE_REGISTRY.tsv") or logical_rel.endswith("CLAIM_EVIDENCE_MATRIX.tsv"):
                return "PROVENANCE_MANIFEST", "RELEASE_PROVENANCE", "PERMANENT_SOURCE"
            if logical_rel.startswith("figures/"):
                return "FIGURE_EXPORT", "PUBLICATION_FIGURE", "PUBLICATION_EVIDENCE"
            if logical_rel.startswith("tables/") or logical_rel.startswith("raw/"):
                return "DERIVED_DATA", "ANALYSIS_RESULT", "COMPACT_RESULT"
            if logical_rel.startswith("processed/"):
                return "MODEL_INTERMEDIATE", "ANALYSIS_INPUT", "REPRODUCIBLE_INTERMEDIATE"
            if logical_rel.endswith((".json", ".yaml", ".yml", ".tsv", ".md")):
                return "PROVENANCE_FILE", "REPRODUCIBILITY_EVIDENCE", "PERMANENT_SOURCE"
            return "FILE", "DUALX_EVIDENCE", "PERMANENT_SOURCE"

        artifact_ids: dict[str, int] = {}
        for rel, (path, expected_sha, category) in sorted(files.items()):
            if not path.is_file():
                raise RuntimeError(f"Expected evidence file is missing: {rel}")
            digest = sha256(path)
            size = path.stat().st_size
            if digest != expected_sha:
                raise RuntimeError(f"Evidence identity changed during import: {rel}")
            kind, semantic, retention = classify(rel, category)
            role_status = "UNKNOWN" if category == "SOURCE_DATA" else "PROVENANCE_PARTIAL"
            key = f"DUALX-LOCAL:{rel}:{digest[:16]}"
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,
                    sha256,byte_size,scientific_status,retention_class,notes)
                   VALUES('DUALX',?,?,?,'USER_CONFIRMED',?,?,?,?,?)""",
                (key, kind, semantic, digest, size, role_status, retention,
                 f"Exact DUAL-X Safe Core v6 file; source category={category}; path is under the local project checkout."),
            )
            artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()[0]
            con.execute(
                "UPDATE artifact SET artifact_kind=?,semantic_role=?,retention_class=? WHERE artifact_id=?",
                (kind, semantic, retention, artifact_id),
            )
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
                        (ROOT_ID, rel, observed_utc))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, rel)).fetchone()[0]
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""", (artifact_id, path_id, observed_utc, now)
            )
            artifact_ids[rel] = artifact_id
            if rel.startswith(BUNDLE_REL.as_posix() + "/"):
                con.execute(
                    """INSERT OR IGNORE INTO registry_entity_ref
                       (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                       VALUES(?,'manifest_path',?,'ARTIFACT',?,'HASHES',?)""",
                    (source_ids["evidence_hashes"], rel, key, f"sha256={digest}; bytes={size}"),
                )

        # External checkpoint metadata is retained as unresolved input records.
        external_model_ids: dict[str, int] = {}
        for row in model_rows:
            key = f"DUALX-EXTERNAL-MODEL:{row['model_id']}:{row['sha256'][:16]}"
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,
                    sha256,byte_size,scientific_status,retention_class,notes)
                   VALUES('DUALX',?,'EXTERNAL_POINTER','MODEL_CHECKPOINT','UNKNOWN',?,?,?,
                          'REPRODUCIBLE_INPUT',?)""",
                (key, row["sha256"], int(row["bytes"]), row["lineage_status"],
                 f"External model not located by this audit; manifest placeholder={row['path']}; training_seed={row['training_seed']}; source_checkpoint={row['source_checkpoint']}"),
            )
            external_model_ids[row["model_id"]] = con.execute(
                "SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)
            ).fetchone()[0]

        evidence_rel = BUNDLE_REL.as_posix()
        project_description = (
            "DUAL-X project. This catalog includes the local Safe Core M1-M6 v6 evidence package and "
            "its exact figure/data/script lineage. External model checkpoints remain unresolved pointers; "
            "Rockfish X-FORCE paths are a separate storage scope and are not conflated with this local release."
        )
        con.execute(
            "UPDATE project SET description=?, canonical_source_id=COALESCE(canonical_source_id, ?) WHERE project_id='DUALX'",
            (project_description, source_ids["release"]),
        )
        con.execute(
            """INSERT OR IGNORE INTO campaign
               (campaign_id,project_id,campaign_name,source_campaign_key,status,notes)
               VALUES(?,?,?,?,?,?)""",
            (CAMPAIGN_ID, "DUALX", "Safe Core M1-M6 v6 evidence rebuild",
             release["release_tag"], "COMPLETE",
             "Source release manifest says READY_FOR_INDEPENDENT_REVIEW; clean checkout rebuilt analysis and 21 figure files. Model checkpoints are external; retain claim-specific scientific limits."),
        )

        repo_key = "DUALX-SAFE-CORE"
        con.execute(
            """INSERT OR IGNORE INTO git_repository
               (repository_key,local_path,remote_url,project_id,ownership_status,canonical_branch,notes)
               VALUES(?,?,?,?,?,?,?)""",
            (repo_key, str(REPO), remote, "DUALX", "USER", None,
             f"Observed local branch {branch} at {head}, upstream {upstream}; four untracked .DS_Store paths; canonical default branch not checked."),
        )
        repository_id = con.execute("SELECT repository_id FROM git_repository WHERE repository_key=?", (repo_key,)).fetchone()[0]
        for ref_name, ref_kind, commit, upstream_ref, dirty_flag, note in [
            (f"refs/heads/{branch}", "BRANCH", head, upstream, int(dirty), "Local branch HEAD observed from git snapshot."),
            ("refs/tags/dualx-safe-core-m1-m6-v6", "TAG", tag_commit, None, None,
             "Annotated tag target peeled to release commit per git rev-parse <tag>^{} and RELEASE_MANIFEST.json."),
        ]:
            if not con.execute(
                "SELECT 1 FROM git_ref_observation WHERE repository_id=? AND observed_utc=? AND ref_name=?",
                (repository_id, observed_utc, ref_name),
            ).fetchone():
                con.execute(
                    """INSERT INTO git_ref_observation
                       (repository_id,observed_utc,ref_name,ref_kind,commit_sha,upstream_ref,
                        dirty_worktree,notes)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (repository_id, observed_utc, ref_name, ref_kind, commit, upstream_ref, dirty_flag, note),
                )

        con.execute(
            """INSERT OR IGNORE INTO paper_release
               (release_id,project_id,release_label,manuscript_commit,repository_id,
                released_utc,status,manifest_artifact_id,notes)
               VALUES(?,?,?,?,?,?,'REVIEW',?,?)""",
            (RELEASE_ID, "DUALX", release["release_tag"], release["release_commit"], repository_id,
             observed_utc, artifact_ids[(BUNDLE_REL / "RELEASE_MANIFEST.json").as_posix()],
             "Source release manifest says READY_FOR_INDEPENDENT_REVIEW; no manuscript or Overleaf release is asserted."),
        )

        def ensure_run(run_id: str, label: str, workflow: str, method: str,
                       status: str, scientific: str, notes: str) -> None:
            con.execute(
                """INSERT OR IGNORE INTO workflow_run
                   (run_id,project_id,campaign_id,run_label,workflow_type,method,
                    source_root_id,source_relative_path,scheduler_state,scientific_status,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, "DUALX", CAMPAIGN_ID, label, workflow, method,
                 ROOT_ID, evidence_rel, status, scientific, notes),
            )
            con.execute(
                "UPDATE workflow_run SET scheduler_state=?,scientific_status=?,notes=? WHERE run_id=?",
                (status, scientific, notes, run_id),
            )

        ensure_run(RUNS["inference"], "safe_core_portable_config", "ML_INFERENCE",
                   "M1-M6 inference from two source structuresets across 3 models; run manifest records 377 static and 58 NEB frames and seeds 11/23/37.",
                   "UNKNOWN", "PROVENANCE_PARTIAL",
                   "run_manifest.json records outputs and inference commit 06a7d4; original exact command/config path is temporary and not captured; model checkpoints are external and their training lineage is partial.")
        ensure_run(RUNS["analysis"], "Safe Core M1-M6 frozen-inference analysis", "ML_ANALYSIS",
                   "Analysis rebuilt from frozen inference outputs; scientific status remains claim-specific and provenance partial where source/model lineage is missing.",
                   "UNKNOWN", "PROVENANCE_PARTIAL",
                   "clean_checkout_validation.json records analysis rebuild and validator PASS; this is local execution with no Slurm job, so scheduler state remains UNKNOWN.")
        ensure_run(RUNS["figures"], "Safe Core v6 figure regeneration", "FIGURE_RENDER",
                   "Render seven registered figure IDs into 21 PNG/PDF/SVG files from exact source tables.",
                   "UNKNOWN", "PROVENANCE_PARTIAL",
                   "Figure registry records exact make_figures script, configs, source-table hashes, render command, and locked environment; local execution has no Slurm job, so scheduler state remains UNKNOWN.")
        ensure_run(RUNS["validation"], "Safe Core v6 clean checkout validation", "BUNDLE_VALIDATION",
                   "Validate the frozen bundle and clean-checkout reproduction outputs.",
                   "UNKNOWN", "LINEAGE_SUPPORTED",
                   "clean_checkout_validation.json records four passing tests, analysis rebuild, figure render, and bundle validator PASS; local execution has no Slurm job, so scheduler state remains UNKNOWN. Captured status is source evidence, not a new test run here.")

        recipe_specs = [
            (RUNS["inference"], "v6-original-inference", "NOT_CAPTURED: see run_manifest.json; the documented reference command is in analysis_manifest.json, but it is not asserted to be the original invocation.", "INCOMPLETE", "run_safe_core.py", "inference"),
            (RUNS["analysis"], "v6-clean-checkout-analysis", analysis_manifest["commands"]["analysis"], "REPRODUCED", "analyze_bundle.py", "analysis"),
            (RUNS["figures"], "v6-clean-checkout-figures", analysis_manifest["commands"]["figures"], "REPRODUCED", "make_figures.py", "figures"),
            (RUNS["validation"], "v6-clean-checkout-validation", "python safe_core/scripts/validate_bundle.py --bundle <bundle> --repo-root <repo-root>", "REPRODUCED", "validate_bundle.py", "validation"),
        ]
        recipe_ids: dict[str, int] = {}
        for run_id, version, command, recipe_status, script_basename, role in recipe_specs:
            script_rel = next(rel for rel in artifact_ids if rel.endswith("/" + script_basename) or rel == "safe_core/scripts/" + script_basename)
            script_id = artifact_ids[script_rel]
            manifest_id = artifact_ids[(BUNDLE_REL / "INPUT_MANIFEST.tsv").as_posix()]
            note = "Source command and clean checkout outcome recorded in manifests."
            if role == "inference":
                note = "Original invocation unavailable; model checkpoints are external and exact historical config path no longer exists. Do not treat the documented reference command as the original command."
            if role == "analysis":
                note = "Clean checkout validation records this command passed; analysis resumed from frozen inference outputs."
            con.execute(
                """INSERT OR IGNORE INTO reproduction_recipe
                   (run_id,recipe_version,exact_command,working_directory,executable_version,
                    environment_description,module_list,script_artifact_id,input_manifest_artifact_id,
                    recipe_status,validation_note)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, version, command, evidence_rel, "Python (version recorded in validation environment)",
                 "see safe_core/requirements.lock; model cache path external" if role == "inference" else "see safe_core/requirements.lock",
                 "PYTHONPATH=safe_core", script_id, manifest_id, recipe_status, note),
            )
            recipe_ids[role] = con.execute(
                "SELECT recipe_id FROM reproduction_recipe WHERE run_id=? AND recipe_version=?",
                (run_id, version),
            ).fetchone()[0]

        # Bind run code to exact source commits; no commit is pushed or updated.
        commit_artifact = artifact_ids[(BUNDLE_REL / "run_manifest.json").as_posix()]
        release_artifact = artifact_ids[(BUNDLE_REL / "RELEASE_MANIFEST.json").as_posix()]
        for run_id, commit, role, evidence_id in [
            (RUNS["inference"], run_manifest["repo_commit"], "INFERENCE_CODE", commit_artifact),
            (RUNS["analysis"], release["release_commit"], "ANALYSIS_CODE", release_artifact),
            (RUNS["figures"], release["release_commit"], "FIGURE_CODE", release_artifact),
            (RUNS["validation"], release["release_commit"], "VALIDATION_CODE", artifact_ids[(BUNDLE_REL / "clean_checkout_validation.json").as_posix()]),
        ]:
            con.execute(
                """INSERT OR IGNORE INTO run_git_binding
                   (run_id,repository_id,commit_sha,binding_role,evidence_artifact_id,provenance_status)
                   VALUES(?,?,?,?,?,'LINEAGE_SUPPORTED')""",
                (run_id, repository_id, commit, role, evidence_id),
            )

        con.execute("INSERT OR IGNORE INTO run_parent(child_run_id,parent_run_id,relationship,notes) VALUES(?,?,?,?)",
                    (RUNS["analysis"], RUNS["inference"], "USES_FROZEN_INFERENCE_OUTPUTS", "analysis_manifest.json says analysis_resumed_from_frozen_inference=true"))
        con.execute("INSERT OR IGNORE INTO run_parent(child_run_id,parent_run_id,relationship,notes) VALUES(?,?,?,?)",
                    (RUNS["figures"], RUNS["analysis"], "RENDERS_ANALYSIS_OUTPUTS", "Figure registry lists exact derived table inputs and make_figures.py command."))
        con.execute("INSERT OR IGNORE INTO run_parent(child_run_id,parent_run_id,relationship,notes) VALUES(?,?,?,?)",
                    (RUNS["validation"], RUNS["figures"], "VALIDATES_BUNDLE_OUTPUTS", "Clean checkout validation records figures regenerated and validator PASS."))

        # Register exact source/model inputs for the reusable workflow recipes.
        def add_recipe_input(role: str, artifact_id: int, run_key: str) -> None:
            recipe_id = recipe_ids[run_key]
            con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",
                        (recipe_id, artifact_id, role))

        input_manifest_id = artifact_ids[(BUNDLE_REL / "INPUT_MANIFEST.tsv").as_posix()]
        model_registry_id = artifact_ids[(BUNDLE_REL / "MODEL_REGISTRY.tsv").as_posix()]
        analysis_config_id = artifact_ids[(BUNDLE_REL / "analysis_config_snapshot.json").as_posix()]
        predictions_id = artifact_ids[(BUNDLE_REL / "tables/predictions.tsv").as_posix()]
        pooled_id = artifact_ids[(BUNDLE_REL / "processed/pooled_features.npz").as_posix()]
        for aid, role in [(input_manifest_id, "FRAME_INPUT_MANIFEST"), (model_registry_id, "MODEL_REGISTRY"),
                          (analysis_config_id, "ANALYSIS_CONFIG")]:
            add_recipe_input(role, aid, "inference")
            add_recipe_input(role, aid, "analysis")
        for model, aid in external_model_ids.items():
            add_recipe_input("EXTERNAL_MODEL:" + model, aid, "inference")
        for (source_path, _), rel in source_data.items():
            add_recipe_input("SOURCE_DATA:" + source_path, artifact_ids[rel], "inference")
        for aid, role in [(predictions_id, "FROZEN_INFERENCE_PREDICTIONS"), (pooled_id, "FROZEN_POOLED_FEATURES")]:
            add_recipe_input(role, aid, "analysis")
        add_recipe_input("LOCKED_ENVIRONMENT", artifact_ids[lock_rel], "analysis")
        add_recipe_input("LOCKED_ENVIRONMENT", artifact_ids[lock_rel], "figures")

        for row in model_rows:
            aid = external_model_ids[row["model_id"]]
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,'MODEL',1,?)""",
                (RUNS["inference"], aid, "MODEL:" + row["model_id"],
                 "external model metadata/hashes from MODEL_REGISTRY.tsv; location unresolved"),
            )
        for (source_path, _), rel in source_data.items():
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,'INPUT',1,?)""",
                (RUNS["inference"], artifact_ids[rel], "SOURCE_DATA:" + source_path,
                 "exact path and source SHA are recorded in INPUT_MANIFEST.tsv"),
            )
        for rel in ["tables/predictions.tsv", "processed/pooled_features.npz"]:
            aid = artifact_ids[(BUNDLE_REL / rel).as_posix()]
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,'OUTPUT',0,?)""",
                (RUNS["inference"], aid, rel, "listed in DUAL-X figure/table registry"),
            )
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,'INPUT',1,?)""",
                (RUNS["analysis"], aid, rel, "frozen inference output consumed by analysis bundle rebuild"),
            )

        # Map every registered output to its source tool, exact source tables,
        # and run. Hashes were validated above against the immutable manifest.
        script_to_run = {
            "safe_core/scripts/run_safe_core.py": RUNS["inference"],
            "safe_core/scripts/run_reference_inference.py": RUNS["inference"],
            "safe_core/scripts/analyze_bundle.py": RUNS["analysis"],
            "safe_core/scripts/audit_hidden_provenance.py": RUNS["analysis"],
            "safe_core/scripts/finalize_bundle.py": RUNS["validation"],
            "safe_core/scripts/make_figures.py": RUNS["figures"],
        }
        script_run_kind = {
            "safe_core/scripts/run_safe_core.py": "INFERENCE",
            "safe_core/scripts/run_reference_inference.py": "REFERENCE_INFERENCE",
            "safe_core/scripts/analyze_bundle.py": "ANALYSIS",
            "safe_core/scripts/audit_hidden_provenance.py": "HIDDEN_AUDIT",
            "safe_core/scripts/finalize_bundle.py": "FINALIZATION",
            "safe_core/scripts/make_figures.py": "FIGURE_RENDER",
        }
        for row in figure_rows:
            target_rel = (BUNDLE_REL / row["path"]).as_posix()
            target_id = artifact_ids[target_rel]
            script_rel = row["source_script"]
            script_id = artifact_ids[script_rel]
            run_id = script_to_run[script_rel]
            if row["artifact_type"] == "figure":
                direction, role = "FIGURE", "FIGURE_EXPORT"
            else:
                direction, role = "OUTPUT", row["artifact_type"].upper() + ":" + row["artifact_id"]
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,?,0,?)""",
                (run_id, target_id, role, direction,
                 f"Source registry status={row['status']}; exact source script={script_rel}."),
            )
            source_rels = [x for x in row["source_tables"].split(";") if x]
            for source_rel in source_rels:
                source_art_rel = (BUNDLE_REL / source_rel).as_posix()
                if source_art_rel not in artifact_ids:
                    raise RuntimeError(f"Figure/table registry source path is not hash-registered: {source_rel}")
                parent_id = artifact_ids[source_art_rel]
                if parent_id == target_id:
                    continue
                con.execute(
                    """INSERT INTO derivation_edge
                       (parent_artifact_id,child_artifact_id,run_id,operation,tool_name,
                        tool_version,script_artifact_id,parameter_json,provenance_status,notes)
                       SELECT ?,?,?,?,?,?,?,?,?,?
                       WHERE NOT EXISTS (
                         SELECT 1 FROM derivation_edge WHERE parent_artifact_id=? AND child_artifact_id=?
                           AND run_id=? AND operation=? AND script_artifact_id=?
                       )""",
                    (parent_id, target_id, run_id,
                     "FIGURE_RENDER" if row["artifact_type"] == "figure" else "TABLE_GENERATION",
                     script_rel, release["release_commit"], script_id,
                     json.dumps({"command": row["render_command"], "config": row["source_config"], "panel_or_cell": row["panel_or_cell"]}, sort_keys=True),
                     "LINEAGE_SUPPORTED",
                     f"Source registry lists exact source path/hash; workflow category={script_run_kind[script_rel]}.",
                     parent_id, target_id, run_id,
                     "FIGURE_RENDER" if row["artifact_type"] == "figure" else "TABLE_GENERATION", script_id),
                )

        # Figure entities and their exact output/data/script/config/environment assets.
        figure_ids: dict[str, str] = {}
        for row in figure_rows:
            if row["artifact_type"] != "figure":
                continue
            source_id = "DUALX-SAFECORE-" + row["artifact_id"]
            figure_ids[row["artifact_id"]] = source_id
            con.execute(
                """INSERT OR IGNORE INTO paper_figure
                   (figure_id,project_id,campaign_id,source_campaign_key,title,chart_kind,
                    chart_status,provenance_status,source_registry_source_id,data_query,
                    processing_command,scientific_boundary,notes)
                   VALUES(?,?,?,?,?,'SCIENTIFIC_FIGURE',?,'PROVENANCE_PARTIAL',?,?,?,?,?)""",
                (source_id, "DUALX", CAMPAIGN_ID, release["release_tag"], row["artifact_id"],
                 row["status"], source_ids["figure_registry"], row["source_tables"],
                 row["render_command"], row["caption_status"],
                 f"Source figure registry panel_or_cell={row['panel_or_cell']}; 3 output formats are separate artifacts."),
            )

        for row in figure_rows:
            if row["artifact_type"] != "figure":
                continue
            figure_id = figure_ids[row["artifact_id"]]
            export_rel = (BUNDLE_REL / row["path"]).as_posix()
            con.execute(
                """INSERT OR IGNORE INTO paper_figure_asset
                   (figure_id,artifact_id,asset_role,source_path,availability,relationship,notes)
                   VALUES(?,?,'FIGURE_EXPORT',?,'PRESENT','RENDERED_OUTPUT',?)""",
                (figure_id, artifact_ids[export_rel], row["path"], f"sha256={row['sha256']}; status={row['status']}"),
            )
            for source_rel in [x for x in row["source_tables"].split(";") if x]:
                source_art_rel = (BUNDLE_REL / source_rel).as_posix()
                con.execute(
                    """INSERT OR IGNORE INTO paper_figure_asset
                       (figure_id,artifact_id,asset_role,source_path,availability,relationship,notes)
                       VALUES(?,?,'DERIVED_DATA',?,'PRESENT','RENDER_SOURCE_TABLE',?)""",
                    (figure_id, artifact_ids[source_art_rel], source_rel,
                     f"source_table_hash={next((h.split('=',1)[1] for h in row['source_table_hashes'].split(';') if h.startswith(source_rel+'=')), 'not listed')}"),
                )
            for asset_role, rel, relationship in [
                ("REBUILD_METHOD", row["source_script"], "FIGURE_RENDERER"),
                ("PROVENANCE_MANIFEST", row["source_config"], "FIGURE_CONFIGURATION"),
                ("SOURCE_MANIFEST", row["render_environment"], "LOCKED_RENDER_ENVIRONMENT"),
            ]:
                asset_rel = rel if rel in artifact_ids else (BUNDLE_REL / rel).as_posix()
                if asset_rel in artifact_ids:
                    con.execute(
                        """INSERT OR IGNORE INTO paper_figure_asset
                           (figure_id,artifact_id,asset_role,source_path,availability,relationship,notes)
                           VALUES(?,?,?,?,'PRESENT',?,?)""",
                        (figure_id, artifact_ids[asset_rel], asset_role, rel, relationship,
                         f"sha256={sha256(REPO / asset_rel)}"),
                    )

        # Source claims remain provenance-partial until a separate scientific review.
        claim_prefix_ids: dict[str, str] = {}
        for row in claim_rows:
            claim_id = "DUALX-SAFECORE-" + row["claim_id"]
            claim_prefix_ids[row["claim_id"]] = claim_id
            con.execute(
                """INSERT OR IGNORE INTO paper_claim
                   (claim_id,release_id,manuscript_path,section_or_panel,claim_text,
                    scientific_status,publication_action,notes)
                   VALUES(?,?,NULL,?,?,'PROVENANCE_PARTIAL','PENDING',?)""",
                (claim_id, RELEASE_ID, row["panel_or_cell"], row["claim"],
                 f"Current catalog status deliberately remains PROVENANCE_PARTIAL; exact source status={row['status']}; allowed wording is preserved in source observation."),
            )
            refs = {
                "evidence": [x for x in row["evidence"].split(";") if x],
                "artifact_type": row["artifact_type"],
                "artifact_id": row["artifact_id"],
                "panel_or_cell": row["panel_or_cell"],
                "allowed_wording": row["allowed_wording"],
            }
            con.execute(
                """INSERT OR IGNORE INTO paper_claim_source_observation
                   (source_id,source_claim_key,claim_id,source_status,source_publication_action,
                    source_job_or_run,source_target,source_evidence_refs,observed_utc)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (source_ids["claims"], row["claim_id"], claim_id, row["status"], "", "",
                 row["allowed_wording"], json.dumps(refs, sort_keys=True), observed_utc),
            )
            for evidence_rel in refs["evidence"]:
                artifact_rel = (BUNDLE_REL / evidence_rel).as_posix()
                if artifact_rel not in artifact_ids:
                    raise RuntimeError(f"Claim evidence is not present in the release hash manifest: {evidence_rel}")
                con.execute(
                    """INSERT INTO claim_evidence
                       (claim_id,artifact_id,run_id,evidence_role,support_status,notes)
                       SELECT ?,?,NULL,?,'UNKNOWN',?
                       WHERE NOT EXISTS (
                         SELECT 1 FROM claim_evidence
                         WHERE claim_id=? AND artifact_id=? AND run_id IS NULL AND evidence_role=?
                       )""",
                    (claim_id, artifact_ids[artifact_rel], "SOURCE_MATRIX:" + row["claim_id"],
                     "Source matrix cites this exact path; catalog support decision remains UNKNOWN.",
                     claim_id, artifact_ids[artifact_rel], "SOURCE_MATRIX:" + row["claim_id"]),
                )
            segments = [x for x in row["panel_or_cell"].split(";") if x]
            for segment in segments:
                figure_source_key = segment.split(":", 1)[0]
                if figure_source_key not in figure_ids:
                    continue
                figure_id = figure_ids[figure_source_key]
                con.execute(
                    """INSERT OR IGNORE INTO paper_claim_figure
                       (claim_id,figure_id,source_id,source_claim_key,relationship,notes)
                       VALUES(?,?,?,?, 'ILLUSTRATES',?)""",
                    (claim_id, figure_id, source_ids["claims"], row["claim_id"],
                     f"Explicit figure/panel mapping from source CLAIM_EVIDENCE_MATRIX.tsv: {segment}"),
                )

        # Tie figure/data/script artifacts back to the analysis and render runs.
        for row in figure_rows:
            rel = (BUNDLE_REL / row["path"]).as_posix()
            aid = artifact_ids[rel]
            run_id = RUNS["figures"] if row["artifact_type"] == "figure" else script_to_run[row["source_script"]]
            direction = "FIGURE" if row["artifact_type"] == "figure" else "OUTPUT"
            role = "FIGURE_EXPORT" if row["artifact_type"] == "figure" else "TABLE_OR_MANIFEST:" + row["artifact_id"]
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,?,0,?)""",
                (run_id, aid, role, direction, "exact output path/hash from FIGURE_TABLE_REGISTRY.tsv"),
            )

        # Record direct evidence links from each source registry into the catalog.
        for source_name, source_id in source_ids.items():
            if source_name in ("claims", "figure_registry", "release"):
                rel = str((BUNDLE / {"claims":"CLAIM_EVIDENCE_MATRIX.tsv", "figure_registry":"FIGURE_TABLE_REGISTRY.tsv", "release":"RELEASE_MANIFEST.json"}[source_name]).relative_to(REPO).as_posix())
                art_id = artifact_ids[rel]
                art_key = con.execute("SELECT artifact_key FROM artifact WHERE artifact_id=?", (art_id,)).fetchone()[0]
                con.execute(
                    """INSERT OR IGNORE INTO registry_entity_ref
                       (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                       VALUES(?, 'registry_file', ?, 'ARTIFACT', ?, 'DOCUMENTS', ?)""",
                    (source_id, rel, art_key, f"SHA-256 source snapshot; {source_name}"),
                )

        # SQLite UNIQUE constraints treat NULL run_id values as distinct, so
        # preserve one exact source-matrix edge per claim/artifact/role.
        con.execute(
            """DELETE FROM claim_evidence
               WHERE claim_id IN (SELECT claim_id FROM paper_claim WHERE release_id=?)
                 AND rowid NOT IN (
                   SELECT MIN(rowid) FROM claim_evidence
                   WHERE claim_id IN (SELECT claim_id FROM paper_claim WHERE release_id=?)
                   GROUP BY claim_id, artifact_id, evidence_role
                 )""",
            (RELEASE_ID, RELEASE_ID),
        )

        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError(f"Foreign key violations before commit: {fk[:5]}")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect(DB) as check:
        print("dualx_project=" + str(check.execute("SELECT COUNT(*) FROM project WHERE project_id='DUALX'").fetchone()[0]))
        print("dualx_campaigns=" + str(check.execute("SELECT COUNT(*) FROM campaign WHERE project_id='DUALX'").fetchone()[0]))
        print("dualx_runs=" + str(check.execute("SELECT COUNT(*) FROM workflow_run WHERE project_id='DUALX'").fetchone()[0]))
        print("dualx_artifacts=" + str(check.execute("SELECT COUNT(*) FROM artifact WHERE project_id='DUALX'").fetchone()[0]))
        print("dualx_figures=" + str(check.execute("SELECT COUNT(*) FROM paper_figure WHERE project_id='DUALX'").fetchone()[0]))
        print("dualx_claims=" + str(check.execute("SELECT COUNT(*) FROM paper_claim WHERE release_id=?", (RELEASE_ID,)).fetchone()[0]))
        print("dualx_figure_edges=" + str(check.execute("SELECT COUNT(*) FROM derivation_edge WHERE run_id=?", (RUNS["figures"],)).fetchone()[0]))
        print("foreign_key_violations=" + str(len(check.execute("PRAGMA foreign_key_check").fetchall())))


if __name__ == "__main__":
    main()
