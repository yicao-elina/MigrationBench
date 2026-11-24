#!/usr/bin/env python3
"""Stage the single proven-unused Conda package cache entry in the SQL catalog."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DATA = AUDIT / "phase2_data"
DB = AUDIT / "provenance_catalog.sqlite"
MANIFEST = DATA / "conda_libstdcxx_14_2_guard_manifest_20261122.json"
PREFIX_EVIDENCE = DATA / "conda_libstdcxx_14_2_prefix_evidence_20261122.tsv"
REPODATA = DATA / "conda_libstdcxx_14_2_repodata_record_20261122.json"
RECIPE = DATA / "recreate_conda_package_cache.sh"
BATCH = "RF-DATA-CONDA-PKG-LIBSTDCXX-UNUSED-20261122-R01"
RUN = "RF-CONDAPKG-LIBSTDCXX-14_2-H8F9B012-REGEN-RECIPE-20261122"
LOCAL_ROOT = "LOCAL-MAIN"
REMOTE_ROOT = "RF-DATA-YI"
EXPECTED_MANIFEST_SHA = "8b1e7978413ded29a2cb41658dccd625e9d6dbdf4bffbd62ecca4f3fb6a96dab"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def source_and_artifact(con: sqlite3.Connection, key: str, path: Path, kind: str,
                        role: str, notes: str, project: str = "HPC_ENVIRONMENTS") -> int:
    now = utc_now()
    path = path.resolve()
    digest = sha256(path)
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if prior and tuple(prior) != (str(path), digest):
        raise RuntimeError(f"Source key already refers to changed evidence: {key}")
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
           VALUES(?,?,?,?,?,?,1,?)""",
        (key, str(path), kind, digest, now, now, notes),
    )
    source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0]
    root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT,)).fetchone()[0]
    relative = path.relative_to(Path(root)).as_posix()
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (LOCAL_ROOT, relative, now))
    path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT, relative)).fetchone()[0]
    artifact_key = "CONDAPKG-EVIDENCE:" + key
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES(?,?,?,?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
        (project, artifact_key, kind, role, digest, path.stat().st_size, notes),
    )
    artifact_id, artifact_hash = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if artifact_hash != digest:
        raise RuntimeError(f"Evidence artifact hash mismatch: {artifact_key}")
    con.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""", (artifact_id, path_id, now, now)
    )
    return artifact_id


def main() -> None:
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Reviewed Conda package manifest digest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    leaves = [row for row in entries if row["type"] in ("file", "symlink")]
    counts = manifest["entry_counts"]
    if counts != {"directories": 15, "regular_files": 88, "special_files": 0, "symlinks": 2}:
        raise RuntimeError("Conda candidate entry counts differ from reviewed scan")
    if len(leaves) != 90 or not manifest["candidate_scan_stable"]:
        raise RuntimeError("Candidate tree is incomplete or unstable")
    if manifest["candidate_identity_in_any_scanned_conda_meta"] or manifest["conda_meta_read_errors"]:
        raise RuntimeError("Candidate package is referenced or one or more Conda prefixes were unreadable")
    if not manifest["all_file_and_symlink_links_private_to_candidate"]:
        raise RuntimeError("Candidate contains a hard-linked file or symlink")
    repodata = json.loads(REPODATA.read_text(encoding="utf-8"))
    if (repodata["name"], repodata["version"], repodata["build"], repodata["sha256"]) != (
        "libstdcxx", "14.2.0", "h8f9b012_2", "8f5bd92e4a24e1d35ba015c5252e8f818898478cb3bc50bd8b12ab54707dc4da"
    ):
        raise RuntimeError("Conda repodata package identity/digest differs from the reviewed package")

    now = utc_now()
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        manifest_artifact = source_and_artifact(con, "conda_package_cache_guard_manifest_20261122", MANIFEST,
            "CONDA_PACKAGE_CACHE_GUARD_MANIFEST", "Exact package-cache path/hash manifest",
            "105-entry candidate tree; regular files and symlink payloads are hashed, all metadata is guarded; no prefix references.")
        prefix_artifact = source_and_artifact(con, "conda_prefix_reference_audit_20261122", PREFIX_EVIDENCE,
            "CONDA_PREFIX_REFERENCE_AUDIT", "Conda environment reference audit",
            "All 20 visible Conda prefixes were scanned across 2,190 conda-meta records with zero read errors and zero candidate identity matches.")
        repodata_artifact = source_and_artifact(con, "conda_libstdcxx_repodata_record_20261122", REPODATA,
            "CONDA_PACKAGE_REPODATA_RECORD", "Exact source package metadata",
            "Conda-forge package record preserves archive URL, version, build, MD5 and SHA-256.")
        recipe_artifact = source_and_artifact(con, "conda_package_cache_reproduction_recipe_20261122", RECIPE,
            "CONDA_CACHE_REPRODUCTION_SCRIPT", "Pinned cache regeneration script",
            "Creates a disposable prefix to force Conda to fetch and extract the pinned conda-forge package into CONDA_PKGS_DIRS.")

        yi_root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (REMOTE_ROOT,)).fetchone()[0]
        candidate_root = manifest["candidate_root"]
        if not candidate_root.startswith(yi_root + "/"):
            raise RuntimeError("Candidate escaped the user-owned Rockfish root")
        relative_root = candidate_root[len(yi_root) + 1:]

        cache_key = "CONDAPKG-CACHE-TREE:libstdcxx-14.2.0-h8f9b012_2"
        con.execute(
            """INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc)
               VALUES(?,?,?)""", (REMOTE_ROOT, relative_root, now)
        )
        root_path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (REMOTE_ROOT, relative_root)).fetchone()[0]
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                scientific_status,retention_class,notes)
               VALUES('HPC_ENVIRONMENTS',?,'CONDA_PACKAGE_CACHE_TREE','Unused extracted package cache tree',
                      'USER_CONFIRMED',?,?,'RECONSTRUCTION','CACHE',?)""",
            (cache_key, manifest["tree_entries_sha256"], manifest["logical_bytes"],
             "Not referenced by any of 20 scanned prefixes; safe to re-fetch from the pinned source package record. Exact tree evidence is in the linked cleanup manifest."),
        )
        cache_id, cache_hash = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (cache_key,)).fetchone()
        if cache_hash != manifest["tree_entries_sha256"]:
            raise RuntimeError("Synthetic cache-tree artifact has a changed tree hash")
        con.execute(
            """INSERT OR IGNORE INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES(?,?,'PRESENT',?,?)""", (cache_id, root_path_id, now, now)
        )

        package_key = "CONDAPKG-REMOTE:libstdcxx-14.2.0-h8f9b012_2"
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                scientific_status,retention_class,notes)
               VALUES('HPC_ENVIRONMENTS',?,'REMOTE_CONDA_PACKAGE','Pinned conda-forge archive',
                      'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','REPRODUCIBLE_INPUT',?)""",
            (package_key, repodata["sha256"], repodata["size"],
             f"Exact source URL {repodata['url']}; Conda repodata SHA-256 verified against the package record."),
        )
        package_id = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (package_key,)).fetchone()
        if package_id[1] != repodata["sha256"]:
            raise RuntimeError("Remote package artifact digest mismatch")
        package_id = package_id[0]

        con.execute(
            """INSERT OR IGNORE INTO workflow_run
               (run_id,project_id,run_label,workflow_type,method,source_root_id,source_relative_path,
                scheduler_state,scientific_status,notes)
               VALUES(?,'HPC_ENVIRONMENTS','Pinned Conda package-cache regeneration recipe',
                      'CONDA_PACKAGE_CACHE_REGENERATION','conda create package extraction',?,?,
                      'UNKNOWN','RECONSTRUCTION',?)""",
            (RUN, REMOTE_ROOT, relative_root,
             "This catalog entry documents how to regenerate an environment cache payload; it does not claim the old cache tree belongs to a specific scientific run."),
        )
        exact_command = (
            "CONDA_PKGS_DIRS=/data/pclancy3/yi/.conda/pkgs \"$CONDA_EXE\" create --yes --no-deps "
            "--prefix \"$TMP_BASE/env\" --override-channels --channel https://conda.anaconda.org/conda-forge "
            "--no-default-packages libstdcxx=14.2.0=h8f9b012_2"
        )
        con.execute(
            """INSERT OR IGNORE INTO reproduction_recipe
               (run_id,recipe_version,exact_command,working_directory,executable_version,
                environment_description,script_artifact_id,input_manifest_artifact_id,recipe_status,validation_note)
               VALUES(?, '2026-11-22-r01', ?, '/data/pclancy3/yi', 'Conda version not pinned in the preserved source record',
                      'conda-forge/linux-64; package archive SHA-256 8f5bd92e4a24e1d35ba015c5252e8f818898478cb3bc50bd8b12ab54707dc4da',
                      ?,?,'DOCUMENTED',?)""",
            (RUN, exact_command, recipe_artifact, repodata_artifact,
             "Pinned package identity, URL and archive hash are recorded; the recipe was documented but not executed as a cache-rebuild test."),
        )
        for artifact_id, role, direction, required, notes in (
            (package_id, "PINNED_CONDA_ARCHIVE", "INPUT", 1, "Remote input; URL and SHA-256 are in artifact notes and the linked repodata JSON."),
            (repodata_artifact, "REPODATA_RECORD", "PROVENANCE", 1, "Exact package record captured from the cached Conda metadata."),
            (recipe_artifact, "CACHE_REGENERATION_SCRIPT", "PROVENANCE", 1, "Executable shell recipe for a future cache rebuild."),
            (prefix_artifact, "PREFIX_REFERENCE_AUDIT", "PROVENANCE", 1, "Evidence that no scanned environment refers to the package."),
            (manifest_artifact, "DELETION_GUARD_MANIFEST", "PROVENANCE", 1, "Exact metadata/hash guard for the deleted tree."),
            (cache_id, "EXTRACTED_PACKAGE_CACHE", "INTERMEDIATE", 0, "Historical cache tree, planned for verified removal."),
        ):
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,?,?,?)""", (RUN, artifact_id, role, direction, required, notes)
            )

        existing = con.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        expected_logical = sum(int(row["size"]) for row in leaves)
        if existing:
            if tuple(existing) != ("PLANNED", expected_logical):
                raise RuntimeError("Existing cleanup batch is not the expected planned batch")
        else:
            con.execute(
                """INSERT INTO cleanup_batch
                   (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,
                    expected_bytes,evidence_artifact_id)
                   VALUES(?,?,?,'EXACT_CACHE_MANIFEST_GUARDED',?,'PLANNED',?,?,?)""",
                (BATCH, now, REMOTE_ROOT,
                 "Remove only one extracted Conda package tree absent from all 20 readable prefixes and with private leaf link counts; preserve the pinned archive URL/hash and cache regeneration recipe.",
                 "User authorized removal of reproducible intermediates and caches after the source path and recreation method are cataloged.",
                 expected_logical, manifest_artifact),
            )

        for row in leaves:
            relative = str(row["path"])[len(yi_root) + 1:]
            kind = "FILE" if row["type"] == "file" else "SYMLINK"
            digest = row.get("sha256") if kind == "FILE" else row["target_sha256"]
            con.execute(
                """INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc)
                   VALUES(?,?,?)""", (REMOTE_ROOT, relative, now)
            )
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (REMOTE_ROOT, relative)).fetchone()[0]
            artifact_key = "RF-DATA-YI:" + relative
            existing_artifact = con.execute("SELECT artifact_id,artifact_kind,byte_size,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if not existing_artifact:
                con.execute(
                    """INSERT INTO artifact
                       (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                        scientific_status,retention_class,notes)
                       VALUES('HPC_ENVIRONMENTS',?,?,'unused Conda extracted-package cache leaf',
                              'USER_CONFIRMED',?,?,'RECONSTRUCTION','CACHE',?)""",
                    (artifact_key, kind, digest, int(row["size"]), "Exact pre-delete identity is in the linked candidate-tree manifest."),
                )
                artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()[0]
            else:
                artifact_id = existing_artifact[0]
                if existing_artifact[1] != kind or int(existing_artifact[2]) != int(row["size"]):
                    raise RuntimeError(f"Catalog path identity changed: {relative}")
                if existing_artifact[3] not in (None, digest):
                    raise RuntimeError(f"Catalog path hash conflicts with fresh exact manifest: {relative}")
                con.execute(
                    """UPDATE artifact SET project_id='HPC_ENVIRONMENTS',semantic_role='unused Conda extracted-package cache leaf',
                       owner_status='USER_CONFIRMED',sha256=?,byte_size=?,scientific_status='RECONSTRUCTION',
                       retention_class='CACHE',notes=notes || ? WHERE artifact_id=?""",
                    (digest, int(row["size"]), f" Exact manifest SHA/tree receipt: {EXPECTED_MANIFEST_SHA}.", artifact_id),
                )
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""", (artifact_id, path_id, now, now)
            )
            current_location = con.execute("SELECT path_id,availability FROM artifact_location WHERE artifact_id=?", (artifact_id,)).fetchone()
            if current_location != (path_id, "PRESENT"):
                raise RuntimeError(f"Candidate artifact does not have one present location: {relative}")
            con.execute(
                """INSERT OR IGNORE INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,
                    pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes)
                   VALUES(?,?,'DELETE',?,'CLEARED',?,?,?,'PLANNED',?)""",
                (BATCH, artifact_id,
                 "Unused Conda package cache leaf; exact hash or symlink target and lstat identity match the guarded tree manifest; package can be re-fetched from conda-forge.",
                 digest, int(row["size"]), int(row["mtime_ns"]),
                 f"entry={row['type']}; allocated_bytes={row['allocated_bytes']}; nlink={row['nlink']}; manifest_sha256={EXPECTED_MANIFEST_SHA}"),
            )

        staged = con.execute("SELECT count(*),sum(pre_delete_size_bytes),sum(pre_delete_sha256 IS NOT NULL) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        if staged != (90, expected_logical, 90):
            raise RuntimeError(f"Unexpected staged cleanup item totals: {staged}")
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print(json.dumps({"batch": BATCH, "status": "PLANNED", "delete_items": 90,
                      "logical_bytes": expected_logical, "allocated_leaf_bytes": manifest["allocated_regular_and_symlink_bytes"],
                      "allocated_all_path_entries_bytes": manifest["allocated_path_entry_bytes"],
                      "manifest_sha256": EXPECTED_MANIFEST_SHA}, sort_keys=True))


if __name__ == "__main__":
    main()
