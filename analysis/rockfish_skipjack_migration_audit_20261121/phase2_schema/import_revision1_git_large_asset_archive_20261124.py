#!/usr/bin/env python3
"""Register the verified 2026-11-24 MigrationBench model/vector archive in SQLite."""
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from import_rockfish_asset_archives import (
    DB, STORAGE_ROOT, add_location_and_observation, ensure_artifact,
    ensure_storage_path, register_source,
)

AUDIT = Path(__file__).resolve().parents[1]
DATA = AUDIT / "phase2_data"
ARCHIVE = DATA / "revision1_git_large_assets_20261124"
MANIFEST = ARCHIVE / "ARCHIVE_MANIFEST.tsv"
REPORT = ARCHIVE / "VERIFICATION_REPORT.json"
ROCKFISH_REL_ROOT = "MigrationBench_data_archive/revision1_git_large_assets_20261124"
SNAPSHOT_ID = "MB-ROCKFISH-LARGE-ASSET-ARCHIVE-20261124"
RUN_ID = SNAPSHOT_ID


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for path in (MANIFEST, REPORT, ARCHIVE / "README.md"):
        if not path.is_file():
            raise RuntimeError("archive control file missing: " + str(path))
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8"), delimiter="\t"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("verified_sha256_and_size") is not True:
        raise RuntimeError("Rockfish archive report is not a successful full-file verification")
    if len(rows) != int(report["file_count"]):
        raise RuntimeError("archive manifest count disagrees with verification report")
    if sum(int(row["byte_size"]) for row in rows) != int(report["logical_bytes"]):
        raise RuntimeError("archive manifest bytes disagree with verification report")
    report_rows = {row["archive_rel"]: row for row in report["payloads"]}
    for row in rows:
        verified = report_rows.get(row["archive_rel"])
        if not verified:
            raise RuntimeError("payload missing from verification report: " + row["archive_rel"])
        for field in ("sha256", "git_blob_oid", "source_commit", "source_repository_path", "byte_size"):
            expected = str(row[field])
            actual = str(verified[field])
            if expected != actual:
                raise RuntimeError("manifest/report mismatch in %s for %s" % (field, row["archive_rel"]))

    now = utc_now()
    manifest_sha = sha256_file(MANIFEST)
    con = sqlite3.connect(str(DB), timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        con.execute("BEGIN IMMEDIATE")
        if not con.execute("SELECT 1 FROM project WHERE project_id='MIGRATIONBENCH'").fetchone():
            raise RuntimeError("MIGRATIONBENCH project is missing")
        if not con.execute("SELECT 1 FROM storage_root WHERE root_id=?", (STORAGE_ROOT,)).fetchone():
            raise RuntimeError("Rockfish storage root is missing: " + STORAGE_ROOT)

        notes = (
            "Five archived payloads totaling %d bytes; source size/SHA-256 and "
            "Git blob identity where applicable were verified on Rockfish. "
            "Git and source provenance do not establish scientific validity."
        ) % int(report["logical_bytes"])
        con.execute(
            """INSERT OR IGNORE INTO workflow_run
               (run_id,project_id,run_label,workflow_type,method,system_key,source_root_id,
                source_relative_path,scheduler_state,scientific_status,start_utc,end_utc,notes)
               VALUES(?,?,'Revision1 large Git assets','ARCHIVE_MIGRATION',
               'Rockfish archive copy plus full-file SHA-256 and Git blob verification',
               'Rockfish',?,?, 'COMPLETED','PROVENANCE_PARTIAL',?,?,?)""",
            (RUN_ID, "MIGRATIONBENCH", STORAGE_ROOT, ROCKFISH_REL_ROOT, now, now, notes),
        )
        con.execute(
            """UPDATE workflow_run SET scheduler_state='COMPLETED',scientific_status='PROVENANCE_PARTIAL',
               end_utc=?,notes=? WHERE run_id=?""",
            (now, notes, RUN_ID),
        )
        con.execute(
            """INSERT OR IGNORE INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES(?,?,?,?,?,'archive-import-v1','COMPLETE',?,NULL,NULL,0,?,NULL,?,?,?)""",
            (SNAPSHOT_ID, STORAGE_ROOT, now, now,
             "Pinned Git asset index plus live Rockfish archive verification",
             len(rows), int(report["logical_bytes"]), str(MANIFEST), manifest_sha, notes),
        )
        con.execute(
            """UPDATE inventory_snapshot SET finished_utc=?,completion_state='COMPLETE',files_seen=?,
               logical_bytes=?,manifest_path=?,manifest_sha256=?,notes=? WHERE snapshot_id=?""",
            (now, len(rows), int(report["logical_bytes"]), str(MANIFEST), manifest_sha, notes, SNAPSHOT_ID),
        )

        asset_ids = []
        for row in rows:
            size = int(row["byte_size"])
            category = "models" if row["source_role"].startswith("MODEL_CHECKPOINT") else "figures"
            kind = "MODEL_CHECKPOINT" if category == "models" else "FIGURE"
            semantic_role = "model checkpoint" if category == "models" else "editable vector figure"
            retention = "PERMANENT_SOURCE" if category == "models" else "PUBLICATION_EVIDENCE"
            source_path = row["source_repository_path"]
            notes_text = (
                "archive_rel=%s; source_path=%s; source_commit=%s; git_blob_oid=%s; "
                "acquisition_method=%s; sha256=%s"
            ) % (
                row["archive_rel"], source_path, row["source_commit"],
                row["git_blob_oid"], row["acquisition_method"], row["sha256"],
            )
            if kind == "MODEL_CHECKPOINT":
                notes_text += (
                    "; canonical_model_id=MACE-CHECKPOINT-SHA256-%s; "
                    "checkpoint_identity_status=VERIFIED; training_lineage_status=PROVENANCE_PARTIAL"
                ) % row["sha256"][:16]
            if not row["git_blob_oid"]:
                notes_text += "; source was not tracked in candidate Git tree"
            artifact_key = "MIGRATIONBENCH-ROCKFISH-ARCHIVE:20261124:" + row["archive_rel"]
            artifact_id = ensure_artifact(
                con, "MIGRATIONBENCH", artifact_key, kind, semantic_role,
                row["sha256"], size, retention, notes_text,
            )
            rel = ROCKFISH_REL_ROOT + "/" + row["archive_rel"]
            path_id = add_location_and_observation(
                con, SNAPSHOT_ID, artifact_id, STORAGE_ROOT, rel, size, None,
                row["sha256"], now,
            )
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,'PROVENANCE',0,?)""",
                (RUN_ID, artifact_id, row["source_role"], "archive_path_id=%d; full-file verification passed" % path_id),
            )
            # Record the archived copy for any paper evidence that used the old Git path.
            for evidence in con.execute(
                """SELECT release_item_id,evidence_role FROM paper_release_item_evidence
                   WHERE artifact_id=? AND source_path=?""",
                (artifact_id, source_path),
            ).fetchall():
                con.execute(
                    """INSERT OR IGNORE INTO paper_release_item_evidence
                       (release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes)
                       VALUES(?,?,?,?,'PRESENT','ARCHIVED_COPY_AFTER_GIT_CLEANUP',?)""",
                    (evidence["release_item_id"], artifact_id, rel, evidence["evidence_role"],
                     "Rockfish archived copy; SHA-256=" + row["sha256"]),
                )
            asset_ids.append(artifact_id)

        for name, path, rel, notes_text in (
            ("MANIFEST", MANIFEST, ROCKFISH_REL_ROOT + "/ARCHIVE_MANIFEST.tsv", "Archive payload manifest."),
            ("VERIFICATION", REPORT, ROCKFISH_REL_ROOT + "/VERIFICATION_REPORT.json", "Full-file checksum verification report."),
            ("README", ARCHIVE / "README.md", ROCKFISH_REL_ROOT + "/README.md", "Archive restore and layout guide."),
            ("INDEX", DATA / "MIGRATIONBENCH_ROCKFISH_ARCHIVE_INDEX_20261124.md",
             "MigrationBench_data_archive/README.md", "Top-level MigrationBench and DUAL-X archive map."),
        ):
            source_sha = sha256_file(path)
            key = "MB-REV1-LARGE-ASSET-ARCHIVE-%s-20261124-%s" % (name, source_sha[:16])
            source_id, source_sha = register_source(
                con, key, path, "ROCKFISH_ARCHIVE_CONTROL_FILE",
                "Local captured copy of the Rockfish archive control file; remote archive path independently verified.",
                notes_text, now,
            )
            control_id = ensure_artifact(
                con, "MIGRATIONBENCH", "ARCHIVE-CONTROL:" + key,
                "ARCHIVE_CONTROL_FILE", "archive manifest/status/readme",
                source_sha, path.stat().st_size, "PUBLICATION_EVIDENCE",
                "sha256=%s; %s" % (source_sha, path),
            )
            path_id = ensure_storage_path(con, STORAGE_ROOT, rel, now)
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""",
                (control_id, path_id, now, now),
            )
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,'ARCHIVE_CONTROL','PROVENANCE',0,?)""",
                (RUN_ID, control_id, "registry_source_id=%s" % source_id),
            )

        con.commit()
        print("snapshot_id", SNAPSHOT_ID)
        print("payloads", len(rows))
        print("logical_bytes", int(report["logical_bytes"]))
        print("artifact_ids", asset_ids)
        print("manifest_sha256", manifest_sha)
        print("quick_check", con.execute("PRAGMA quick_check").fetchone()[0])
        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        print("foreign_key_violations", len(violations))
        if violations:
            raise RuntimeError("foreign key check failed: " + repr([tuple(v) for v in violations[:10]]))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
