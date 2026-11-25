#!/usr/bin/env python3
"""Register the verified six-run Revision 1 MD evidence archive in SQLite."""

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
BUNDLE = DATA / "revision1_md_fig6_table_s8_20261124"
MANIFEST = BUNDLE / "ARCHIVE_MANIFEST.tsv"
SUMMARY = BUNDLE / "RUN_SUMMARY.tsv"
REPORT = BUNDLE / "VERIFICATION_REPORT.json"
README = BUNDLE / "README.md"
INDEX = DATA / "MIGRATIONBENCH_ROCKFISH_ARCHIVE_INDEX_20261124.md"
ROCKFISH_REL_ROOT = "MigrationBench_data_archive/revision1_md_fig6_table_s8_20261124"
SNAPSHOT_ID = "MB-ROCKFISH-MD-ARCHIVE-20261124"
RUN_ID_PREFIX = "MB-R1-MD-ARCHIVE-20261124:"
RELEASE_ID = "MB-REVISION1-SUBMITTED-20261110"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def root_for_path(con, full_path):
    matches = con.execute(
        "SELECT root_id,scoped_path FROM storage_root WHERE ?=scoped_path OR ? LIKE scoped_path || '/%'",
        (full_path, full_path),
    ).fetchall()
    if not matches:
        raise RuntimeError("no catalog storage root for source path: " + full_path)
    return max(matches, key=lambda row: len(row["scoped_path"]))


def kind_for_role(role):
    role = role.lower()
    if "trajectory" in role:
        return "TRAJECTORY", "MD coordinate trajectory segment", "REPRODUCIBLE_INPUT"
    if "model" in role:
        return "MODEL_CHECKPOINT", "model checkpoint used by the MD run", "PERMANENT_SOURCE"
    if "checkpoint" in role or "restart" in role:
        return "CHECKPOINT", "MD coordinate checkpoint", "REPRODUCIBLE_INPUT"
    if "structure" in role or "input" in role:
        return "COORDINATE_INPUT", "shared initial coordinate structure", "REPRODUCIBLE_INPUT"
    if "metric" in role or "result" in role:
        return "RESULT", "MD metrics output", "PUBLICATION_EVIDENCE"
    if "status" in role:
        return "RUN_STATUS", "MD run status and stop reason", "PUBLICATION_EVIDENCE"
    if "manifest" in role or "ledger" in role:
        return "RUN_PROVENANCE", "MD run manifest or continuation ledger", "PERMANENT_SOURCE"
    return "ARCHIVE_EVIDENCE", "verified MD archive evidence", "PUBLICATION_EVIDENCE"


def direction_for_role(role):
    role = role.lower()
    if "model" in role:
        return "MODEL"
    if "input" in role or "structure" in role or "source_smoke_checkpoint" in role:
        return "INPUT"
    if "trajectory" in role or "latest_coordinate_checkpoint" in role or "metric" in role:
        return "OUTPUT"
    if "status" in role or "ledger" in role or "manifest" in role:
        return "PROVENANCE"
    return "PROVENANCE"


def main():
    for path in (MANIFEST, SUMMARY, REPORT, README, INDEX):
        if not path.is_file():
            raise RuntimeError("MD archive control file is missing: " + str(path))
    rows = load_rows(MANIFEST)
    summary_rows = load_rows(SUMMARY)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("verification_status") != "PASS":
        raise RuntimeError("Rockfish MD archive verification report is not PASS")
    if len(rows) != int(report["payload_file_count"]):
        raise RuntimeError("MD archive manifest/report file count mismatch")
    if sum(int(row["byte_size"]) for row in rows) != int(report["payload_logical_bytes"]):
        raise RuntimeError("MD archive manifest/report byte total mismatch")
    if {row["run_label"] for row in summary_rows} != {run["run_label"] for run in report["runs"]}:
        raise RuntimeError("MD run summary/report run labels differ")
    if len(summary_rows) != len(report["runs"]):
        raise RuntimeError("MD run count differs from the verification report")
    if not rows or any(not row.get("source_copy_verification", "").startswith("PASS_SIZE_SHA256") for row in rows):
        raise RuntimeError("one or more archived MD payloads lack a full size/SHA copy check")
    for row in rows:
        if len(row["sha256"]) != 64 or int(row["byte_size"]) < 0:
            raise RuntimeError("invalid size or SHA-256 in MD archive manifest")
    for name, expected in report["control_files"].items():
        path = {"ARCHIVE_MANIFEST.tsv": MANIFEST, "README.md": README,
                "RUN_SUMMARY.tsv": SUMMARY}.get(name)
        if path is None or sha256_file(path) != expected:
            raise RuntimeError("MD archive control-file hash mismatch: " + name)

    report_runs = {row["run_label"]: row for row in report["runs"]}
    summary_by_label = {row["run_label"]: row for row in summary_rows}
    outcomes_by_label = {}
    for label in summary_by_label:
        outcome = dict(report_runs[label])
        outcome.update(summary_by_label[label])
        if outcome["outcome_label"] != report_runs[label]["outcome_label"]:
            raise RuntimeError("run outcome differs between verification and summary: " + label)
        outcomes_by_label[label] = outcome
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
            raise RuntimeError("Rockfish archive storage root is missing")

        notes = (
            "%d files totaling %d bytes; six grouped-A2 runs are a subset of the "
            "15-model Fig. 6/Table S8/R7 evidence cohort. One reached 1,000,000 "
            "steps and five stopped at scientific thresholds earlier; this is "
            "an archive/provenance record, not scientific acceptance."
        ) % (len(rows), int(report["payload_logical_bytes"]))
        con.execute(
            """INSERT OR IGNORE INTO workflow_run
               (run_id,project_id,run_label,workflow_type,method,system_key,source_root_id,
                source_relative_path,scheduler_state,scientific_status,start_utc,end_utc,notes)
               VALUES(?,'MIGRATIONBENCH','Revision 1 grouped-A2 MD archive import',
               'ARCHIVE_MIGRATION','Rockfish MD evidence copy with full-file SHA-256 verification',
               'Rockfish',?,?, 'COMPLETED','PROVENANCE_PARTIAL',?,?,?)""",
            (SNAPSHOT_ID, STORAGE_ROOT, ROCKFISH_REL_ROOT, now, now, notes),
        )
        con.execute(
            """UPDATE workflow_run SET scheduler_state='COMPLETED',scientific_status='PROVENANCE_PARTIAL',
               end_utc=?,notes=? WHERE run_id=?""",
            (now, notes, SNAPSHOT_ID),
        )
        con.execute(
            """INSERT OR IGNORE INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES(?,?,?,?,?,'md-archive-import-v1','COMPLETE',?,NULL,NULL,0,?,NULL,?,?,?)""",
            (SNAPSHOT_ID, STORAGE_ROOT, now, now,
             "Pinned archive manifest plus independent Rockfish destination size/SHA verification",
             len(rows), int(report["payload_logical_bytes"]), str(MANIFEST), manifest_sha, notes),
        )
        con.execute(
            """UPDATE inventory_snapshot SET finished_utc=?,completion_state='COMPLETE',files_seen=?,
               logical_bytes=?,manifest_path=?,manifest_sha256=?,notes=? WHERE snapshot_id=?""",
            (now, len(rows), int(report["payload_logical_bytes"]), str(MANIFEST), manifest_sha,
             notes, SNAPSHOT_ID),
        )

        # Keep a deterministic run crosswalk; reuse an exact source-path row if present.
        run_ids = {}
        run_rows = {}
        for label, item in summary_by_label.items():
            trajectory = next(r for r in rows if r["run_label"] == label and "trajectory" in r["role"].lower())
            run_dir = str(Path(trajectory["source_path"]).parent)
            root = root_for_path(con, run_dir)
            source_rel = run_dir[len(root["scoped_path"].rstrip("/")) + 1:]
            existing = con.execute(
                """SELECT run_id,source_root_id,source_relative_path FROM workflow_run
                   WHERE project_id='MIGRATIONBENCH' AND run_label=?""", (label,),
            ).fetchall()
            exact = [r for r in existing if r["source_root_id"] == root["root_id"]
                     and r["source_relative_path"] == source_rel]
            if len(exact) > 1:
                raise RuntimeError("duplicate exact workflow_run rows for " + label)
            if exact:
                run_id = exact[0]["run_id"]
            else:
                run_id = RUN_ID_PREFIX + label
                outcome = outcomes_by_label[label]
                run_notes = (
                    "Source run crosswalk from the 2026-11-24 verified MD archive. "
                    "outcome=%s; last_step=%s/%s; scientific_stop_reason=%s. "
                    "The archived set is six of the 15-model paper cohort; no "
                    "one-nanosecond completion or scientific acceptance is inferred."
                ) % (outcome["outcome_label"], outcome["last_step"], outcome["target_step"],
                     outcome["scientific_stop_reason"])
                con.execute(
                    """INSERT OR IGNORE INTO workflow_run
                       (run_id,project_id,run_label,workflow_type,method,system_key,source_root_id,
                        source_relative_path,scheduler_state,scientific_status,notes)
                       VALUES(?,'MIGRATIONBENCH',?,'MLIP_MD',?,'Sb2Te3_2L_octo_Cr2',?,?,'COMPLETED',
                       'PROVENANCE_PARTIAL',?)""",
                    (run_id, label,
                     "Grouped-A2 FT-600K/FT-MultiT continuation beginning from step 220000",
                     root["root_id"], source_rel, run_notes),
                )
            run_ids[label] = run_id
            run_rows[label] = item

        artifact_by_payload = {}
        source_by_hash_path = {}
        for row in rows:
            size = int(row["byte_size"])
            kind, semantic_role, retention = kind_for_role(row["role"])
            label = row["run_label"]
            outcome = outcomes_by_label.get(label)
            notes_text = (
                "run_label=%s; archive_rel=%s; source_path=%s; role=%s; "
                "sha256=%s; source_copy_verification=%s"
            ) % (label, row["archive_relative_path"], row["source_path"], row["role"],
                 row["sha256"], row["source_copy_verification"])
            if outcome:
                notes_text += "; outcome=%s; last_step=%s/%s; reason=%s" % (
                    outcome["outcome_label"], outcome["last_step"], outcome["target_step"],
                    outcome["scientific_stop_reason"])
            key = "MIGRATIONBENCH-R1-MD-ARCHIVE:20261124:" + row["archive_relative_path"]
            artifact_id = ensure_artifact(
                con, "MIGRATIONBENCH", key, kind, semantic_role, row["sha256"],
                size, retention, notes_text,
            )
            archive_rel = ROCKFISH_REL_ROOT + "/" + row["archive_relative_path"]
            archive_path_id = add_location_and_observation(
                con, SNAPSHOT_ID, artifact_id, STORAGE_ROOT, archive_rel,
                size, None, row["sha256"], now,
            )
            artifact_by_payload[(label, row["role"], row["source_path"], row["sha256"])] = artifact_id
            source_by_hash_path[(row["source_path"], row["sha256"])] = artifact_id
            if label in run_ids:
                direction = direction_for_role(row["role"])
                role_name = "ARCHIVED_" + row["role"].upper()
                con.execute(
                    """INSERT OR IGNORE INTO run_artifact
                       (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                       VALUES(?,?,?,?,0,?)""",
                    (run_ids[label], artifact_id, role_name, direction,
                     "Archive path=%s; copy SHA-256 and size verification passed." % archive_rel),
                )
            source_root = root_for_path(con, row["source_path"])
            source_rel = row["source_path"][len(source_root["scoped_path"].rstrip("/")) + 1:]
            source_path_id = ensure_storage_path(con, source_root["root_id"], source_rel, now)
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""",
                (artifact_id, source_path_id, row["captured_utc"], row["captured_utc"]),
            )
            con.execute(
                """UPDATE artifact_location SET availability='PRESENT',last_verified_utc=?
                   WHERE artifact_id=? AND path_id=?""",
                (row["captured_utc"], artifact_id, source_path_id),
            )

        # Link source model/input hashes to each run. Payloads are archived once,
        # while multiple runs may refer to the same shared input structure.
        for label, item in summary_by_label.items():
            for role, source_path, sha in (
                ("MD_MODEL_CHECKPOINT", item["model_source_path"], item["model_sha256"]),
                ("MD_INITIAL_STRUCTURE", item["input_source_path"], item["input_sha256"]),
            ):
                artifact_id = source_by_hash_path.get((source_path, sha))
                if artifact_id is None:
                    existing = con.execute(
                        "SELECT artifact_id FROM artifact WHERE project_id='MIGRATIONBENCH' AND sha256=? ORDER BY artifact_id",
                        (sha,),
                    ).fetchall()
                    if len(existing) == 1:
                        artifact_id = existing[0][0]
                    else:
                        artifact_key = "MIGRATIONBENCH-R1-MD-SOURCE:%s:%s" % (role, sha)
                        artifact_id = ensure_artifact(
                            con, "MIGRATIONBENCH", artifact_key,
                            "MODEL_CHECKPOINT" if role == "MD_MODEL_CHECKPOINT" else "COORDINATE_INPUT",
                            role.lower().replace("_", " "), sha, None,
                            "PERMANENT_SOURCE" if role == "MD_MODEL_CHECKPOINT" else "REPRODUCIBLE_INPUT",
                            "Source identity recorded by the verified archive run summary; source bytes were not copied into the archive bundle.",
                        )
                root = root_for_path(con, source_path)
                source_rel = source_path[len(root["scoped_path"].rstrip("/")) + 1:]
                source_path_id = ensure_storage_path(con, root["root_id"], source_rel, now)
                con.execute(
                    """INSERT OR IGNORE INTO artifact_location
                       (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                       VALUES(?,?,'PRESENT',?,?)""",
                    (artifact_id, source_path_id, now, now),
                )
                con.execute(
                    """UPDATE artifact_location SET availability='PRESENT',last_verified_utc=?
                       WHERE artifact_id=? AND path_id=?""",
                    (now, artifact_id, source_path_id),
                )
                con.execute(
                    """INSERT OR IGNORE INTO run_artifact
                       (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                       VALUES(?,?,?, ?,0,?)""",
                    (run_ids[label], artifact_id, role, "MODEL" if role == "MD_MODEL_CHECKPOINT" else "INPUT",
                     "Source path and SHA-256 are recorded in RUN_SUMMARY.tsv; a central archive copy is linked when present."),
                )

        control_paths = [
            ("ARCHIVE_MANIFEST.tsv", MANIFEST, ROCKFISH_REL_ROOT + "/ARCHIVE_MANIFEST.tsv"),
            ("RUN_SUMMARY.tsv", SUMMARY, ROCKFISH_REL_ROOT + "/RUN_SUMMARY.tsv"),
            ("VERIFICATION_REPORT.json", REPORT, ROCKFISH_REL_ROOT + "/VERIFICATION_REPORT.json"),
            ("README.md", README, ROCKFISH_REL_ROOT + "/README.md"),
        ]
        control_artifact_ids = {}
        for name, path, rel in control_paths:
            digest = sha256_file(path)
            source_key = "MB-REV1-MD-ARCHIVE-%s-20261124-%s" % (name.replace(".", "_"), digest[:16])
            source_id, source_sha = register_source(
                con, source_key, path, "ROCKFISH_MD_ARCHIVE_CONTROL_FILE",
                "Captured archive control file; remote payload verification is described by the pinned report.",
                "archive_path=%s; sha256=%s" % (rel, digest), now,
            )
            artifact_id = ensure_artifact(
                con, "MIGRATIONBENCH", "ARCHIVE-CONTROL:" + source_key,
                "ARCHIVE_CONTROL_FILE", "Revision 1 MD archive control file",
                source_sha, path.stat().st_size, "PUBLICATION_EVIDENCE",
                "sha256=%s; local_capture=%s" % (source_sha, path),
            )
            path_id = ensure_storage_path(con, STORAGE_ROOT, rel, now)
            control_artifact_ids[rel] = artifact_id
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""",
                (artifact_id, path_id, now, now),
            )
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,'ARCHIVE_CONTROL','PROVENANCE',0,?)""",
                (SNAPSHOT_ID, artifact_id, "registry_source_id=%s" % source_id),
            )

        release_items = con.execute(
            """SELECT release_item_id,source_item_id,section,panel
               FROM paper_release_item WHERE release_id=?
                 AND (source_item_id='Fig6' OR source_item_id='Table S8' OR source_item_id='Table R7')""",
            (RELEASE_ID,),
        ).fetchall()
        if not release_items:
            raise RuntimeError("submitted release does not expose Fig6/Table S8/Table R7 catalog rows")
        for item in release_items:
            selected = []
            for row in rows:
                role = row["role"].lower()
                if item["source_item_id"] == "Fig6" and "trajectory" not in role and "model" not in role and "structure" not in role and "input" not in role:
                    continue
                selected.append(row)
            for row in selected:
                artifact_id = artifact_by_payload[(row["run_label"], row["role"], row["source_path"], row["sha256"])]
                source_path = "/scratch16/pclancy3/yi/" + ROCKFISH_REL_ROOT + "/" + row["archive_relative_path"]
                evidence_role = "MD_TRAJECTORY_ARCHIVE_SUBSET" if "trajectory" in row["role"].lower() else "MD_CONTINUATION_EVIDENCE_SUBSET"
                relationship = "ARCHIVED_SIX_RUN_SUBSET_OF_15_MODEL_COHORT"
                evidence_notes = (
                    "Selected grouped-A2 run only; not the complete 15-model Fig. 6/Table S8/R7 cohort. "
                    "See RUN_SUMMARY.tsv for target-vs-threshold outcomes and the bundle manifest for hashes."
                )
                con.execute(
                    """INSERT OR IGNORE INTO paper_release_item_evidence
                       (release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes)
                       VALUES(?,?,?,?,'PRESENT',?,?)""",
                    (item["release_item_id"], artifact_id, source_path, evidence_role,
                     relationship, evidence_notes),
                )
            for rel, artifact_id in control_artifact_ids.items():
                source_path = "/scratch16/pclancy3/yi/" + rel
                con.execute(
                    """INSERT OR IGNORE INTO paper_release_item_evidence
                       (release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes)
                       VALUES(?,?,?,'MD_ARCHIVE_CONTROL_FILE','PRESENT',?,?)""",
                    (item["release_item_id"], artifact_id, source_path,
                     "ARCHIVED_SIX_RUN_SUBSET_OF_15_MODEL_COHORT",
                     "Control file for the six-run subset; it does not contain the complete 15-model cohort."),
                )

        for name, path, rel in (("ARCHIVE_INDEX", INDEX, "MigrationBench_data_archive/README.md"),):
            digest = sha256_file(path)
            source_key = "MB-REV1-MD-ARCHIVE-INDEX-20261124-" + digest[:16]
            source_id, source_sha = register_source(
                con, source_key, path, "ROCKFISH_ARCHIVE_INDEX",
                "Top-level Rockfish archive index for MigrationBench and DUAL-X bundles.",
                rel, now,
            )
            artifact_id = ensure_artifact(
                con, "MIGRATIONBENCH", "ARCHIVE-CONTROL:" + source_key,
                "ARCHIVE_CONTROL_FILE", "top-level archive index", source_sha,
                path.stat().st_size, "PUBLICATION_EVIDENCE", str(path),
            )
            path_id = ensure_storage_path(con, STORAGE_ROOT, rel, now)
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""",
                (artifact_id, path_id, now, now),
            )
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,'ARCHIVE_INDEX','PROVENANCE',0,?)""",
                (SNAPSHOT_ID, artifact_id, "registry_source_id=%s" % source_id),
            )

        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign-key check failed during MD archive import")
        con.commit()
        print("snapshot_id", SNAPSHOT_ID)
        print("payload_files", len(rows))
        print("logical_bytes", int(report["payload_logical_bytes"]))
        print("run_count", len(run_ids))
        print("release_item_links", len(release_items))
        print("manifest_sha256", manifest_sha)
        print("quick_check", con.execute("PRAGMA quick_check").fetchone()[0])
        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        print("foreign_key_violations", len(violations))
        if violations:
            raise RuntimeError("foreign-key check failed")
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
