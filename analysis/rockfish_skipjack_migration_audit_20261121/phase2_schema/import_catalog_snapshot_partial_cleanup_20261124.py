#!/usr/bin/env python3
"""Preflight/import the guarded removal of two superseded Rockfish SQLite partial transfers."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
EVIDENCE = AUDIT / "phase2_data/catalog_snapshot_recovery_20261124"
PREDELETE = EVIDENCE / "partial_files_predelete_20261124.json"
ARCHIVE = EVIDENCE / "catalog_snapshot_precleanup_archive_receipt_20261124.json"
RECEIPT = EVIDENCE / "partial_files_cleanup_receipt_20261124.json"
BATCH = "RF-SCRATCH16-SQLITE-PARTIAL-SNAPSHOT-CLEANUP-20261124"
ROOT_ID = "RF-SCRATCH16-YI"
PROJECT_ID = "ROCKFISH_STORAGE_AUDIT"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs():
    pre = json.loads(PREDELETE.read_text(encoding="utf-8"))
    arc = json.loads(ARCHIVE.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if pre.get("uid") != 3901 or len(pre.get("files", [])) != 2:
        raise RuntimeError("predelete capture does not describe two UID 3901 files")
    if receipt.get("files_deleted") != 2 or len(receipt.get("files", [])) != 2:
        raise RuntimeError("cleanup receipt does not verify two deletions")
    if receipt.get("logical_bytes_freed") != sum(f["size_bytes"] for f in pre["files"]):
        raise RuntimeError("logical-byte total differs from predelete file list")
    if receipt.get("allocated_bytes_by_predelete_stat") != sum(f["allocated_bytes"] for f in pre["files"]):
        raise RuntimeError("allocated-byte total differs from predelete stat observations")
    expected = {f["path"]: f for f in pre["files"]}
    observed = {f["path"]: f for f in receipt["files"]}
    if set(expected) != set(observed) or any(observed[p].get("state") != "ABSENT" for p in expected):
        raise RuntimeError("predelete and postdelete path sets do not reconcile")
    remote = arc.get("remote", {})
    verified = receipt.get("verified_replacement_snapshot", {})
    if remote.get("sha256") != verified.get("sha256") or remote.get("final_path") != verified.get("path"):
        raise RuntimeError("cleanup receipt replacement snapshot differs from archived snapshot receipt")
    if remote.get("quick_check") != "ok" or remote.get("foreign_key_violations") != 0:
        raise RuntimeError("replacement snapshot was not validated")
    if receipt.get("queue_check", {}).get("exit_code") != 0 or receipt["queue_check"].get("matching_workdirs"):
        raise RuntimeError("queue gate failed or overlapped archive directory")
    return pre, arc, receipt


def preflight(con, pre, arc, receipt):
    root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
    if not root:
        raise RuntimeError("missing storage root " + ROOT_ID)
    scope = root[0].rstrip("/")
    batch = con.execute("SELECT status FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
    if batch and batch[0] != "COMPLETED":
        raise RuntimeError("existing partial-snapshot cleanup batch is not completed")
    evidence_artifact = con.execute(
        """SELECT artifact_id FROM artifact WHERE artifact_key LIKE ? AND sha256=? AND byte_size=?
           ORDER BY artifact_id DESC LIMIT 1""",
        ("AUDIT-DELIVERABLE:%phase2_data/catalog_snapshot_recovery_20261124/partial_files_cleanup_receipt_20261124.json:%",
         sha256(RECEIPT), RECEIPT.stat().st_size),
    ).fetchone()
    if not evidence_artifact:
        raise RuntimeError("cleanup receipt must first be registered as an audit deliverable")
    plan = []
    for row in pre["files"]:
        path = row["path"]
        if not path.startswith(scope + "/"):
            raise RuntimeError("candidate outside registered storage root: " + path)
        rel = path[len(scope) + 1:]
        existing = con.execute(
            "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, rel)
        ).fetchone()
        artifact_key = "SQLITE-SNAPSHOT-PARTIAL:%s:%s" % (Path(path).name, row["sha256"][:16])
        if existing:
            links = con.execute("""SELECT a.artifact_key,a.sha256,a.byte_size,al.availability
                FROM artifact_location al JOIN artifact a USING(artifact_id) WHERE al.path_id=?""", (existing[0],)).fetchall()
            if links and not (len(links) == 1 and links[0]["artifact_key"] == artifact_key
                and links[0]["sha256"] == row["sha256"] and links[0]["byte_size"] == row["size_bytes"]
                and links[0]["availability"] == "REMOVED" and batch and batch[0] == "COMPLETED"):
                raise RuntimeError("unexpected existing artifact location for candidate: " + path)
        plan.append((path, rel, row))
    return evidence_artifact[0], plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply the verified cleanup batch in one SQLite transaction")
    args = parser.parse_args()
    pre, arc, receipt = load_inputs()
    uri = "file:%s?mode=%s" % (DB, "rw" if args.apply else "ro")
    con = sqlite3.connect(uri, uri=True, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        if args.apply:
            con.execute("BEGIN IMMEDIATE")
        evidence_artifact_id, plan = preflight(con, pre, arc, receipt)
        if not args.apply:
            print(json.dumps({"mode":"READ_ONLY_PREFLIGHT","cleanup_batch_id":BATCH,
                "files":len(plan),"logical_bytes":sum(x[2]["size_bytes"] for x in plan),
                "allocated_bytes_by_stat":sum(x[2]["allocated_bytes"] for x in plan),
                "replacement_snapshot_sha256":arc["remote"]["sha256"],
                "cleanup_receipt_sha256":sha256(RECEIPT),"preflight":"PASS"},indent=2,sort_keys=True))
            return

        source_specs = [
            ("predelete", PREDELETE, sha256(PREDELETE), "SQLITE_SNAPSHOT_PARTIAL_PREDELETE_CAPTURE", pre["captured_utc"],
             "Exact lstat/hash capture of two same-owner incomplete snapshot transfers; all-user queue and process checks recorded."),
            ("cleanup", RECEIPT, sha256(RECEIPT), "SQLITE_SNAPSHOT_PARTIAL_CLEANUP_RECEIPT", receipt["captured_utc"],
             "Exact guarded removal receipt for the two superseded transfer partials; includes immediate queue summary and absence checks."),
            ("replacement", ARCHIVE, sha256(ARCHIVE), "SQLITE_CATALOG_SNAPSHOT_ARCHIVE_RECEIPT", arc["remote"]["verified_utc"],
             "Consistent SQLite backup installed by atomic rename; remote size/SHA/quick_check/foreign_key_check validated."),
        ]
        src_ids = {}
        for label, path, digest, kind, stamp, notes in source_specs:
            key = "sqlite_snapshot_partial_%s:%s:%s" % (label, BATCH, digest[:20])
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                    immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (key, str(path.resolve()), kind, digest, stamp, stamp,
                 "User-authorized cleanup of exact temporary SQLite transfer files after verified replacement archive.", notes),
            )
            src = con.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
            if src["source_path"] != str(path.resolve()) or src["sha256"] != digest:
                raise RuntimeError("registry source identity conflict: " + label)
            src_ids[label] = src["source_id"]

        created = pre["captured_utc"]
        completed = receipt["captured_utc"]
        logical = sum(row["size_bytes"] for row in pre["files"])
        allocated = sum(row["allocated_bytes"] for row in pre["files"])
        con.execute(
            """INSERT OR IGNORE INTO cleanup_batch
               (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,
                authorization_note,expected_bytes,actual_bytes,completed_utc,evidence_artifact_id)
               VALUES(?,?,?,?,?,'COMPLETED',?,?,?,?,?)""",
            (BATCH, created, ROOT_ID, "USER_DIRECTED_EXACT_HASHED_TEMP_TRANSFER_CLEANUP",
             "Remove only two same-owner incomplete/corrupt SQLite transfer partials after a verified consistent replacement snapshot was atomically archived.",
             "User-directed request to remove outdated/duplicate temporary large files after preserving reproducible SQL state.",
             logical, logical, completed, evidence_artifact_id),
        )
        con.execute("UPDATE cleanup_batch SET status='COMPLETED',expected_bytes=?,actual_bytes=?,completed_utc=?,evidence_artifact_id=? WHERE cleanup_batch_id=?",
                    (logical, logical, completed, evidence_artifact_id, BATCH))

        for path, relative, row in plan:
            artifact_key = "SQLITE-SNAPSHOT-PARTIAL:%s:%s" % (Path(path).name, row["sha256"][:16])
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                    scientific_status,retention_class,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (PROJECT_ID, artifact_key, "SQLITE_SNAPSHOT_TRANSFER_PARTIAL",
                 "Incomplete or corrupt temporary transfer of the provenance catalog; superseded by verified consistent snapshot.",
                 "USER_CONFIRMED", row["sha256"], row["size_bytes"], "UNKNOWN", "TEMPORARY",
                 "Pre-delete content hash and stat identity recorded; no consumer or active WorkDir; exact path was removed after replacement archive validation."),
            )
            artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()[0]
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (ROOT_ID, relative, created))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, relative)).fetchone()[0]
            con.execute("""INSERT OR IGNORE INTO artifact_location
                (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                VALUES(?,?,'PRESENT',?,?)""", (artifact_id, path_id, created, created))
            note = ("Verified ABSENT at %s after unlink; pre-delete uid=%s gid=%s mode=%s device=%s inode=%s "
                    "nlink=%s mtime_ns=%s; SHA-256=%s; logical bytes=%s; allocated bytes by pre-delete st_blocks=%s; "
                    "replacement snapshot path=%s SHA-256=%s quick_check=ok foreign_key_violations=0. "
                    "Quota delta was not measured.") % (
                completed, row["uid"], row["gid"], row["mode"], row["device"], row["inode"], row["nlink"],
                row["mtime_ns"], row["sha256"], row["size_bytes"], row["allocated_bytes"],
                arc["remote"]["final_path"], arc["remote"]["sha256"])
            con.execute(
                """INSERT INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,
                    pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,
                    deleted_utc,actual_bytes,legacy_cleanup_source_id,legacy_cleanup_table,
                    legacy_cleanup_row_key,notes)
                   VALUES(?,?,'DELETE',?,'CLEARED',?,?,?,'VERIFIED',?,?,?,'sqlite_snapshot_partial_cleanup_receipt',?,?)
                   ON CONFLICT(cleanup_batch_id,artifact_id) DO UPDATE SET
                    disposition='DELETE',decision_reason=excluded.decision_reason,owner_clearance='CLEARED',
                    pre_delete_sha256=excluded.pre_delete_sha256,pre_delete_size_bytes=excluded.pre_delete_size_bytes,
                    pre_delete_mtime_ns=excluded.pre_delete_mtime_ns,deletion_state='VERIFIED',
                    deleted_utc=excluded.deleted_utc,actual_bytes=excluded.actual_bytes,
                    legacy_cleanup_source_id=excluded.legacy_cleanup_source_id,
                    legacy_cleanup_table=excluded.legacy_cleanup_table,legacy_cleanup_row_key=excluded.legacy_cleanup_row_key,
                    notes=excluded.notes""",
                (BATCH, artifact_id, "User-directed removal of an exact hash-verified superseded SQLite transfer partial.",
                 row["sha256"], row["size_bytes"], row["mtime_ns"], completed, row["size_bytes"],
                 src_ids["cleanup"], path, note),
            )
            con.execute("UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE path_id=?", (completed, path_id))

        con.execute(
            """INSERT INTO cleanup_execution_receipt
               (cleanup_batch_id,receipt_source_id,observed_utc,execution_state,deleted_file_count,
                actual_logical_bytes,actual_allocated_bytes,queue_snapshot_source_id,notes)
               VALUES(?,?,?,'COMPLETED',2,?,?,?,?)
               ON CONFLICT(cleanup_batch_id,receipt_source_id) DO UPDATE SET
                observed_utc=excluded.observed_utc,execution_state='COMPLETED',deleted_file_count=2,
                actual_logical_bytes=excluded.actual_logical_bytes,actual_allocated_bytes=excluded.actual_allocated_bytes,
                queue_snapshot_source_id=excluded.queue_snapshot_source_id,notes=excluded.notes""",
            (BATCH, src_ids["cleanup"], completed, logical, allocated, src_ids["cleanup"],
             "Two exact temporary files absent post-delete; logical bytes from lstat sizes and allocated bytes from pre-delete st_blocks×512. Immediate all-user squeue succeeded with no WorkDir match; raw full queue was not retained. No personal quota delta was measured."),
        )
        for label, relationship, text in [
            ("predelete", "PRECHECKS_CLEANUP", "Exact pre-delete stat/hash, identity, process check, and queue check source."),
            ("cleanup", "VERIFIES_CLEANUP", "Exact two-path deletion receipt and post-delete absence evidence."),
            ("replacement", "VERIFIES_REPLACEMENT_ARCHIVE", "Consistent replacement SQLite database archived and remotely verified before partial deletion."),
        ]:
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'CLEANUP_EVIDENCE',?,'CLEANUP_BATCH',?,?,?)""",
                (src_ids[label], BATCH + ":" + label, BATCH, relationship, text),
            )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign key violation during snapshot-partial cleanup import")
        con.commit()
        print(json.dumps({"mode":"APPLY","cleanup_batch_id":BATCH,"files":2,
            "logical_bytes":logical,"allocated_bytes_by_stat":allocated,
            "sources":src_ids,"quick_check":con.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_violations":len(con.execute("PRAGMA foreign_key_check").fetchall())},indent=2,sort_keys=True))
    except Exception:
        if con.in_transaction:
            con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
