#!/usr/bin/env python3
"""Preflight/import the receipt and independent post-delete check for eight empty QE mixes."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
EVIDENCE_DIR = AUDIT / "phase2_data/wfc_mix_residual_20261124"
RECEIPT = EVIDENCE_DIR / "delete_empty_mix_receipt.json"
POSTCHECK = EVIDENCE_DIR / "postdelete_path_check_20261124.json"
BATCH = "RF-YI-QE-ZERO-MIX-CLEANUP-20261124"
OLD_BATCH = "RF-YI-QE-RESTART-CLEANUP-20261123"
ROOT_ID = "RF-DATA-YI"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_evidence():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    postcheck = json.loads(POSTCHECK.read_text(encoding="utf-8"))
    if receipt.get("operation") != "delete empty QE mixing restart placeholders":
        raise RuntimeError("unexpected cleanup operation")
    rows = receipt.get("files", [])
    if receipt.get("files_deleted") != 8 or receipt.get("logical_bytes_freed") != 0 or len(rows) != 8:
        raise RuntimeError("receipt count or byte total is inconsistent")
    if any(int(r["size_bytes"]) != 0 or r.get("sha256") != EMPTY_SHA256 for r in rows):
        raise RuntimeError("a pre-delete file was not verified as empty")
    if postcheck.get("path_count") != 8 or postcheck.get("present_count") != 0 or postcheck.get("absent_count") != 8 or postcheck.get("error_count") != 0:
        raise RuntimeError("independent post-delete check is not a clean 8/8 absence check")
    receipt_paths = {str(Path(r["path"])) for r in rows}
    post_paths = {r["path"] for r in postcheck.get("results", []) if r.get("state") == "ABSENT"}
    if receipt_paths != post_paths or len(receipt_paths) != 8:
        raise RuntimeError("receipt and post-delete paths differ")
    if postcheck.get("source_receipt_path") != str(RECEIPT.resolve()):
        raise RuntimeError("post-delete check points at an unexpected receipt")
    return receipt, postcheck, rows


def inspect(con, receipt, postcheck, rows):
    root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
    if not root:
        raise RuntimeError("missing storage root " + ROOT_ID)
    scoped_path = root[0].rstrip("/")
    existing_batch = con.execute("SELECT status FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
    if existing_batch and existing_batch[0] != "COMPLETED":
        raise RuntimeError("existing cleanup batch is not completed")
    if not con.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id=?", (OLD_BATCH,)).fetchone():
        raise RuntimeError("historical hold batch is missing")
    expected = []
    for row in rows:
        path = str(Path(row["path"]))
        if not path.startswith(scoped_path + "/"):
            raise RuntimeError("path is outside registered root: " + path)
        rel = path[len(scoped_path) + 1:]
        path_row = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, rel)).fetchone()
        if not path_row:
            raise RuntimeError("path missing from catalog: " + path)
        path_id = path_row[0]
        aliases = con.execute(
            """SELECT a.artifact_id,a.sha256,a.byte_size,al.availability
               FROM artifact_location al JOIN artifact a USING(artifact_id)
               WHERE al.path_id=?""", (path_id,)
        ).fetchall()
        old = con.execute(
            """SELECT ci.artifact_id,a.sha256,a.byte_size
               FROM cleanup_item ci JOIN artifact a USING(artifact_id)
               WHERE ci.cleanup_batch_id=? AND ci.disposition='HOLD'
                 AND EXISTS (SELECT 1 FROM artifact_location al WHERE al.artifact_id=a.artifact_id AND al.path_id=?)""",
            (OLD_BATCH, path_id),
        ).fetchall()
        if len(old) != 1:
            raise RuntimeError("expected one historical HOLD alias at " + path)
        if old[0]["byte_size"] not in (None, 0) or old[0]["sha256"] not in (None, EMPTY_SHA256):
            raise RuntimeError("historical alias metadata conflicts with empty file receipt")
        if not aliases or any(a["byte_size"] not in (None, 0) or a["sha256"] not in (None, EMPTY_SHA256) for a in aliases):
            raise RuntimeError("catalogued artifact content conflicts with empty file receipt")
        expected.append((path, path_id, old[0]["artifact_id"], aliases))
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply catalog updates in one transaction; default is read-only preflight")
    args = parser.parse_args()
    receipt, postcheck, rows = load_evidence()
    receipt_sha = sha256_file(RECEIPT)
    postcheck_sha = sha256_file(POSTCHECK)
    captured = receipt["utc"].replace("+00:00", "Z")
    verified = postcheck["captured_utc"].replace("+00:00", "Z")

    uri = "file:%s?mode=%s" % (DB, "rw" if args.apply else "ro")
    con = sqlite3.connect(uri, uri=True, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        if args.apply:
            con.execute("BEGIN IMMEDIATE")
        paths = inspect(con, receipt, postcheck, rows)
        report = {
            "mode": "APPLY" if args.apply else "READ_ONLY_PREFLIGHT",
            "cleanup_batch_id": BATCH,
            "paths": len(paths),
            "logical_bytes": 0,
            "allocated_bytes": None,
            "receipt_sha256": receipt_sha,
            "postcheck_sha256": postcheck_sha,
            "queue_evidence": "receipt summary only; raw queue snapshot is not retained",
            "quota_after": None,
            "preflight": "PASS",
        }
        if not args.apply:
            print(json.dumps(report, indent=2, sort_keys=True))
            return

        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
        source_ids = {}
        for label, path, digest, kind, stamp, notes in [
            ("receipt", RECEIPT, receipt_sha, "QE_ZERO_BYTE_MIX_CLEANUP_RECEIPT", captured,
             "Eight exact zero-byte QE .mixN paths; receipt includes pre-delete stat identity and empty-file hash. Queue evidence is summary-only; no raw queue snapshot."),
            ("postcheck", POSTCHECK, postcheck_sha, "QE_ZERO_BYTE_MIX_POSTDELETE_CHECK", verified,
             "Independent exact-path lstat check: all eight paths absent, zero errors."),
        ]:
            source = con.execute(
                """SELECT source_id,sha256,source_path FROM registry_source
                   WHERE source_path=? AND sha256=? AND registry_kind=? ORDER BY source_id LIMIT 1""",
                (str(path.resolve()), digest, kind),
            ).fetchone()
            if not source:
                prefix = "qe_zero_mix_cleanup_receipt" if label == "receipt" else "qe_zero_mix_postdelete_check"
                source_key = "%s:%s:%s" % (prefix, BATCH, digest[:20])
                con.execute(
                    """INSERT OR IGNORE INTO registry_source
                       (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                        immutable_snapshot,authority_note,notes)
                       VALUES(?,?,?,?,?,?,1,?,?)""",
                    (source_key, str(path.resolve()), kind, digest, stamp, stamp,
                     "User-directed cleanup; evidence is bounded to exact listed paths.", notes),
                )
                source = con.execute("SELECT source_id,sha256,source_path FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
            if source["sha256"] != digest or source["source_path"] != str(path.resolve()):
                raise RuntimeError("source registry identity conflict for " + label)
            source_ids[label] = source["source_id"]

        evidence = con.execute(
            """SELECT artifact_id FROM artifact WHERE artifact_key LIKE ? AND sha256=? AND byte_size=?
               ORDER BY artifact_id DESC LIMIT 1""",
            ("AUDIT-DELIVERABLE:%phase2_data/wfc_mix_residual_20261124/delete_empty_mix_receipt.json:%", receipt_sha, RECEIPT.stat().st_size),
        ).fetchone()
        if not evidence:
            raise RuntimeError("registered receipt evidence artifact is missing")
        con.execute(
            """INSERT OR IGNORE INTO cleanup_batch
               (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,
                expected_bytes,actual_bytes,completed_utc,evidence_artifact_id)
               VALUES(?,?,?,? ,?,'COMPLETED',?,0,0,?,?)""",
            (BATCH, captured, ROOT_ID, "USER_DIRECTED_EXACT_METADATA_VERIFIED_QE_RESTART_CLEANUP",
             "Remove only eight verified zero-byte QE mixing restart placeholders after the all-user queue check.",
             "User-directed restart cleanup authorization, 2026-11-24.", captured, evidence[0]),
        )
        con.execute("UPDATE cleanup_batch SET status='COMPLETED',expected_bytes=0,actual_bytes=0,completed_utc=?,evidence_artifact_id=? WHERE cleanup_batch_id=?",
                    (captured, evidence[0], BATCH))

        for row, (path, path_id, artifact_id, aliases) in zip(rows, paths):
            notes = (
                "Direct post-delete lstat confirmed ABSENT at %s. Pre-delete uid=%s gid=%s mode=%s inode=%s mtime_ns=%s; "
                "all-user queue check is summarized in receipt but raw queue output was not retained. Empty-file SHA-256=%s. "
                "Logical bytes freed=0; allocated bytes and quota delta are unmeasured."
            ) % (verified, row["uid"], row["gid"], row["mode"], row["inode"], row["mtime_ns"], row["sha256"])
            con.execute(
                """INSERT INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,pre_delete_sha256,
                    pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,deleted_utc,actual_bytes,
                    legacy_cleanup_source_id,legacy_cleanup_table,legacy_cleanup_row_key,notes)
                   VALUES(?,?,'DELETE',?,'CLEARED',?,0,?,'VERIFIED',?,0,?,?,?,?)
                   ON CONFLICT(cleanup_batch_id,artifact_id) DO UPDATE SET
                    disposition='DELETE',decision_reason=excluded.decision_reason,owner_clearance='CLEARED',
                    pre_delete_sha256=excluded.pre_delete_sha256,pre_delete_size_bytes=0,
                    pre_delete_mtime_ns=excluded.pre_delete_mtime_ns,deletion_state='VERIFIED',
                    deleted_utc=excluded.deleted_utc,actual_bytes=0,legacy_cleanup_source_id=excluded.legacy_cleanup_source_id,
                    legacy_cleanup_table=excluded.legacy_cleanup_table,legacy_cleanup_row_key=excluded.legacy_cleanup_row_key,
                    notes=excluded.notes""",
                (BATCH, artifact_id, "User-directed deletion of an exact empty QE .mixN restart path; no active matching WorkDir.",
                 EMPTY_SHA256, int(row["mtime_ns"]), captured, source_ids["receipt"],
                 "zero_byte_mix_cleanup_receipt", path, notes),
            )
            con.execute("UPDATE artifact SET sha256=?,byte_size=0 WHERE artifact_id=?", (EMPTY_SHA256, artifact_id))
            con.execute("UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE path_id=?", (verified, path_id))

        receipt_notes = (
            "Eight exact paths absent in independent lstat check at %s. Logical bytes freed=0. "
            "Allocated bytes and personal quota delta are unmeasured (NULL). Queue check succeeded per receipt summary; "
            "raw queue snapshot is not retained."
        ) % verified
        con.execute(
            """INSERT INTO cleanup_execution_receipt
               (cleanup_batch_id,receipt_source_id,observed_utc,execution_state,deleted_file_count,
                actual_logical_bytes,actual_allocated_bytes,queue_snapshot_source_id,notes)
               VALUES(?,?,?,'COMPLETED',8,0,NULL,NULL,?)
               ON CONFLICT(cleanup_batch_id,receipt_source_id) DO UPDATE SET
                observed_utc=excluded.observed_utc,execution_state='COMPLETED',deleted_file_count=8,
                actual_logical_bytes=0,actual_allocated_bytes=NULL,queue_snapshot_source_id=NULL,notes=excluded.notes""",
            (BATCH, source_ids["receipt"], captured, receipt_notes),
        )
        for label, relationship, note in [
            ("receipt", "VERIFIES_CLEANUP", "Eight zero-byte paths, exact pre-delete stat/hash evidence, and reported queue gate."),
            ("postcheck", "VERIFIES_POSTDELETE_ABSENCE", "Independent exact-path lstat verification: 8 absent, 0 present, 0 errors."),
        ]:
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'CLEANUP_EVIDENCE',?,'CLEANUP_BATCH',?,?,?)""",
                (source_ids[label], BATCH + ":" + label, BATCH, relationship, note),
            )
        con.commit()
        report["applied"] = True
        report["postcheck_captured_utc"] = verified
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception:
        if con.in_transaction:
            con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
