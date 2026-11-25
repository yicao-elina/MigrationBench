#!/usr/bin/env python3
"""Commit a completed WSe2 WFC deletion receipt to the provenance catalog."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
RECEIPT = AUDIT / "phase2_scratch4/wse2_wfc_delete_receipt_20261122.jsonl"
BATCH = "RF-S4-WSE2-WFC-CLEANUP-20261122"
ROOT_ID = "LOCAL-MAIN"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main():
    events = [json.loads(line) for line in RECEIPT.read_text().splitlines() if line.strip()]
    preflight = [e for e in events if e.get("event") == "preflight_passed"]
    deleted = [e for e in events if e.get("event") == "deleted_verified"]
    complete = [e for e in events if e.get("event") == "complete"]
    if len(preflight) != 1 or len(deleted) != 8 or len(complete) != 1:
        raise SystemExit("receipt must contain one passed preflight, eight verified deletions, and one completion")
    if any(e.get("verified_missing") is not True for e in deleted):
        raise SystemExit("receipt contains an unverified deletion")
    expected_bytes = sum(e["size_bytes"] for e in deleted)
    if complete[0].get("bytes") != expected_bytes or complete[0].get("file_count") != 8:
        raise SystemExit("receipt summary does not match the individual records")
    receipt_hash = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    receipt_size = RECEIPT.stat().st_size

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute("SELECT * FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        if batch is None or batch["status"] != "EXECUTING" or batch["expected_bytes"] != expected_bytes:
            raise SystemExit("catalog batch is not executing or its expected byte count differs")
        prior = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", ("RF-S4-WSE2-WFC-DELETE-RECEIPT-20261122",)).fetchone()
        if prior:
            raise SystemExit("receipt artifact already exists; refusing duplicate update")

        for event in deleted:
            target = Path(event["path"])
            if not target.is_absolute() or not str(target).startswith("/scratch4/pclancy3/yi/3.WSe2/1.pwrelax/"):
                raise SystemExit("receipt path is outside the approved WSe2 subtree")
            rel = str(target).split("/scratch4/pclancy3/yi/", 1)[1]
            row = con.execute(
                """SELECT ci.artifact_id, ci.pre_delete_sha256, ci.pre_delete_size_bytes,
                          ci.pre_delete_mtime_ns, ci.deletion_state, l.path_id, l.availability
                   FROM cleanup_item ci JOIN artifact_location l USING (artifact_id)
                   JOIN storage_path sp USING (path_id) JOIN storage_root sr USING (root_id)
                   WHERE ci.cleanup_batch_id=? AND sr.root_id='RF-SCRATCH4-YI' AND sp.relative_path=?""",
                (BATCH, rel),
            ).fetchone()
            if row is None or row["deletion_state"] != "EVIDENCE_READY" or row["availability"] != "PRESENT":
                raise SystemExit("catalog path is missing or not in the expected pre-delete state: " + rel)
            if (row["pre_delete_sha256"], row["pre_delete_size_bytes"], row["pre_delete_mtime_ns"]) != (
                event["sha256"], event["size_bytes"], event["mtime_ns"]
            ):
                raise SystemExit("receipt metadata/hash does not match catalog for " + rel)
            con.execute(
                """UPDATE cleanup_item SET deletion_state='VERIFIED', deleted_utc=?, actual_bytes=?,
                          notes=notes || ' Guarded remote unlink receipt verified.'
                   WHERE cleanup_batch_id=? AND artifact_id=?""",
                (event["deleted_utc"], event["size_bytes"], BATCH, row["artifact_id"]),
            )
            con.execute("UPDATE artifact_location SET availability='REMOVED', last_verified_utc=? WHERE path_id=?", (event["deleted_utc"], row["path_id"]))

        local_root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
        if local_root is None:
            raise SystemExit("local audit root is not registered")
        relative = str(RECEIPT.relative_to(Path(local_root[0])))
        con.execute(
            """INSERT INTO artifact(project_id, artifact_key, artifact_kind, semantic_role, owner_status,
                      sha256, byte_size, scientific_status, retention_class, notes)
               VALUES(NULL, ?, 'CLEANUP_RECEIPT', 'verified WSe2 WFC deletion receipt', 'USER_CONFIRMED',
                      ?, ?, 'VERIFIED', 'PERMANENT_SOURCE',
                      'Records exact pre-delete hashes and post-unlink checks for all eight WSe2 WFC files.')""",
            ("RF-S4-WSE2-WFC-DELETE-RECEIPT-20261122", receipt_hash, receipt_size),
        )
        artifact_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        con.execute(
            "INSERT INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
            (ROOT_ID, relative, utc_now()),
        )
        path_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        con.execute(
            "INSERT INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?, 'PRESENT', ?, ?)",
            (artifact_id, path_id, utc_now(), utc_now()),
        )
        con.execute(
            """UPDATE cleanup_batch SET status='COMPLETED', actual_bytes=?, completed_utc=?, evidence_artifact_id=?,
                      authorization_note=authorization_note || ' All eight files were unlinked after live queue/hash/ownership/link-count checks; each path was verified missing.'
               WHERE cleanup_batch_id=? AND status='EXECUTING'""",
            (expected_bytes, complete[0]["utc"], artifact_id, BATCH),
        )
        if con.execute("SELECT changes()").fetchone()[0] != 1:
            raise SystemExit("failed to complete cleanup batch")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    verify = sqlite3.connect(DB)
    result = verify.execute("PRAGMA integrity_check").fetchone()[0]
    fk = verify.execute("PRAGMA foreign_key_check").fetchall()
    batch = verify.execute("SELECT status,expected_bytes,actual_bytes,completed_utc FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
    states = verify.execute("SELECT deletion_state,count(*),sum(actual_bytes) FROM cleanup_item WHERE cleanup_batch_id=? GROUP BY deletion_state", (BATCH,)).fetchall()
    verify.close()
    if result != "ok" or fk or batch[0] != "COMPLETED" or batch[1] != batch[2]:
        raise SystemExit("post-update SQL verification failed")
    print(json.dumps({"integrity_check": result, "foreign_key_violations": len(fk), "batch": batch, "items": states, "receipt_sha256": receipt_hash}, default=list))


if __name__ == "__main__":
    main()
