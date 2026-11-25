#!/usr/bin/env python3
"""Validate a guarded deletion receipt and reconcile it into the SQL catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
LOCAL_ROOT_ID = "LOCAL-MAIN"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iso_utc(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def receipt_records(path: Path) -> tuple[list[dict], str, int | None]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        preflight = next((event for event in events if event.get("event") == "preflight_passed"), None)
        complete = next((event for event in reversed(events) if event.get("event") == "complete"), None)
        if not preflight or preflight.get("queue_overlaps") != 0:
            raise RuntimeError("JSONL receipt has no passing zero-overlap queue preflight")
        records = [event for event in events if event.get("event") == "deleted_verified" and event.get("verified_missing") is True]
        if not complete or complete.get("file_count") != len(records):
            raise RuntimeError("JSONL receipt completion count does not match per-file deletion receipts")
        completed = iso_utc(complete.get("utc"))
        if complete.get("logical_bytes") != sum(int(row["size_bytes"]) for row in records):
            raise RuntimeError("JSONL receipt logical byte total does not match per-file records")
        return records, completed, preflight.get("queue_overlaps")

    payload = json.loads(raw)
    if payload.get("status") != "DELETED_AND_VERIFIED" or payload.get("errors"):
        raise RuntimeError("JSON receipt is not a complete verified deletion")
    if payload.get("queue_status") != "CHECKED" or payload.get("queue_overlap_count") != 0:
        raise RuntimeError("JSON receipt has no passing zero-overlap queue check")
    records = []
    for row in payload.get("deleted", []):
        records.append({
            "path": row["candidate"],
            "size_bytes": row["logical_bytes"],
            "allocated_bytes": row["allocated_bytes"],
            "sha256": row["sha256"],
            "deleted_utc": row.get("deleted_utc", payload.get("timestamp_utc")),
            "verified_missing": True,
            "event": "deleted_verified",
        })
    if payload.get("verified_count") != len(records):
        raise RuntimeError("JSON receipt file count does not match its deleted rows")
    return records, iso_utc(payload.get("timestamp_utc")), payload.get("queue_overlap_count")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--quota-before", required=True, type=Path)
    parser.add_argument("--quota-after", required=True, type=Path)
    parser.add_argument("--quota-root", choices=("RF-DATA-FS", "RF-SCRATCH4-FS"), required=True)
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    if not receipt.is_file():
        raise SystemExit("receipt file does not exist: " + str(receipt))
    records, completed_utc, queue_overlaps = receipt_records(receipt)
    if not records or queue_overlaps != 0:
        raise RuntimeError("Receipt contains no verified deletions or a queue overlap")
    digest = sha256(receipt)
    source_key = "cleanup_execution_receipt_" + args.batch

    con = sqlite3.connect(DB, timeout=120)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute(
            "SELECT status,expected_bytes,source_root_id FROM cleanup_batch WHERE cleanup_batch_id=?",
            (args.batch,),
        ).fetchone()
        if not batch or batch["status"] != "EXECUTING":
            raise RuntimeError("Cleanup batch must be EXECUTING before receipt reconciliation")
        expected = con.execute(
            """SELECT a.artifact_id,ci.pre_delete_sha256,ci.pre_delete_size_bytes,
                      sr.scoped_path || CASE WHEN sp.relative_path='' THEN '' ELSE '/' || sp.relative_path END AS path
               FROM cleanup_item ci JOIN artifact a USING(artifact_id)
               JOIN artifact_location al USING(artifact_id) JOIN storage_path sp USING(path_id)
               JOIN storage_root sr USING(root_id)
               WHERE ci.cleanup_batch_id=? AND ci.disposition='DELETE'
                 AND ci.deletion_state='EVIDENCE_READY' AND al.availability='PRESENT'""",
            (args.batch,),
        ).fetchall()
        expected_by_path = {row["path"]: row for row in expected}
        if len(expected_by_path) != len(expected):
            raise RuntimeError("Catalog has multiple delete locations for a cleanup candidate")
        actual_by_path = {}
        for row in records:
            path = row.get("path")
            if not path or path in actual_by_path:
                raise RuntimeError("Receipt has a missing or duplicate path")
            actual_by_path[path] = row
        if set(expected_by_path) != set(actual_by_path):
            missing = sorted(set(expected_by_path) - set(actual_by_path))[:10]
            extra = sorted(set(actual_by_path) - set(expected_by_path))[:10]
            raise RuntimeError(f"Receipt paths differ from staged cleanup rows; missing={missing}, extra={extra}")

        logical_total = 0
        allocated_total = 0
        for path, row in actual_by_path.items():
            staged = expected_by_path[path]
            size = int(row.get("size_bytes", row.get("logical_bytes", -1)))
            file_hash = row.get("sha256")
            if size != staged["pre_delete_size_bytes"] or file_hash != staged["pre_delete_sha256"]:
                raise RuntimeError("Receipt size/hash differs from SQL pre-delete evidence: " + path)
            logical_total += size
            allocated_total += int(row.get("allocated_bytes", size))
        if batch["expected_bytes"] != logical_total:
            raise RuntimeError("Receipt total differs from cleanup_batch.expected_bytes")

        quota_ids = []
        for quota_path in (args.quota_before.resolve(), args.quota_after.resolve()):
            source = con.execute("SELECT source_id,sha256 FROM registry_source WHERE source_path=?", (str(quota_path),)).fetchone()
            if not source:
                raise RuntimeError("Quota evidence has not been imported into registry_source: " + str(quota_path))
            quota = con.execute(
                """SELECT quota_observation_id FROM quota_observation
                   WHERE raw_output_path=? AND root_id=? AND quota_subject_kind='GROUP' AND quota_subject='pclancy3'
                   ORDER BY quota_observation_id DESC LIMIT 1""",
                (str(quota_path), args.quota_root),
            ).fetchone()
            if not quota:
                raise RuntimeError("Group quota observation is missing for " + str(quota_path))
            quota_ids.append(quota[0])

        prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
        expected_source = (str(receipt), digest)
        if prior and (prior[0], prior[1]) != expected_source:
            raise RuntimeError("Cleanup receipt source key already refers to different content")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES (?,?, 'CLEANUP_EXECUTION_RECEIPT_JSON',?,?,?,1,?)""",
            (source_key, str(receipt), digest, completed_utc, completed_utc,
             "Guarded per-file deletion receipt with queue preflight, full hashes, identity checks, and verified-missing results."),
        )
        source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]

        local_root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT_ID,)).fetchone()
        if not local_root:
            raise RuntimeError("LOCAL-MAIN root is missing")
        relative = str(receipt.relative_to(Path(local_root[0])))
        con.execute(
            "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
            (LOCAL_ROOT_ID, relative, completed_utc),
        )
        path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT_ID, relative)).fetchone()[0]
        artifact_key = "CLEANUP-EXECUTION-RECEIPT:" + args.batch
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                scientific_status,retention_class,notes)
               VALUES (?,?, 'CLEANUP_RECEIPT','Verified deletion execution receipt','USER_CONFIRMED',?,?,'VERIFIED','PERMANENT_SOURCE',?)""",
            (con.execute("SELECT project_id FROM storage_root WHERE root_id=?", (batch["source_root_id"],)).fetchone()[0],
             artifact_key, digest, receipt.stat().st_size, "Immutable JSON receipt for cleanup batch " + args.batch),
        )
        receipt_artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()[0]
        con.execute(
            """INSERT OR IGNORE INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES(?,?,'PRESENT',?,?)""",
            (receipt_artifact_id, path_id, completed_utc, completed_utc),
        )

        for path, row in actual_by_path.items():
            artifact_id = expected_by_path[path]["artifact_id"]
            deleted_utc = iso_utc(row.get("deleted_utc") or completed_utc)
            con.execute(
                """UPDATE cleanup_item SET deletion_state='VERIFIED',deleted_utc=?,actual_bytes=?,
                          notes=notes || ? WHERE cleanup_batch_id=? AND artifact_id=?""",
                (deleted_utc, int(row.get("size_bytes", row.get("logical_bytes"))),
                 f" Guarded receipt {receipt.name} SHA-256={digest}; remote path verified missing.", args.batch, artifact_id),
            )
            con.execute(
                "UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE artifact_id=? AND availability='PRESENT'",
                (completed_utc, artifact_id),
            )

        con.execute(
            """INSERT OR IGNORE INTO cleanup_execution_receipt
               (cleanup_batch_id,receipt_source_id,observed_utc,execution_state,deleted_file_count,
                actual_logical_bytes,actual_allocated_bytes,quota_before_observation_id,quota_after_observation_id,notes)
               VALUES(?,?,?,'COMPLETED',?,?,?,?,?,?)""",
            (args.batch, source_id, completed_utc, len(records), logical_total, allocated_total,
             quota_ids[0], quota_ids[1], f"Receipt artifact_id={receipt_artifact_id}; all candidate hashes matched; zero queue overlap."),
        )
        con.execute(
            "UPDATE cleanup_batch SET status='COMPLETED',actual_bytes=?,completed_utc=? WHERE cleanup_batch_id=?",
            (logical_total, completed_utc, args.batch),
        )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign key check failed while recording cleanup receipt")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print(json.dumps({"cleanup_batch_id": args.batch, "status": "COMPLETED", "deleted_files": len(records),
                      "logical_bytes": logical_total, "allocated_bytes": allocated_total,
                      "receipt_sha256": digest, "quota_before_id": quota_ids[0], "quota_after_id": quota_ids[1]}, sort_keys=True))


if __name__ == "__main__":
    main()
