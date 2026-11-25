#!/usr/bin/env python3
"""Extract and link the all-user queue snapshot embedded in the QE cleanup receipt."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DATA = AUDIT / "phase2_data" / "yi_refresh_20261123"
DB = AUDIT / "provenance_catalog.sqlite"
RECEIPT = DATA / "qe_mix_guarded_cleanup_receipt_20261123.jsonl"
QUEUE_CAPTURE = DATA / "qe_mix_live_alluser_squeue_20261123.jsonl"
BATCH_ID = "RF-DATA-YI-CR-IN-GAP-MIX-CLEANUP-20261123"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_queue_snapshot() -> tuple[dict, list[dict]]:
    events = [json.loads(line) for line in RECEIPT.read_text(encoding="utf-8").splitlines() if line.strip()]
    preflight = next((event for event in events if event.get("event") == "preflight_passed"), None)
    completed = next((event for event in reversed(events) if event.get("event") == "complete"), None)
    deletions = [event for event in events if event.get("event") == "deleted_verified"]
    if not preflight or preflight.get("queue_overlaps") != 0 or not completed or len(deletions) != 12:
        raise RuntimeError("successful receipt does not contain the expected queue/deletion evidence")
    if any(event.get("queue_overlaps") != 0 for event in deletions):
        raise RuntimeError("one of the immediate pre-unlink queue checks reported an overlap")
    rows = preflight.get("queue_rows")
    if not isinstance(rows, list) or not rows or any(not row.get(k) for row in rows for k in ("job_id", "user", "state", "workdir")):
        raise RuntimeError("all-user preflight queue rows are incomplete")
    metadata = {
        "event": "capture_metadata",
        "captured_utc": preflight["utc"],
        "source_command": preflight["queue_command"],
        "scope": "all_users",
        "job_count": len(rows),
        "queue_overlaps": 0,
        "staged_cleanup_batch": BATCH_ID,
        "source_receipt_sha256": sha256(RECEIPT),
    }
    payload = [metadata] + [{"event": "queue_job", **row} for row in rows]
    QUEUE_CAPTURE.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in payload), encoding="utf-8")
    return metadata, rows


def main() -> None:
    metadata, rows = write_queue_snapshot()
    captured = metadata["captured_utc"]
    observed = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    path = QUEUE_CAPTURE.resolve(strict=True)
    relative = path.relative_to(REPO).as_posix()
    digest = sha256(path)
    source_key = "qe_mix_cleanup_alluser_queue:" + relative + ":" + digest[:16]
    artifact_key = "QE-MIX-CLEANUP-QUEUE:" + digest[:16]

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute("SELECT status FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()
        if not batch or batch[0] != "COMPLETED":
            raise RuntimeError("QE cleanup batch is not COMPLETED")
        source_prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
        if source_prior and source_prior != (str(path), digest):
            raise RuntimeError("queue source key identity conflict")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
               VALUES(?,?, 'SLURM_LIVE_QUEUE_ALL_USER_CAPTURE',?,?,?,1,?,?)""",
            (source_key, str(path), digest, observed, captured,
             "All-user Rockfish squeue output captured by the immediate QE cleanup guard.",
             "1,385 workdir rows; zero candidate overlaps. Per-file recheck rows and zero-overlap results remain in the linked execution receipt."),
        )
        source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                scientific_status,retention_class,notes)
               VALUES('ROCKFISH_STORAGE_AUDIT',?,'SLURM_QUEUE_SNAPSHOT',?,'USER_CONFIRMED',?,?,'VERIFIED','PERMANENT_SOURCE',?)""",
            (artifact_key, "All-user Rockfish queue snapshot before QE mix cleanup", digest, path.stat().st_size,
             "Parsed all-user jobs: %d; cleanup workdir overlaps: 0." % len(rows)),
        )
        artifact = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
        if artifact[1:] != (digest, path.stat().st_size):
            raise RuntimeError("queue snapshot artifact identity conflict")
        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()
        if not root or Path(root[0]).resolve() != REPO.resolve():
            raise RuntimeError("LOCAL-MAIN root mismatch")
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)", (relative, observed))
        path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (relative,)).fetchone()[0]
        con.execute(
            """INSERT OR IGNORE INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES(?,?,'PRESENT',?,?)""", (artifact[0], path_id, observed, observed)
        )
        con.execute(
            """INSERT OR IGNORE INTO registry_entity_ref
               (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
               VALUES(?, 'qe_cleanup_queue_snapshot', ?, 'CLEANUP_BATCH', ?, 'EVIDENCE_FOR', ?)""",
            (source_id, BATCH_ID, BATCH_ID, "Immediate all-user queue preflight; zero workdir overlaps."),
        )
        receipt = con.execute("SELECT cleanup_execution_receipt_id,queue_snapshot_source_id FROM cleanup_execution_receipt WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()
        if not receipt:
            raise RuntimeError("cleanup execution receipt row is missing")
        if receipt[1] not in (None, source_id):
            raise RuntimeError("cleanup receipt is already linked to a different queue capture")
        con.execute("UPDATE cleanup_execution_receipt SET queue_snapshot_source_id=? WHERE cleanup_batch_id=?", (source_id, BATCH_ID))
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign-key check failed while linking queue evidence")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"batch": BATCH_ID, "queue_source_id": source_id, "job_count": len(rows),
                      "queue_overlaps": 0, "queue_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
