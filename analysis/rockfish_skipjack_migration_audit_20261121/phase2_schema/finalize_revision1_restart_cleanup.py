#!/usr/bin/env python3
"""Record the completed metadata-only Revision 1 restart cleanup in SQLite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
ROOT_ID = "RF-SCRATCH16-YI"
BATCH = "RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123"
REMOTE_BASE = "MigrationBench/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/revision1_restart_audit_20261123T081721Z"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.manifest, args.receipt, args.status, args.verification):
        path.resolve(strict=True)

    with args.manifest.open(newline="", encoding="utf-8") as f:
        candidates = list(csv.DictReader(f, delimiter="\t"))
    events = [json.loads(line) for line in args.receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
    status = json.loads(args.status.read_text(encoding="utf-8"))
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    if status.get("receipt_sha256") != sha256(args.receipt):
        raise SystemExit("remote status does not match the copied cleanup receipt")
    deleted = [e for e in events if e.get("event") == "deleted_verified"]
    by_path = {e["path"]: e for e in deleted}
    manifest_paths = {r["absolute_path"] for r in candidates}
    if len(candidates) != 4298 or len(deleted) != 4298 or set(by_path) != manifest_paths:
        raise SystemExit("receipt paths/count do not exactly match the audited manifest")
    if status.get("state") != "COMPLETED" or status.get("deleted_file_count") != 4298:
        raise SystemExit("remote cleanup status is not COMPLETED for all 4,298 files")
    if not verification.get("verified") or verification.get("candidate_paths_remaining"):
        raise SystemExit("independent remote verification did not pass")
    if verification.get("retained_coordinate_or_neb_paths_missing") or verification.get("unlisted_wfc_or_mixing_files_remaining"):
        raise SystemExit("retained coordinate or residual restart path verification failed")
    if status.get("queue_overlap_count") != 0 or status.get("queue_check_count") != 114:
        raise SystemExit("fresh per-group queue guards do not match the completed run")

    item_by_path = {r["absolute_path"]: r for r in candidates}
    logical = sum(int(e["identity"]["size_bytes"]) for e in deleted)
    allocated = sum(int(e["identity"]["allocated_bytes"]) for e in deleted)
    if logical != 1036297176856 or allocated != 1036429164544:
        raise SystemExit("receipt byte totals differ from the reviewed scope")
    finished = verification["captured_utc"]

    db = sqlite3.connect(DB, timeout=180)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=180000")
    try:
        batch = db.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        if not batch or batch["status"] not in ("EVIDENCE_READY", "COMPLETED") or batch["expected_bytes"] != logical:
            raise SystemExit("catalog cleanup batch is absent or not evidence-ready")
        update_items = []
        update_locations = []
        for path, event in by_path.items():
            path_digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
            artifact_key = "RF-REV1-QE-RESTART:" + path_digest
            artifact = db.execute("SELECT artifact_id,byte_size,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if not artifact or artifact["byte_size"] != int(event["identity"]["size_bytes"]) or artifact["sha256"] is not None:
                raise RuntimeError("catalog artifact does not match the metadata-only receipt: " + path)
            aid = int(artifact["artifact_id"])
            update_items.append((finished, int(event["identity"]["size_bytes"]),
                                 "Verified absent after unlink. Full payload SHA-256 was intentionally not collected; identity, queue guard and retained coordinate evidence are in the cleanup receipt.", BATCH, aid))
            update_locations.append((finished, aid))
        db.executemany("""UPDATE cleanup_item SET deletion_state='VERIFIED',deleted_utc=?,actual_bytes=?,notes=notes||' '||?
                        WHERE cleanup_batch_id=? AND artifact_id=? AND disposition='DELETE'""", update_items)
        if db.execute("SELECT changes()").fetchone()[0] < 1:
            raise RuntimeError("cleanup item updates were not applied")
        for finished_utc, aid in update_locations:
            db.execute("UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE artifact_id=?", (finished_utc, aid))

        def register_evidence(path: Path, role: str, kind: str, remote_name: str, note: str) -> int:
            digest = sha256(path)
            size = path.stat().st_size
            source_key = "rev1_cleanup_receipt:" + role + ":" + digest[:20]
            db.execute("""INSERT OR IGNORE INTO registry_source
                (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
                VALUES(?,?,?,?,?,?,1,?,?)""",
                (source_key, str(path.resolve()), kind, digest, utc_now(), finished,
                 "User-authorized metadata-only Revision 1 cleanup, execution receipt, and post-delete check.", note))
            source = db.execute("SELECT source_id,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
            if not source or source["sha256"] != digest:
                raise RuntimeError("cleanup evidence source identity mismatch: " + role)
            akey = "RF-REV1-CLEANUP-EXECUTION:" + role + ":" + digest[:20]
            db.execute("""INSERT OR IGNORE INTO artifact
                (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                VALUES('MIGRATIONBENCH',?,?,?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PUBLICATION_EVIDENCE',?)""",
                (akey, kind, role, digest, size, note))
            artifact = db.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (akey,)).fetchone()
            if not artifact or artifact["sha256"] != digest or artifact["byte_size"] != size:
                raise RuntimeError("cleanup evidence artifact identity mismatch: " + role)
            relative = REMOTE_BASE + "/" + remote_name
            db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (ROOT_ID, relative, finished))
            path_id = db.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, relative)).fetchone()[0]
            db.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                       (artifact["artifact_id"], path_id, finished, finished))
            db.execute("""INSERT OR IGNORE INTO registry_entity_ref
                (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                VALUES(?,'CLEANUP_BATCH',?,'CLEANUP_BATCH',?,'EXECUTION_EVIDENCE_FOR',?)""",
                (source["source_id"], BATCH, BATCH, note))
            return int(artifact["artifact_id"])

        receipt_art = register_evidence(args.receipt, "DELETION_RECEIPT", "QE_RESTART_CLEANUP_RECEIPT_JSONL", "cleanup_receipt.jsonl",
                                        "Per-file no-follow unlink, post-unlink absence confirmation, metadata, and 114 fresh queue guard records. Private raw queue captures are outside the repository and excluded.")
        status_art = register_evidence(args.status, "EXECUTION_STATUS", "QE_RESTART_CLEANUP_STATUS_JSON", "cleanup_status.json",
                                       "Completed status, counts, totals, df before/after, and quota command failure due site permission denial.")
        verify_art = register_evidence(args.verification, "POSTDELETE_VERIFICATION", "QE_RESTART_CLEANUP_VERIFICATION_JSON", "cleanup_verification.json",
                                       "Independent exact-path absence check, retained coordinate and NEB path presence, residual WFC/mixing filename scan, and storage snapshot.")
        db.execute("""UPDATE cleanup_batch SET status='COMPLETED',actual_bytes=?,completed_utc=? WHERE cleanup_batch_id=?""",
                   (logical, finished, BATCH))
        if db.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE' AND deletion_state='VERIFIED'", (BATCH,)).fetchone()[0] != 4298:
            raise RuntimeError("verified cleanup item count is not 4,298")
        if db.execute("SELECT count(*) FROM artifact_location l JOIN cleanup_item i USING(artifact_id) WHERE i.cleanup_batch_id=? AND i.disposition='DELETE' AND l.availability='REMOVED'", (BATCH,)).fetchone()[0] != 4298:
            raise RuntimeError("removed artifact location count is not 4,298")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign_key_check failed during cleanup finalization")
        db.commit()
        print(json.dumps({"cleanup_batch_id": BATCH, "state": "COMPLETED_VERIFIED",
                          "candidate_count": len(candidates), "logical_bytes": logical,
                          "allocated_bytes": allocated, "hash_scope": "METADATA_ONLY_NO_RESTART_CONTENT_SHA256",
                          "receipt_artifact_id": receipt_art, "status_artifact_id": status_art,
                          "verification_artifact_id": verify_art, "queue_check_count": 114,
                          "retained_coordinate_or_neb_paths": verification["retained_coordinate_and_neb_path_count"],
                          "private_queue_captures_mirrored": False}, sort_keys=True))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
