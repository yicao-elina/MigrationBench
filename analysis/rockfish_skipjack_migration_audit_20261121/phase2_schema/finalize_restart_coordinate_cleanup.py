#!/usr/bin/env python3
"""Validate the guarded Rockfish deletion receipt and finalize its SQLite batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
BATCH = "RF-YI-QE-RESTART-CLEANUP-20261123"
ROOTS = {
    "/scratch16/pclancy3/yi": "RF-SCRATCH16-YI",
    "/data/pclancy3/yi": "RF-DATA-YI",
}
ARCHIVE = "/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_restart_cleanup_20261123"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--postscan", required=True, type=Path)
    p.add_argument("--quota-df", required=True, type=Path)
    p.add_argument("--archive-hashes", required=True, type=Path)
    args = p.parse_args()
    for path in (args.manifest, args.receipt, args.postscan, args.quota_df, args.archive_hashes):
        path.resolve(strict=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = read_jsonl(args.receipt)
    postscan = json.loads(args.postscan.read_text(encoding="utf-8"))
    archive_hashes = json.loads(args.archive_hashes.read_text(encoding="utf-8"))
    archive_by_path = {r["path"]: r for r in archive_hashes["files"]}
    for path in (args.receipt, args.postscan, args.quota_df, args.manifest):
        remote_name = path.name
        archived = archive_by_path.get(remote_name)
        if not archived or archived["sha256"] != sha256(path) or archived["size_bytes"] != path.stat().st_size:
            raise SystemExit("private Rockfish archive checksum inventory does not match: " + remote_name)
    final = receipt[-1]
    deleted = [e for e in receipt if e.get("event") == "deleted_verified"]
    deleted_by_path = {e["path"]: e for e in deleted}
    manifest_files = manifest["files"]
    manifest_by_path = {f["path"]: f for f in manifest_files}
    if len(manifest_by_path) != len(manifest_files):
        raise SystemExit("manifest has duplicate paths")
    if set(deleted_by_path) != set(manifest_by_path) or len(deleted) != 7581:
        raise SystemExit("per-file deletion receipt does not exactly match the 7,581-path manifest")
    if final.get("status") != "POST_DELETE_VERIFIED" or final.get("errors"):
        raise SystemExit("guarded executor did not report a clean POST_DELETE_VERIFIED state")
    if final.get("deleted_count") != 7581 or final.get("remaining_manifest_paths") != 0:
        raise SystemExit("guarded executor final counts do not match the manifest")
    if final.get("unlisted_restart_candidates_remaining") != 0 or final.get("run_root_queue_overlaps") != 0:
        raise SystemExit("guarded executor reported residual restart candidates or queue overlap")
    if final.get("queue_checks_immediately_before_each_root_mutation") != 46:
        raise SystemExit("the 46 immediate per-root queue checks are not present")

    for path, record in manifest_by_path.items():
        if deleted_by_path[path].get("identity") != record.get("identity"):
            raise SystemExit("pre-unlink stat identity mismatch: " + path)
        if deleted_by_path[path].get("verified_missing") is not True:
            raise SystemExit("path lacks post-unlink absence confirmation: " + path)

    logical = sum(int(e["identity"]["size_bytes"]) for e in deleted)
    allocated = sum(int(e["identity"]["allocated_bytes"]) for e in deleted)
    if logical != 1225323826928 or allocated != 1225535520768:
        raise SystemExit("deleted byte totals differ from the reviewed manifest")
    if manifest.get("eligible_logical_bytes") != logical or manifest.get("eligible_allocated_bytes") != allocated:
        raise SystemExit("manifest aggregate byte totals do not match per-file identities")

    counts = postscan["counts"]
    if counts["QE_WFC_N"]["count"] != 0 or counts["QE_WFC_DAT"]["count"] != 0:
        raise SystemExit("post-delete all-tree scan found WFC restart files")
    if counts["QE_MIX_N"] != {"count": 8, "logical_bytes": 0}:
        raise SystemExit("post-delete all-tree scan did not match the eight zero-byte holds")
    if postscan.get("total_files") != 8 or postscan.get("total_logical_bytes") != 0:
        raise SystemExit("post-delete all-tree scan totals are inconsistent")

    finished = postscan["captured_utc"]
    db = sqlite3.connect(DB, timeout=180)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=180000")
    try:
        batch = db.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        if not batch or batch["status"] != "EXECUTING" or batch["expected_bytes"] != logical:
            raise SystemExit("cleanup batch is not EXECUTING with the reviewed expected bytes")

        update_items = []
        update_locations = []
        for path, event in deleted_by_path.items():
            root = next((r for r in ROOTS if path.startswith(r + "/")), None)
            if root is None:
                raise SystemExit("manifest path is outside expected user roots: " + path)
            relative = path[len(root) + 1:]
            aid = db.execute("SELECT ci.artifact_id FROM storage_path sp JOIN artifact_location al USING(path_id) JOIN cleanup_item ci ON ci.artifact_id=al.artifact_id WHERE sp.root_id=? AND sp.relative_path=? AND ci.cleanup_batch_id=? AND ci.disposition='DELETE'", (ROOTS[root], relative, BATCH)).fetchone()
            if aid is None:
                raise SystemExit("no catalog artifact location for manifest path: " + path)
            artifact = db.execute("SELECT byte_size,sha256 FROM artifact WHERE artifact_id=?", (aid[0],)).fetchone()
            if not artifact or artifact["byte_size"] != event["identity"]["size_bytes"] or artifact["sha256"] is not None:
                raise SystemExit("catalog artifact size/hash scope differs from metadata-only receipt: " + path)
            loc = db.execute("SELECT availability FROM artifact_location WHERE artifact_id=? AND path_id=(SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?)", (aid[0], ROOTS[root], relative)).fetchone()
            if not loc or loc["availability"] != "PRESENT":
                raise SystemExit("catalog location is not PRESENT before finalization: " + path)
            update_items.append((event["identity"]["size_bytes"],
                                 "Absent after guarded unlink. Restart payload content SHA-256 was intentionally not collected; pre-unlink identity, live queue guards, and retained coordinate evidence are recorded in the execution receipt.",
                                 BATCH, aid[0]))
            update_locations.append((finished, aid[0], ROOTS[root], relative))

        if len(update_items) != 7581:
            raise SystemExit("catalog item count differs from receipt")
        db.executemany("""UPDATE cleanup_item SET deletion_state='VERIFIED',actual_bytes=?,notes=notes||' '||?
                        WHERE cleanup_batch_id=? AND artifact_id=? AND disposition='DELETE' AND deletion_state='EVIDENCE_READY'""", update_items)
        if db.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE' AND deletion_state='VERIFIED'", (BATCH,)).fetchone()[0] != 7581:
            raise SystemExit("not all manifest cleanup items reached VERIFIED")
        for when, aid, root_id, relative in update_locations:
            db.execute("UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE artifact_id=? AND path_id=(SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?)", (when, aid, root_id, relative))

        held_paths = {r[0] for r in db.execute("""SELECT sr.scoped_path||'/'||sp.relative_path FROM cleanup_item ci JOIN artifact a USING(artifact_id) JOIN artifact_location al USING(artifact_id) JOIN storage_path sp USING(path_id) JOIN storage_root sr USING(root_id)
            WHERE ci.cleanup_batch_id=? AND ci.disposition='HOLD'""", (BATCH,))}
        scan_paths = {m["path"] for m in postscan["matches"]}
        if held_paths != scan_paths or len(held_paths) != 8:
            raise SystemExit("the eight retained empty placeholder paths do not exactly match the independent scan")

        def register(path: Path, role: str, kind: str, remote_name: str, note: str) -> tuple[int, int]:
            digest = sha256(path)
            size = path.stat().st_size
            source_key = "qe_restart_coordinate_cleanup:" + role + ":" + digest[:20]
            remote_path = ARCHIVE + "/" + remote_name
            db.execute("""INSERT OR IGNORE INTO registry_source
                (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
                VALUES(?,?,?,?,?,?,1,?,?)""",
                (source_key, remote_path, kind, digest, utc_now(), finished,
                 "User-authorized Rockfish WFC/mixing restart cleanup; private archive hashes verified after copy.",
                 note + " Local evidence file: " + str(path.resolve())))
            source = db.execute("SELECT source_id,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
            if not source or source["sha256"] != digest:
                raise RuntimeError("registry source hash mismatch: " + role)
            artifact_key = "RF-YI-RESTART-CLEANUP:" + role + ":" + digest[:20]
            db.execute("""INSERT OR IGNORE INTO artifact
                (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                VALUES('MIGRATIONBENCH',?,?,?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
                (artifact_key, kind, role, digest, size, note))
            artifact = db.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if not artifact or artifact["sha256"] != digest or artifact["byte_size"] != size:
                raise RuntimeError("evidence artifact mismatch: " + role)
            relative = remote_path.removeprefix("/scratch16/pclancy3/yi/")
            db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('RF-SCRATCH16-YI',?,?)", (relative, finished))
            path_id = db.execute("SELECT path_id FROM storage_path WHERE root_id='RF-SCRATCH16-YI' AND relative_path=?", (relative,)).fetchone()[0]
            db.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'ARCHIVED',?,?)", (artifact["artifact_id"], path_id, finished, finished))
            db.execute("""INSERT OR IGNORE INTO registry_entity_ref
                (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                VALUES(?,'CLEANUP_BATCH',?,'CLEANUP_BATCH',?,'EXECUTION_EVIDENCE_FOR',?)""", (source["source_id"], BATCH, BATCH, note))
            return int(source["source_id"]), int(artifact["artifact_id"])

        receipt_source, receipt_artifact = register(args.receipt, "DELETION_RECEIPT", "QE_RESTART_CLEANUP_RECEIPT_JSONL", "guarded_cleanup_execution_20261123.jsonl",
            "7,581 per-file stat identity and verified-absent records, 46 preflight and 46 immediate pre-root queue checks, post-delete summary; payload content hashes omitted by design.")
        scan_source, scan_artifact = register(args.postscan, "POSTDELETE_NAME_SCAN", "QE_RESTART_POSTDELETE_SCAN_JSON", "postdelete_exact_name_scan_20261123T122347Z.json",
            "Independent all-tree exact-name scan of both user roots: zero WFC/WFC DAT and eight zero-byte held mixes.")
        quota_source, quota_artifact = register(args.quota_df, "QUOTA_AND_FILESYSTEM_READOUT", "ROCKFISH_QUOTA_DF_CAPTURE_TEXT", "quota_disk_observation_20261123.txt",
            "Personal quota RPC was denied by site permissions. The recorded df rows are filesystem-wide and are not personal quota values.")
        _, _ = register(args.archive_hashes, "ARCHIVE_SHA256_MANIFEST", "RESTART_CLEANUP_ARCHIVE_SHA256_JSON", "SHA256SUMS.json",
            "SHA-256 inventory for the private centralized cleanup evidence archive.")

        queue_source = db.execute("SELECT source_id FROM registry_source WHERE source_key LIKE 'qe_restart_coordinate_cleanup:ALL_USER_QUEUE_SNAPSHOT:%' ORDER BY source_id DESC LIMIT 1").fetchone()
        db.execute("""INSERT OR IGNORE INTO cleanup_execution_receipt
            (cleanup_batch_id,receipt_source_id,observed_utc,execution_state,deleted_file_count,actual_logical_bytes,actual_allocated_bytes,queue_snapshot_source_id,notes)
            VALUES(?,?,?,'COMPLETED',?,?,?,?,?)""",
            (BATCH, receipt_source, finished, len(deleted), logical, allocated,
             queue_source[0] if queue_source else None,
             "Guarded deletion receipt validates every exact manifest path and original stat identity. All-user queue checks were repeated immediately before each of 46 run-root mutations. Personal quota-after observation unavailable: quota -s returned site permission errors; df was filesystem-wide only."))
        execution = db.execute("SELECT cleanup_execution_receipt_id,deleted_file_count,actual_logical_bytes,actual_allocated_bytes FROM cleanup_execution_receipt WHERE cleanup_batch_id=? AND receipt_source_id=?", (BATCH, receipt_source)).fetchone()
        if not execution or tuple(execution[k] for k in ("deleted_file_count", "actual_logical_bytes", "actual_allocated_bytes")) != (7581, logical, allocated):
            raise SystemExit("cleanup_execution_receipt row did not persist expected totals")
        db.execute("UPDATE cleanup_batch SET status='COMPLETED',actual_bytes=?,completed_utc=?,evidence_artifact_id=? WHERE cleanup_batch_id=?", (logical, finished, receipt_artifact, BATCH))

        if db.execute("SELECT count(*) FROM artifact_location al JOIN cleanup_item ci USING(artifact_id) WHERE ci.cleanup_batch_id=? AND ci.disposition='DELETE' AND al.availability='REMOVED'", (BATCH,)).fetchone()[0] != 7581:
            raise SystemExit("not all deleted artifact locations are marked REMOVED")
        if db.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='HOLD' AND deletion_state='HELD'", (BATCH,)).fetchone()[0] != 8:
            raise SystemExit("held items changed unexpectedly")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise SystemExit("foreign_key_check failed during finalization")
        db.commit()
        print(json.dumps({"cleanup_batch_id": BATCH, "state": "COMPLETED", "deleted_files": len(deleted),
            "logical_bytes": logical, "allocated_bytes": allocated, "held_zero_byte_mix_files": 8,
            "payload_hashes_collected": False, "receipt_artifact_id": receipt_artifact,
            "postscan_artifact_id": scan_artifact, "quota_capture_artifact_id": quota_artifact,
            "queue_snapshot_source_id": queue_source[0] if queue_source else None}, sort_keys=True))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
