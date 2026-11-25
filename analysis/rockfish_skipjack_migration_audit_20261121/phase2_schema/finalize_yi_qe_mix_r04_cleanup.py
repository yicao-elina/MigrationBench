#!/usr/bin/env python3
"""Import quota/queue/deletion receipts and finalize exact SQL item availability."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
BATCH_ID = "RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123"
IMPORT_QUOTA = AUDIT / "phase2_schema/import_quota_capture.py"
CAPTURE_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc(value: str | None = None) -> str:
    if value:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def import_quota(path: Path, label: str, command: str) -> str:
    key = "rockfish_qe_mix_cleanup_quota_" + label + "_20261123"
    resolved = str(path.resolve(strict=True))
    digest = sha256(path)
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=20) as check:
        prior = check.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
        if prior:
            if prior != (resolved, digest):
                raise RuntimeError("Quota source key already points at different bytes: " + key)
            if check.execute("SELECT count(*) FROM quota_observation WHERE raw_output_path=?", (resolved,)).fetchone()[0] >= 3:
                return key
    subprocess.run(["python3", str(IMPORT_QUOTA), str(path.resolve(strict=True)), command, key],
                   check=True, timeout=600, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return key


def register_evidence(con: sqlite3.Connection, path: Path, kind: str, note: str = "") -> tuple[int, int]:
    path = path.resolve(strict=True)
    rel = path.relative_to(REPO).as_posix()
    digest = sha256(path)
    size = path.stat().st_size
    key = f"yi_qe_mix_cleanup_evidence:{rel}:{digest[:16]}"
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (key, str(path), kind, digest, CAPTURE_UTC, CAPTURE_UTC,
         "Exact immutable local receipt for the user-authorized guarded QE mix cleanup.", note),
    )
    source = con.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if not source or tuple(source[1:]) != (str(path), digest):
        raise RuntimeError("Local receipt source identity conflict: " + key)
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)", (rel, CAPTURE_UTC))
    path_id = int(con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (rel,)).fetchone()[0])
    artifact_key = f"RF-AUDIT:{rel}:{digest[:16]}"
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
           VALUES('ROCKFISH_STORAGE_AUDIT',?,'CLEANUP_RECEIPT',?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
        (artifact_key, kind, digest, size, note or "Exact cleanup receipt or related audit artifact."),
    )
    art = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if not art or tuple(art[1:]) != (digest, size):
        raise RuntimeError("Local receipt artifact identity conflict: " + artifact_key)
    con.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""",
        (int(art[0]), path_id, CAPTURE_UTC, CAPTURE_UTC),
    )
    con.execute("UPDATE artifact_location SET availability='PRESENT',last_verified_utc=? WHERE artifact_id=? AND path_id=?", (CAPTURE_UTC, int(art[0]), path_id))
    return int(source[0]), int(art[0])


def add_ref(con: sqlite3.Connection, source_id: int, entity_type: str, entity_key: str, notes: str) -> None:
    con.execute(
        """INSERT OR IGNORE INTO registry_entity_ref
           (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
           VALUES(?,?,?,'CLEANUP_BATCH',?,'EVIDENCE_FOR',?)""",
        (source_id, entity_type, BATCH_ID, BATCH_ID, notes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--receipt-status", required=True, type=Path)
    parser.add_argument("--postverify-json", required=True, type=Path)
    parser.add_argument("--postverify-status", required=True, type=Path)
    parser.add_argument("--quota-before", required=True, type=Path)
    parser.add_argument("--quota-before-raw", required=True, type=Path)
    parser.add_argument("--quota-before-status", required=True, type=Path)
    parser.add_argument("--quota-before-normalization", required=True, type=Path)
    parser.add_argument("--quota-after", required=True, type=Path)
    parser.add_argument("--quota-after-raw", required=True, type=Path)
    parser.add_argument("--quota-after-status", required=True, type=Path)
    parser.add_argument("--quota-after-normalization", required=True, type=Path)
    parser.add_argument("--queue-json", required=True, type=Path)
    parser.add_argument("--queue-status", required=True, type=Path)
    args = parser.parse_args()
    paths = [args.receipt, args.receipt_status, args.postverify_json, args.postverify_status,
             args.quota_before, args.quota_before_raw, args.quota_before_status, args.quota_before_normalization,
             args.quota_after, args.quota_after_raw, args.quota_after_status, args.quota_after_normalization,
             args.queue_json, args.queue_status]
    for path in paths:
        path.resolve(strict=True)

    before_meta = json.loads(args.quota_before_status.read_text(encoding="utf-8"))
    after_meta = json.loads(args.quota_after_status.read_text(encoding="utf-8"))
    before_command = before_meta["remote_command"]
    after_command = after_meta["remote_command"]
    before_key = import_quota(args.quota_before, "before", before_command)
    after_key = import_quota(args.quota_after, "after", after_command)

    receipt_events = [json.loads(line) for line in args.receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
    exec_status = json.loads(args.receipt_status.read_text(encoding="utf-8"))
    postverify = json.loads(args.postverify_json.read_text(encoding="utf-8").strip().splitlines()[-1])
    postverify_status = json.loads(args.postverify_status.read_text(encoding="utf-8"))
    queue_capture = json.loads(args.queue_json.read_text(encoding="utf-8"))
    queue_status = json.loads(args.queue_status.read_text(encoding="utf-8"))
    if queue_status.get("ssh_returncode") != 0 or queue_capture.get("returncode") != 0 or queue_capture.get("unknown_or_unparseable_workdir_rows") != 0 or queue_capture.get("matching_rows"):
        raise SystemExit("pre-cleanup selected-root queue evidence does not show a clean, parseable queue")
    deleted_events = [e for e in receipt_events if e.get("event") == "deleted_verified"]
    deleted_paths = {e.get("path") for e in deleted_events}
    if len(deleted_paths) != len(deleted_events):
        raise SystemExit("cleanup receipt contains duplicate deleted paths")
    if any(e.get("verified_missing") is not True for e in deleted_events):
        raise SystemExit("cleanup receipt contains a deletion without immediate missing-path verification")
    if postverify.get("candidate_count") != 432:
        raise SystemExit("post-cleanup verifier did not check the exact 432-path set")
    post_items = {e["path"]: e for e in postverify.get("items", [])}
    if len(post_items) != 432:
        raise SystemExit("post-cleanup verifier path set is incomplete")
    if postverify_status.get("ssh_returncode") != 0:
        raise SystemExit("post-cleanup Rockfish identity verification failed")

    delete_paths = {p for e in deleted_events for p in [e["path"]]}
    missing = {p for p, e in post_items.items() if e["state"] == "MISSING"}
    present_verified = {p for p, e in post_items.items() if e["state"] == "PRESENT_VERIFIED"}
    unexpected = {p for p, e in post_items.items() if e["state"] not in ("MISSING", "PRESENT_VERIFIED")}
    if missing != delete_paths:
        raise SystemExit("post-cleanup missing path set differs from executor receipt")
    if unexpected:
        raise SystemExit("post-cleanup verification found an unexpected path state: " + repr(sorted(unexpected)[:5]))

    manifest_path = AUDIT / "phase2_data/yi_mix_repro_review_20261123/yi_qe_mix_reproduction_manifest_20261123.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_delete = {i["path"] for g in manifest["groups"] if g["disposition"] == "DELETE" for i in g["items"]}
    held_paths = {i["path"] for g in manifest["groups"] if g["disposition"] == "HOLD" for i in g["items"]}
    if not delete_paths.issubset(expected_delete) or not held_paths.issubset(present_verified):
        raise SystemExit("executor path set or held-path retention does not match reviewed manifest")
    batch_complete_event = next((e for e in receipt_events if e.get("event") == "batch_complete"), None)
    fully_complete = (batch_complete_event is not None and len(delete_paths) == 384 and
                      len(missing) == 384 and len(present_verified) == 48 and not unexpected and
                      exec_status.get("ssh_returncode") == 0 and exec_status.get("remote_batch_complete") is True)
    logical_bytes = sum(int(e["identity"]["size_bytes"]) for e in deleted_events)
    allocated_bytes = sum(int(e["identity"]["allocated_bytes"]) for e in deleted_events)
    if fully_complete and (logical_bytes != 7_500_169_152 or allocated_bytes != 7_518_289_920):
        raise SystemExit("deletion receipt totals differ from the staged SQL batch")
    if len(missing) + len(present_verified) != 432:
        raise SystemExit("post-cleanup states do not account for all candidates")

    def capture_timestamp(path: Path) -> str:
        header = path.read_text(encoding="utf-8").splitlines()[0].strip()
        if header.startswith("capture_utc="):
            header = header.split("=", 1)[1]
        return utc(header)
    before_capture_utc = capture_timestamp(args.quota_before)
    after_capture_utc = capture_timestamp(args.quota_after)
    if before_capture_utc >= after_capture_utc:
        raise SystemExit("quota captures are not in before/after order")
    source_keys = {}
    receipt_kinds = [
        (args.receipt, "QE_MIX_GUARDED_CLEANUP_EXECUTION_RECEIPT", "Per-file all-user queue, full SHA/identity, unlink, and immediate absence records."),
        (args.receipt_status, "QE_MIX_GUARDED_CLEANUP_EXECUTION_STATUS", "Local capture status, exact manifest/ledger hashes, queue age, and remote return status."),
        (args.postverify_json, "QE_MIX_POST_CLEANUP_PATH_IDENTITY_VERIFICATION", "Independent post-delete verification of all absent DELETE paths and all retained files."),
        (args.postverify_status, "QE_MIX_POST_CLEANUP_VERIFICATION_STATUS", "Post-cleanup Rockfish verification command and exact result hash."),
        (args.quota_before, "QE_MIX_CLEANUP_QUOTA_BEFORE", "Fresh timestamped user/group GPFS quota immediately before the guarded cleanup."),
        (args.quota_before_raw, "QE_MIX_CLEANUP_QUOTA_BEFORE_RAW", "Byte-for-byte SSH stdout from the pre-cleanup quota capture; original cluster-local date line preserved."),
        (args.quota_before_status, "QE_MIX_CLEANUP_QUOTA_BEFORE_STATUS", "Execution receipt for the before-cleanup quota read."),
        (args.quota_before_normalization, "QE_MIX_CLEANUP_QUOTA_BEFORE_TIMESTAMP_NORMALIZATION", "UTC timestamp normalization cross-checked exactly against the SSH capture receipt."),
        (args.quota_after, "QE_MIX_CLEANUP_QUOTA_AFTER", "Fresh timestamped user/group GPFS quota after the guarded cleanup."),
        (args.quota_after_raw, "QE_MIX_CLEANUP_QUOTA_AFTER_RAW", "Byte-for-byte SSH stdout from the post-cleanup quota capture; original cluster-local date line preserved."),
        (args.quota_after_status, "QE_MIX_CLEANUP_QUOTA_AFTER_STATUS", "Execution receipt for the after-cleanup quota read."),
        (args.quota_after_normalization, "QE_MIX_CLEANUP_QUOTA_AFTER_TIMESTAMP_NORMALIZATION", "UTC timestamp normalization cross-checked exactly against the SSH capture receipt."),
        (args.queue_json, "QE_MIX_CLEANUP_SELECTED_ROOT_QUEUE", "Fresh all-user queue snapshot restricted to the selected cleanup roots, zero overlaps."),
        (args.queue_status, "QE_MIX_CLEANUP_SELECTED_ROOT_QUEUE_STATUS", "Queue snapshot execution status and raw output hash."),
        (AUDIT / "phase2_schema/execute_yi_qe_mix_r04_guarded_cleanup.py", "QE_MIX_GUARDED_CLEANUP_EXECUTOR", "Exact guarded cleanup implementation with per-file queue and SHA identity checks."),
        (AUDIT / "phase2_schema/verify_yi_qe_mix_r04_cleanup_result.py", "QE_MIX_POST_CLEANUP_VERIFIER", "Exact read-only post-cleanup path/hash verifier."),
        (AUDIT / "phase2_data/yi_mix_repro_review_20261123/capture_qe_mix_cleanup_quota.py", "QE_MIX_QUOTA_CAPTURE_SCRIPT", "Read-only GPFS quota capture script."),
        (AUDIT / "phase2_data/yi_mix_repro_review_20261123/normalize_qe_mix_quota_timestamp.py", "QE_MIX_QUOTA_TIMESTAMP_NORMALIZER", "Converts the cluster-local timestamp to UTC and records the original raw bytes."),
    ]

    con = sqlite3.connect(DB, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        con.execute("BEGIN IMMEDIATE")
        source_artifacts = {}
        for path, kind, note in receipt_kinds:
            sid, aid = register_evidence(con, path, kind, note)
            source_artifacts[path.resolve()] = (sid, aid)
        for label, key in (("before", before_key), ("after", after_key)):
            row = con.execute("SELECT source_id,source_path FROM registry_source WHERE source_key=?", (key,)).fetchone()
            expected_path = str((args.quota_before if label == "before" else args.quota_after).resolve())
            if not row or row[1] != expected_path:
                raise RuntimeError("Imported quota source path mismatch for " + label)
            source_artifacts[(args.quota_before if label == "before" else args.quota_after).resolve()] = (int(row[0]), 0)

        def quota_id(when: str) -> int:
            source_path = str((args.quota_before if when == "before" else args.quota_after).resolve())
            row = con.execute(
                """SELECT quota_observation_id FROM quota_observation
                   WHERE root_id='RF-DATA-FS' AND observed_utc=? AND quota_subject_kind='GROUP' AND quota_subject='pclancy3'""",
                (before_capture_utc if when == "before" else after_capture_utc,),
            ).fetchone()
            if not row:
                raise RuntimeError("Missing /data pclancy3 group quota observation from " + source_path)
            return int(row[0])

        before_quota_id = quota_id("before")
        after_quota_id = quota_id("after")
        normalization_notes = {
            "before": json.loads(args.quota_before_normalization.read_text(encoding="utf-8")),
            "after": json.loads(args.quota_after_normalization.read_text(encoding="utf-8")),
        }
        for label, observation_id in (("before", before_quota_id), ("after", after_quota_id)):
            note = normalization_notes[label]
            if note["normalized_path"] != str((args.quota_before if label == "before" else args.quota_after).resolve()):
                raise RuntimeError("Quota normalized path mismatch for " + label)
            raw_path = args.quota_before_raw if label == "before" else args.quota_after_raw
            raw_path_text = str(raw_path.resolve())
            raw_digest = sha256(raw_path)
            con.execute(
                "UPDATE registry_source SET notes=notes || ? WHERE source_key=?",
                (f" UTC timestamp normalized from the preserved cluster-local header; validated against the local SSH capture at {note['local_capture_receipt_utc']} (difference {note['timestamp_difference_seconds']} s). Byte-for-byte raw stdout: {raw_path_text} SHA256={raw_digest}.",
                 before_key if label == "before" else after_key),
            )
            con.execute(
                "UPDATE quota_observation SET notes=notes || ? WHERE quota_observation_id=?",
                (f" Original output first line used {note['raw_timestamp_line']}; UTC timestamp was normalized and cross-checked against the local capture receipt with {note['timestamp_difference_seconds']} s difference.", observation_id),
            )
        queue_path = args.queue_json.resolve()
        qpath = queue_path.relative_to(REPO).as_posix()
        qsource_row = con.execute("SELECT source_id FROM registry_source WHERE source_path=? AND sha256=? ORDER BY source_id LIMIT 1",
                                  (str(queue_path), sha256(queue_path))).fetchone()
        if qsource_row:
            queue_source_id = int(qsource_row[0])
        else:
            queue_capture_utc = utc(queue_capture["observed_utc"])
            qkey = "rockfish_selected_cleanup_queue_yi_qe_mix_" + queue_capture_utc.replace(":", "").replace("-", "")
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
                   VALUES(?,?,'SLURM_SELECTED_ROOT_QUEUE',?,?,?,1,?,?)""",
                (qkey, str(queue_path), sha256(queue_path), queue_capture_utc, queue_capture_utc,
                 "Read-only all-user squeue snapshot filtered to selected roots.", "0 matching jobs and zero unknown/unparseable WorkDir rows."),
            )
            queue_source_id = int(con.execute("SELECT source_id FROM registry_source WHERE source_path=? AND sha256=? ORDER BY source_id LIMIT 1",
                                              (str(queue_path), sha256(queue_path))).fetchone()[0])

        receipt_path = args.receipt.resolve()
        receipt_source_id = source_artifacts[receipt_path][0]
        receipt_state = "COMPLETED" if fully_complete else ("PARTIAL" if delete_paths else "FAILED")
        completed_utc = batch_complete_event.get("utc") if fully_complete else None
        receipt_utc = utc(completed_utc) if completed_utc else CAPTURE_UTC
        prior = con.execute("SELECT cleanup_execution_receipt_id FROM cleanup_execution_receipt WHERE cleanup_batch_id=? AND receipt_source_id=?", (BATCH_ID, receipt_source_id)).fetchone()
        if not prior:
            con.execute(
                """INSERT INTO cleanup_execution_receipt
                   (cleanup_batch_id,receipt_source_id,observed_utc,execution_state,deleted_file_count,
                    actual_logical_bytes,actual_allocated_bytes,quota_before_observation_id,
                    quota_after_observation_id,queue_snapshot_source_id,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (BATCH_ID, receipt_source_id, receipt_utc, receipt_state, len(delete_paths), logical_bytes,
                 allocated_bytes, before_quota_id, after_quota_id, queue_source_id,
                 "Fresh selected-root queue capture had zero matching rows and zero unknown WorkDirs. Remote executor repeated all-user queue check after hashing and immediately before each unlink. Post-verifier accounted for all 432 paths; 48 provenance-mismatched candidates remain held."),
            )
        for event in deleted_events:
            path = event["path"]
            identity = event["identity"]
            item = con.execute(
                """SELECT ci.artifact_id,ci.pre_delete_sha256,ci.pre_delete_size_bytes,ci.disposition
                   FROM cleanup_item ci JOIN artifact a USING(artifact_id)
                   WHERE ci.cleanup_batch_id=? AND a.artifact_key=?""",
                (BATCH_ID, "RF-DATA-YI:" + str(PurePosixPath(path).relative_to(PurePosixPath("/data/pclancy3/yi")))),
            ).fetchone()
            if not item or item[3] != "DELETE" or item[1] != identity["sha256"] or int(item[2]) != int(identity["size_bytes"]):
                raise RuntimeError("SQL predelete identity disagrees with executor receipt: " + path)
            con.execute(
                """UPDATE cleanup_item SET deletion_state='VERIFIED',deleted_utc=?,actual_bytes=?
                   WHERE cleanup_batch_id=? AND artifact_id=? AND disposition='DELETE'""",
                (utc(event["utc"]), int(identity["size_bytes"]), BATCH_ID, int(item[0])),
            )
            rel = str(PurePosixPath(path).relative_to(PurePosixPath("/data/pclancy3/yi")))
            location = con.execute(
                """SELECT al.location_id FROM artifact_location al JOIN storage_path sp USING(path_id)
                   WHERE al.artifact_id=? AND sp.root_id='RF-DATA-YI' AND sp.relative_path=?""",
                (int(item[0]), rel),
            ).fetchone()
            if not location:
                raise RuntimeError("SQL artifact location missing for deleted candidate: " + path)
            con.execute("UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE location_id=?", (utc(event["utc"]), int(location[0])))

        new_status = "COMPLETED" if fully_complete else ("PARTIAL" if delete_paths else "EVIDENCE_READY")
        con.execute(
            """UPDATE cleanup_batch SET status=?,actual_bytes=?,completed_utc=? WHERE cleanup_batch_id=?""",
            (new_status, logical_bytes, utc(completed_utc) if fully_complete else None, BATCH_ID),
        )
        for path, kind, _note in receipt_kinds:
            sid, _aid = source_artifacts[path.resolve()]
            add_ref(con, sid, "qe_mix_cleanup_receipt", path.name,
                    "Receipt or guard code for the exact QE mix cleanup batch; file content SHA is stored on its immutable source row.")
        add_ref(con, queue_source_id, "qe_mix_cleanup_queue", queue_path.name,
                "Fresh all-user queue capture for the selected roots before this batch; the executor repeated the queue check per file.")
        for label, key in (("before", before_key), ("after", after_key)):
            sid = int(con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0])
            add_ref(con, sid, "qe_mix_cleanup_quota", label, "Timestamped GPFS user/group quota bracket; group totals are shared, not path-localized.")

        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("Foreign-key violations: " + repr(violations[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    try:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = len(con.execute("PRAGMA foreign_key_check").fetchall())
        batches = con.execute("SELECT status,expected_bytes,actual_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()
        items = con.execute("SELECT disposition,deletion_state,count(*),sum(actual_bytes) FROM cleanup_item WHERE cleanup_batch_id=? GROUP BY disposition,deletion_state", (BATCH_ID,)).fetchall()
        print(json.dumps({"cleanup_batch": batches, "items": [tuple(x) for x in items],
                          "execution_receipt_state": receipt_state, "deleted_count": len(delete_paths),
                          "logical_bytes": logical_bytes, "allocated_bytes": allocated_bytes,
                          "postverify_missing": len(missing), "postverify_present_verified": len(present_verified),
                          "quota_before_observed_utc": before_capture_utc, "quota_after_observed_utc": after_capture_utc,
                          "integrity_check": integrity, "foreign_key_violations": foreign_keys}, sort_keys=True))
        if integrity != "ok" or foreign_keys:
            raise SystemExit("catalog integrity validation failed")
    finally:
        con.close()


if __name__ == "__main__":
    main()
