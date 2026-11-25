#!/usr/bin/env python3
"""Register a passing live guard preflight and move the cache batch to EVIDENCE_READY."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DATA = AUDIT / "phase2_data"
DB = AUDIT / "provenance_catalog.sqlite"
CAPTURE = DATA / "conda_package_cleanup_preflight_20261122_r02.json"
GUARD = DATA / "guard_delete_conda_package_cache.py"
BATCH = "RF-DATA-CONDA-PKG-LIBSTDCXX-UNUSED-20261122-R01"
RUN = "RF-CONDAPKG-LIBSTDCXX-14_2-H8F9B012-REGEN-RECIPE-20261122"
LOCAL_ROOT = "LOCAL-MAIN"
EXPECTED_MANIFEST = "8b1e7978413ded29a2cb41658dccd625e9d6dbdf4bffbd62ecca4f3fb6a96dab"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    result = json.loads(CAPTURE.read_text(encoding="utf-8"))
    queue = result.get("queue", {})
    if result.get("event") != "preflight_passed" or result.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("Guard preflight did not pass for the reviewed manifest")
    if (result.get("candidate_files"), result.get("candidate_symlinks"), result.get("candidate_directories"),
        result.get("verified_entries"), result.get("logical_leaf_bytes"), result.get("allocated_leaf_bytes")) != (
        88, 2, 15, 105, 14223129, 14785024
    ):
        raise RuntimeError("Live guard preflight tree totals differ from the staged candidate")
    if queue.get("overlap_count") != 0 or queue.get("unknown_workdir_count") != 0 or queue.get("active_conda_process_count") != 0:
        raise RuntimeError("Live queue/process guard is incomplete or has conflicts")
    if result.get("candidate_root") != "/data/pclancy3/yi/.conda/pkgs/libstdcxx-14.2.0-h8f9b012_2":
        raise RuntimeError("Preflight candidate path is not the reviewed package-cache path")
    if result.get("local_guard_script_sha256") != sha(GUARD):
        raise RuntimeError("Guard script changed after preflight capture")

    observed = result["utc"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        existing = con.execute("SELECT status,expected_bytes,evidence_artifact_id FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        if not existing or existing[0] not in ("PLANNED", "EVIDENCE_READY") or existing[1] != 14223129:
            raise RuntimeError("Cleanup batch must be the exact planned/evidence-ready Conda package batch")
        prior_items = con.execute("SELECT count(*),sum(pre_delete_size_bytes),sum(deletion_state='EVIDENCE_READY') FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE'", (BATCH,)).fetchone()
        if prior_items not in ((90, 14223129, 0), (90, 14223129, 90)):
            raise RuntimeError("Cleanup candidate rows no longer match the exact 90 leaf plan")

        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT,)).fetchone()[0]
        evidence = [
            ("conda_package_live_preflight_20261122_r02", CAPTURE, "CONDA_PACKAGE_CACHE_LIVE_PREFLIGHT", "Exact candidate-tree and live Slurm guard result, R02"),
            ("conda_package_cleanup_guard_script_20261122_r02", GUARD, "CONDA_PACKAGE_CACHE_GUARD_SCRIPT", "Fail-closed remote manifest, queue, process and deletion guard, R02"),
        ]
        source_ids = {}
        for key, path, kind, role in evidence:
            path = path.resolve(); digest = sha(path)
            prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
            if prior and tuple(prior) != (str(path), digest):
                raise RuntimeError("Source key points at changed evidence: " + key)
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
                   VALUES(?,?,?,?,?,?,1,?)""",
                (key, str(path), kind, digest, now, observed,
                 "Guarded cleanup evidence; preflight records the current full queue-workdir overlap and Conda-process checks."),
            )
            source_ids[key] = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0]
            rel = path.relative_to(Path(root)).as_posix()
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (LOCAL_ROOT, rel, now))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT, rel)).fetchone()[0]
            artifact_key = "CONDAPKG-EVIDENCE:" + key
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                    scientific_status,retention_class,notes)
                   VALUES('HPC_ENVIRONMENTS',?,?,?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
                (artifact_key, kind, role, digest, path.stat().st_size,
                 "SHA-256 verified local evidence for batch " + BATCH),
            )
            artifact_id, artifact_sha = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if artifact_sha != digest:
                raise RuntimeError("Evidence artifact SHA mismatch: " + key)
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""", (artifact_id, path_id, now, now)
            )
            if key == "conda_package_cleanup_guard_script_20261122_r02":
                con.execute(
                    """INSERT OR IGNORE INTO run_artifact
                       (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                       VALUES(?,?,'GUARDED_CLEANUP_EXECUTOR','PROVENANCE',1,?)""",
                    (RUN, artifact_id, "The same pinned manifest guard was run in preflight and execution mode."),
                )

        preflight_source_id = source_ids["conda_package_live_preflight_20261122_r02"]
        con.execute("UPDATE cleanup_batch SET status='EVIDENCE_READY' WHERE cleanup_batch_id=?", (BATCH,))
        con.execute("UPDATE cleanup_item SET deletion_state='EVIDENCE_READY' WHERE cleanup_batch_id=? AND disposition='DELETE' AND deletion_state='PLANNED'", (BATCH,))
        staged = con.execute("SELECT count(*),sum(pre_delete_size_bytes),sum(deletion_state='EVIDENCE_READY') FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE'", (BATCH,)).fetchone()
        if staged != (90, 14223129, 90):
            raise RuntimeError("Staged cleanup rows did not all advance to EVIDENCE_READY: " + repr(staged))
        con.execute(
            """INSERT OR IGNORE INTO cleanup_execution_receipt
               (cleanup_batch_id,receipt_source_id,observed_utc,execution_state,deleted_file_count,
                actual_logical_bytes,actual_allocated_bytes,queue_snapshot_source_id,notes)
               VALUES(?,?,?,'PRECHECKED',0,0,0,?,?)""",
            (BATCH, preflight_source_id, observed, preflight_source_id,
             "Preflight re-hashed every one of 88 files and verified two symlink targets across the complete 105-entry tree; all 1,355 queue rows had parseable workdirs, candidate overlap 0, and no Conda package-management process was active."),
        )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"batch": BATCH, "status": "EVIDENCE_READY", "preflight_utc": observed,
                      "tree_entries": result["verified_entries"], "queue_jobs": queue["job_count"],
                      "overlap_count": queue["overlap_count"], "delete_items": 90}, sort_keys=True))


if __name__ == "__main__":
    main()
