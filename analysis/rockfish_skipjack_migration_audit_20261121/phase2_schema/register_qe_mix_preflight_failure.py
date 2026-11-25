#!/usr/bin/env python3
"""Preserve the preflight-only Python 3.6 compatibility failure in the catalog."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
DATA = AUDIT / "phase2_data" / "yi_refresh_20261123"
INITIAL_SCRIPT = DATA / "execute_qe_mix_guarded_cleanup_attempt1_20261123.py"
FAILURE_RECEIPT = DATA / "qe_mix_guarded_cleanup_preflight_failure_20261123.jsonl"
BATCH_ID = "RF-DATA-YI-CR-IN-GAP-MIX-CLEANUP-20261123"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register_file(con: sqlite3.Connection, path: Path, kind: str, role: str, science: str, key_prefix: str, observed: str) -> int:
    path = path.resolve(strict=True)
    relative = path.relative_to(REPO).as_posix()
    sha = digest(path)
    source_key = key_prefix + ":" + relative + ":" + sha[:16]
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (source_key, str(path), kind, sha, observed, observed,
         "Exact local evidence for the first guarded QE mix cleanup attempt.",
         "Remote compatibility failure occurred before queue parsing, candidate hashing, or any unlink."),
    )
    source = con.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
    if source[1:] != (str(path), sha):
        raise RuntimeError("failure evidence source key conflict")
    project = "ROCKFISH_STORAGE_AUDIT"
    artifact_key = key_prefix + ":" + relative + ":" + sha[:16]
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES(?,?,?,?, 'USER_CONFIRMED',?,?,?,'PERMANENT_SOURCE',?)""",
        (project, artifact_key, kind, role, sha, path.stat().st_size, science,
         "Preserved verbatim for audit; this preflight attempt made no remote file changes."),
    )
    artifact = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if artifact[1:] != (sha, path.stat().st_size):
        raise RuntimeError("failure evidence artifact hash/size conflict")
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
           VALUES(?, 'qe_cleanup_preflight_attempt', ?, 'CLEANUP_BATCH', ?, 'EVIDENCE_FOR', ?)""",
        (source[0], BATCH_ID, BATCH_ID, "Records a guard-only failure; no candidate was deleted."),
    )
    return int(artifact[0])


def main() -> None:
    if not INITIAL_SCRIPT.is_file() or not FAILURE_RECEIPT.is_file():
        raise SystemExit("preserved attempt-1 script and preflight receipt are required")
    events = [json.loads(line) for line in FAILURE_RECEIPT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(events) != 1 or events[0].get("event") != "guard_or_execution_failed" or "unexpected keyword argument 'text'" not in events[0].get("error", ""):
        raise SystemExit("receipt does not establish a single Python 3.6 queue-preflight compatibility failure")
    initial_hash = digest(INITIAL_SCRIPT)
    observed = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute("SELECT status FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()
        if not batch or batch[0] != "EXECUTING":
            raise RuntimeError("cleanup batch is not in EXECUTING state after the preserved preflight failure")
        previous = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key LIKE 'QE-MIX-EXECUTOR:%' AND sha256=?", (initial_hash,)).fetchone()
        if not previous:
            raise RuntimeError("initial executor version is missing from the catalog")
        old_root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()[0]
        old_relative = str((AUDIT / "phase2_schema" / "execute_qe_mix_guarded_cleanup.py").resolve().relative_to(Path(old_root).resolve()))
        old_path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (old_relative,)).fetchone()[0]
        # The original code is now preserved at a second, immutable attempt-1 path.
        con.execute("UPDATE artifact_location SET availability='MISSING',last_verified_utc=? WHERE artifact_id=? AND path_id=?",
                    (observed, previous[0], old_path_id))
        relative_copy = INITIAL_SCRIPT.resolve().relative_to(Path(old_root).resolve()).as_posix()
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)", (relative_copy, observed))
        copy_path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (relative_copy,)).fetchone()[0]
        con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                    (previous[0], copy_path_id, observed, observed))
        register_file(con, INITIAL_SCRIPT, "GUARDED_CLEANUP_EXECUTOR_ATTEMPT", "Initial QE mix guarded cleanup executor", "PROVENANCE_PARTIAL", "QE-MIX-EXECUTOR-ATTEMPT1", observed)
        register_file(con, FAILURE_RECEIPT, "GUARDED_CLEANUP_PREFLIGHT_FAILURE", "Python 3.6 queue-preflight compatibility failure receipt", "FAILURE_MODE", "QE-MIX-PREFLIGHT-FAILURE", observed)
        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("foreign-key check failed: " + repr(violations[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"status": "PRESERVED", "initial_script_sha256": initial_hash,
                      "failure_receipt_sha256": digest(FAILURE_RECEIPT), "deleted_files": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
