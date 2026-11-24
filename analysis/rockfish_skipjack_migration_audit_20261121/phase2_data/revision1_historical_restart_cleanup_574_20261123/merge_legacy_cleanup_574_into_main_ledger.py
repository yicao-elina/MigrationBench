#!/usr/bin/env python3
"""Add the old master-worktree 574-file receipt to the unified cleanup ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
TARGET = AUDIT / "cleanup/cleanup_audit.sqlite"
SOURCE = HERE / "cleanup_audit_master_snapshot_20261123.sqlite"
BACKUP = HERE / "cleanup_audit_before_574_merge_20261123.sqlite"
ACTION_ID = "remove_stale_wfc_mix_restarts_20261122_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not TARGET.is_file() or not SOURCE.is_file():
        raise SystemExit("target/source cleanup database is missing")
    if not BACKUP.exists():
        source = sqlite3.connect(TARGET)
        backup = sqlite3.connect(BACKUP)
        try:
            source.backup(backup)
        finally:
            backup.close()
            source.close()
    before = sha256(BACKUP)
    con = sqlite3.connect(TARGET, timeout=60)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("ATTACH DATABASE ? AS legacy", (str(SOURCE),))
        action = con.execute("SELECT * FROM legacy.cleanup_action WHERE action_id=?", (ACTION_ID,)).fetchone()
        rows = con.execute("SELECT * FROM legacy.cleanup_artifact WHERE action_id=? ORDER BY artifact_path", (ACTION_ID,)).fetchall()
        if not action or len(rows) != 574:
            raise RuntimeError("legacy source action/path count does not match the expected 574-file receipt")
        if sum(int(row["size_bytes"]) for row in rows) != 1_536_705_954_176:
            raise RuntimeError("legacy source logical-byte total does not match")
        columns = [row[1] for row in con.execute("PRAGMA table_info(cleanup_action)")]
        existing_action = con.execute("SELECT * FROM main.cleanup_action WHERE action_id=?", (ACTION_ID,)).fetchone()
        action_values = tuple(action[column] for column in columns)
        if existing_action and tuple(existing_action) != action_values:
            raise RuntimeError("target action id exists with different metadata")
        overlap = con.execute(
            """SELECT COUNT(*) FROM legacy.cleanup_artifact AS s
               JOIN main.cleanup_artifact AS t USING(artifact_path)
               WHERE s.action_id=?""", (ACTION_ID,)
        ).fetchone()[0]
        if overlap not in (0, 574):
            raise RuntimeError(f"partial path overlap with target ledger: {overlap}")
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            """INSERT OR IGNORE INTO main.cleanup_action
               (action_id,created_utc,source_root,rule,rationale,status,expected_bytes,actual_bytes,completed_utc)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            action_values,
        )
        if overlap == 0:
            con.execute(
                """INSERT INTO main.cleanup_artifact
                   (action_id,run_id,slurm_job_id,artifact_path,role,size_bytes,mtime_ns,file_mode,pre_delete_sha256,status)
                   SELECT action_id,run_id,slurm_job_id,artifact_path,role,size_bytes,mtime_ns,file_mode,pre_delete_sha256,status
                   FROM legacy.cleanup_artifact WHERE action_id=?""", (ACTION_ID,)
            )
        after_rows = con.execute("SELECT COUNT(*) FROM main.cleanup_artifact WHERE action_id=?", (ACTION_ID,)).fetchone()[0]
        if after_rows != 574:
            raise RuntimeError(f"target artifact count after merge is {after_rows}, expected 574")
        con.commit()
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError("target cleanup ledger integrity check failed: " + integrity)
        payload = {
            "action_id": ACTION_ID,
            "target_database": str(TARGET),
            "source_snapshot": str(SOURCE),
            "source_snapshot_sha256": sha256(SOURCE),
            "pre_merge_backup": str(BACKUP),
            "pre_merge_backup_sha256": before,
            "merged_path_count": 0 if overlap == 574 else 574,
            "final_path_count": after_rows,
            "source_content_hash_count": sum(bool(row["pre_delete_sha256"]) for row in rows),
            "target_action_count": con.execute("SELECT COUNT(*) FROM main.cleanup_action").fetchone()[0],
            "target_artifact_count": con.execute("SELECT COUNT(*) FROM main.cleanup_artifact").fetchone()[0],
            "integrity_check": integrity,
            "merged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        (HERE / "legacy_cleanup_ledger_merge_receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
