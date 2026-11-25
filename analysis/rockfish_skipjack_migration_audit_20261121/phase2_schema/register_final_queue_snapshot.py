#!/usr/bin/env python3
"""Register the filtered all-user Rockfish queue snapshot used for triage."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
CAPTURE = AUDIT / "phase2_data" / "current_queue_data_scratch4_allusers_20261122.txt"
SOURCE_KEY = "rockfish_current_alluser_queue_data_scratch4_20261122"


def main() -> None:
    lines = CAPTURE.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("Queue capture is empty")
    observed = datetime.fromisoformat(lines[0]).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = [line for line in lines[1:] if line.strip()]
    if any(not (row.split("|")[3].startswith("/data/pclancy3/") or row.split("|")[3].startswith("/scratch4/pclancy3/")) for row in rows):
        raise SystemExit("Queue capture includes a workdir outside the stated scope")
    digest = hashlib.sha256(CAPTURE.read_bytes()).hexdigest()

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (SOURCE_KEY,)).fetchone()
        if prior and prior != (str(CAPTURE.resolve()), digest):
            raise RuntimeError("Queue source key already points at changed evidence")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES(?,?,'SLURM_LIVE_QUEUE_FILTERED_CAPTURE',?,?,?,1,?)""",
            (SOURCE_KEY, str(CAPTURE.resolve()), digest, observed, observed,
             "Filtered all-user Rockfish squeue rows for /data/pclancy3 and /scratch4/pclancy3; triage snapshot only, not a substitute for a per-batch live guard."),
        )
        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()[0]
        relative = CAPTURE.resolve().relative_to(Path(root)).as_posix()
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)", (relative, observed))
        path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (relative,)).fetchone()[0]
        artifact_key = "ROCKFISH-QUEUE-SNAPSHOT:" + CAPTURE.name
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
               VALUES(?,'SLURM_QUEUE_SNAPSHOT','Filtered live all-user queue capture','USER_CONFIRMED',?,?,'VERIFIED','PERMANENT_SOURCE',?)""",
            (artifact_key, digest, CAPTURE.stat().st_size, "Filtered result rows: " + str(len(rows)) + "; capture timestamp " + observed),
        )
        artifact_id = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
        if artifact_id[1] != digest:
            raise RuntimeError("Queue artifact hash mismatch")
        con.execute(
            """INSERT OR IGNORE INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES(?,?,'PRESENT',?,?)""",
            (artifact_id[0], path_id, observed, observed),
        )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed")
        con.commit()
        print(f"registered {SOURCE_KEY}: rows={len(rows)} sha256={digest}")
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
