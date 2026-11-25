#!/usr/bin/env python3
"""Import incomplete 2026-11-23 metadata streams as explicitly PARTIAL snapshots."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"

SPECS = [
    {
        "snapshot": "RF-DATA-PCLANCY3-20261123-R02-PARTIAL",
        "root_id": "RF-DATA-PCLANCY3",
        "root_path": "/data/pclancy3",
        "ledger": AUDIT / "phase2_data/data_refresh_20261123/data_pclancy3_refresh_20261123.jsonl.gz",
        "source_key": "rockfish_data_pclancy3_partial_inventory_20261123",
        "source_kind": "PARTIAL_ROCKFISH_METADATA_JSONL_GZIP",
        "start_field": "started_epoch",
        "record_field": "record_type",
        "entry_value": "entry",
        "error_value": "not_present",
        "status_note": (
            "PARTIAL: SSH exited 255 without a path_scan_summary or scan_finished marker. "
            "The stream records the base boundary, current-UID top-level trees, and metadata-only entries; "
            "only part of /data/pclancy3/yi was traversed. No payloads were read and symlinks were not followed."
        ),
    },
    {
        "snapshot": "RF-SCRATCH4-PCLANCY3-20261123-R02-PARTIAL",
        "root_id": "RF-SCRATCH4-PCLANCY3",
        "root_path": "/scratch4/pclancy3",
        "ledger": AUDIT / "phase2_scratch4/scratch4_refresh_20261123/scratch4_pclancy3_metadata_20261123.jsonl.gz",
        "source_key": "rockfish_scratch4_pclancy3_partial_inventory_20261123",
        "source_kind": "PARTIAL_ROCKFISH_METADATA_JSONL_GZIP",
        "start_field": "started_utc",
        "record_field": "record",
        "entry_value": None,
        "error_value": "error",
        "status_note": (
            "PARTIAL: no scan_end marker. The read-only walk was stopped after the remote scanner entered "
            "uninterruptible GPFS I/O under a third-party Python environment path. It records 98 permission "
            "errors and only the emitted prefix; do not treat unseen paths as absent."
        ),
    },
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_epoch(value: float) -> str:
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_record(spec: dict, row: dict):
    record = row.get(spec["record_field"])
    is_entry = (record == spec["entry_value"] if spec["entry_value"] is not None
                else record in ("file", "dir", "symlink", "special"))
    if is_entry:
        path = row["path"]
        entry_type = {"file": "FILE", "directory": "DIRECTORY", "dir": "DIRECTORY",
                      "symlink": "SYMLINK"}.get(row.get("type", record), "OTHER")
        rel = "" if path == spec["root_path"] else str(PurePosixPath(path).relative_to(PurePosixPath(spec["root_path"])))
        size = int(row["size_bytes"]) if entry_type == "FILE" and row.get("size_bytes") is not None else None
        return (rel, entry_type, row.get("uid"), row.get("gid"), row.get("mode_octal"), size,
                int(row["allocated_bytes"]) if row.get("allocated_bytes") is not None else None,
                int(row["mtime_ns"]) if row.get("mtime_ns") is not None else None,
                str(row["inode"]) if row.get("inode") is not None else None,
                str(row["device"]) if row.get("device") is not None else None,
                row.get("symlink_target"), int(row["nlink"]) if row.get("nlink") is not None else None,
                "PRESENT")
    if record == spec["error_value"]:
        path = row["path"]
        rel = "" if path == spec["root_path"] else str(PurePosixPath(path).relative_to(PurePosixPath(spec["root_path"])))
        return (rel, "OTHER", None, None, None, None, None, None, None, None, None, None, "UNREADABLE")
    return None


def import_one(con: sqlite3.Connection, spec: dict, now: str) -> dict:
    ledger = spec["ledger"].resolve(strict=True)
    sha = digest(ledger)
    started = None
    counts = {"FILE": 0, "DIRECTORY": 0, "SYMLINK": 0, "OTHER": 0, "UNREADABLE": 0}
    logical_file_bytes = 0
    allocated_file_bytes = 0
    rows = 0

    con.execute("DROP TABLE IF EXISTS temp.refresh_stage")
    con.execute("""CREATE TEMP TABLE refresh_stage (
        relative_path TEXT PRIMARY KEY,
        entry_type TEXT NOT NULL,
        uid INTEGER, gid INTEGER, mode_octal TEXT,
        logical_bytes INTEGER, allocated_bytes INTEGER, mtime_ns INTEGER,
        inode TEXT, device TEXT, symlink_target TEXT, link_count INTEGER,
        observation_status TEXT NOT NULL
    ) WITHOUT ROWID""")
    batch = []
    errors_seen = 0
    with gzip.open(ledger, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            record = row.get(spec["record_field"])
            if record in ("scan_started", "scan_start") and started is None:
                value = row.get(spec["start_field"])
                if spec["start_field"] == "started_epoch":
                    started = utc_epoch(value)
                else:
                    started = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                continue
            if record == spec["error_value"]:
                item = normalize_record(spec, row)
                if item:
                    rel, *values = item
                    con.execute("""INSERT INTO refresh_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(relative_path) DO UPDATE SET observation_status='UNREADABLE'""", (rel, *values))
                    errors_seen += 1
                continue
            item = normalize_record(spec, row)
            if item is None:
                continue
            batch.append(item)
            if len(batch) >= 20000:
                con.executemany("INSERT INTO refresh_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                rows += len(batch)
                batch.clear()
    if batch:
        con.executemany("INSERT INTO refresh_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
    rows = con.execute("SELECT COUNT(*) FROM refresh_stage").fetchone()[0]
    if not started:
        raise RuntimeError(f"No scan-start record in {ledger}")

    for entry_type, count, logical, allocated, unreadable in con.execute(
        """SELECT entry_type,COUNT(*),SUM(logical_bytes),SUM(allocated_bytes),
                  SUM(observation_status='UNREADABLE')
           FROM refresh_stage GROUP BY entry_type"""
    ):
        counts[entry_type] = count
        counts["UNREADABLE"] += unreadable or 0
        if entry_type == "FILE":
            logical_file_bytes = logical or 0
            allocated_file_bytes = allocated or 0
    # A directory that failed scandir is still one observed path. Preserve the
    # error count separately in unreadable_count even if its parent stat is absent.
    unreadable_count = max(errors_seen, counts["UNREADABLE"])

    prior = con.execute("SELECT manifest_sha256 FROM inventory_snapshot WHERE snapshot_id=?", (spec["snapshot"],)).fetchone()
    if prior:
        if prior[0] != sha:
            raise RuntimeError(f"Snapshot ID already exists with a different source hash: {spec['snapshot']}")
        existing = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (spec["snapshot"],)).fetchone()[0]
        return {"snapshot_id": spec["snapshot"], "already_imported": True, "rows": existing, "sha256": sha}

    source_path = str(ledger)
    prior_source = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (spec["source_key"],)).fetchone()
    if prior_source and prior_source != (source_path, sha):
        raise RuntimeError(f"Source key already identifies different bytes: {spec['source_key']}")
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,NULL,1,?,?)""",
        (spec["source_key"], source_path, spec["source_kind"], sha, now,
         "Raw read-only Rockfish metadata stream; incomplete scan explicitly marked PARTIAL.", spec["status_note"]),
    )
    con.execute(
        """INSERT INTO inventory_snapshot
           (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
            files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
            manifest_path,manifest_sha256,notes)
           VALUES(?,?,?,NULL,?,'partial-refresh-v1','PARTIAL',?,?,?,?,?,?,?, ?,?)""",
        (spec["snapshot"], spec["root_id"], started,
         "Metadata-only lstat stream; symlinks are not followed and payloads are not read.",
         counts["FILE"], counts["DIRECTORY"], counts["SYMLINK"], unreadable_count,
         logical_file_bytes, allocated_file_bytes, source_path, sha, spec["status_note"]),
    )
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) SELECT ?,relative_path,? FROM refresh_stage",
                (spec["root_id"], started))
    con.execute(
        """INSERT INTO inventory_observation
           (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,
            mtime_ns,inode,device,symlink_target,hash_scope,observation_status,link_count)
           SELECT ?,p.path_id,s.entry_type,s.uid,s.gid,s.mode_octal,s.logical_bytes,s.allocated_bytes,
                  s.mtime_ns,s.inode,s.device,s.symlink_target,'NONE',s.observation_status,s.link_count
           FROM refresh_stage AS s
           JOIN storage_path AS p ON p.root_id=? AND p.relative_path=s.relative_path""",
        (spec["snapshot"], spec["root_id"]),
    )
    source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (spec["source_key"],)).fetchone()[0]
    con.execute(
        """INSERT OR IGNORE INTO registry_entity_ref
           (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
           VALUES(?,'filesystem_inventory',?,'INVENTORY_SNAPSHOT',?,'CAPTURES_PARTIAL',?)""",
        (source_id, spec["snapshot"], spec["snapshot"], spec["status_note"]),
    )
    observed = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (spec["snapshot"],)).fetchone()[0]
    if observed != rows:
        raise RuntimeError(f"Observation count mismatch for {spec['snapshot']}: parsed={rows}, stored={observed}")
    return {"snapshot_id": spec["snapshot"], "already_imported": False, "rows": observed,
            "files": counts["FILE"], "directories": counts["DIRECTORY"], "symlinks": counts["SYMLINK"],
            "unreadable": unreadable_count, "logical_bytes": logical_file_bytes,
            "allocated_bytes": allocated_file_bytes, "sha256": sha}


def main() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=300)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=300000")
    con.execute("PRAGMA synchronous=NORMAL")
    results = []
    try:
        con.execute("BEGIN IMMEDIATE")
        for spec in SPECS:
            results.append(import_one(con, spec, now))
        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Foreign key violations: {violations[:5]}")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"imported_utc": now, "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
