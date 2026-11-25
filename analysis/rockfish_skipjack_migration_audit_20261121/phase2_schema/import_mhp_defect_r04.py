#!/usr/bin/env python3
"""Import a complete, no-follow R04 metadata refresh of MHP_defect."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
EVIDENCE = AUDIT / "phase2_data/mhp_defect_refresh_20261123"
ROOT_ID = "RF-DATA-MHP-DEFECT"
ROOT = PurePosixPath("/data/pclancy3/MHP_defect")
LEDGER = EVIDENCE / "mhp_defect_current_r04.jsonl"
STATUS = EVIDENCE / "scan_status.json"
SUMMARY = EVIDENCE / "reconciliation_summary.json"
SNAPSHOT_ID = "RF-DATA-MHP-DEFECT-R04-20261123T085118Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")


def main() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if (status.get("ssh_returncode") != 0 or status.get("root") != str(ROOT)
            or status.get("stderr_bytes") != 0):
        raise RuntimeError("MHP_defect scan status is not a clean exact-root scan")
    if status.get("stdout_sha256") != sha256(LEDGER):
        raise RuntimeError("MHP_defect scan ledger SHA differs from execution status")
    if summary.get("completion_state") != "COMPLETE" or summary.get("root") != str(ROOT):
        raise RuntimeError("MHP_defect reconciliation is not complete")
    if summary.get("current_counts") != {
        "directories_below_root": 23, "errors": 0, "files": 168, "symlinks": 2
    }:
        raise RuntimeError("MHP_defect summary counts changed")

    started = finished = None
    rows: dict[str, tuple] = {}
    logical = allocated_files = allocated_all_paths = 0
    type_counts = {"FILE": 0, "DIRECTORY": 0, "SYMLINK": 0, "OTHER": 0}
    with LEDGER.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            record = row.get("record")
            if record == "scan_start":
                started = utc(row["started_utc"])
                if row.get("root") != str(ROOT):
                    raise RuntimeError("unexpected MHP_defect scan root")
                continue
            if record == "scan_complete":
                finished = utc(row["finished_utc"])
                counts = row.get("counts", {})
                if counts != {"file": 168, "dir": 23, "symlink": 2, "other": 0, "unreadable": 0}:
                    raise RuntimeError("MHP_defect scanner ended with errors")
                continue
            if record == "error":
                raise RuntimeError("MHP_defect scan contains an error record")
            kind = {"file": "FILE", "dir": "DIRECTORY", "symlink": "SYMLINK"}.get(record)
            if kind is None:
                raise RuntimeError("unexpected MHP_defect ledger record: " + repr(record))
            path = PurePosixPath(row["path"])
            if path != ROOT and ROOT not in path.parents:
                raise RuntimeError("path escaped the exact MHP_defect root: " + str(path))
            rel = str(path.relative_to(ROOT))
            if not rel or rel == "." or rel in rows:
                raise RuntimeError("empty or duplicate MHP_defect relative path: " + rel)
            size = int(row.get("size_bytes") or 0) if kind == "FILE" else None
            alloc = int(row.get("blocks_512") or 0) * 512
            rows[rel] = (
                kind, row.get("uid"), row.get("gid"), row.get("mode_octal"), size,
                alloc, row.get("mtime_ns"), str(row["inode"]) if row.get("inode") is not None else None,
                str(row["device"]) if row.get("device") is not None else None,
                row.get("symlink_target"), row.get("link_count"),
            )
            type_counts[kind] += 1
            allocated_all_paths += alloc
            if kind == "FILE":
                logical += int(size or 0)
                allocated_files += alloc

    if not started or not finished:
        raise RuntimeError("MHP_defect scan lacks start or completion marker")
    if type_counts != {"FILE": 168, "DIRECTORY": 23, "SYMLINK": 2, "OTHER": 0}:
        raise RuntimeError("MHP_defect ledger type counts changed: " + repr(type_counts))
    if logical != 1_372_309_347:
        raise RuntimeError("MHP_defect logical byte total differs from reviewed summary")
    if logical != int(summary["current_logical_bytes"]):
        raise RuntimeError("MHP_defect logical total disagrees with reconciliation")

    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    manifest_sha = sha256(LEDGER)
    evidence_files = [p for p in sorted(EVIDENCE.iterdir()) if p.is_file()]
    con = sqlite3.connect(DB, timeout=600)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=600000")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute(
            "SELECT scoped_path,filesystem_name,mount_path FROM storage_root WHERE root_id=?",
            (ROOT_ID,),
        ).fetchone()
        if root != (str(ROOT), "data", "/data"):
            raise RuntimeError("catalog MHP_defect root is missing or changed")
        prior = con.execute(
            "SELECT manifest_sha256 FROM inventory_snapshot WHERE snapshot_id=?",
            (SNAPSHOT_ID,),
        ).fetchone()
        if prior and prior[0] != manifest_sha:
            raise RuntimeError("snapshot ID is already bound to different bytes")

        source_ids = []
        for path in evidence_files:
            digest = sha256(path)
            key = f"mhp_defect_r04:{path.name}:{digest[:16]}"
            as_of = finished
            kind = "MHP_DEFECT_R04_INVENTORY_EVIDENCE"
            if path.name == LEDGER.name:
                kind = "MHP_DEFECT_R04_RAW_METADATA_LEDGER"
            elif path.name == STATUS.name:
                kind = "MHP_DEFECT_R04_SCAN_STATUS"
            elif path.name == SUMMARY.name:
                kind = "MHP_DEFECT_R04_RECONCILIATION_SUMMARY"
            elif path.name == "queue_capture.json":
                kind = "SLURM_QUEUE_ALL_USERS_CAPTURE"
                as_of = utc(json.loads(path.read_text(encoding="utf-8"))["observed_utc"])
            elif path.name == "queue_status.json":
                as_of = utc(json.loads(path.read_text(encoding="utf-8"))["finished_utc"])
            source = con.execute(
                "SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)
            ).fetchone()
            if source and source != (str(path.resolve()), digest):
                raise RuntimeError("MHP_defect source key collision: " + key)
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                    immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (key, str(path.resolve()), kind, digest, captured, as_of,
                 "Exact local read-only R04 scan or reconciliation evidence identified by SHA-256.",
                 "MHP_defect metadata refresh evidence; file payloads were not read or hashed."),
            )
            source_id = con.execute(
                "SELECT source_id FROM registry_source WHERE source_key=?", (key,)
            ).fetchone()[0]
            con.execute(
                "UPDATE registry_source SET as_of_utc=? WHERE source_id=? AND source_path=? AND sha256=?",
                (as_of, source_id, str(path.resolve()), digest),
            )
            source_ids.append((source_id, key, path.name))
            artifact_key = f"ROCKFISH-SOURCE:{key}"
            size = path.stat().st_size
            old = con.execute(
                "SELECT sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)
            ).fetchone()
            if old and old != (digest, size):
                raise RuntimeError("MHP_defect evidence artifact identity changed: " + path.name)
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,
                    byte_size,scientific_status,retention_class,notes)
                   VALUES('ROCKFISH_STORAGE_AUDIT',?,?,?,?,?,?,'UNKNOWN','PERMANENT_SOURCE',?)""",
                (artifact_key, kind, "Immutable MHP_defect R04 inventory/audit evidence",
                 "USER_CONFIRMED", digest, size, "No scientific file payloads were hashed."),
            )
            rel_evidence = path.resolve().relative_to(AUDIT.parents[1]).as_posix()
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)",
                        (rel_evidence, captured))
            evidence_path_id = con.execute(
                "SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?",
                (rel_evidence,),
            ).fetchone()[0]
            evidence_artifact_id = con.execute(
                "SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)
            ).fetchone()[0]
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""",
                (evidence_artifact_id, evidence_path_id, captured, captured),
            )

        if not prior:
            con.execute(
                """INSERT INTO inventory_snapshot
                   (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                    files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                    manifest_path,manifest_sha256,notes)
                   VALUES(?,?,?,?,'os.scandir+lstat, no-follow, metadata-only','mhp-r04-v1','COMPLETE',?,?,?,0,?,?,?,?,?)""",
                (SNAPSHOT_ID, ROOT_ID, started, finished, type_counts["FILE"],
                 type_counts["DIRECTORY"], type_counts["SYMLINK"], logical,
                 allocated_files, str(LEDGER.resolve()), manifest_sha,
                 "Exact complete current subtree scan; 490 removed baseline paths reconcile to the prior verified MHP WFC cleanup. Symlinks were not followed."),
            )
            batch = []
            for rel, item in rows.items():
                kind, uid, gid, mode, size, alloc, mtime, inode, device, target, nlink = item
                batch.append((rel, kind, uid, gid, mode, size, alloc, mtime, inode, device, target, nlink))
            con.execute("CREATE TEMP TABLE mhp_stage(relative_path TEXT PRIMARY KEY,entry_type TEXT,uid INTEGER,gid INTEGER,mode_octal TEXT,logical_bytes INTEGER,allocated_bytes INTEGER,mtime_ns INTEGER,inode TEXT,device TEXT,symlink_target TEXT,link_count INTEGER) WITHOUT ROWID")
            con.executemany("INSERT INTO mhp_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) SELECT ?,relative_path,? FROM mhp_stage", (ROOT_ID, finished))
            con.execute(
                """INSERT INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,
                    inode,device,symlink_target,hash_scope,observation_status,link_count)
                   SELECT ?,p.path_id,s.entry_type,s.uid,s.gid,s.mode_octal,s.logical_bytes,s.allocated_bytes,
                          s.mtime_ns,s.inode,s.device,s.symlink_target,'NONE','PRESENT',s.link_count
                   FROM mhp_stage s JOIN storage_path p ON p.root_id=? AND p.relative_path=s.relative_path""",
                (SNAPSHOT_ID, ROOT_ID),
            )
        for source_id, key, filename in source_ids:
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'inventory_evidence',?,'INVENTORY_SNAPSHOT',?,'CAPTURES',?)""",
                (source_id, filename, SNAPSHOT_ID, "Current R04 MHP_defect inventory, reconciliation, or queue evidence."),
            )
        stored = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()[0]
        if stored != len(rows):
            raise RuntimeError(f"MHP_defect SQL rows differ from manifest: {stored} != {len(rows)}")
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign key violations: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({
        "snapshot_id": SNAPSHOT_ID, "status": "IMPORTED_COMPLETE" if not prior else "ALREADY_IMPORTED",
        "paths": len(rows), "types": type_counts, "logical_bytes": logical,
        "allocated_file_bytes": allocated_files, "allocated_all_path_bytes": allocated_all_paths,
        "manifest_sha256": manifest_sha, "evidence_files": len(evidence_files),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
