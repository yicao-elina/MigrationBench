#!/usr/bin/env python3
"""Import the exact R04 refresh for three former-collaborator/home roots."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
DATA = AUDIT / "phase2_data"
RAW = DATA / "collab_home_path_inventory_20261123.jsonl.gz"
SUMMARY = DATA / "collab_home_summary_20261123.json"
SCANNER = DATA / "refresh_collab_home_inventory_20261123.py"
REPORT = DATA / "collab_home_refresh_report_20261123.md"
OUTPUT = DATA / "collab_roots_current_metadata_r04.jsonl.gz"
ROOT_ID = "RF-DATA-PCLANCY3"
PARENT = PurePosixPath("/data/pclancy3")
ROOTS = {
    PurePosixPath("/data/pclancy3/ycao73_home"),
    PurePosixPath("/data/pclancy3/for-victor"),
    PurePosixPath("/data/pclancy3/for-yuqi"),
}
SNAPSHOT_ID = "RF-DATA-PCLANCY3-COLLAB-ROOTS-R04-20261123T071500Z"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if set(summary.get("roots", [])) != set(map(str, ROOTS)):
        raise RuntimeError("unexpected exact root list in collaborator refresh")
    if summary.get("ssh_exit_code") != 0 or summary.get("stat_errors") or summary.get("missing_roots"):
        raise RuntimeError("collaborator refresh is incomplete or has errors")

    started = finished = None
    seen_roots: set[PurePosixPath] = set()
    counts = {"FILE": 0, "DIRECTORY": 0, "SYMLINK": 0, "OTHER": 0}
    logical = allocated_files = allocated_all_paths = 0
    manifest_sha = hashlib.sha256()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(RAW, "rt", encoding="utf-8") as stream, OUTPUT.open("wb") as raw_out:
        with gzip.GzipFile(fileobj=raw_out, filename="", mode="wb", compresslevel=6, mtime=0) as out:
            for line in stream:
                row = json.loads(line)
                record = row.get("record")
                if record == "scan_start":
                    started = utc_epoch(float(row["started_epoch"]))
                    if set(map(PurePosixPath, row.get("roots", []))) != ROOTS:
                        raise RuntimeError("scan header root set changed")
                    continue
                if record == "scan_end":
                    finished = utc_epoch(float(row["finished_epoch"]))
                    continue
                if record not in ("dir", "file", "symlink"):
                    raise RuntimeError("unexpected collaborator scan record: " + repr(record))
                path = PurePosixPath(row["path"])
                declared = PurePosixPath(row["root"])
                if declared not in ROOTS or not (path == declared or declared in path.parents):
                    raise RuntimeError("path escaped its exact scanned root: " + str(path))
                seen_roots.add(declared)
                kind = {"dir": "DIRECTORY", "file": "FILE", "symlink": "SYMLINK"}[record]
                size = int(row.get("size_bytes") or 0)
                alloc = int(row.get("blocks_512") or 0) * 512
                item = {
                    "path": str(path), "type": kind,
                    "uid": row.get("uid"), "gid": row.get("gid"),
                    "mode_octal": row.get("mode_octal"),
                    "size_bytes": size, "allocated_bytes": alloc,
                    "mtime_ns": row.get("mtime_ns"), "device": row.get("dev"),
                    "inode": row.get("ino"), "nlink": row.get("nlink"),
                    "symlink_target": row.get("target"),
                }
                encoded = (json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                out.write(encoded)
                manifest_sha.update(encoded)
                counts[kind] += 1
                allocated_all_paths += alloc
                if kind == "FILE":
                    logical += size
                    allocated_files += alloc

    if not started or not finished or seen_roots != ROOTS:
        raise RuntimeError("scan stream lacks completion markers or exact root coverage")
    expected = summary["counts"]
    if counts != {"FILE": expected["file"], "DIRECTORY": expected["dir"],
                  "SYMLINK": expected["symlink"], "OTHER": 0}:
        raise RuntimeError("inventory counts differ from summary")
    if counts["FILE"] != 25232 or logical != 1268083798:
        raise RuntimeError("collaborator file count/size does not match the reviewed current summary")

    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    output_sha = sha256(OUTPUT)
    con = sqlite3.connect(DB, timeout=600)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=600000")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute("SELECT scoped_path,filesystem_name,mount_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
        if root != (str(PARENT), "data", "/data"):
            raise RuntimeError("expected /data/pclancy3 catalog root is missing or changed")
        prior = con.execute("SELECT manifest_sha256,allocated_bytes FROM inventory_snapshot WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()
        if prior:
            if prior[0] != output_sha:
                raise RuntimeError("snapshot ID is already bound to different evidence")
            n = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()[0]
            if prior[1] != allocated_files:
                con.execute("UPDATE inventory_snapshot SET allocated_bytes=? WHERE snapshot_id=?", (allocated_files, SNAPSHOT_ID))
                con.commit()
                print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "ALLOCATED_FILE_BYTES_CORRECTED",
                                  "observations": n, "allocated_file_bytes": allocated_files}))
                return
            con.commit()
            print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "ALREADY_IMPORTED", "observations": n}))
            return

        sources = []
        for path, key, kind, note in (
            (OUTPUT, "rockfish_collab_roots_r04_composite", "COLLAB_ROOTS_R04_COMPOSITE_MANIFEST", "Deterministic current metadata subset for exactly three declared /data/pclancy3 roots."),
            (RAW, "rockfish_collab_roots_r04_raw", "COLLAB_ROOTS_R04_RAW_METADATA", "Raw no-follow metadata refresh for ycao73_home, for-victor and for-yuqi only."),
            (SUMMARY, "rockfish_collab_roots_r04_summary", "COLLAB_ROOTS_R04_SCAN_SUMMARY", "Completion, count, exact root list and large-file summary for the three selected roots."),
            (SCANNER, "rockfish_collab_roots_r04_scanner", "COLLAB_ROOTS_R04_SCAN_SCRIPT", "Read-only scanner used for the exact three-root metadata refresh."),
            (REPORT, "rockfish_collab_roots_r04_report", "COLLAB_ROOTS_R04_REPORT", "Human-readable current comparison and candidate/hold summary for the three roots."),
        ):
            resolved = path.resolve(strict=True)
            digest = sha256(resolved)
            source_key = key + ":" + digest[:16]
            previous = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
            if previous and previous != (str(resolved), digest):
                raise RuntimeError("source key collision: " + source_key)
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                    immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (source_key, str(resolved), kind, digest, captured, finished,
                 "Exact captured file bytes identified by SHA-256.", note),
            )
            sources.append(con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0])

        con.execute("CREATE TEMP TABLE collab_stage(relative_path TEXT PRIMARY KEY,entry_type TEXT,uid INTEGER,gid INTEGER,mode_octal TEXT,logical_bytes INTEGER,allocated_bytes INTEGER,mtime_ns INTEGER,inode TEXT,device TEXT,symlink_target TEXT,link_count INTEGER) WITHOUT ROWID")
        batch = []
        with gzip.open(OUTPUT, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                path = PurePosixPath(row["path"])
                rel = str(path.relative_to(PARENT))
                batch.append((rel, row["type"], row.get("uid"), row.get("gid"), row.get("mode_octal"),
                              row["size_bytes"] if row["type"] == "FILE" else None, row["allocated_bytes"],
                              row.get("mtime_ns"), str(row["inode"]) if row.get("inode") is not None else None,
                              str(row["device"]) if row.get("device") is not None else None,
                              row.get("symlink_target"), row.get("nlink")))
                if len(batch) >= 10000:
                    con.executemany("INSERT INTO collab_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            con.executemany("INSERT INTO collab_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        path_count = con.execute("SELECT COUNT(*) FROM collab_stage").fetchone()[0]
        type_counts = dict(con.execute("SELECT entry_type,COUNT(*) FROM collab_stage GROUP BY entry_type").fetchall())
        if path_count != sum(counts.values()):
            raise RuntimeError("staged collaborator paths differ from manifest counts")

        note = ("Complete no-follow current metadata for exactly /data/pclancy3/ycao73_home, "
                "/data/pclancy3/for-victor and /data/pclancy3/for-yuqi; this is not a full /data/pclancy3 scan. "
                "The two former-collaborator duplicate-cleanup batches are separately recorded; remaining unique paths are retained. "
                "Git pack payloads were not read or removed.")
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES(?,?,?,?,'Exact disjoint selected-root no-follow metadata scan','collab-r04-v1','COMPLETE',?,?,?,0,?,?,?,?,?)""",
            (SNAPSHOT_ID, ROOT_ID, started, finished, counts["FILE"], counts["DIRECTORY"], counts["SYMLINK"],
             logical, allocated_files, str(OUTPUT.resolve()), output_sha, note),
        )
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) SELECT ?,relative_path,? FROM collab_stage", (ROOT_ID, finished))
        con.execute(
            """INSERT INTO inventory_observation
               (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,
                inode,device,symlink_target,hash_scope,observation_status,link_count)
               SELECT ?,p.path_id,s.entry_type,s.uid,s.gid,s.mode_octal,s.logical_bytes,s.allocated_bytes,
                      s.mtime_ns,s.inode,s.device,s.symlink_target,'NONE','PRESENT',s.link_count
               FROM collab_stage s JOIN storage_path p ON p.root_id=? AND p.relative_path=s.relative_path""",
            (SNAPSHOT_ID, ROOT_ID),
        )
        for source_id in sorted(set(sources)):
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'inventory_source',?,'INVENTORY_SNAPSHOT',?,'CAPTURES',?)""",
                (source_id, str(source_id), SNAPSHOT_ID, "Evidence source for the exact selected-root R04 metadata snapshot."),
            )
        stored = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()[0]
        if stored != path_count:
            raise RuntimeError(f"stored observations differ from selected-root path count: {stored} != {path_count}")
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign-key check failed: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "IMPORTED_COMPLETE_SELECTED_ROOTS",
                      "paths": path_count, "types": type_counts, "logical_file_bytes": logical,
                      "allocated_file_bytes": allocated_files, "allocated_all_path_bytes": allocated_all_paths,
                      "manifest_sha256": output_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
