#!/usr/bin/env python3
"""Import the bounded metadata-only R05 census of accessible /data roots.

This is a PARTIAL mount snapshot: 625 immediate directories were not
traversable, 33 readable roots had incomplete scans, and payload bytes were
never read. Missing descendants are not interpreted as deleted.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
DATA = AUDIT / "phase2_data/data_mount_accessible_census_20261123"
ACCESS = DATA / "all_other_data_root_access_20261123.json"
ROOT_LIST = DATA / "accessible_other_data_roots_20261123.json"
SUMMARY = DATA / "accessible_data_census_summary_20261123.json"
REPORT = DATA / "ACCESSIBLE_DATA_ROOT_CENSUS_20261123.md"
SCANNER = DATA / "scan_accessible_data_roots_r04.py"
LIST_CAPTURE = DATA / "capture_accessible_data_root_list.py"
OUT = DATA / "accessible_data_mount_r05_normalized_manifest_20261123.jsonl.gz"
ROOT_ID = "RF-DATA-FS"
MOUNT = PurePosixPath("/data")
SNAPSHOT_ID = "RF-DATA-FS-R05-ACCESSIBLE-ROOT-CENSUS-20261123T085425Z-PARTIAL"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_epoch(value: float | str) -> str:
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def mtime_ns(value: str | float | int | None) -> int | None:
    if value is None:
        return None
    return int(Decimal(str(value)) * Decimal(1_000_000_000))


def relative(path: str) -> str:
    absolute = PurePosixPath(path)
    try:
        return str(absolute.relative_to(MOUNT))
    except ValueError as exc:
        raise RuntimeError(f"out-of-mount path in census: {path}") from exc


def build_manifest() -> tuple[str, dict]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    access = json.loads(ACCESS.read_text(encoding="utf-8"))
    root_list = json.loads(ROOT_LIST.read_text(encoding="utf-8"))
    if access.get("mount") != "/data" or access.get("counts", {}).get("denied") != 625:
        raise RuntimeError("unexpected /data boundary capture")
    if summary.get("expected_accessible_other_roots") != 99 or summary.get("scanned_root_summaries") != 99:
        raise RuntimeError("unexpected accessible-root coverage")
    if len(root_list.get("roots", [])) != 99:
        raise RuntimeError("root list does not contain exactly 99 accessible roots")

    rows: dict[str, dict] = {}
    starts: list[float] = []
    finishes: list[float] = []
    state_by_root: dict[str, str] = {}
    raw_paths = sorted(DATA.glob("batch_*.jsonl.gz"))
    if len(raw_paths) != 20:
        raise RuntimeError(f"expected 20 compressed scan ledgers, found {len(raw_paths)}")
    for raw in raw_paths:
        with gzip.open(raw, "rt", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                record = event.get("record")
                if record == "root_start":
                    starts.append(float(event["started_epoch"]))
                    continue
                if record == "root_summary":
                    finishes.append(float(event["finished_epoch"]))
                    state_by_root[event["root"]] = event["completion_state"]
                    continue
                if record != "entry":
                    raise RuntimeError(f"unknown raw census event {record!r}")
                path = event["path"]
                typ = {"f": "FILE", "d": "DIRECTORY", "l": "SYMLINK"}.get(event["type"], "OTHER")
                row = {
                    "path": path,
                    "type": typ,
                    "uid": event.get("uid"), "gid": event.get("gid"),
                    "mode_octal": event.get("mode_octal"),
                    "size_bytes": int(event["size_bytes"]) if typ == "FILE" else None,
                    "allocated_bytes": int(event["blocks_512"]) * 512 if event.get("blocks_512") is not None else None,
                    "mtime_ns": mtime_ns(event.get("mtime_epoch")),
                    "device": str(event["device"]) if event.get("device") is not None else None,
                    "inode": str(event["inode"]) if event.get("inode") is not None else None,
                    "nlink": event.get("nlink"),
                    "symlink_target": event.get("symlink_target"),
                    "observation_status": "PRESENT",
                }
                if path in rows:
                    raise RuntimeError(f"duplicate census path: {path}")
                rows[path] = row

    expected = int(summary["observed_entries"])
    if len(rows) != expected:
        raise RuntimeError(f"raw ledger path count changed: {len(rows)} != {expected}")
    if sum(row["type"] == "FILE" for row in rows.values()) != int(summary["observed_files"]):
        raise RuntimeError("raw ledger file count differs from census summary")
    if len(state_by_root) != 99 or sum(x == "COMPLETE" for x in state_by_root.values()) != 66:
        raise RuntimeError("per-root completion states differ from census report")

    # Root-level lstat rows preserve known directory/file metadata even where
    # traversal was denied. Mark denied directories UNREADABLE; descendants
    # are absent because they were not observable, not because they were deleted.
    for entry in access["other_directories"]:
        path = entry["path"]
        if path in rows:
            existing = rows[path]
            if existing["uid"] != entry["uid"] or existing["gid"] != entry["gid"]:
                raise RuntimeError(f"root lstat differs from traversal record: {path}")
            continue
        rows[path] = {
            "path": path, "type": "DIRECTORY", "uid": entry.get("uid"), "gid": entry.get("gid"),
            "mode_octal": entry.get("mode_octal"), "size_bytes": None, "allocated_bytes": None,
            "mtime_ns": entry.get("mtime_ns"), "device": None, "inode": None, "nlink": None,
            "symlink_target": None,
            "observation_status": "UNREADABLE" if entry["access"] == "DENIED" else "PRESENT",
        }
    project_root = access["project_root"]
    rows[project_root["path"]] = {
        "path": project_root["path"], "type": "DIRECTORY", "uid": project_root.get("uid"),
        "gid": project_root.get("gid"), "mode_octal": project_root.get("mode_octal"),
        "size_bytes": None, "allocated_bytes": None, "mtime_ns": project_root.get("mtime_ns"),
        "device": None, "inode": None, "nlink": None, "symlink_target": None,
        "observation_status": "PRESENT",
    }
    for entry in access["root_files"]:
        rows[entry["path"]] = {
            "path": entry["path"], "type": "FILE", "uid": entry.get("uid"), "gid": entry.get("gid"),
            "mode_octal": entry.get("mode_octal"), "size_bytes": None, "allocated_bytes": None,
            "mtime_ns": entry.get("mtime_ns"), "device": None, "inode": None, "nlink": None,
            "symlink_target": None, "observation_status": "PRESENT",
        }

    normalized = sorted(rows.values(), key=lambda row: row["path"])
    counts = {typ: sum(row["type"] == typ for row in normalized) for typ in ("FILE", "DIRECTORY", "SYMLINK", "OTHER")}
    logical_known = sum(int(row["size_bytes"] or 0) for row in normalized if row["type"] == "FILE")
    allocated_known_files = sum(int(row["allocated_bytes"] or 0) for row in normalized if row["type"] == "FILE")
    allocated_all_paths = sum(int(row["allocated_bytes"] or 0) for row in normalized)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha = hashlib.sha256()
    with OUT.open("wb") as raw_out:
        with gzip.GzipFile(fileobj=raw_out, filename="", mode="wb", compresslevel=6, mtime=0) as out:
            for row in normalized:
                encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
                out.write(encoded)
                manifest_sha.update(encoded)
    return manifest_sha.hexdigest(), {
        "summary": summary, "counts": counts, "logical_known": logical_known,
        "allocated_known_files": allocated_known_files, "allocated_all_paths": allocated_all_paths,
        "manifest_sha256": manifest_sha.hexdigest(), "rows": len(normalized),
        "started_utc": utc_epoch(min(starts)), "finished_utc": utc_epoch(max(finishes)),
        "state_by_root": state_by_root,
    }


def register_source(con: sqlite3.Connection, path: Path, key: str, kind: str, captured: str,
                    as_of: str, note: str) -> int:
    resolved = path.resolve(strict=True)
    digest = sha256(resolved)
    source_key = f"{key}:{digest[:16]}"
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
    if prior and tuple(prior) != (str(resolved), digest):
        raise RuntimeError("registry source key collision: " + source_key)
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (source_key, str(resolved), kind, digest, captured, as_of,
         "Exact evidence bytes identified by SHA-256; this source contains metadata, not payload data.", note),
    )
    return con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]


def main() -> None:
    manifest_sha, data = build_manifest()
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    output_sha = sha256(OUT)
    con = sqlite3.connect(DB, timeout=600)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=600000")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute("SELECT scoped_path,filesystem_name,mount_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
        if not root or tuple(root) != ("/data", "data", "/data"):
            raise RuntimeError("expected /data storage root is missing or changed")
        prior = con.execute("SELECT manifest_sha256 FROM inventory_snapshot WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()
        if prior:
            if prior[0] != output_sha:
                raise RuntimeError("snapshot ID is already bound to different manifest bytes")
            count = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()[0]
            con.commit()
            print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "ALREADY_IMPORTED", "observations": count,
                              "counts": data["counts"], "logical_known_bytes": data["logical_known"],
                              "manifest_sha256": output_sha}, sort_keys=True))
            return

        sources: list[int] = []
        fixed = [
            (OUT, "rockfish_data_mount_r05_normalized_manifest", "DATA_MOUNT_R05_NORMALIZED_METADATA_MANIFEST", "Deterministic path-sorted combined metadata manifest for the accessible-root R05 census."),
            (ACCESS, "rockfish_data_mount_access_boundary", "DATA_MOUNT_IMMEDIATE_ROOT_ACCESS_CAPTURE", "Immediate /data lstat and traversal results: 625 denied directories, 99 readable roots, the separately scanned project root, and two root files."),
            (ROOT_LIST, "rockfish_data_mount_accessible_root_list", "DATA_MOUNT_ACCESSIBLE_ROOT_LIST", "Metadata-only list of the 99 accessible immediate roots scanned recursively."),
            (SUMMARY, "rockfish_data_mount_accessible_census_summary", "DATA_MOUNT_ACCESSIBLE_CENSUS_SUMMARY", "Counts and lower-bound aggregates for the 99 bounded scans."),
            (REPORT, "rockfish_data_mount_accessible_census_report", "DATA_MOUNT_ACCESSIBLE_CENSUS_REPORT", "Coverage, lower bounds, ownership limits and evidence index."),
            (SCANNER, "rockfish_data_mount_accessible_census_scanner", "DATA_MOUNT_CENSUS_SCANNER", "Bounded metadata-only find -xdev scanner used for each accessible root."),
            (LIST_CAPTURE, "rockfish_data_mount_root_list_scanner", "DATA_MOUNT_ROOT_LIST_CAPTURE_SCRIPT", "Metadata-only script that captured root access and lstat information."),
            (Path(__file__).resolve(), "rockfish_data_mount_r05_importer", "PROVENANCE_IMPORT_SCRIPT", "Idempotent importer for the partial mount-level R05 census."),
        ]
        for path, key, kind, note in fixed:
            sources.append(register_source(con, path, key, kind, captured, data["finished_utc"], note))
        for path in sorted(DATA.glob("batch_*.jsonl.gz")) + sorted(DATA.glob("batch_*_summary.json")):
            sources.append(register_source(con, path, "rockfish_data_mount_census_evidence:" + path.stem,
                "DATA_MOUNT_CENSUS_RAW_LEDGER" if path.suffix == ".gz" else "DATA_MOUNT_CENSUS_BATCH_SUMMARY",
                captured, data["finished_utc"], "Immutable per-batch raw metadata ledger or aggregate completion receipt."))

        con.execute("CREATE TEMP TABLE mount_stage(relative_path TEXT PRIMARY KEY,entry_type TEXT,uid INTEGER,gid INTEGER,mode_octal TEXT,logical_bytes INTEGER,allocated_bytes INTEGER,mtime_ns INTEGER,inode TEXT,device TEXT,symlink_target TEXT,link_count INTEGER,observation_status TEXT) WITHOUT ROWID")
        batch = []
        with gzip.open(OUT, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                rel = relative(row["path"])
                batch.append((rel, row["type"], row.get("uid"), row.get("gid"), row.get("mode_octal"),
                    row.get("size_bytes"), row.get("allocated_bytes"), row.get("mtime_ns"), row.get("inode"),
                    row.get("device"), row.get("symlink_target"), row.get("nlink"), row["observation_status"]))
                if len(batch) >= 10000:
                    con.executemany("INSERT INTO mount_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            con.executemany("INSERT INTO mount_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        path_count = con.execute("SELECT COUNT(*) FROM mount_stage").fetchone()[0]
        if path_count != data["rows"]:
            raise RuntimeError("staged path count differs from normalized manifest")
        unreadable = con.execute("SELECT COUNT(*) FROM mount_stage WHERE observation_status='UNREADABLE'").fetchone()[0]
        if unreadable != 625:
            raise RuntimeError(f"expected 625 non-traversable roots, observed {unreadable}")
        file_count = con.execute("SELECT COUNT(*) FROM mount_stage WHERE entry_type='FILE'").fetchone()[0]
        directory_count = con.execute("SELECT COUNT(*) FROM mount_stage WHERE entry_type='DIRECTORY'").fetchone()[0]
        symlink_count = con.execute("SELECT COUNT(*) FROM mount_stage WHERE entry_type='SYMLINK'").fetchone()[0]
        if (file_count, directory_count, symlink_count) != (96899, 9020, 1046):
            raise RuntimeError(f"unexpected normalized entry counts: {file_count=} {directory_count=} {symlink_count=}")

        note = ("PARTIAL mount-level metadata-only census. Recursively scanned 99 other-owner immediate roots outside /data/pclancy3: "
                "66 completed and 33 truncated/errored. Captures lstat metadata for 625 non-traversable immediate roots, the /data/pclancy3 boundary, "
                "and two top-level files whose sizes were not captured. Readable roots have 106,336 observed paths before boundary rows; "
                "descendants of inaccessible and incomplete roots are unknown. /data/pclancy3 descendants are tracked in separate scoped snapshots. "
                "No payload was read or hashed. Missing paths are not deletion evidence; no other-owner cleanup decision is implied.")
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES(?,?,?,?,'Bounded find -xdev metadata scan of accessible immediate roots plus mount boundary lstat','data-mount-r05-v1','PARTIAL',?,?,?,?,?,?,?, ?,?)""",
            (SNAPSHOT_ID, ROOT_ID, data["started_utc"], data["finished_utc"], file_count, directory_count,
             symlink_count, unreadable, data["logical_known"], data["allocated_known_files"], str(OUT.resolve()), output_sha, note),
        )
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) SELECT ?,relative_path,? FROM mount_stage", (ROOT_ID, data["finished_utc"]))
        con.execute(
            """INSERT INTO inventory_observation
               (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,
                inode,device,symlink_target,hash_scope,observation_status,link_count)
               SELECT ?,p.path_id,s.entry_type,s.uid,s.gid,s.mode_octal,s.logical_bytes,s.allocated_bytes,s.mtime_ns,
                      s.inode,s.device,s.symlink_target,'NONE',s.observation_status,s.link_count
               FROM mount_stage s JOIN storage_path p ON p.root_id=? AND p.relative_path=s.relative_path""",
            (SNAPSHOT_ID, ROOT_ID),
        )
        for source_id in sorted(set(sources)):
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'inventory_source',?,'INVENTORY_SNAPSHOT',?,'CAPTURES',?)""",
                (source_id, str(source_id), SNAPSHOT_ID, "Source evidence for the partial mount-level R05 metadata census."),
            )
        stored = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()[0]
        if stored != path_count:
            raise RuntimeError(f"stored observation count differs from manifest: {stored} != {path_count}")
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign-key check failed: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "IMPORTED_PARTIAL", "observations": data["rows"],
                      "counts": data["counts"], "unreadable_roots": 625, "incomplete_accessible_roots": 33,
                      "logical_known_bytes": data["logical_known"], "allocated_known_file_bytes": data["allocated_known_files"],
                      "allocated_all_observed_paths": data["allocated_all_paths"], "manifest_sha256": output_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
