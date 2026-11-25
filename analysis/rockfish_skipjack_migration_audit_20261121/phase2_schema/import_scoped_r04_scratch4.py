#!/usr/bin/env python3
"""Import exact isolated R04 Scratch4 metadata captures, preserving partial scans."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
SCAN_DIR = AUDIT / "phase2_scratch4/scratch4_refresh_20261123_r04_remaining"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iso(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")


def specifications() -> list[dict]:
    result = []
    for status_path in sorted(SCAN_DIR.glob("*.status.json")):
        status = json.loads(status_path.read_text())
        ledger_path = Path(status.get("stdout_gzip", ""))
        if not ledger_path.is_absolute():
            ledger_path = AUDIT / ledger_path
        if not ledger_path.is_file():
            ledger_path = status_path.with_name(status_path.name.removesuffix(".status.json") + ".jsonl.gz")
        if not ledger_path.is_file():
            if status_path.name == "cganley_users_direct_lstat.status.json":
                # This exact-target sample is imported by import_r04_targeted_lstat.py.
                continue
            raise RuntimeError("status has no matching gzip ledger: " + str(status_path))
        root = status.get("root")
        if not root or not root.startswith("/scratch4/pclancy3/"):
            raise RuntimeError("unexpected R04 subtree root: " + repr(root))
        result.append({"status_path": status_path, "ledger_path": ledger_path, "status": status, "root": root})
    return result


def source(con: sqlite3.Connection, path: Path, key: str, kind: str,
           captured: str, as_of: str, note: str) -> int:
    path = path.resolve(strict=True)
    sha = digest(path)
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if prior and prior != (str(path), sha):
        raise RuntimeError("source key identifies different bytes: " + key)
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (key, str(path), kind, sha, captured, as_of,
         "Read-only no-follow metadata scan; exact source file identified by SHA-256.", note),
    )
    return con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0]


def relative_path(root: str, path: str) -> str:
    absolute, scope = PurePosixPath(path), PurePosixPath(root)
    try:
        return "" if absolute == scope else str(absolute.relative_to(scope))
    except ValueError as exc:
        raise RuntimeError("ledger path escaped declared scope: " + path) from exc


def import_one(con: sqlite3.Connection, spec: dict, captured: str) -> dict:
    status, ledger = spec["status"], spec["ledger_path"].resolve(strict=True)
    ledger_sha = digest(ledger)
    expected_sha = status.get("archive_sha256") or status.get("stdout_sha256")
    if expected_sha and expected_sha != ledger_sha:
        raise RuntimeError("status digest differs from ledger: " + str(ledger))
    status_name = status.get("status")
    complete = (status_name == "COMPLETE" and status.get("returncode") == 0 and
                status.get("scan_end_record_present") is True and status.get("traversal_errors", 0) == 0 and
                status.get("parse_error") is None)
    partial = (status_name == "PARTIAL" and status.get("scan_end_record_present") is False and
               status.get("returncode") not in (None, 0))
    if not (complete or partial):
        return {"root": spec["root"], "status": "SKIPPED_UNSUPPORTED_STATUS"}

    root = spec["root"]
    start = None
    finish = None
    declared = None
    staged: dict[str, tuple] = {}
    parse_errors = 0
    truncated = False
    try:
        with gzip.open(ledger, "rt", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if partial:
                        truncated = True
                        break
                    raise
                record = row.get("record")
                if record == "scan_start":
                    if row.get("root") != root:
                        raise RuntimeError("scan start root mismatch: " + root)
                    start = row.get("started_utc")
                    continue
                if record == "scan_end":
                    if row.get("root") != root:
                        raise RuntimeError("scan end root mismatch: " + root)
                    finish = row.get("finished_utc") or row.get("finished_at")
                    declared = row.get("counts")
                    continue
                path = row.get("path")
                if not path:
                    continue
                kind = {"file": "FILE", "dir": "DIRECTORY", "symlink": "SYMLINK", "special": "OTHER"}.get(record)
                if record == "error":
                    parse_errors += 1
                    continue
                if kind is None:
                    continue
                rel = relative_path(root, path)
                item = (rel, kind, row.get("uid"), row.get("gid"), row.get("mode_octal"),
                        int(row["size_bytes"]) if kind == "FILE" and row.get("size_bytes") is not None else None,
                        int(row["allocated_bytes"]) if row.get("allocated_bytes") is not None else None,
                        int(row["mtime_ns"]) if row.get("mtime_ns") is not None else None,
                        str(row["inode"]) if row.get("inode") is not None else None,
                        str(row["device"]) if row.get("device") is not None else None,
                        row.get("symlink_target"), int(row["nlink"]) if row.get("nlink") is not None else None,
                        row.get("owner"), row.get("group"))
                if rel in staged:
                    raise RuntimeError("duplicate path row in ledger: " + path)
                staged[rel] = item
    except EOFError:
        if not partial:
            raise
        truncated = True
    except OSError:
        if not partial:
            raise
        truncated = True

    if not start:
        raise RuntimeError("R04 ledger lacks scan_start: " + str(ledger))
    if complete and (not finish or not isinstance(declared, dict)):
        raise RuntimeError("complete R04 ledger lacks scan_end: " + str(ledger))
    if partial and finish:
        raise RuntimeError("partial status unexpectedly includes a scan_end marker: " + root)

    files = sum(item[1] == "FILE" for item in staged.values())
    directories = sum(item[1] == "DIRECTORY" for item in staged.values())
    symlinks = sum(item[1] == "SYMLINK" for item in staged.values())
    special = sum(item[1] == "OTHER" for item in staged.values())
    if complete:
        got = {"file": files, "dir": directories, "symlink": symlinks, "special": special, "error": parse_errors}
        expected = {key: int(declared.get(key, 0)) for key in got}
        if got != expected:
            raise RuntimeError("R04 ledger counts differ from scan_end for " + root + ": " + repr((got, expected)))
        if (files, directories, symlinks, special) != (
                int(status["counts"].get("file", -1)), int(status["counts"].get("dir", -1)),
                int(status["counts"].get("symlink", -1)), int(status["counts"].get("special", -1))):
            raise RuntimeError("R04 status counts differ from ledger for " + root)

    observed = iso(finish or status.get("observed_utc") or status.get("scan_start", {}).get("started_utc"))
    started = iso(start)
    snapshot_suffix = started.replace(":", "").replace("-", "")
    snapshot_id = "RF-R04-SCRATCH4-" + slug(root.removeprefix("/scratch4/")) + "-" + snapshot_suffix + ("-PARTIAL" if partial else "")
    root_id = "RF-R04-SCRATCH4-" + slug(root.removeprefix("/scratch4/"))
    root_owner = next((item[12] for rel, item in staged.items() if rel == ""), None)
    root_group = next((item[13] for rel, item in staged.items() if rel == ""), None)
    if root_owner is None:
        root_owner = status.get("root_owner")
    if root_group is None:
        root_group = status.get("root_group") or "pclancy3"
    scope_kind = "USER_SUBTREE" if str(root_owner) in ("ycao73", "3901") else "GROUP_SUBTREE"
    prior_root = con.execute("SELECT scoped_path,filesystem_name,mount_path FROM storage_root WHERE root_id=?", (root_id,)).fetchone()
    expected_root = (root, "scratch4", "/scratch4")
    if prior_root and prior_root != expected_root:
        raise RuntimeError("storage root identifier conflict: " + root_id)
    con.execute(
        """INSERT OR IGNORE INTO storage_root
           (root_id,cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,scope_kind,status,notes)
           VALUES(?,'Rockfish','scratch4','/scratch4',?,?,?,?, 'ACTIVE',?)""",
        (root_id, root, root_owner, root_group, scope_kind,
         "R04 isolated no-follow metadata subtree; no file payloads or hashes read. " +
         ("Owner is " + str(root_owner) + "; per-path UID/GID remain authoritative." if root_owner else "Root owner unavailable; per-path UID/GID are authoritative.")),
    )
    if con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (root_id,)).fetchone()[0] != root:
        raise RuntimeError("failed to register exact R04 storage root: " + root)

    existing = con.execute("SELECT manifest_sha256 FROM inventory_snapshot WHERE snapshot_id=?", (snapshot_id,)).fetchone()
    if existing:
        if existing[0] != ledger_sha:
            raise RuntimeError("snapshot ID already maps to a different ledger: " + snapshot_id)
        count = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (snapshot_id,)).fetchone()[0]
        return {"root": root, "snapshot_id": snapshot_id, "status": "ALREADY_IMPORTED", "observations": count}

    ledger_source = source(con, ledger, "rockfish_r04_inventory:" + snapshot_id + ":" + ledger_sha[:16],
                           "ROCKFISH_R04_SUBTREE_JSONL_GZIP", captured, observed,
                           ("Complete isolated R04 scan of " if complete else "Timed-out partial R04 lower-bound scan of ") + root + "; not a mount-wide scan.")
    status_path = spec["status_path"].resolve(strict=True)
    status_sha = digest(status_path)
    status_source = source(con, status_path, "rockfish_r04_status:" + snapshot_id + ":" + status_sha[:16],
                           "ROCKFISH_R04_SUBTREE_STATUS_JSON", captured, observed,
                           "Exact R04 per-root completion and timeout status record.")
    logical = sum(item[5] or 0 for item in staged.values() if item[1] == "FILE")
    allocated = sum(item[6] or 0 for item in staged.values() if item[1] == "FILE")
    note = "No-follow metadata only; no payload reads or hashes. "
    if complete:
        note += "COMPLETE for this exact subtree only."
    else:
        note += ("PARTIAL lower bound; timeout return code=" + str(status.get("returncode")) +
                 ", no scan_end marker, compressed_stream_truncated=" + str(truncated) +
                 "; absent paths must not be treated as deleted.")
    con.execute(
        """INSERT INTO inventory_snapshot
           (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
            files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
            manifest_path,manifest_sha256,notes)
           VALUES(?,?,?,?,?,'scoped-r04-v1',?,?,?,?,?,?,?,?,?,?)""",
        (snapshot_id, root_id, started, observed,
         "Independent os.scandir/lstat metadata scan; symlinks not followed; no content reads or hashes.",
         "COMPLETE" if complete else "PARTIAL", files, directories, symlinks, parse_errors, logical,
         allocated, str(ledger), ledger_sha, note),
    )
    for item in staged.values():
        rel, kind, uid, gid, mode, size, alloc, mtime, inode, device, target, nlink = item[:12]
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (root_id, rel, observed))
        path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, rel)).fetchone()[0]
        con.execute(
            """INSERT INTO inventory_observation
               (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,symlink_target,
                hash_scope,observation_status,link_count)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,'NONE','PRESENT',?)""",
            (snapshot_id, path_id, kind, uid, gid, mode, size, alloc, mtime, inode, device, target, nlink),
        )
    for source_id, note_text in ((ledger_source, "Exact raw metadata ledger for this R04 subtree snapshot."),
                                 (status_source, "Completion/timeout status for this exact R04 subtree snapshot.")):
        con.execute(
            """INSERT OR IGNORE INTO registry_entity_ref
               (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
               VALUES(?,'filesystem_inventory',?,'INVENTORY_SNAPSHOT',?,'CAPTURES',?)""",
            (source_id, snapshot_id, snapshot_id, note_text),
        )
    count = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (snapshot_id,)).fetchone()[0]
    if count != len(staged):
        raise RuntimeError("R04 inventory observation count mismatch: " + snapshot_id)
    return {"root": root, "snapshot_id": snapshot_id, "status": "IMPORTED_" + ("COMPLETE" if complete else "PARTIAL"),
            "observations": count, "files": files, "directories": directories, "symlinks": symlinks,
            "logical_bytes": logical, "allocated_bytes": allocated, "sha256": ledger_sha}


def main() -> None:
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=300)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=300000")
    results = []
    try:
        con.execute("BEGIN IMMEDIATE")
        for spec in specifications():
            results.append(import_one(con, spec, captured))
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign-key check failed: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"imported": results, "foreign_key_violations": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
