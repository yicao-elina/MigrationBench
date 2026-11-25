#!/usr/bin/env python3
"""Import the exact R04 direct-target lstat sample without implying tree coverage."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
SCAN = AUDIT / "phase2_scratch4/scratch4_refresh_20261123_r04_remaining"
STATUS = SCAN / "cganley_users_direct_lstat.status.json"
LEDGER = SCAN / "cganley_users_direct_lstat.jsonl"
ROOT = "/scratch4/pclancy3/scratch_archive_cganley2/users"
SNAPSHOT_ID = "RF-R04-LSTAT-CGANLEY-USERS-20261123T060735Z-PARTIAL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slug(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")


def main() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if status.get("returncode") != 0 or len(status.get("targets", [])) != 31:
        raise RuntimeError("R04 exact-target lstat status is not a clean 31-target capture")
    if status.get("output") != "phase2_scratch4/scratch4_refresh_20261123_r04_remaining/cganley_users_direct_lstat.jsonl":
        raise RuntimeError("unexpected lstat output path")
    if status.get("sha256") != sha256(LEDGER):
        raise RuntimeError("lstat ledger hash differs from status")

    targets = set(status.get("targets", []))
    staged: dict[str, tuple] = {}
    start = finish = None
    with LEDGER.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record") == "lstat_start":
                start = row.get("started_utc")
                if row.get("count") != len(targets) or row.get("uid") != 3901:
                    raise RuntimeError("lstat header disagrees with its exact target set")
                continue
            if row.get("record") == "lstat_end":
                finish = row.get("finished_utc")
                if row.get("counts") != {"ok": len(targets), "error": 0}:
                    raise RuntimeError("lstat footer reports incomplete targets")
                continue
            path = row.get("path")
            if path not in targets or (path != ROOT and not path.startswith(ROOT + "/")):
                raise RuntimeError("lstat returned an unexpected target path: " + repr(path))
            if path in staged:
                raise RuntimeError("duplicate lstat path: " + path)
            kind = {"file": "FILE", "dir": "DIRECTORY", "symlink": "SYMLINK", "special": "OTHER"}.get(row.get("record"))
            if kind is None:
                raise RuntimeError("unsupported lstat record: " + repr(row.get("record")))
            rel = str(PurePosixPath(path).relative_to(PurePosixPath(ROOT)))
            staged[path] = (
                rel, kind, row.get("uid"), row.get("gid"), row.get("mode_octal"),
                row.get("size_bytes") if kind == "FILE" else None,
                row.get("allocated_bytes"), row.get("mtime_ns"),
                str(row["inode"]) if row.get("inode") is not None else None,
                str(row["device"]) if row.get("device") is not None else None,
                row.get("symlink_target"), row.get("nlink"),
            )
    if not start or not finish or set(staged) != targets:
        raise RuntimeError("lstat stream does not exactly cover the declared target set")

    started, observed = iso(start), iso(finish)
    root_id = "RF-R04-SCRATCH4-" + slug(ROOT.removeprefix("/scratch4/"))
    logical = sum(int(x[5] or 0) for x in staged.values() if x[1] == "FILE")
    allocated = sum(int(x[6] or 0) for x in staged.values() if x[1] == "FILE")
    type_counts = {kind: sum(x[1] == kind for x in staged.values()) for kind in ("FILE", "DIRECTORY", "SYMLINK", "OTHER")}
    ledger_sha, status_sha = sha256(LEDGER), sha256(STATUS)
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    con = sqlite3.connect(DB, timeout=300)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=300000")
    try:
        con.execute("BEGIN IMMEDIATE")
        root_row = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (root_id,)).fetchone()
        if not root_row or root_row[0] != ROOT:
            raise RuntimeError("R04 parent storage root must be imported before this targeted sample")
        existing = con.execute("SELECT manifest_sha256 FROM inventory_snapshot WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()
        if existing:
            if existing[0] != ledger_sha:
                raise RuntimeError("snapshot ID is already bound to different lstat bytes")
            con.commit()
            print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "ALREADY_IMPORTED", "observations": len(staged)}))
            return

        for path, key, kind, as_of, note in (
            (LEDGER, "rockfish_r04_direct_lstat_20261123T060735Z", "ROCKFISH_R04_TARGETED_LSTAT_JSONL", observed,
             "Exact no-follow metadata sample of 31 declared ancestor/direct-file paths; not a recursive inventory."),
            (STATUS, "rockfish_r04_direct_lstat_status_20261123T060735Z", "ROCKFISH_R04_TARGETED_LSTAT_STATUS_JSON", observed,
             "Status and exact target list for the 31-path R04 no-follow lstat sample."),
        ):
            path = path.resolve(strict=True)
            digest = sha256(path)
            prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
            if prior and prior != (str(path), digest):
                raise RuntimeError("lstat registry source key conflict: " + key)
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (key, str(path), kind, digest, captured, as_of,
                 "Read-only no-follow metadata capture; no content reads or hashes.", note),
            )

        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES(?,?,?,?,?,'scoped-r04-lstat-v1','PARTIAL',?,?,?,?,?,?,?,?,?)""",
            (SNAPSHOT_ID, root_id, started, observed,
             "Exact-target os.lstat sample; symlinks not followed; no payload reads or hashes.",
             type_counts["FILE"], type_counts["DIRECTORY"], type_counts["SYMLINK"], 0,
             logical, allocated, str(LEDGER.resolve()), ledger_sha,
             "PARTIAL sample of exactly 31 declared ancestor/direct-file paths. "
             "This does not represent complete directory contents; absence from this sample is not deletion evidence."),
        )
        for item in staged.values():
            rel, kind, uid, gid, mode, size, alloc, mtime, inode, device, target, nlink = item
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (root_id, rel, observed))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, rel)).fetchone()[0]
            con.execute(
                """INSERT INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,
                    inode,device,symlink_target,hash_scope,observation_status,link_count)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,'NONE','PRESENT',?)""",
                (SNAPSHOT_ID, path_id, kind, uid, gid, mode, size, alloc, mtime, inode, device, target, nlink),
            )
        for key, note in (
            ("rockfish_r04_direct_lstat_20261123T060735Z", "Exact path metadata for the R04 31-target lstat sample."),
            ("rockfish_r04_direct_lstat_status_20261123T060735Z", "Scope and completion status of the R04 31-target lstat sample."),
        ):
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES((SELECT source_id FROM registry_source WHERE source_key=?),
                          'filesystem_inventory',?,'INVENTORY_SNAPSHOT',?,'CAPTURES',?)""",
                (key, SNAPSHOT_ID, SNAPSHOT_ID, note),
            )
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign-key check failed: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "IMPORTED_PARTIAL_TARGET_SAMPLE",
                      "observations": len(staged), "files": type_counts["FILE"],
                      "directories": type_counts["DIRECTORY"], "logical_bytes": logical,
                      "allocated_bytes": allocated, "ledger_sha256": ledger_sha,
                      "status_sha256": status_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
