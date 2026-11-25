#!/usr/bin/env python3
"""Import complete, isolated R03 Rockfish subtree metadata scans into SQL.

Each file is a separate subtree snapshot. This never promotes a partial stream
to COMPLETE and never interprets an absent path as deleted.
"""

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
DATA_DIR = AUDIT / "phase2_data/data_refresh_20261123_r03"
SCRATCH_DIR = AUDIT / "phase2_scratch4/scratch4_refresh_20261123_r03/root_results"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iso(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")


def specs() -> list[dict]:
    found = []
    for ledger in sorted(DATA_DIR.glob("conda_env_*_r03.jsonl.gz")):
        status_path = ledger.with_name(ledger.name.removesuffix(".jsonl.gz") + ".status.json")
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        root = status["root"]
        root_id = "RF-R03-DATA-" + safe_id(root.removeprefix("/data/"))
        found.append({"ledger": ledger, "status": status_path, "root": root,
                      "root_id": root_id, "mount_path": "/data", "filesystem_name": "data",
                      "snapshot": root_id + "-20261123"})
    for ledger in sorted(SCRATCH_DIR.glob("*.jsonl.gz")):
        status_path = ledger.with_name(ledger.name.removesuffix(".jsonl.gz") + ".status.json")
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        root = status.get("root")
        if not root:
            continue
        suffix = ""
        if status.get("pruned_prefix"):
            suffix = "-PRUNED-" + safe_id(str(status["pruned_prefix"]).rsplit("/", 1)[-1])
        root_id = "RF-R03-SCRATCH4-" + safe_id(root.removeprefix("/scratch4/")) + suffix
        found.append({"ledger": ledger, "status": status_path, "root": root,
                      "root_id": root_id, "mount_path": "/scratch4", "filesystem_name": "scratch4",
                      "snapshot": root_id + "-20261123"})
    return found


def register_source(con: sqlite3.Connection, path: Path, key: str, kind: str,
                    now: str, as_of: str, note: str) -> int:
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
        (key, str(path), kind, sha, now, as_of,
         "Read-only metadata capture; SHA-256 identifies the exact compressed ledger or status record.", note),
    )
    return con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0]


def normalize(root: str, row: dict):
    path = row.get("path")
    if not path:
        return None
    absolute = PurePosixPath(path)
    scope = PurePosixPath(root)
    try:
        relative = "" if absolute == scope else str(absolute.relative_to(scope))
    except ValueError as exc:
        raise RuntimeError("scan entry escaped declared root: " + path) from exc
    kind = {"file": "FILE", "directory": "DIRECTORY", "dir": "DIRECTORY",
            "symlink": "SYMLINK", "special": "OTHER", "excluded_subtree": "DIRECTORY"}.get(row.get("type") or row.get("record"))
    if not kind:
        raise RuntimeError("unknown entry type for " + path + ": " + repr(row.get("type")))
    return (relative, kind, row.get("uid"), row.get("gid"), row.get("mode_octal"),
            int(row["size_bytes"]) if kind == "FILE" and row.get("size_bytes") is not None else None,
            int(row["allocated_bytes"]) if row.get("allocated_bytes") is not None else None,
            int(row["mtime_ns"]) if row.get("mtime_ns") is not None else None,
            str(row["inode"]) if row.get("inode") is not None else None,
            str(row["device"]) if row.get("device") is not None else None,
            row.get("symlink_target"), int(row["nlink"]) if row.get("nlink") is not None else None)


def import_one(con: sqlite3.Connection, spec: dict, now: str) -> dict:
    status = json.loads(spec["status"].read_text())
    complete_marker = (status.get("root_complete_marker") is True or
                       (status.get("status") == "COMPLETE" and status.get("scan_end_record_present") is True))
    if status.get("root") != spec["root"] or not complete_marker:
        return {"snapshot": spec["snapshot"], "status": "SKIPPED_NOT_COMPLETE"}
    exit_code = status.get("ssh_exit_code", status.get("returncode"))
    parse_errors = status.get("parse_error_count", 0 if status.get("parse_error") is None else 1)
    timed_out = status.get("timed_out", status.get("local_timeout", False))
    if exit_code != 0 or parse_errors != 0 or timed_out or status.get("traversal_errors", 0) != 0:
        return {"snapshot": spec["snapshot"], "status": "SKIPPED_FAILED_STATUS"}

    ledger = spec["ledger"].resolve(strict=True)
    ledger_sha = digest(ledger)
    expected_sha = status.get("archive_sha256") or status.get("stdout_sha256")
    if expected_sha and expected_sha != ledger_sha:
        raise RuntimeError("status hash does not match exact ledger bytes: " + str(ledger))
    start = None
    finish = None
    declared = None
    root_entry_in_scan = False
    staged: dict[str, tuple] = {}
    with gzip.open(ledger, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            record = row.get("record_type") or row.get("record")
            if record in ("root_started", "scan_start"):
                start = row.get("started_utc") or row.get("started_at")
                identity = row.get("root_identity")
                if identity:
                    item = normalize(spec["root"], identity)
                    if item:
                        staged[item[0]] = item
            elif record in ("root_complete", "scan_end"):
                finish = row.get("completed_utc") or row.get("finished_utc")
                declared = row.get("counts")
            elif record in ("file", "dir", "directory", "symlink", "special", "entry", "excluded_subtree"):
                item = normalize(spec["root"], row)
                if item:
                    staged[item[0]] = item
                    if row.get("path") == spec["root"]:
                        root_entry_in_scan = True
    if not start or not finish or not isinstance(declared, dict):
        return {"snapshot": spec["snapshot"], "status": "SKIPPED_MISSING_COMPLETE_RECORD"}

    observed = iso(finish)
    files = sum(item[1] == "FILE" for item in staged.values())
    directories = sum(item[1] == "DIRECTORY" for item in staged.values())
    symlinks = sum(item[1] == "SYMLINK" for item in staged.values())
    logical = sum(item[5] or 0 for item in staged.values() if item[1] == "FILE")
    allocated = sum(item[6] or 0 for item in staged.values() if item[1] == "FILE")
    declared_files = int(declared.get("regular_files", declared.get("file", -1)))
    declared_dirs = int(declared.get("directories", declared.get("dir", -1)))
    declared_links = int(declared.get("symlinks", declared.get("symlink", -1)))
    # The Conda scanner stores the root identity inside its start record,
    # outside the child counts. The Scratch4 scanner emits the root as an
    # ordinary row and includes it in the end-record counts.
    root_rel = ""
    counted_rows = list(staged.values()) if root_entry_in_scan else [item for rel, item in staged.items() if rel != root_rel]
    child_rows = [item for rel, item in staged.items() if rel != root_rel]
    counted_files = sum(item[1] == "FILE" for item in counted_rows)
    counted_dirs = sum(item[1] == "DIRECTORY" for item in counted_rows)
    counted_links = sum(item[1] == "SYMLINK" for item in counted_rows)
    child_files = sum(item[1] == "FILE" for item in child_rows)
    child_dirs = sum(item[1] == "DIRECTORY" for item in child_rows)
    child_links = sum(item[1] == "SYMLINK" for item in child_rows)
    if (counted_files, counted_dirs, counted_links) != (declared_files, declared_dirs, declared_links):
        raise RuntimeError("scan count mismatch for %s: captured=%r declared=%r" %
                           (spec["root"], (counted_files, counted_dirs, counted_links),
                            (declared_files, declared_dirs, declared_links)))

    existing = con.execute("SELECT manifest_sha256 FROM inventory_snapshot WHERE snapshot_id=?",
                           (spec["snapshot"],)).fetchone()
    if existing:
        if existing[0] != ledger_sha:
            raise RuntimeError("snapshot ID already exists with a different ledger hash: " + spec["snapshot"])
        count = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?",
                            (spec["snapshot"],)).fetchone()[0]
        return {"snapshot": spec["snapshot"], "status": "ALREADY_IMPORTED", "observations": count}

    root_observation = staged.get(root_rel)
    root_owner = None
    root_group = None
    if root_observation:
        root_owner = status.get("root_owner")
        root_group = status.get("root_group")
    # Resolve the actual root identity from the scan record, when the scanner
    # emitted it. Root owner is descriptive only; per-path UID/GID remain the
    # authority for file ownership.
    with gzip.open(ledger, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            record = row.get("record_type") or row.get("record")
            identity = row.get("root_identity") if record in ("root_started", "scan_start") else None
            if identity or (row.get("path") == spec["root"] and row.get("type", row.get("record")) in ("directory", "dir")):
                ident = identity or row
                root_owner = ident.get("owner")
                root_group = ident.get("group")
                break
    scope_kind = "USER_SUBTREE" if root_owner == "ycao73" else "GROUP_SUBTREE"
    root_note = ("R03 isolated metadata-only subtree scan; root owner " + str(root_owner or "not observed") +
                 "; per-path UID/GID observations are authoritative. No symlink traversal, payload reads, or hashes.")
    existing_scope = con.execute("SELECT root_id FROM storage_root WHERE scoped_path=?", (spec["root"],)).fetchone()
    if existing_scope:
        spec["root_id"] = existing_scope[0]
    prior_root = con.execute("SELECT scoped_path,filesystem_name,mount_path FROM storage_root WHERE root_id=?",
                             (spec["root_id"],)).fetchone()
    expected_root = (spec["root"], spec["filesystem_name"], spec["mount_path"])
    if prior_root and prior_root != expected_root:
        raise RuntimeError("storage_root ID already identifies another path: " + spec["root_id"])
    con.execute(
        """INSERT OR IGNORE INTO storage_root
           (root_id,cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,scope_kind,status,notes)
           VALUES(?,'Rockfish',?,?,?,?,?,?,'ACTIVE',?)""",
        (spec["root_id"], spec["filesystem_name"], spec["mount_path"], spec["root"], root_owner,
         root_group or "pclancy3", scope_kind, root_note),
    )
    root_after = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (spec["root_id"],)).fetchone()
    if not root_after or root_after[0] != spec["root"]:
        raise RuntimeError("failed to register exact subtree storage root: %s -> %r" % (spec["root_id"], root_after))
    source_id = register_source(con, ledger, "rockfish_r03_inventory:" + spec["snapshot"] + ":" + ledger_sha[:16],
                                "ROCKFISH_R03_SUBTREE_JSONL_GZIP", now, observed,
                                "Complete isolated metadata scan of " + spec["root"] + "; this is not a mount-wide scan.")
    status_id = register_source(con, spec["status"], "rockfish_r03_status:" + spec["snapshot"] + ":" + digest(spec["status"])[:16],
                                "ROCKFISH_R03_SUBTREE_STATUS_JSON", now, observed,
                                "Completion record for isolated metadata scan of " + spec["root"] + ".")
    completion_counts = declared
    con.execute(
        """INSERT INTO inventory_snapshot
           (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
            files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
            manifest_path,manifest_sha256,notes)
           VALUES(?,?,?,?,?,'scoped-r03-v1','COMPLETE',?,?,?,?,?,?,?,?,?)""",
        (spec["snapshot"], spec["root_id"], iso(start), observed,
         "Isolated os.scandir/lstat metadata scan; no symlink traversal, payload reads, or hashing.",
         declared_files, declared_dirs, declared_links, int(completion_counts.get("errors", completion_counts.get("error", 0))),
         sum(item[5] or 0 for item in child_rows if item[1] == "FILE"),
         sum(item[6] or 0 for item in child_rows if item[1] == "FILE"), str(ledger), ledger_sha,
         "COMPLETE only for the exact subtree " + spec["root"] + "; paths outside this subtree were not observed. "
         + ("The observed prune marker identifies the untraversed subtree " + str(status["pruned_prefix"]) + ". " if status.get("pruned_prefix") else "")
         + "UID 3901 file paths in scan: " + str(int(status.get("uid3901_regular_files", completion_counts.get("regular_files", -1))) if "uid3901_regular_files" in status else "not separately reported") + ".")
    )
    for item in staged.values():
        rel, kind, uid, gid, mode, size, alloc, mtime, inode, device, target, nlink = item
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
                    (spec["root_id"], rel, observed))
        path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
                              (spec["root_id"], rel)).fetchone()[0]
        con.execute(
            """INSERT INTO inventory_observation
               (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,
                mtime_ns,inode,device,symlink_target,hash_scope,observation_status,link_count)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,'NONE','PRESENT',?)""",
            (spec["snapshot"], path_id, kind, uid, gid, mode, size, alloc, mtime, inode, device, target, nlink),
        )
    for source_id_ref, key, note in (
        (source_id, spec["snapshot"], "Exact compressed metadata stream for this complete subtree scan."),
        (status_id, spec["snapshot"], "Completion/status record for this complete subtree scan."),
    ):
        con.execute(
            """INSERT OR IGNORE INTO registry_entity_ref
               (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
               VALUES(?,'filesystem_inventory',?,'INVENTORY_SNAPSHOT',?,'CAPTURES',?)""",
            (source_id_ref, key, key, note),
        )
    stored = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?",
                         (spec["snapshot"],)).fetchone()[0]
    if stored != len(staged):
        raise RuntimeError("observation count mismatch for " + spec["snapshot"])
    return {"snapshot": spec["snapshot"], "status": "IMPORTED", "observations": stored,
            "files": declared_files, "directories": declared_dirs, "symlinks": declared_links,
            "logical_bytes": sum(item[5] or 0 for item in child_rows if item[1] == "FILE"),
            "allocated_bytes": sum(item[6] or 0 for item in child_rows if item[1] == "FILE"),
            "sha256": ledger_sha}


def main() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    results = []
    con = sqlite3.connect(DB, timeout=300)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=300000")
    try:
        con.execute("BEGIN IMMEDIATE")
        for spec in specs():
            results.append(import_one(con, spec, now))
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign key check failed: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"imported_utc": now, "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
