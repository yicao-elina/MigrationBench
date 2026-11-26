#!/usr/bin/env python3
"""Read-only, no-follow metadata inventory for the three owner-scoped roots."""

from __future__ import print_function

import argparse
import json
import os
import socket
import sqlite3
import stat
import sys
import time
from datetime import datetime


SCANNER_VERSION = "scoped-personal-inventory-r1"
ROOTS = (
    ("data_yi", "/data/pclancy3/yi"),
    ("scratch4_yi", "/scratch4/pclancy3/yi"),
    ("scratch16_yi", "/scratch16/pclancy3/yi"),
)
EXCLUDED_PATHS = (
    "/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/"
    "scoped_personal_root_inventories/20261126_r1",
)
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def capture_time():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def excluded(path):
    for item in EXCLUDED_PATHS:
        if path == item or path.startswith(item + os.sep):
            return True
    return False


def insert_entry(con, values):
    con.execute(
        """INSERT INTO inventory_entry
           (root_id,relative_path,parent_relative_path,top_level_name,depth,
            entry_type,logical_bytes,allocated_bytes,uid,gid,mode_octal,
            device,inode,link_count,mtime_ns)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)


def scan_root(con, root_id, root_path):
    root_stat = os.lstat(root_path)
    if not stat.S_ISDIR(root_stat.st_mode) or os.path.realpath(root_path) != root_path:
        raise RuntimeError("authorized root missing, not a real directory, or resolves elsewhere: " + root_path)
    insert_entry(con, (root_id, "", None, ".", 0, "ROOT", 0,
                       int(root_stat.st_blocks * 512) if hasattr(root_stat, "st_blocks") else None,
                       root_stat.st_uid, root_stat.st_gid, format(stat.S_IMODE(root_stat.st_mode), "04o"),
                       root_stat.st_dev, root_stat.st_ino, root_stat.st_nlink, root_stat.st_mtime_ns))
    stack = [(root_path, "", 0)]
    counts = {"entry_count": 1, "file_count": 0, "directory_count": 1,
              "symlink_count": 0, "special_count": 0, "error_count": 0,
              "logical_bytes": 0, "allocated_path_sum_bytes": 0}
    rows_since_commit = 0
    while stack:
        directory, relative_directory, depth = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError as error:
            con.execute("INSERT OR REPLACE INTO inventory_error VALUES(?,?,?,?)",
                        (root_id, directory, error.__class__.__name__, str(error)))
            counts["error_count"] += 1
            continue
        for entry in entries:
            path = entry.path
            if excluded(path):
                continue
            relative = entry.name if not relative_directory else relative_directory + "/" + entry.name
            parent = relative_directory or None
            try:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    kind = "SYMLINK"
                    counts["symlink_count"] += 1
                elif stat.S_ISDIR(info.st_mode):
                    kind = "DIR"
                    counts["directory_count"] += 1
                    stack.append((path, relative, depth + 1))
                elif stat.S_ISREG(info.st_mode):
                    kind = "FILE"
                    counts["file_count"] += 1
                    counts["logical_bytes"] += info.st_size
                else:
                    kind = "SPECIAL"
                    counts["special_count"] += 1
                allocated = int(info.st_blocks * 512) if hasattr(info, "st_blocks") else None
                if kind == "FILE" and allocated is not None:
                    counts["allocated_path_sum_bytes"] += allocated
                top_level = relative.split("/", 1)[0]
                insert_entry(con, (root_id, relative, parent, top_level, depth + 1, kind,
                                   info.st_size if kind == "FILE" else 0, allocated,
                                   info.st_uid, info.st_gid, format(stat.S_IMODE(info.st_mode), "04o"),
                                   info.st_dev, info.st_ino, info.st_nlink, info.st_mtime_ns))
                counts["entry_count"] += 1
                rows_since_commit += 1
                if rows_since_commit >= 10000:
                    con.commit()
                    rows_since_commit = 0
            except OSError as error:
                con.execute("INSERT OR REPLACE INTO inventory_error VALUES(?,?,?,?)",
                            (root_id, path, error.__class__.__name__, str(error)))
                counts["error_count"] += 1
    con.commit()
    unique_allocated = con.execute(
        """SELECT COALESCE(SUM(allocated_bytes),0) FROM (
             SELECT MAX(allocated_bytes) AS allocated_bytes
             FROM inventory_entry WHERE root_id=? AND entry_type='FILE'
             GROUP BY device,inode)""", (root_id,)).fetchone()[0]
    counts["allocated_unique_inode_bytes"] = int(unique_allocated or 0)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot-id", default="personal-roots-r1-20261126")
    args = parser.parse_args()
    output = os.path.abspath(args.output)
    partial = output + ".partial"
    if os.path.exists(output) or os.path.exists(partial):
        raise SystemExit("refusing to overwrite existing output or partial: " + output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.umask(0o077)
    started = time.time()
    captured = capture_time()
    con = sqlite3.connect(partial)
    summary = {}
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript(open(SCHEMA_PATH, "r").read())
        con.execute("INSERT INTO inventory_snapshot VALUES(?,?,?,?,?,?,?,?)",
                    (args.snapshot_id, captured, None, socket.gethostname(), SCANNER_VERSION,
                     "COMPLETE", "Only the three explicitly allowlisted personal roots; lstat metadata; no contents read; symlink targets not traversed.",
                     json.dumps(list(EXCLUDED_PATHS))))
        for root_id, root_path in ROOTS:
            con.execute("INSERT INTO inventory_root(root_id,snapshot_id,root_path) VALUES(?,?,?)",
                        (root_id, args.snapshot_id, root_path))
            summary[root_id] = scan_root(con, root_id, root_path)
            values = summary[root_id]
            columns = ("entry_count","file_count","directory_count","symlink_count","special_count",
                       "error_count","logical_bytes","allocated_path_sum_bytes","allocated_unique_inode_bytes")
            con.execute("UPDATE inventory_root SET " + ",".join(k+"=?" for k in columns) + " WHERE root_id=?",
                        tuple(values[k] for k in columns) + (root_id,))
            con.commit()
        con.execute("PRAGMA user_version=1")
        con.commit()
        errors = con.execute("SELECT COUNT(*) FROM inventory_error").fetchone()[0]
        state = "COMPLETE_WITH_ERRORS" if errors else "COMPLETE"
        finished = capture_time()
        con.execute("UPDATE inventory_snapshot SET completion_state=?,finished_utc=? WHERE snapshot_id=?", (state,finished,args.snapshot_id))
        con.commit()
        quick = con.execute("PRAGMA quick_check").fetchone()[0]
        fk = len(con.execute("PRAGMA foreign_key_check").fetchall())
        if quick != "ok" or fk:
            raise RuntimeError("database integrity failure: {} foreign_key_violations={}".format(quick,fk))
        print(json.dumps({"snapshot_id":args.snapshot_id,"captured_utc":captured,
                          "finished_utc":finished,"completion_state":state,"roots":summary,"errors":errors,
                          "quick_check":quick,"foreign_key_violations":fk,
                          "elapsed_seconds":round(time.time()-started,2),
                          "database_bytes":os.path.getsize(output)},sort_keys=True))
    finally:
        con.close()
    os.chmod(partial,0o600)
    os.replace(partial,output)


if __name__ == "__main__":
    main()
