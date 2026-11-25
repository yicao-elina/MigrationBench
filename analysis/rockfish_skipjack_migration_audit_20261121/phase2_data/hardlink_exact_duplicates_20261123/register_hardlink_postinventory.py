#!/usr/bin/env python3
"""Record current per-path inode observations after verified hardlink dedup."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


AUDIT = Path(__file__).resolve().parents[2]
DB = AUDIT / "provenance_catalog.sqlite"
ROOT_ID = "RF-DATA-YI"
ROOT_PATH = "/data/pclancy3/yi"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution", required=True, type=Path)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    receipt_path = args.execution.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("mode") != "execute" or len(receipt.get("actions", [])) != 5:
        raise SystemExit("expected a verified five-pair execution receipt")
    rows = []
    for action in receipt["actions"]:
        if action.get("state") != "HARDLINKED_VERIFIED":
            raise SystemExit("execution has an unverified pair: " + action.get("group", "?"))
        post = action["post_link"]
        if post["a"]["device"] != post["b"]["device"] or post["a"]["inode"] != post["b"]["inode"]:
            raise SystemExit("post-link inode mismatch: " + action["group"])
        if post["a"]["nlink"] != 2 or post["b"]["nlink"] != 2:
            raise SystemExit("post-link nlink is not 2: " + action["group"])
        for side in ("a", "b"):
            path = action["pre_link"][side]["path"]
            stat = post[side]
            digest = post["sha256_" + side]
            if not path.startswith(ROOT_PATH + "/") or digest != action["pre_link"][side]["sha256"]:
                raise SystemExit("post-link path/hash does not match pre-link evidence: " + path)
            rows.append({"group": action["group"], "path": path, "stat": stat, "sha256": digest})
    if len(rows) != 10 or len({r["path"] for r in rows}) != 10:
        raise SystemExit("expected ten unique path observations")

    con = sqlite3.connect(DB, timeout=120)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        for row in rows:
            rel = row["path"][len(ROOT_PATH) + 1:]
            found = con.execute(
                """SELECT sp.path_id,a.artifact_id,a.sha256,a.byte_size,al.availability
                   FROM storage_path sp JOIN artifact_location al USING(path_id)
                   JOIN artifact a USING(artifact_id)
                   WHERE sp.root_id=? AND sp.relative_path=?""", (ROOT_ID, rel)
            ).fetchall()
            if len(found) != 1 or found[0]["availability"] != "PRESENT":
                raise ValueError("expected exactly one present catalog artifact at " + row["path"])
            if found[0]["sha256"] != row["sha256"] or found[0]["byte_size"] != row["stat"]["size_bytes"]:
                raise ValueError("catalog artifact hash/size mismatch at " + row["path"])
            row["path_id"] = int(found[0]["path_id"])
        snapshot_id = "RF-HARDLINK-POSTVERIFY-20261123T125323Z"
        manifest_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        logical = sum(r["stat"]["size_bytes"] for r in rows)
        started = receipt["captured_utc"]
        finished = (receipt.get("post_queue") or {}).get("captured_utc") or started
        if not args.apply:
            print(json.dumps({"state": "PREFLIGHT_ONLY", "snapshot_id": snapshot_id,
                              "paths": len(rows), "logical_bytes_by_path": logical,
                              "allocated_bytes": None, "started_utc": started, "finished_utc": finished,
                              "receipt_sha256": manifest_hash}, sort_keys=True))
            return
        con.execute("BEGIN IMMEDIATE")
        existing = con.execute("SELECT manifest_sha256 FROM inventory_snapshot WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if existing:
            if existing[0] != manifest_hash:
                raise ValueError("snapshot id exists with a different receipt hash")
            old = con.execute("SELECT count(*) FROM inventory_observation WHERE snapshot_id=?", (snapshot_id,)).fetchone()[0]
            if old != 10:
                raise ValueError("existing snapshot is partial; refusing an ambiguous rerun")
            con.rollback()
            print(json.dumps({"state": "ALREADY_REGISTERED", "snapshot_id": snapshot_id, "paths": old}, sort_keys=True))
            return
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES(?,?,?,?,?,'guarded_hardlink_duplicates.py','COMPLETE',10,0,0,0,?,NULL,?,?,?)""",
            (snapshot_id, ROOT_ID, started, finished,
             "guarded_hardlink_postverify", logical, str(receipt_path), manifest_hash,
             "Focused post-link observation for ten retained pathnames across five exact duplicate groups. Per-path allocated bytes are intentionally NULL at snapshot level because hardlink paths share inodes; exact per-path stat and inode values remain in inventory_observation."),
        )
        for row in rows:
            s = row["stat"]
            con.execute(
                """INSERT INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,
                    inode,device,symlink_target,sha256,hash_scope,observation_status,link_count)
                   VALUES(?,?,'FILE',?,?,?,?,?,?,?,?,NULL,?,'FULL_FILE','PRESENT',?)""",
                (snapshot_id,row["path_id"],s["uid"],s["gid"],s["mode_octal"],s["size_bytes"],
                 s["allocated_bytes"],s["mtime_ns"],str(s["inode"]),str(s["device"]),row["sha256"],s["nlink"]),
            )
        if con.execute("SELECT count(*) FROM inventory_observation WHERE snapshot_id=?", (snapshot_id,)).fetchone()[0] != 10:
            raise ValueError("post-link snapshot did not register ten observations")
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise ValueError("foreign_key_check returned rows")
        con.commit()
        print(json.dumps({"state": "REGISTERED", "snapshot_id": snapshot_id, "paths": 10,
                          "logical_bytes_by_path": logical, "allocated_bytes_snapshot": None,
                          "receipt_sha256": manifest_hash, "foreign_key_errors": 0}, sort_keys=True))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
