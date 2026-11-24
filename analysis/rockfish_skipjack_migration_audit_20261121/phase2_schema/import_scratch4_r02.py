#!/usr/bin/env python3
"""Import the post-cleanup `/scratch4/pclancy3/yi` refresh without overwriting R01."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DB = AUDIT / "provenance_catalog.sqlite"
LEDGER = AUDIT / "phase2_scratch4/scratch4_inventory_R02_20261122.jsonl"
ROOT_PATH = "/scratch4/pclancy3/yi"
ROOT_ID = "RF-SCRATCH4-YI"
SNAPSHOT = "RF-SCRATCH4-YI-20261122-R02"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def project_for(relative):
    if relative.startswith("3.WSe2/"):
        return "WSE2"
    if relative.startswith("2.Cr-Sb2Te3/"):
        return "MIGRATIONBENCH"
    return "OTHER_RESEARCH"


def main():
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    start = next(x for x in records if x["record"] == "scan_start")
    end = next(x for x in records if x["record"] == "scan_end")
    entries = [x for x in records if x["record"] in ("file", "dir", "symlink", "special")]
    errors = [x for x in records if x["record"] == "error"]
    checksum = digest(LEDGER)
    observed = iso(end["finished_epoch"])
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys=ON")
    try:
        con.execute("BEGIN IMMEDIATE")
        if con.execute("SELECT 1 FROM inventory_snapshot WHERE snapshot_id=?", (SNAPSHOT,)).fetchone():
            raise SystemExit("R02 snapshot already exists; refusing overwrite")
        prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key='rockfish_scratch4_user_inventory_20261122_R02'").fetchone()
        identity = (str(LEDGER.resolve()), checksum)
        if prior and prior != identity:
            raise SystemExit("registered R02 source differs from the local ledger")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES (?,?, 'JSONL_FILESYSTEM_INVENTORY',?,?,?,1,?)""",
            ("rockfish_scratch4_user_inventory_20261122_R02", identity[0], identity[1], observed, observed,
             "Complete post-cleanup metadata-only lstat scan of `/scratch4/pclancy3/yi`; zero traversal errors."),
        )
        file_count = end["counts"]["file"]
        dir_count = end["counts"]["dir"]
        link_count = end["counts"]["symlink"]
        logical = end["logical_file_bytes"]
        allocated = end["allocated_file_bytes"]
        state = "COMPLETE" if not errors and end["counts"]["error"] == 0 else "PARTIAL"
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (SNAPSHOT, ROOT_ID, iso(start["started_epoch"]), observed,
             "Sorted os.scandir + lstat; metadata only", "inventory_scratch4_r02.py", state,
             file_count, dir_count, link_count, len(errors), logical, allocated,
             str(LEDGER.resolve()), checksum,
             "Post-WSe2-WFC-cleanup complete tree refresh. The R01 snapshot and all removed-path rows remain immutable; removed candidates have artifact_location=REMOVED."),
        )
        counts = {"FILE": 0, "DIRECTORY": 0, "SYMLINK": 0, "OTHER": 0}
        for row in entries:
            kind = {"file": "FILE", "dir": "DIRECTORY", "symlink": "SYMLINK", "special": "OTHER"}[row["record"]]
            absolute = PurePosixPath(row["path"])
            relative = str(absolute.relative_to(PurePosixPath(ROOT_PATH)))
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (ROOT_ID, relative, observed))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, relative)).fetchone()[0]
            mode = row.get("mode_octal")
            con.execute(
                """INSERT INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,
                    mtime_ns,inode,device,symlink_target,hash_scope,observation_status,link_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'NONE','PRESENT',?)""",
                (SNAPSHOT, path_id, kind, row.get("uid"), row.get("gid"), mode,
                 row["size_bytes"] if kind != "DIRECTORY" else None,
                 row["allocated_bytes"] if kind == "FILE" else row.get("allocated_bytes"),
                 row.get("mtime_ns"), str(row.get("inode")) if row.get("inode") is not None else None,
                 str(row.get("device")) if row.get("device") is not None else None,
                 row.get("symlink_target"), row.get("nlink")),
            )
            if kind in ("FILE", "SYMLINK"):
                key = ROOT_ID + ":" + relative
                con.execute(
                    """INSERT OR IGNORE INTO artifact(project_id,artifact_key,artifact_kind,semantic_role,owner_status,
                              byte_size,scientific_status,retention_class,notes)
                       VALUES (?,?,?,'UNCLASSIFIED','UNKNOWN',?,'UNKNOWN','UNCLASSIFIED',?)""",
                    (project_for(relative), key, kind, row["size_bytes"], "Metadata-only R02 inventory; owner not inferred."),
                )
                artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()[0]
                con.execute(
                    """INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                       VALUES (?,?,'PRESENT',?,?)""",
                    (artifact_id, path_id, observed, observed),
                )
                con.execute("UPDATE artifact_location SET availability='PRESENT',last_verified_utc=? WHERE artifact_id=? AND path_id=?", (observed, artifact_id, path_id))
            counts[kind] += 1
        if counts["FILE"] != file_count or counts["DIRECTORY"] != dir_count or counts["SYMLINK"] != link_count:
            raise SystemExit("imported R02 path counts differ from scan summary")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    with sqlite3.connect(DB) as check:
        print("integrity_check=" + check.execute("PRAGMA integrity_check").fetchone()[0])
        print("foreign_key_violations=" + str(len(check.execute("PRAGMA foreign_key_check").fetchall())))
        print("snapshot=" + repr(check.execute("SELECT snapshot_id,completion_state,files_seen,directories_seen,symlinks_seen,logical_bytes,allocated_bytes FROM inventory_snapshot WHERE snapshot_id=?", (SNAPSHOT,)).fetchone()))


if __name__ == "__main__":
    main()
