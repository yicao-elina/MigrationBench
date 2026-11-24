#!/usr/bin/env python3
"""Register/import four additional data roots and a UID-filtered ownership view."""

import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DATA = AUDIT / "phase2_data"
DB = AUDIT / "provenance_catalog.sqlite"
ROOT_LEDGER = DATA / "data_owner_roots_path_inventory_20261122.jsonl.gz"
ROOT_SUMMARY = DATA / "data_owner_roots_summary_20261122.json"
UID_LEDGER = DATA / "data_uid3901_fullpath_inventory_20261122.jsonl.gz"
UID_SUMMARY = DATA / "data_uid3901_fullpath_summary_20261122.json"

GROUP_ROOT = "/data/pclancy3"
GROUP_ROOT_ID = "RF-DATA-PCLANCY3"
ROOTS = {
    "/data/pclancy3/ycao73_home": ("RF-DATA-YCAO73-HOME", "FLARE_LAMMPS"),
    "/data/pclancy3/for-victor": ("RF-DATA-FOR-VICTOR", "DEFECT_FORMATION_ENERGY"),
    "/data/pclancy3/for-yuqi": ("RF-DATA-FOR-YUQI", "QE_PSEUDOPOTENTIALS"),
    "/data/pclancy3/MHP_defect": ("RF-DATA-MHP-DEFECT", "MHP_DEFECT"),
}

PROJECTS = [
    ("FLARE_LAMMPS", "FLARE and LAMMPS source", "USER", "Rockfish user-home FLARE/LAMMPS source and build tree; per-repository state still needs review."),
    ("DEFECT_FORMATION_ENERGY", "Defect formation-energy workflow", "USER", "Small QE formation-energy workflow with source code, config, input logs, and compact result JSON."),
    ("QE_PSEUDOPOTENTIALS", "QE pseudopotential files", "USER", "Pseudopotential files referenced by QE workflows; content duplicates and provenance must be hash-checked."),
    ("MHP_DEFECT", "Metal-halide-perovskite defect calculations", "SHARED", "Historical MHP defect NEB and charge-density outputs; scheduler and publication lineage remain under review."),
]

ROOT_META = {
    "/data/pclancy3/ycao73_home": ("RF-DATA-YCAO73-HOME", "USER_SUBTREE", "ycao73 home-source material under /data; UID 3901 was observed."),
    "/data/pclancy3/for-victor": ("RF-DATA-FOR-VICTOR", "USER_SUBTREE", "Former-collaborator handoff; process with the same evidence pipeline, no blanket retention/deletion."),
    "/data/pclancy3/for-yuqi": ("RF-DATA-FOR-YUQI", "USER_SUBTREE", "Former-collaborator handoff; process with the same evidence pipeline, no blanket retention/deletion."),
    "/data/pclancy3/MHP_defect": ("RF-DATA-MHP-DEFECT", "USER_SUBTREE", "MHP defect calculations and outputs; per-calculation lineage is being reviewed."),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso(epoch):
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def register_source(con, key, path, kind, observed, note):
    identity = (str(path.resolve()), sha256(path))
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if prior and prior != identity:
        raise RuntimeError("source key changed: " + key)
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
           VALUES (?,?,?,?,?,?,1,?)""",
        (key, identity[0], kind, identity[1], observed, observed, note),
    )


def ensure_root(con, root_id, scoped_path, scope_kind, notes):
    values = ("Rockfish", "data", "/data", scoped_path, "ycao73", "pclancy3", scope_kind, "ACTIVE", notes)
    old = con.execute(
        "SELECT cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,scope_kind,status,notes FROM storage_root WHERE root_id=?",
        (root_id,),
    ).fetchone()
    if old and old != values:
        raise RuntimeError("storage root changed: " + root_id)
    con.execute(
        """INSERT OR IGNORE INTO storage_root
           (root_id,cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,scope_kind,status,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (root_id,) + values,
    )


def project_for_record(root, rel):
    if root.endswith("/ycao73_home"):
        return "FLARE_LAMMPS"
    if root.endswith("/for-victor"):
        return "DEFECT_FORMATION_ENERGY"
    if root.endswith("/for-yuqi"):
        return "QE_PSEUDOPOTENTIALS"
    if root.endswith("/MHP_defect"):
        return "MHP_DEFECT"
    if rel.startswith("yi/flare-data/1-Cr-Sb2Te3/") or rel.startswith("yi/3.WSe2/"):
        return "MIGRATIONBENCH" if "Sb2Te3" in rel else "WSE2"
    return "OTHER_RESEARCH"


def add_observation(con, snapshot, root_id, root_path, record, observed):
    absolute = PurePosixPath(record["path"])
    relative = str(absolute.relative_to(PurePosixPath(root_path)))
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (root_id, relative, observed))
    path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, relative)).fetchone()[0]
    kind = {"file": "FILE", "dir": "DIRECTORY", "symlink": "SYMLINK"}[record["record"]]
    size = int(record.get("size_bytes", 0))
    allocated = int(record.get("blocks_512", 0)) * 512
    mode = record.get("mode_octal")
    if mode and not mode.startswith("0o"):
        mode = "0o" + mode.zfill(4)
    mtime = record.get("mtime_ns")
    if mtime is None and record.get("mtime_epoch") is not None:
        mtime = int(float(record["mtime_epoch"]) * 1_000_000_000)
    device = record.get("dev", record.get("device"))
    inode = record.get("ino", record.get("inode"))
    con.execute(
        """INSERT INTO inventory_observation
           (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,
            symlink_target,hash_scope,observation_status,link_count)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'NONE','PRESENT',?)""",
        (snapshot, path_id, kind, record.get("uid"), int(record["gid_name"]) if str(record.get("gid_name", "")).isdigit() else record.get("gid"),
         mode, size, allocated, mtime, str(inode) if inode is not None else None,
         str(device) if device is not None else None, record.get("symlink_target", record.get("target")), record.get("nlink")),
    )
    key = root_id + ":" + relative
    if kind in ("FILE", "SYMLINK"):
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,byte_size,scientific_status,retention_class,notes)
               VALUES (?,?,?,'UNCLASSIFIED','UNKNOWN',?,'UNKNOWN','UNCLASSIFIED','Metadata-only extended Rockfish data inventory; semantic role not inferred.')""",
            (project_for_record(root_path, relative), key, kind, size),
        )
        artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()[0]
        con.execute(
            "INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES (?,?,'PRESENT',?,?)",
            (artifact_id, path_id, observed, observed),
        )
    return kind, size, allocated


def main():
    root_summary = json.loads(ROOT_SUMMARY.read_text())
    uid_summary = json.loads(UID_SUMMARY.read_text())
    root_records = [json.loads(line) for line in gzip.open(ROOT_LEDGER, "rt", encoding="utf-8")]
    scan_start = next(x for x in root_records if x.get("record") == "scan_start")
    scan_end = next(x for x in root_records if x.get("record") == "scan_end")
    root_rows = [x for x in root_records if x.get("record") in ("dir", "file", "symlink")]
    uid_rows = (json.loads(line) for line in gzip.open(UID_LEDGER, "rt", encoding="utf-8"))
    uid_finish = datetime.fromtimestamp(UID_SUMMARY.stat().st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    uid_start = datetime.fromtimestamp(UID_SUMMARY.stat().st_mtime - uid_summary["elapsed_seconds"], timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys=ON")
    try:
        con.execute("BEGIN IMMEDIATE")
        ensure_root(con, GROUP_ROOT_ID, GROUP_ROOT, "GROUP_SUBTREE", "Aggregate pclancy3 project view; UID-filtered inventory observations are path-summed and overlap nested roots." )
        for path, (root_id, scope_kind, notes) in ROOT_META.items():
            ensure_root(con, root_id, path, scope_kind, notes)
        for project in PROJECTS:
            con.execute("INSERT OR IGNORE INTO project(project_id,project_name,owner_scope,description,status) VALUES(?,?,?,?, 'ACTIVE')", project)
        register_source(con, "rockfish_data_additional_roots_20261122", ROOT_LEDGER, "GZIP_JSONL_ROOT_INVENTORY", iso(scan_end["finished_epoch"]), "Complete lstat inventory for ycao73_home, for-victor, for-yuqi, and MHP_defect; no symlink traversal.")
        register_source(con, "rockfish_data_uid3901_paths_20261122", UID_LEDGER, "GZIP_JSONL_UID_FILTERED_INVENTORY", uid_finish, "UID 3901 filtered find/lstat inventory across /data/pclancy3. Path totals include hardlink repeats; a separate summary records unique-inode bytes.")

        for root_path, (root_id, _project) in ROOTS.items():
            summary = root_summary["totals_by_root"][root_path]
            dir_count = sum(1 for row in root_rows if row.get("root") == root_path and row.get("record") == "dir")
            snap = "RF-DATA-" + root_id.removeprefix("RF-DATA-") + "-20261122-R01"
            con.execute(
                """INSERT INTO inventory_snapshot
                   (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                    files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,manifest_path,manifest_sha256,notes)
                   VALUES (?,?,?,?,?,'inventory_additional_data_roots.py','COMPLETE',?,?,?,?,?,?,?,?,?)""",
                (snap, root_id, iso(scan_start["started_epoch"]), iso(scan_end["finished_epoch"]),
                 "Sorted os.scandir + lstat; metadata only", summary["file_count"], dir_count,
                 summary.get("symlink_count", 0), 0, summary["file_logical_bytes"], summary["file_allocated_bytes"],
                 str(ROOT_LEDGER.resolve()), sha256(ROOT_LEDGER),
                 "One root-level view from a single complete scan. Owner IDs recorded; file bytes are path-sized."),
            )
            totals = {"FILE": 0, "DIRECTORY": 0, "SYMLINK": 0, "logical": 0, "allocated": 0}
            for row in root_rows:
                if row.get("root") != root_path:
                    continue
                kind, size, allocated = add_observation(con, snap, root_id, root_path, row, iso(scan_end["finished_epoch"]))
                totals[kind] += 1
                if kind == "FILE":
                    totals["logical"] += size
                    totals["allocated"] += allocated
            expected = (summary["file_count"], dir_count, summary.get("symlink_count", 0), summary["file_logical_bytes"], summary["file_allocated_bytes"])
            actual = (totals["FILE"], totals["DIRECTORY"], totals["SYMLINK"], totals["logical"], totals["allocated"])
            if actual != expected:
                raise RuntimeError("additional root totals differ: %s expected=%s actual=%s" % (root_path, expected, actual))

        owner_snapshot = "RF-DATA-PCLANCY3-UID3901-20261122-R01"
        uid_counts = uid_summary["counts_by_type"]
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,manifest_path,manifest_sha256,notes)
               VALUES (?,?,?,?,?,'inventory_uid3901_data.py','COMPLETE',?,0,?,0,?,?,?, ?,?)""",
            (owner_snapshot, GROUP_ROOT_ID, uid_start, uid_finish,
             "UID 3901 filtered find -P /data/pclancy3 -xdev with lstat metadata", uid_counts["regular_file_paths"],
             uid_counts["symlink_paths"], uid_summary["logical_bytes_path_sum"],
             uid_summary["allocated_bytes_unique_inode_sum"], str(UID_LEDGER.resolve()), sha256(UID_LEDGER),
             "Filtered owner view, not a full directory inventory. Logical bytes count all paths; allocated bytes count each inode once. "
             "There are %d hardlink-extra paths; path-summed allocated bytes are %d. No owned paths above 100 MB were found outside the inventoried child roots." %
             (uid_summary["hardlink_extra_path_count"], uid_summary["allocated_bytes_path_sum"])),
        )
        roots_for_uid = [("/data/pclancy3/yi", "RF-DATA-YI"), ("/data/pclancy3/ycao73_home", "RF-DATA-YCAO73-HOME"),
                         ("/data/pclancy3/for-victor", "RF-DATA-FOR-VICTOR"), ("/data/pclancy3/for-yuqi", "RF-DATA-FOR-YUQI"),
                         ("/data/pclancy3/MHP_defect", "RF-DATA-MHP-DEFECT")]
        owner_totals = {"file": 0, "symlink": 0, "logical": 0, "unique_inode_allocated": 0}
        for row in uid_rows:
            absolute = row.get("path")
            if not absolute:
                continue
            selected = None
            for prefix, root_id in roots_for_uid:
                if absolute.startswith(prefix + "/"):
                    selected = (prefix, root_id)
                    break
            if selected is None:
                selected = (GROUP_ROOT, GROUP_ROOT_ID)
            root_path, root_id = selected
            relative = str(PurePosixPath(absolute).relative_to(PurePosixPath(root_path)))
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (root_id, relative, uid_finish))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, relative)).fetchone()[0]
            kind = "FILE" if row["type"] == "f" else "SYMLINK"
            size = int(row.get("size_bytes") or 0)
            allocated = int(row.get("allocated_bytes") or (int(row.get("blocks_512") or 0) * 512))
            mode = row.get("mode_octal")
            con.execute(
                """INSERT INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,
                    symlink_target,hash_scope,observation_status,link_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'NONE','PRESENT',?)""",
                (owner_snapshot, path_id, kind, row.get("uid"), int(row.get("gid_name")) if str(row.get("gid_name", "")).isdigit() else None,
                 mode, size, allocated, int(float(row["mtime_epoch"])*1_000_000_000) if row.get("mtime_epoch") is not None else None,
                 str(row.get("inode")) if row.get("inode") is not None else None,
                 str(row.get("device")) if row.get("device") is not None else None,
                 row.get("symlink_target"), row.get("nlink")),
            )
            owner_totals["file" if kind == "FILE" else "symlink"] += 1
            owner_totals["logical"] += size
            if kind == "FILE":
                art_key = root_id + ":" + relative
                con.execute(
                    """INSERT OR IGNORE INTO artifact
                       (project_id,artifact_key,artifact_kind,semantic_role,owner_status,byte_size,scientific_status,retention_class,notes)
                       VALUES (? ,? ,?,'UNCLASSIFIED','UNKNOWN',?,'UNKNOWN','UNCLASSIFIED','UID 3901 owner-filtered inventory; semantic role not inferred.')""",
                    (project_for_record(root_path, relative), art_key, kind, size),
                )
                art_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (art_key,)).fetchone()[0]
                con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES (?,?,'PRESENT',?,?)", (art_id, path_id, uid_finish, uid_finish))
        if owner_totals["file"] != uid_counts["regular_file_paths"] or owner_totals["symlink"] != uid_counts["symlink_paths"] or owner_totals["logical"] != uid_summary["logical_bytes_path_sum"]:
            raise RuntimeError("UID snapshot counts/bytes differ from summary: " + repr(owner_totals))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    with sqlite3.connect(DB) as check:
        print("integrity_check=" + check.execute("PRAGMA integrity_check").fetchone()[0])
        print("foreign_key_violations=" + str(len(check.execute("PRAGMA foreign_key_check").fetchall())))
        for row in check.execute("SELECT snapshot_id,completion_state,files_seen,directories_seen,symlinks_seen,logical_bytes,allocated_bytes FROM inventory_snapshot WHERE snapshot_id LIKE 'RF-DATA-%20261122-R01' ORDER BY snapshot_id"):
            print("\t".join(str(x) if x is not None else "NULL" for x in row))


if __name__ == "__main__":
    main()
