#!/usr/bin/env python3
"""Load the full metadata-only /scratch4/pclancy3 walk into the catalog."""

import csv
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
MANIFEST = AUDIT / "phase2_scratch4_group" / "scratch4_pclancy3_full_inventory.tsv"
SUMMARY = AUDIT / "phase2_scratch4_group" / "inventory_totals.tsv"
ERRORS = AUDIT / "phase2_scratch4_group" / "traversal_errors.tsv"
SNAPSHOT = "RF-SCRATCH4-PCLANCY3-20261122-R01"
ROOT_ID = "RF-SCRATCH4-PCLANCY3"
ROOT_PATH = "/scratch4/pclancy3"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_mtime(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


totals = {}
with open(SUMMARY, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        totals[row["metric"]] = int(row["value"])

# The remote scan did not log its start/end clock. The inventory file's local
# mtime is retained only as the evidence-package capture time and explicitly
# described as such in snapshot.notes.
captured = utc_mtime(MANIFEST)
manifest_hash = sha256(MANIFEST)
errors_hash = sha256(ERRORS)
now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

error_paths = set()
seen = {"file": 0, "dir": 0, "symlink": 0, "special": 0, "error": 0}
with open(MANIFEST, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        kind = r["record_type"]
        seen[kind] = seen.get(kind, 0) + 1
        path = r["path"]
        if path != ROOT_PATH and not path.startswith(ROOT_PATH + "/"):
            raise SystemExit("path escaped registered root: " + path)
        if kind == "error":
            error_paths.add(path)

con = sqlite3.connect(DB, timeout=120)
con.execute("PRAGMA foreign_keys=ON")
con.execute("PRAGMA busy_timeout=120000")
con.execute("PRAGMA synchronous=NORMAL")

existing = con.execute("SELECT 1 FROM inventory_snapshot WHERE snapshot_id=?", (SNAPSHOT,)).fetchone()
if existing:
    raise SystemExit("snapshot already imported: " + SNAPSHOT)

con.execute("CREATE TEMP TABLE stage (record_type TEXT,relative_path TEXT,logical_bytes INTEGER,allocated_bytes INTEGER,mtime_ns INTEGER,mode TEXT,uid INTEGER,gid INTEGER,nlink INTEGER,device TEXT,inode TEXT,symlink_target TEXT,unreadable INTEGER)")
insert_sql = "INSERT INTO stage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
batch = []
with open(MANIFEST, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        kind = r["record_type"]
        path = r["path"]
        if kind == "error":
            continue
        relative_path = "" if path == ROOT_PATH else path[len(ROOT_PATH) + 1:]
        batch.append((kind, relative_path,
                      int(r["logical_bytes"]) if r["logical_bytes"] else None,
                      int(r["allocated_bytes_entry"]) if r["allocated_bytes_entry"] else None,
                      int(r["mtime_epoch_ns"]) if r["mtime_epoch_ns"] else None,
                      r["mode"] or None,
                      int(r["uid"]) if r["uid"] else None,
                      int(r["gid"]) if r["gid"] else None,
                      int(r["nlink"]) if r["nlink"] else None,
                      r["device"] or None, r["inode"] or None,
                      r["symlink_target"] or None, int(path in error_paths)))
        if len(batch) >= 20000:
            con.executemany(insert_sql, batch)
            batch = []
if batch:
    con.executemany(insert_sql, batch)

if seen.get("file") != totals["file_count"] or seen.get("dir") != totals["dir_count"] or seen.get("symlink") != totals["symlink_count"] or seen.get("error") != totals["error_count"]:
    raise SystemExit("inventory TSV row counts differ from summary: " + repr((seen, totals)))
if len(error_paths) != seen["error"]:
    raise SystemExit("error paths are not unique; refusing ambiguous import")
normal_paths = con.execute("SELECT COUNT(*) FROM stage").fetchone()[0]
unreadable_paths = con.execute("SELECT COUNT(*) FROM stage WHERE unreadable=1").fetchone()[0]
expected_normal = sum(seen[k] for k in ("file", "dir", "symlink", "special"))
if normal_paths != expected_normal or unreadable_paths != len(error_paths):
    raise SystemExit("normal/error path mapping differs from manifest: " + repr((normal_paths, expected_normal, unreadable_paths, len(error_paths))))

allocated_unique = totals.get("allocated_bytes_unique_hardlink_inodes_estimate", totals.get("allocated_bytes_per_path_entry"))
# End the temporary staging transaction before starting the catalog transaction.
con.commit()
con.execute("BEGIN IMMEDIATE")
con.execute("""INSERT OR IGNORE INTO storage_root
    (root_id,cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,scope_kind,project_id,status,notes)
    VALUES (?, 'Rockfish','scratch4','/scratch4',?,'ycao73','pclancy3','GROUP_SUBTREE',NULL,'ACTIVE',?)""",
    (ROOT_ID, ROOT_PATH, "Full accessible pclancy3 group subtree; multiple UIDs own files."))
con.execute("INSERT INTO registry_source(source_key,source_path,registry_kind,sha256,captured_utc,immutable_snapshot,authority_note,notes) VALUES (?,?,?,?,?,1,?,?)",
            ("scratch4_pclancy3_full_walk_20261122", str(MANIFEST.resolve()), "ROCKFISH_METADATA_INVENTORY_TSV", manifest_hash, captured,
             "Read-only os.scandir/lstat walk on Rockfish; does not follow symlinks.",
             "Remote scan clock was not captured. Manifest mtime is recorded as evidence-package time, not asserted as scan start/end. 98 permission-denied paths are separately recorded."))
con.execute("INSERT INTO registry_source(source_key,source_path,registry_kind,sha256,captured_utc,immutable_snapshot,authority_note,notes) VALUES (?,?,?,?,?,1,?,?)",
            ("scratch4_pclancy3_traversal_errors_20261122", str(ERRORS.resolve()), "ROCKFISH_INVENTORY_ERROR_TSV", errors_hash, captured,
             "Read-only inventory error report.", "Permission errors are retained as unknown scope, not interpreted as empty paths."))
con.execute("""INSERT INTO inventory_snapshot
    (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
     files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
     manifest_path,manifest_sha256,notes)
    VALUES (?,?,?,?,?,'inventory_pclancy3.py','PARTIAL',?,?,?,?,?,?,?,?,?)""",
    (SNAPSHOT, ROOT_ID, captured, captured,
     "Remote Python os.scandir/lstat metadata-only walk; no symlink following; timestamps reflect local evidence-file capture because remote scan clock was not logged.",
     totals["file_count"], totals["dir_count"], totals["symlink_count"], totals["error_count"],
     totals["logical_bytes"], allocated_unique, str(MANIFEST.resolve()), manifest_hash,
     "PARTIAL: 98 permission errors. allocated_bytes is unique-inode estimate to compare with GPFS quota; path-entry allocation was %s bytes. Manifest includes %s rows. Owners span multiple UIDs; do not infer user ownership from group membership." % (totals["allocated_bytes_per_path_entry"], sum(seen.values()))))

con.execute("""INSERT INTO quota_observation
    (root_id,observed_utc,quota_subject_kind,quota_subject,used_bytes,quota_bytes,hard_limit_bytes,in_doubt_bytes,grace_state,
     source_command,raw_output_path,notes)
    VALUES (?,?,'GROUP','pclancy3',?,?,?,?,?,?,?,?)""",
    (ROOT_ID, "2026-11-22T19:14:11Z", totals["quota_blocks_kib"] * 1024 if "quota_blocks_kib" in totals else 9750809088 * 1024,
     10737418240 * 1024, 10737418240 * 1024, 959504 * 1024, "none",
     "mmlsquota -Y -g pclancy3 scratch4", str((AUDIT / "phase2_data" / "rockfish_quota_after_wse2_20261122.txt").resolve()),
     "Group-wide quota, not identical to this subtree; captured 19:14:11Z after WSe2 WFC deletion. Count 1,456,725 files. Stored before next verified deletion batch."))

con.execute("""INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc)
    SELECT ?,relative_path,? FROM stage""",
    (ROOT_ID, captured))
con.execute("""INSERT INTO inventory_observation
    (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,
     symlink_target,sha256,hash_scope,observation_status,link_count)
    SELECT ?,p.path_id,
      CASE s.record_type WHEN 'file' THEN 'FILE' WHEN 'dir' THEN 'DIRECTORY' WHEN 'symlink' THEN 'SYMLINK' ELSE 'OTHER' END,
      s.uid,s.gid,s.mode,s.logical_bytes,s.allocated_bytes,s.mtime_ns,s.inode,s.device,s.symlink_target,NULL,'NONE',
      CASE WHEN s.unreadable=1 THEN 'UNREADABLE' ELSE 'PRESENT' END,s.nlink
    FROM stage s CROSS JOIN storage_path p
    WHERE p.root_id=? AND p.relative_path=s.relative_path""",
    (SNAPSHOT, ROOT_ID))

con.commit()
rows = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (SNAPSHOT,)).fetchone()[0]
unreadable = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=? AND observation_status='UNREADABLE'", (SNAPSHOT,)).fetchone()[0]
con.execute("DROP TABLE stage")
con.commit()
print(json.dumps({"snapshot_id": SNAPSHOT, "manifest_sha256": manifest_hash, "manifest_rows": sum(seen.values()),
                  "inventory_observation_rows": rows, "unreadable_observations": unreadable,
                  "summary": totals, "import_completed_utc": now}, sort_keys=True))
con.close()
