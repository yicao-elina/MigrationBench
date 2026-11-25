#!/usr/bin/env python3
"""Create a new immutable catalog snapshot with 2026-11-25 yi-root observations.

The scan summary covers every path beneath the two authorized roots. Detailed
SQL observations are imported for all in-tree hardlink members under data/yi,
the 200 largest data/yi files, and the 100 largest scratch4/yi files. The
remaining per-path metadata stays in the hash-verified Rockfish inventory
archive; this importer does not claim row-level coverage for every scanned path.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
from pathlib import Path


DEFAULT_BASE = Path(
    "/scratch16/pclancy3/yi/MigrationBench_data_archive/"
    "provenance_catalog_snapshots/20261124/provenance_catalog_20261124.sqlite"
)
DEFAULT_OUT = Path(
    "/scratch16/pclancy3/yi/MigrationBench_data_archive/"
    "provenance_catalog_snapshots/20261125/provenance_catalog_20261125.sqlite"
)
DATA_DIR = Path(
    "/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/"
    "data_pclancy3_yi_refresh_20261125/data_yi_refresh_20261125"
)
SCRATCH4_DIR = Path(
    "/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/"
    "scratch4_yi_refresh_20261125/scratch4_yi_refresh_20261125"
)
ROOTS = {
    "RF-DATA-YI": "/data/pclancy3/yi",
    "RF-SCRATCH4-YI": "/scratch4/pclancy3/yi",
}
CAPTURE_UTC = "2026-11-25T10:35:13Z"
SCRATCH4_CAPTURE_UTC = "2026-11-25T10:42:33Z"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_hash(directory, name):
    rows = (directory / "SHA256SUMS.tsv").read_text().splitlines()
    for row in rows[1:]:
        rel, _size, digest = row.split("\t")
        if rel == name:
            path = directory / name
            actual_size = path.stat().st_size
            expected_size = int(_size)
            if actual_size != expected_size or sha256_file(path) != digest:
                raise RuntimeError(f"archive checksum mismatch: {path}")
            return digest
    raise RuntimeError(f"checksum row absent: {directory}/SHA256SUMS.tsv: {name}")


def parse_mode(value):
    if not value:
        return None
    return format(int(value, 8), "04o")


def add_observation(con, snapshot_id, root_id, root_path, record):
    absolute = record.get("path")
    if not isinstance(absolute, str) or not absolute.startswith(root_path + "/"):
        raise RuntimeError(f"path outside authorized root {root_path}: {absolute!r}")
    relative = absolute[len(root_path) + 1:]
    cur = con.execute(
        "INSERT INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?) "
        "ON CONFLICT(root_id,relative_path) DO NOTHING",
        (root_id, relative, CAPTURE_UTC if root_id == "RF-DATA-YI" else SCRATCH4_CAPTURE_UTC),
    )
    row = con.execute(
        "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
        (root_id, relative),
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to upsert storage_path")
    mode = record.get("mode") or record.get("mode_octal")
    if mode and not str(mode).startswith("0o"):
        mode = "0o" + str(mode)
    con.execute(
        """INSERT INTO inventory_observation
           (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,
            allocated_bytes,mtime_ns,inode,device,sha256,hash_scope,observation_status)
           VALUES(?,?,'FILE',?,?,?,?,?,?,?,?,NULL,'NONE','PRESENT')
           ON CONFLICT(snapshot_id,path_id) DO UPDATE SET
             entry_type=excluded.entry_type,uid=excluded.uid,gid=excluded.gid,
             mode_octal=excluded.mode_octal,logical_bytes=excluded.logical_bytes,
             allocated_bytes=excluded.allocated_bytes,mtime_ns=excluded.mtime_ns,
             inode=excluded.inode,device=excluded.device,sha256=NULL,hash_scope='NONE',
             observation_status='PRESENT'""",
        (snapshot_id, row[0], record.get("uid"), record.get("gid"), parse_mode(mode),
         int(record.get("size", record.get("size_bytes", 0))),
         int(record.get("allocated_bytes", 0)), record.get("mtime_ns"),
         str(record.get("inode")) if record.get("inode") is not None else None,
         str(record.get("device")) if record.get("device") is not None else None),
    )


def register_snapshot(con, directory, root_id, snapshot_id, capture, import_kind):
    manifest_path = directory / "inventory.json"
    manifest_digest = manifest_hash(directory, "inventory.json")
    summary = json.loads(manifest_path.read_text())
    root_path = ROOTS[root_id]
    if summary.get("root") != root_path:
        raise RuntimeError(f"unexpected inventory root in {manifest_path}: {summary.get('root')}")
    started = capture
    elapsed = float(summary.get("elapsed_seconds", 0))
    counts = summary["counts"]
    totals = summary["totals"]
    con.execute(
        """INSERT INTO inventory_snapshot
           (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,
            completion_state,files_seen,directories_seen,symlinks_seen,unreadable_count,
            logical_bytes,allocated_bytes,manifest_path,manifest_sha256,notes)
           VALUES(?,?,?,?,?,'scoped-inventory-20261125','COMPLETE',?,?,?,?,?,?,?,?,?)
           ON CONFLICT(snapshot_id) DO UPDATE SET
             finished_utc=excluded.finished_utc,method=excluded.method,
             completion_state=excluded.completion_state,files_seen=excluded.files_seen,
             directories_seen=excluded.directories_seen,symlinks_seen=excluded.symlinks_seen,
             unreadable_count=excluded.unreadable_count,logical_bytes=excluded.logical_bytes,
             allocated_bytes=excluded.allocated_bytes,manifest_path=excluded.manifest_path,
             manifest_sha256=excluded.manifest_sha256,notes=excluded.notes""",
        (snapshot_id, root_id, started, started, f"no-follow metadata walk ({elapsed:.2f}s)",
         int(counts["regular_files"]), int(counts["directories"]),
         int(counts["symlinks"]), int(summary.get("error_count", 0)),
         int(totals["logical_bytes"]), int(totals["allocated_bytes"]),
         str(manifest_path), manifest_digest,
         f"Completed traversal of exact root {root_path}; detail rows imported: {import_kind}. "
         "No file contents were hashed. See adjacent REPORT.md and SHA256SUMS.tsv."),
    )
    con.execute("DELETE FROM inventory_observation WHERE snapshot_id=?", (snapshot_id,))
    records: dict[str, dict] = {}
    if root_id == "RF-DATA-YI":
        for group in summary.get("hardlink_groups", []):
            for item in group.get("paths", []):
                records[item["path"]] = item
        for item in summary.get("top_files", []):
            records[item["path"]] = item
    else:
        for item in summary.get("top_100_files", []):
            records[item["path"]] = item
    for i, item in enumerate(records.values(), 1):
        add_observation(con, snapshot_id, root_id, root_path, item)
        if i % 5000 == 0:
            con.commit()
    con.commit()
    count = con.execute(
        "SELECT count(*) FROM inventory_observation WHERE snapshot_id=?", (snapshot_id,)
    ).fetchone()[0]
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing snapshot: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for d in (DATA_DIR, SCRATCH4_DIR):
        for name in ("inventory.json", "SHA256SUMS.tsv", "REPORT.md"):
            if not (d / name).is_file():
                raise SystemExit(f"missing archived input: {d / name}")
    temp_path = args.output.with_suffix(args.output.suffix + ".partial")
    if temp_path.exists():
        raise SystemExit(f"refusing to overwrite partial output: {temp_path}")
    os.umask(0o077)
    subprocess.check_call(["sqlite3", str(args.base), ".backup " + str(temp_path)])
    out = sqlite3.connect(str(temp_path))
    try:
        out.execute("PRAGMA foreign_keys=ON")
        roots = dict(out.execute("SELECT root_id,scoped_path FROM storage_root WHERE root_id IN (?,?)", tuple(ROOTS)))
        if roots != ROOTS:
            raise RuntimeError(f"catalog roots mismatch: {roots!r}")
        d_count = register_snapshot(out, DATA_DIR, "RF-DATA-YI", "RF-DATA-YI-20261125", CAPTURE_UTC,
                                   "in-tree hardlink members and top 200 files")
        s_count = register_snapshot(out, SCRATCH4_DIR, "RF-SCRATCH4-YI", "RF-SCRATCH4-YI-20261125",
                                   SCRATCH4_CAPTURE_UTC, "top 100 files")
        out.commit()
        quick = out.execute("PRAGMA quick_check").fetchone()[0]
        fk = out.execute("PRAGMA foreign_key_check").fetchmany(10)
        if quick != "ok" or fk:
            raise RuntimeError(f"integrity failure: quick_check={quick!r}; FK sample={fk!r}")
        print(json.dumps({
            "output": str(temp_path), "quick_check": quick, "foreign_key_violations": 0,
            "data_observations": d_count, "scratch4_observations": s_count,
            "snapshots": ["RF-DATA-YI-20261125", "RF-SCRATCH4-YI-20261125"],
            "inventory_sha256": {
                "data": manifest_hash(DATA_DIR, "inventory.json"),
                "scratch4": manifest_hash(SCRATCH4_DIR, "inventory.json"),
            },
        }, indent=2))
    finally:
        out.close()
    os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(temp_path, args.output)
    print("database_sha256", sha256_file(args.output))
    print("database_bytes", args.output.stat().st_size)


if __name__ == "__main__":
    main()
