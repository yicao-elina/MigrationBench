#!/usr/bin/env python3
"""Load the completed /data and /scratch4 path inventories into SQLite.

The full /data ledger is compressed JSONL; the scratch4 ledger is TSV. Hashes
are only populated later for identity-critical evidence and cleanup candidates.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE.parent
DB_PATH = AUDIT_DIR / "provenance_catalog.sqlite"
DATA_DIR = AUDIT_DIR / "phase2_data"
SCRATCH4_DIR = AUDIT_DIR / "phase2_scratch4"

DATA_ROOT = "/data/pclancy3/yi"
SCRATCH4_ROOT = "/scratch4/pclancy3/yi"
DATA_SUMMARY = DATA_DIR / "data_inventory_summary_20261122.json"
DATA_LEDGER = DATA_DIR / "data_path_inventory_20261122.jsonl.gz"
S4_LEDGER = SCRATCH4_DIR / "scratch4_file_inventory.tsv"

PROJECTS = [
    ("MIGRATIONBENCH", "Wiley MigrationBench", "SHARED", "Sb2Te3 migration, MLIP, and associated paper evidence; path-level ownership remains independently classified."),
    ("DUALX", "DUAL-X", "USER", "User-named research project; no path was assigned by this inventory pass."),
    ("WSE2", "WSe2 calculations", "USER", "Historical user-run QE calculations and structure-generation artifacts."),
    ("HPC_ENVIRONMENTS", "Rockfish software environments", "USER", "Installed Conda environments and package caches; individual environments are not presumed disposable."),
    ("ICLR_2026", "2026 ICLR research", "USER", "Separate project tree; retain source and evidence until its own cleanup review."),
    ("NPJ_CM_2025", "2025 npj computational materials", "USER", "Historical paper-related project tree."),
    ("NPJ_2D_2026", "2026 npj 2D materials", "USER", "Separate paper-related project tree."),
    ("LLM_RESEARCH", "LLM research projects", "USER", "Separate research tree; no cleanup inferred from age or size."),
    ("CYQE", "cyqe tooling", "USER", "QE workflow/tooling source tree."),
    ("PBS_QD", "PbS quantum-dot project", "SHARED", "Project subtree within flare-data; owner and paper role require path-level review."),
    ("LAMMPS_SOURCE", "LAMMPS source/build", "UNKNOWN", "Git repository; ownership and branch state pending repository audit."),
    ("PHONOPY_SOURCE", "phonopy source", "UNKNOWN", "Git repository; ownership and branch state pending repository audit."),
    ("PSEUDOPOTENTIALS", "ONCV pseudopotential pack", "THIRD_PARTY", "Third-party pseudopotential package; no deletion without exact source/version checks."),
    ("YCAO_PIP", "ycao-pip tooling", "USER", "User-named package/tooling tree."),
    ("OTHER_RESEARCH", "Other and unclassified research", "UNKNOWN", "Mixed files without a current project mapping."),
]


def utc_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def mode_octal(text: str | None) -> str | None:
    if not text or len(text) < 10:
        return None
    chars = text[1:10]
    groups = (chars[:3], chars[3:6], chars[6:9])
    result = 0
    for group in groups:
        bits = 0
        if group[0] == "r":
            bits |= 4
        if group[1] == "w":
            bits |= 2
        if group[2] in "xst":
            bits |= 1
        result = (result << 3) | bits
    if chars[2] in "sS":
        result |= stat.S_ISUID
    if chars[5] in "sS":
        result |= stat.S_ISGID
    if chars[8] in "tT":
        result |= stat.S_ISVTX
    return f"0{result:04o}"


def data_project(relative: str) -> str:
    if relative == ".conda" or relative.startswith(".conda/"):
        return "HPC_ENVIRONMENTS"
    rules = (
        ("flare-data/1-Cr-Sb2Te3/", "MIGRATIONBENCH"),
        ("mace_four_model_MD_transfer_", "MIGRATIONBENCH"),
        ("flare-data/2-WSe2-growth/", "WSE2"),
        ("flare-data/3.PbS_QD/", "PBS_QD"),
        ("25npj_CM/", "NPJ_CM_2025"),
        ("26npj2D/", "NPJ_2D_2026"),
        ("26ICLR/", "ICLR_2026"),
        ("LLM/", "LLM_RESEARCH"),
        ("cyqe/", "CYQE"),
        ("mylammps/", "LAMMPS_SOURCE"),
        ("phonopy/", "PHONOPY_SOURCE"),
        ("ONCVPseudoPack/", "PSEUDOPOTENTIALS"),
        ("ycao-pip/", "YCAO_PIP"),
        ("flare-data/", "MIGRATIONBENCH"),
    )
    for prefix, project_id in rules:
        if relative.startswith(prefix):
            return project_id
    return "OTHER_RESEARCH"


def scratch4_project(relative: str) -> str:
    if relative.startswith("3.WSe2/"):
        return "WSE2"
    if relative.startswith("2.Cr-Sb2Te3/"):
        return "MIGRATIONBENCH"
    return "OTHER_RESEARCH"


def artifact_key(root_id: str, relative: str) -> str:
    return f"{root_id}:{relative}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def insert_source(connection: sqlite3.Connection, key: str, path: Path, kind: str, captured: str, as_of: str, note: str) -> int:
    digest = file_sha256(path)
    resolved = str(path.resolve())
    prior = connection.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if prior and prior != (resolved, digest):
        raise RuntimeError(f"Registered source changed: {key}")
    connection.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
           VALUES (?,?,?,?,?,?,1,?)""",
        (key, resolved, kind, digest, captured, as_of, note),
    )
    return connection.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0]


def ensure_projects(connection: sqlite3.Connection) -> None:
    for project_id, name, owner_scope, notes in PROJECTS:
        connection.execute(
            """INSERT OR IGNORE INTO project(project_id,project_name,owner_scope,description,status)
               VALUES (?,?,?,?,?)""",
            (project_id, name, owner_scope, notes, "ACTIVE"),
        )


def insert_path_record(
    connection: sqlite3.Connection,
    root_id: str,
    absolute_path: str,
    root_path: str,
    entry_type: str,
    size: int | None,
    blocks_512: int | None,
    mtime_ns: int | None,
    mode_text: str | None,
    device: int | str | None,
    inode: int | str | None,
    nlink: int | None,
    symlink_target: str | None,
    snapshot_id: str,
    observed_utc: str,
    project_id: str,
) -> tuple[int, int | None]:
    relative = str(PurePosixPath(absolute_path).relative_to(PurePosixPath(root_path)))
    connection.execute(
        "INSERT INTO storage_path(root_id,relative_path,first_seen_utc) VALUES (?,?,?)",
        (root_id, relative, observed_utc),
    )
    path_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
    logical = int(size) if size is not None else None
    allocated = int(blocks_512) * 512 if blocks_512 is not None else None
    connection.execute(
        """INSERT INTO inventory_observation
           (snapshot_id,path_id,entry_type,mode_octal,logical_bytes,allocated_bytes,
            mtime_ns,inode,device,symlink_target,hash_scope,observation_status,link_count)
           VALUES (?,?,?,?,?,?,?,?,?,?, 'NONE','PRESENT',?)""",
        (snapshot_id, path_id, entry_type, mode_octal(mode_text), logical, allocated,
         mtime_ns, str(inode) if inode is not None else None,
         str(device) if device is not None else None, symlink_target, nlink),
    )
    artifact_id: int | None = None
    if entry_type in ("FILE", "SYMLINK"):
        key = artifact_key(root_id, relative)
        connection.execute(
            """INSERT INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,
                byte_size,scientific_status,retention_class,notes)
               VALUES (?,?,?,?, 'UNKNOWN',?,'UNKNOWN','UNCLASSIFIED','Imported from metadata-only inventory; content and owner not inferred.')""",
            (project_id, key, entry_type, "UNCLASSIFIED", logical),
        )
        artifact_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
        connection.execute(
            """INSERT INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES (?,?,'PRESENT',?,?)""",
            (artifact_id, path_id, observed_utc, observed_utc),
        )
    return path_id, artifact_id


def import_data(connection: sqlite3.Connection) -> None:
    summary = json.loads(DATA_SUMMARY.read_text())
    scan_started = utc_from_epoch(summary["started_local_epoch"])
    scan_finished = utc_from_epoch(summary["finished_local_epoch"])
    digest = file_sha256(DATA_LEDGER)
    source_id = insert_source(connection, "rockfish_data_user_inventory_20261122",
                              DATA_LEDGER, "GZIP_JSONL_FILESYSTEM_INVENTORY",
                              scan_finished, scan_finished,
                              "Complete os.walk/lstat metadata-only inventory; owner IDs were not captured; hashes absent except later candidate manifests.")
    snapshot_id = "RF-DATA-YI-20261122-R01"
    connection.execute(
        """INSERT INTO inventory_snapshot
           (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
            files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
            manifest_path,manifest_sha256,notes)
           VALUES (?, 'RF-DATA-YI', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (snapshot_id, scan_started, scan_finished,
         "Single os.walk/lstat pass; metadata only", str(summary["inventory_version"]),
         "COMPLETE" if summary["ssh_exit_code"] == 0 and not summary.get("errors") else "PARTIAL",
         summary["counts"]["file"], summary["counts"]["dir"], summary["counts"]["symlink"],
         len(summary.get("errors", [])), summary["totals"]["file_logical_bytes"], None,
         str(DATA_LEDGER.resolve()), digest,
         "Full scan of /data/pclancy3/yi only; filesystem-wide and other user-owned roots are outside this snapshot."),
    )
    totals = {"FILE": 0, "DIRECTORY": 0, "SYMLINK": 0, "logical": 0, "allocated": 0}
    with gzip.open(DATA_LEDGER, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get("record")
            if kind not in ("file", "dir", "symlink"):
                continue
            entry_type = {"file": "FILE", "dir": "DIRECTORY", "symlink": "SYMLINK"}[kind]
            path = row["path"]
            rel = str(PurePosixPath(path).relative_to(PurePosixPath(DATA_ROOT)))
            proj = data_project(rel)
            size = row.get("size_bytes")
            blocks = row.get("blocks_512")
            insert_path_record(
                connection, "RF-DATA-YI", path, DATA_ROOT, entry_type,
                size, blocks, row.get("mtime_ns"), row.get("mode"),
                row.get("dev"), row.get("ino"), row.get("nlink"),
                row.get("target"), snapshot_id, scan_started, proj,
            )
            totals[entry_type] += 1
            if entry_type == "FILE" and size is not None:
                totals["logical"] += int(size)
                totals["allocated"] += int(blocks or 0) * 512
    if totals["FILE"] != summary["counts"]["file"] or totals["DIRECTORY"] != summary["counts"]["dir"] or totals["SYMLINK"] != summary["counts"]["symlink"]:
        raise RuntimeError(f"Imported /data counts differ from summary: {totals}")
    connection.execute(
        "UPDATE inventory_snapshot SET allocated_bytes=? WHERE snapshot_id=?",
        (totals["allocated"], snapshot_id),
    )
    print(f"/data: {totals['DIRECTORY']} directories, {totals['FILE']} files, {totals['SYMLINK']} symlinks, {totals['logical']} logical bytes, {totals['allocated']} allocated bytes; source_id={source_id}")


def import_scratch4(connection: sqlite3.Connection) -> None:
    scan_observed = "2026-11-22T18:21:53Z"
    source_id = insert_source(connection, "rockfish_scratch4_user_inventory_20261122",
                              S4_LEDGER, "TSV_FILESYSTEM_INVENTORY",
                              scan_observed, scan_observed,
                              "Complete metadata-only os.scandir/lstat inventory; no content hashes except candidate/evidence files listed separately.")
    snapshot_id = "RF-SCRATCH4-YI-20261122-R01"
    connection.execute(
        """INSERT INTO inventory_snapshot
           (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
            files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
            manifest_path,manifest_sha256,notes)
           VALUES (?, 'RF-SCRATCH4-YI', ?, ?, ?, ?, 'COMPLETE', ?, ?, 0, 0, ?, NULL, ?, ?, ?)""",
        (snapshot_id, scan_observed, scan_observed,
         "Python os.scandir/lstat metadata-only pass", "inventory_scratch4.py",
         2886, 55, 5642682016, str(S4_LEDGER.resolve()), file_sha256(S4_LEDGER),
         "Full scan of /scratch4/pclancy3/yi; 55 directories and 2,886 files; no symlinks or traversal errors."),
    )
    totals = {"FILE": 0, "DIRECTORY": 0, "logical": 0}
    with S4_LEDGER.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            kind = row["record_type"]
            if kind not in ("file", "dir", "symlink"):
                continue
            entry_type = {"file": "FILE", "dir": "DIRECTORY", "symlink": "SYMLINK"}[kind]
            path = row["path"]
            rel = str(PurePosixPath(path).relative_to(PurePosixPath(SCRATCH4_ROOT)))
            proj = scratch4_project(rel)
            size = int(row["bytes"]) if row.get("bytes") else None
            mtime_ns = int(row["mtime_epoch"]) if row.get("mtime_epoch") else None
            mode = row.get("mode")
            insert_path_record(
                connection, "RF-SCRATCH4-YI", path, SCRATCH4_ROOT, entry_type,
                size, None, mtime_ns, mode, None, None, None, None,
                snapshot_id, scan_observed, proj,
            )
            totals[entry_type] += 1
            if entry_type == "FILE" and size is not None:
                totals["logical"] += size
    if totals["FILE"] != 2886 or totals["DIRECTORY"] != 55 or totals["logical"] != 5642682016:
        raise RuntimeError(f"Imported /scratch4 counts differ from summary: {totals}")
    print(f"/scratch4: {totals['DIRECTORY']} directories, {totals['FILE']} files, {totals['logical']} logical bytes; source_id={source_id}")


def main() -> None:
    if not DB_PATH.is_file():
        raise SystemExit(f"Missing catalog database: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        ensure_projects(connection)
        existing = connection.execute("SELECT COUNT(*) FROM inventory_snapshot").fetchone()[0]
        if existing:
            raise SystemExit("Inventory snapshots already exist; refusing to duplicate or overwrite them.")
        import_data(connection)
        import_scratch4(connection)
        connection.commit()
    with sqlite3.connect(DB_PATH) as connection:
        print("SQLite integrity:", connection.execute("PRAGMA integrity_check").fetchone()[0])
        print("Foreign-key violations:", connection.execute("SELECT COUNT(*) FROM pragma_foreign_key_check").fetchone()[0])
        for row in connection.execute(
            """SELECT s.snapshot_id, s.completion_state, s.directories_seen, s.files_seen,
                      s.symlinks_seen, s.logical_bytes, s.allocated_bytes
               FROM inventory_snapshot s ORDER BY s.snapshot_id"""
        ):
            print("\t".join(str(value) if value is not None else "NULL" for value in row))


if __name__ == "__main__":
    main()
