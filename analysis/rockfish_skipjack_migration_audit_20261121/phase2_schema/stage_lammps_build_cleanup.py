#!/usr/bin/env python3
"""Stage LAMMPS/FLARE build lineage and a guarded cleanup plan.

Default mode reads local evidence only and prints counts/checksums. It does not
open the catalog. --apply is reserved for after the active catalog importer
has completed and released its lock. Even --apply only stages inventory,
recipe, and cleanup rows; it never moves or deletes files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DATA = AUDIT / "phase2_data"
EVIDENCE = DATA / "lammps_build_recipe_evidence"
DB_DEFAULT = AUDIT / "provenance_catalog.sqlite"

HOME_ROOT = "/data/pclancy3/ycao73_home"
SOURCE_ROOT = f"{HOME_ROOT}/flare/lammps-flare/lammps"
BUILD_ROOT = f"{SOURCE_ROOT}/build"
KOKKOS_CURRENT = f"{BUILD_ROOT}/_deps/kokkoskernels-src"
KOKKOS_DESTINATION = f"{HOME_ROOT}/flare/lammps-flare/kokkoskernels-afd65f03a52e4d9d2536feb9be93150bdb8dec37"
SCRATCH_BUILD = "/scratch4/pclancy3/yi/builds/lammps-flare-b620c52"
REMOTE_SCRIPT_EXPECTED = "/scratch4/pclancy3/yi/cluster_recipes/lammps-flare/rebuild_lammps_flare_gpu.slurm"

ROOT_HOME_ID = "RF-DATA-YCAO73-HOME"
ROOT_LOCAL_ID = "RF-LOCAL-LAMMPS-AUDIT"
PROJECT_ID = "FLARE_LAMMPS"
RUN_ID = "RF-LAMMPS-FLARE-BUILD-B620C52-20261122-R01"
RECIPE_VERSION = "2026-11-22-r01"
BUILD_SNAPSHOT = "RF-LAMMPS-BUILD-20261122-R01"
SOURCE_SNAPSHOT = "RF-LAMMPS-DIRTY-SOURCE-20261122-R01"
CLEANUP_BATCH_ID = "RF-LAMMPS-BUILD-DELETE-20261122-R01"
GIT_REPO_PATH = SOURCE_ROOT
GIT_HEAD = "b620c52b58f1b5810f3c78e23ab34283f3a2d700"
KOKKOS_GIT_COMMIT = "afd65f03a52e4d9d2536feb9be93150bdb8dec37"

FULL_INVENTORY = DATA / "lammps_build_predelete_inventory_sha256_20261122.jsonl"
CANDIDATES = DATA / "lammps_build_cleanup_candidates_20261122.tsv"
KOKKOS_MANIFEST = DATA / "lammps_kokkoskernels_tree_manifest_20261122.jsonl"
SOURCE_CHANGES = DATA / "lammps_source_changes_sha256_20261122.jsonl"
CLEANUP_SUMMARY = DATA / "lammps_build_cleanup_manifest_20261122.json"
INPUT_MANIFEST = EVIDENCE / "BUILD_INPUT_MANIFEST_20261122.json"
REBUILD_SCRIPT = EVIDENCE / "rebuild_lammps_flare_gpu.slurm"
REPRODUCE_DOC = EVIDENCE / "REPRODUCE_BUILD.md"

EVIDENCE_FILES = [
    (FULL_INVENTORY, "BUILD_PREDELETE_HASH_MANIFEST", "Complete SHA-256 metadata inventory of the old build tree."),
    (CANDIDATES, "BUILD_CLEANUP_CANDIDATE_TSV", "Exact 1,876-file cleanup candidate list with pre-delete hashes and metadata."),
    (KOKKOS_MANIFEST, "KOKKOSKERNELS_TREE_HASH_MANIFEST", "Per-file SHA-256 inventory of the preserved KokkosKernels dependency tree."),
    (SOURCE_CHANGES, "LAMMPS_DIRTY_SOURCE_HASH_MANIFEST", "Exact Git dirty-path inventory; regular files are SHA-256 hashed and symlink targets are recorded."),
    (CLEANUP_SUMMARY, "BUILD_CLEANUP_SUMMARY_JSON", "Candidate/preserved-tree totals and manifest checksum summary."),
    (INPUT_MANIFEST, "LAMMPS_BUILD_INPUT_MANIFEST_JSON", "Reproduction input, dependency, toolchain, and command manifest."),
    (REBUILD_SCRIPT, "LAMMPS_BUILD_SLURM_SCRIPT", "CMake configure/build Slurm script; same artifact is referenced as recipe and scheduler script."),
    (REPRODUCE_DOC, "LAMMPS_BUILD_REPRODUCTION_NOTE", "Human-readable build command, toolchain, and limitations."),
    (EVIDENCE / "CMakeCache.txt", "OLD_BUILD_CMAKE_CACHE", "Prior build configuration; contains the stale former /home source path."),
    (EVIDENCE / "CMakeConfigureLog.yaml", "OLD_BUILD_CMAKE_CONFIGURE_LOG", "Prior CMake configure record."),
    (EVIDENCE / "Makefile", "OLD_BUILD_GENERATED_MAKEFILE", "Prior generated build Makefile."),
    (EVIDENCE / "git_and_dependency_refs.txt", "LAMMPS_GIT_DEPENDENCY_REFS", "Repository status, remotes, cached refs, shallow HEAD, and KokkosKernels commit evidence."),
    (EVIDENCE / "source_change_review.txt", "LAMMPS_SOURCE_CHANGE_REVIEW", "Review summary of modified and untracked FLARE source paths."),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_link_payload(target: str) -> str:
    """Hash the recorded symlink target bytes, not the target file's contents."""
    return hashlib.sha256(os.fsencode(target)).hexdigest()


def utc_iso(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def stat_epoch_to_ns(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def source_manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = next((r for r in rows if r.get("record") == "source_start"), None)
    changes = [r for r in rows if r.get("record") == "source_file"]
    status_counts: dict[str, int] = {}
    for row in changes:
        status_counts[row.get("git_status", "")] = status_counts.get(row.get("git_status", ""), 0) + 1
    regular = [r for r in changes if r.get("sha256")]
    symlinks = [r for r in changes if r.get("symlink_target")]
    return {"start": start, "changes": changes, "status_counts": status_counts,
            "regular_hashes": regular, "symlink_rows": symlinks}


def prepare_inputs() -> dict[str, Any]:
    for path, _kind, _note in EVIDENCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing LAMMPS build evidence: {path}")

    cleanup_summary = json.loads(CLEANUP_SUMMARY.read_text(encoding="utf-8"))
    input_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    candidates = read_tsv(CANDIDATES)
    full_rows = read_jsonl(FULL_INVENTORY)
    kokkos_rows = read_jsonl(KOKKOS_MANIFEST)
    source_rows = read_jsonl(SOURCE_CHANGES)

    for key, path in (("candidate_manifest_sha256", CANDIDATES),
                      ("inventory_sha256", FULL_INVENTORY),
                      ("preserved_manifest_sha256", KOKKOS_MANIFEST)):
        if sha256_file(path) != cleanup_summary[key]:
            raise ValueError(f"Cleanup summary SHA mismatch for {path}")
    if sha256_file(SOURCE_CHANGES) != input_manifest["source_repository"]["dirty_source_manifest_sha256"]:
        raise ValueError("Dirty-source manifest SHA differs from combined build input manifest")
    if sha256_file(REBUILD_SCRIPT) != input_manifest["rebuild"]["slurm_script_sha256"]:
        raise ValueError("Slurm rebuild script SHA differs from combined build input manifest")
    if sha256_file(KOKKOS_MANIFEST) != input_manifest["external_dependencies"][0]["manifest_sha256"]:
        raise ValueError("KokkosKernels manifest SHA differs from combined build input manifest")

    candidate_paths = [r["path"] for r in candidates]
    if len(candidates) != 1876 or len(candidate_paths) != len(set(candidate_paths)):
        raise ValueError(f"Expected 1,876 distinct cleanup candidate files, found {len(candidates)}")
    candidate_logical = sum(int(r["size_bytes"]) for r in candidates)
    candidate_allocated = sum(int(r["allocated_bytes"]) for r in candidates)
    if (candidate_logical != 4052888613 or candidate_allocated != 4090057216 or
            candidate_logical != cleanup_summary["candidate_logical_bytes"] or
            candidate_allocated != cleanup_summary["candidate_allocated_bytes"]):
        raise ValueError("Candidate byte totals differ from the reviewed cleanup summary")

    build_records = [r for r in full_rows if r.get("record") in ("file", "dir", "symlink", "special")]
    build_by_path = {r["path"]: r for r in build_records}
    if len(build_by_path) != len(build_records):
        raise ValueError("Duplicate path in full build pre-delete manifest")
    file_rows = [r for r in build_records if r.get("record") == "file"]
    dir_rows = [r for r in build_records if r.get("record") == "dir"]
    link_rows = [r for r in build_records if r.get("record") == "symlink"]
    if (len(file_rows), len(dir_rows), len(link_rows)) != (7565, 1129, 3):
        raise ValueError("Full build inventory entry counts differ from the reviewed snapshot")
    if sum(int(r["size_bytes"]) for r in file_rows) != 4288637058:
        raise ValueError("Full build inventory logical bytes differ from the reviewed snapshot")
    if sum(int(r["allocated_bytes"]) for r in file_rows) != 4446400000:
        raise ValueError("Full build inventory allocated bytes differ from the reviewed snapshot")
    for candidate in candidates:
        record = build_by_path.get(candidate["path"])
        if not record or record.get("record") != "file":
            raise ValueError(f"Candidate absent from full-file inventory: {candidate['path']}")
        for tsv_key, json_key in (("size_bytes", "size_bytes"), ("allocated_bytes", "allocated_bytes"),
                                  ("mtime_ns", "mtime_ns"), ("mode_octal", "mode_octal"),
                                  ("uid", "uid"), ("gid", "gid"), ("nlink", "nlink"),
                                  ("dev", "dev"), ("ino", "ino"), ("sha256", "sha256")):
            if str(candidate[tsv_key]) != str(record.get(json_key, "")):
                raise ValueError(f"Candidate/full manifest mismatch for {candidate['path']} field {tsv_key}")

    kokkos_root = cleanup_summary["preserved_dependency_root"]
    kokkos_files = [r for r in kokkos_rows if r.get("record") == "file"]
    kokkos_dirs = [r for r in kokkos_rows if r.get("record") == "dir"]
    kokkos_links = [r for r in kokkos_rows if r.get("record") == "symlink"]
    if (len(kokkos_files), len(kokkos_dirs), len(kokkos_links)) != (5689, 911, 3):
        raise ValueError("KokkosKernels manifest entry counts differ from the reviewed preservation summary")
    if sum(int(r["size_bytes"]) for r in kokkos_files) != 235748445:
        raise ValueError("KokkosKernels logical bytes differ from the reviewed preservation summary")
    if sum(int(r["allocated_bytes"]) for r in kokkos_files) != 356342784:
        raise ValueError("KokkosKernels allocated bytes differ from the reviewed preservation summary")
    for row in kokkos_files + kokkos_dirs + kokkos_links:
        build_record = build_by_path.get(row["path"])
        if not build_record or build_record.get("record") != row.get("record"):
            raise ValueError(f"KokkosKernels path absent/different in full build manifest: {row['path']}")
        if row.get("record") == "file" and row.get("sha256") != build_record.get("sha256"):
            raise ValueError(f"KokkosKernels SHA differs from full build manifest: {row['path']}")

    src = source_manifest_summary(source_rows)
    if not src["start"] or len(src["changes"]) != 407 or len(src["regular_hashes"]) != 388 or len(src["symlink_rows"]) != 19:
        raise ValueError("Dirty-source evidence does not cover the expected 407 paths")
    if src["status_counts"].get(" M") != 1 or src["status_counts"].get("??") != 406:
        raise ValueError("Dirty-source evidence is not one tracked modification plus 406 untracked paths")
    source_paths = []
    for row in src["changes"]:
        full_path = str(PurePosixPath(SOURCE_ROOT) / row["relative_path"])
        if row["path"] != full_path:
            raise ValueError(f"Dirty-source path escaped source root: {row['path']}")
        if not row.get("sha256") and not row.get("symlink_target"):
            raise ValueError(f"Dirty source lacks file SHA/symlink target: {row['path']}")
        source_paths.append(full_path)
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("Duplicate dirty-source paths in source manifest")

    source_specs = []
    for path, kind, note in EVIDENCE_FILES:
        source_specs.append({
            "source_key": f"lammps_build_{path.stem}_20261122",
            "path": path,
            "registry_kind": kind,
            "sha256": sha256_file(path),
            "as_of_utc": utc_iso(path.stat().st_mtime),
            "authority_note": note,
        })

    return {
        "cleanup_summary": cleanup_summary,
        "input_manifest": input_manifest,
        "candidates": candidates,
        "build_records": build_records,
        "build_start": next(r for r in full_rows if r.get("record") == "scan_start"),
        "build_end": next(r for r in full_rows if r.get("record") == "scan_end"),
        "kokkos_records": kokkos_files + kokkos_dirs + kokkos_links,
        "kokkos_files": kokkos_files,
        "source_records": src["changes"],
        "source_start": src["start"],
        "source_specs": source_specs,
        "counts": {
            "build_inventory_files": len(file_rows),
            "build_inventory_directories": len(dir_rows),
            "build_inventory_symlinks": len(link_rows),
            "cleanup_candidate_files": len(candidates),
            "cleanup_candidate_logical_bytes": candidate_logical,
            "cleanup_candidate_allocated_bytes": candidate_allocated,
            "preserved_kokkos_files": len(kokkos_files),
            "preserved_kokkos_directories": len(kokkos_dirs),
            "preserved_kokkos_symlinks": len(kokkos_links),
            "preserved_kokkos_allocated_bytes": sum(int(r["allocated_bytes"]) for r in kokkos_files),
            "dirty_source_paths": len(src["changes"]),
            "dirty_source_regular_file_hashes": len(src["regular_hashes"]),
            "dirty_source_symlink_payloads": len(src["symlink_rows"]),
        },
    }


def register_source(con: sqlite3.Connection, spec: dict[str, Any], captured: str) -> int:
    path = spec["path"].resolve()
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (spec["source_key"],)).fetchone()
    expected = (str(path), spec["sha256"])
    if prior and tuple(prior) != expected:
        raise RuntimeError(f"Registered evidence source changed: {spec['source_key']}")
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
           VALUES (?,?,?,?,?,?,1,?)""",
        (spec["source_key"], str(path), spec["registry_kind"], spec["sha256"], captured,
         spec["as_of_utc"], spec["authority_note"]),
    )
    return con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (spec["source_key"],)).fetchone()[0]


def ensure_root(con: sqlite3.Connection, root_id: str, cluster: str, filesystem: str,
                mount: str, scoped: str, owner_user: str | None, owner_group: str | None,
                scope_kind: str, project_id: str | None, status: str, notes: str) -> None:
    expected = (cluster, filesystem, mount, scoped, owner_user, owner_group, scope_kind)
    existing = con.execute(
        "SELECT cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,scope_kind FROM storage_root WHERE root_id=?",
        (root_id,),
    ).fetchone()
    if existing and tuple(existing) != expected:
        raise RuntimeError(f"Storage root identity differs for {root_id}")
    con.execute(
        """INSERT OR IGNORE INTO storage_root
           (root_id,cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,scope_kind,project_id,status,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (root_id, cluster, filesystem, mount, scoped, owner_user, owner_group, scope_kind, project_id, status, notes),
    )


def path_id(con: sqlite3.Connection, root_id: str, root_path: str, absolute: str, observed: str) -> int:
    relative = str(PurePosixPath(absolute).relative_to(PurePosixPath(root_path)))
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES (?,?,?)",
                (root_id, relative, observed))
    return con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
                       (root_id, relative)).fetchone()[0]


def ensure_artifact(con: sqlite3.Connection, *, project_id: str, artifact_key: str,
                    artifact_kind: str, semantic_role: str, owner_status: str,
                    sha256: str | None, byte_size: int | None, retention_class: str,
                    notes: str, scientific_status: str = "UNKNOWN") -> int:
    old = con.execute(
        "SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)
    ).fetchone()
    if old:
        if old[1] and sha256 and old[1] != sha256:
            raise RuntimeError(f"Artifact hash differs for {artifact_key}")
        if old[2] is not None and byte_size is not None and old[2] != byte_size:
            raise RuntimeError(f"Artifact size differs for {artifact_key}")
        con.execute(
            """UPDATE artifact SET project_id=COALESCE(project_id,?),sha256=COALESCE(sha256,?),
               byte_size=COALESCE(byte_size,?),owner_status=?,artifact_kind=?,semantic_role=?,
               retention_class=?,scientific_status=?,notes=? WHERE artifact_id=?""",
            (project_id, sha256, byte_size, owner_status, artifact_kind, semantic_role,
             retention_class, scientific_status, notes, old[0]),
        )
        return old[0]
    con.execute(
        """INSERT INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (project_id, artifact_key, artifact_kind, semantic_role, owner_status, sha256, byte_size,
         scientific_status, retention_class, notes),
    )
    return con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()[0]


def ensure_location(con: sqlite3.Connection, artifact_id: int, pid: int, observed: str,
                    availability: str = "PRESENT") -> None:
    old = con.execute("SELECT availability FROM artifact_location WHERE artifact_id=? AND path_id=?", (artifact_id, pid)).fetchone()
    if old and availability == "PRESENT" and old[0] not in ("PRESENT", "ARCHIVED"):
        raise RuntimeError(f"Existing artifact location is {old[0]}, expected present: artifact_id={artifact_id} path_id={pid}")
    con.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES (?,?,?,?,?)""",
        (artifact_id, pid, availability, observed if availability == "PRESENT" else None,
         observed if availability == "PRESENT" else None),
    )


def add_run_artifact(con: sqlite3.Connection, artifact_id: int, role: str, direction: str,
                     *, required: int = 0, notes: str = "") -> None:
    con.execute(
        """INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes)
           VALUES (?,?,?,?,?,?)""",
        (RUN_ID, artifact_id, role, direction, required, notes),
    )


def apply_catalog(db_path: Path, data: dict[str, Any]) -> None:
    con = sqlite3.connect(db_path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    try:
        schema = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migration'").fetchone()
        if not schema:
            raise RuntimeError("Catalog base schema is not initialized")
        for version, expected_name in (
            (4, "004_git_snapshot_observations.sql"),
            (5, "005_remove_redundant_storage_path_index.sql"),
        ):
            applied = con.execute(
                "SELECT migration_file FROM schema_migration WHERE version=?", (version,)
            ).fetchone()
            if not applied or applied[0] != expected_name:
                raise RuntimeError(
                    f"Required migration {version} ({expected_name}) is not recorded as applied; "
                    "this staging script will not apply or replay migrations."
                )
        already = con.execute("SELECT 1 FROM workflow_run WHERE run_id=?", (RUN_ID,)).fetchone()
        batch_exists = con.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id=?", (CLEANUP_BATCH_ID,)).fetchone()
        if already or batch_exists:
            raise RuntimeError("LAMMPS build run or cleanup batch already exists; review before retrying")

        con.execute("BEGIN IMMEDIATE")
        captured = now_utc()
        con.execute(
            "INSERT OR IGNORE INTO project(project_id,project_name,owner_scope,description,status) VALUES (?,?,? ,?,'ACTIVE')",
            (PROJECT_ID, "FLARE and LAMMPS source", "USER",
             "User-customized FLARE/LAMMPS build; upstream LAMMPS base is third-party."),
        )
        ensure_root(con, ROOT_HOME_ID, "Rockfish", "data", "/data", HOME_ROOT, "ycao73", "pclancy3",
                    "USER_SUBTREE", PROJECT_ID, "ACTIVE", "User-owned ycao73_home subtree; LAMMPS source and old build evidence are inventoried separately.")
        ensure_root(con, ROOT_LOCAL_ID, "Local workstation", "local", "/", str(DATA.resolve()), "alina", None,
                    "OTHER", PROJECT_ID, "READ_ONLY", "Local immutable Rockfish evidence bundle used to stage the LAMMPS build recipe and cleanup plan.")
        # Create the run before attaching its build artifacts through run_artifact.
        con.execute(
            """INSERT INTO workflow_run
               (run_id,project_id,run_label,workflow_type,method,system_key,source_root_id,
                source_relative_path,scheduler_state,scientific_status,notes)
               VALUES (?,?,?,'SOFTWARE_BUILD',?,?,?,?, 'UNKNOWN','PROVENANCE_PARTIAL',?)""",
            (RUN_ID, PROJECT_ID, "LAMMPS-FLARE GPU rebuild b620c52", "CMake Unix Makefiles; MPI/OpenMP/Kokkos-CUDA",
             "Rockfish single-node CUDA Volta 70", ROOT_HOME_ID, "flare/lammps-flare/lammps",
             "Recipe documented; no fresh build/runtime test; old build tree is inventory evidence only."),
        )

        source_ids: dict[str, int] = {}
        evidence_artifact_ids: dict[str, int] = {}
        local_root_path = str(DATA.resolve())
        for spec in data["source_specs"]:
            source_ids[spec["source_key"]] = register_source(con, spec, captured)
            local_pid = path_id(con, ROOT_LOCAL_ID, local_root_path, str(spec["path"].resolve()), spec["as_of_utc"])
            art_id = ensure_artifact(
                con, project_id=PROJECT_ID, artifact_key="LMP-EVIDENCE:" + spec["path"].name,
                artifact_kind="AUDIT_EVIDENCE", semantic_role=spec["registry_kind"],
                owner_status="USER_CONFIRMED", sha256=spec["sha256"], byte_size=spec["path"].stat().st_size,
                retention_class="PERMANENT_SOURCE", notes=spec["authority_note"],
            )
            ensure_location(con, art_id, local_pid, spec["as_of_utc"])
            evidence_artifact_ids[spec["path"].resolve().name] = art_id

        existing_snapshot = con.execute("SELECT 1 FROM inventory_snapshot WHERE snapshot_id IN (?,?)", (BUILD_SNAPSHOT, SOURCE_SNAPSHOT)).fetchone()
        existing_batch = con.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id=?", (CLEANUP_BATCH_ID,)).fetchone()
        if existing_snapshot or existing_batch:
            raise RuntimeError("A LAMMPS staging identity already exists; transaction rolled back")

        # Register the complete build-tree metadata snapshot (all 7,565 files,
        # 1,129 directories, and 3 symlinks) under the existing ycao73_home root.
        build_end = data["build_end"]
        build_start = data["build_start"]
        build_observed = utc_iso(build_end["finished_epoch"])
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES (?,?,?,?,?,'hash_lammps_build_remote.py','COMPLETE',?,?,?,?,?,?,?,?,?)""",
            (BUILD_SNAPSHOT, ROOT_HOME_ID, utc_iso(build_start["started_epoch"]), build_observed,
             "Remote os.scandir/lstat metadata and SHA-256 walk; symlinks are not followed.",
             7565, 1129, 3, 0, 4288637058, 4446400000,
             str(FULL_INVENTORY.resolve()), sha256_file(FULL_INVENTORY),
             "Complete old-build snapshot. The 1,876 cleanup candidates exclude the preserved KokkosKernels subtree; allocated bytes are path allocation, not the future quota receipt."),
        )
        all_paths = [(ROOT_HOME_ID, str(PurePosixPath(r["path"]).relative_to(PurePosixPath(HOME_ROOT))), build_observed)
                     for r in data["build_records"]]
        con.executemany("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES (?,?,?)", all_paths)
        build_path_ids = {}
        for row in data["build_records"]:
            relative = str(PurePosixPath(row["path"]).relative_to(PurePosixPath(HOME_ROOT)))
            pid = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_HOME_ID, relative)).fetchone()[0]
            build_path_ids[row["path"]] = pid
            digest = row.get("sha256")
            # Inventory observations only claim full-file hashes for regular
            # files. Symlink identity is the separately recorded target text.
            hash_scope = "FULL_FILE" if digest else "NONE"
            entry_type = {"file": "FILE", "dir": "DIRECTORY", "symlink": "SYMLINK", "special": "OTHER"}[row["record"]]
            con.execute(
                """INSERT OR IGNORE INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,
                    symlink_target,sha256,hash_scope,observation_status,link_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'PRESENT',?)""",
                (BUILD_SNAPSHOT, pid, entry_type, row.get("uid"), row.get("gid"), row.get("mode_octal"),
                 row.get("size_bytes"), row.get("allocated_bytes"), row.get("mtime_ns"),
                 str(row["ino"]) if row.get("ino") is not None else None,
                 str(row["dev"]) if row.get("dev") is not None else None,
                 row.get("symlink_target"), digest, hash_scope, row.get("nlink")),
            )

        # Capture all 407 exact dirty source paths. Symlink SHA values are
        # hashes of recorded link-target bytes, clearly distinguished in notes.
        source_start = data["source_start"]
        source_observed = utc_iso(source_start["captured_epoch"])
        source_logical_bytes = sum(int(r["size_bytes"]) for r in data["source_records"])
        source_files = [r for r in data["source_records"] if r.get("sha256")]
        source_links = [r for r in data["source_records"] if r.get("symlink_target")]
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES (?,?,?,?,?,'inventory_lammps_source_changes_remote.py','COMPLETE',?,0,?,0,?,NULL,?,?,?)""",
            (SOURCE_SNAPSHOT, ROOT_HOME_ID, source_observed, source_observed,
             "Git-status expanded dirty-path lstat and SHA-256 capture; symlink targets are recorded and target-byte hashed.",
             len(source_files), len(source_links), source_logical_bytes,
             str(SOURCE_CHANGES.resolve()), sha256_file(SOURCE_CHANGES),
             "Complete 407-path dirty-source scope only, not a full source-repository inventory. 388 regular-file hashes plus 19 symlink payload identities."),
        )
        source_artifact_ids: dict[str, int] = {}
        for row in data["source_records"]:
            absolute = str(PurePosixPath(SOURCE_ROOT) / row["relative_path"])
            relative = str(PurePosixPath(absolute).relative_to(PurePosixPath(HOME_ROOT)))
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES (?,?,?)",
                        (ROOT_HOME_ID, relative, source_observed))
            pid = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_HOME_ID, relative)).fetchone()[0]
            is_link = bool(row.get("symlink_target"))
            # The observation stores the target text directly; the artifact
            # record below may additionally carry its explicitly labeled
            # target-payload hash.
            digest = row.get("sha256")
            entry_type = "SYMLINK" if is_link else "FILE"
            con.execute(
                """INSERT OR IGNORE INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,mtime_ns,inode,device,
                    symlink_target,sha256,hash_scope,observation_status,link_count)
                   VALUES (?,?,?,?,?,?,?, ?,NULL,NULL,?,?,?,'PRESENT',?)""",
                (SOURCE_SNAPSHOT, pid, entry_type, row.get("uid"), row.get("gid"), row.get("mode_octal"),
                 row.get("size_bytes"), row.get("mtime_ns"), row.get("symlink_target"), digest,
                 "FULL_FILE" if digest else "NONE", row.get("nlink")),
            )
            art_id = ensure_artifact(
                con, project_id=PROJECT_ID, artifact_key="LMP-SOURCE:" + row["relative_path"],
                artifact_kind="SOURCE_SYMLINK" if is_link else "SOURCE_FILE",
                semantic_role="Dirty source input; git_status=" + row.get("git_status", ""),
                owner_status="USER_CONFIRMED", sha256=digest, byte_size=int(row["size_bytes"]),
                retention_class="REPRODUCIBLE_INPUT",
                notes=("Git HEAD=" + GIT_HEAD + "; relative_path=" + row["relative_path"] +
                       ("; symlink_target=" + row["symlink_target"] + "; sha256 hashes recorded target bytes, not target file contents." if is_link else "; full-file SHA-256 captured.")),
                scientific_status="PROVENANCE_PARTIAL",
            )
            ensure_location(con, art_id, pid, source_observed)
            source_artifact_ids[row["relative_path"]] = art_id

        # Hash-backed candidate artifacts and retention plan.
        candidate_artifact_ids: dict[str, int] = {}
        candidate_by_name: dict[str, dict[str, Any]] = {}
        for row in data["candidates"]:
            absolute = row["path"]
            relative = str(PurePosixPath(absolute).relative_to(PurePosixPath(HOME_ROOT)))
            art_id = ensure_artifact(
                con, project_id=PROJECT_ID, artifact_key=ROOT_HOME_ID + ":" + relative,
                artifact_kind="BUILD_OUTPUT_FILE", semantic_role="Old generated LAMMPS build output",
                owner_status="USER_CONFIRMED", sha256=row["sha256"], byte_size=int(row["size_bytes"]),
                retention_class="REPRODUCIBLE_INTERMEDIATE",
                notes="Full pre-delete SHA-256 and lstat metadata recorded. Rebuild recipe is documented but not yet rerun; no deletion occurs in this staging script.",
                scientific_status="PROVENANCE_PARTIAL",
            )
            pid = build_path_ids[absolute]
            ensure_location(con, art_id, pid, build_observed)
            candidate_artifact_ids[absolute] = art_id
            candidate_by_name[Path(absolute).name] = row
            add_run_artifact(con, art_id, "CLEANUP_CANDIDATE", "INTERMEDIATE",
                             notes="Regenerable CMake/MPI/OpenMP/Kokkos/CUDA build output; exact pre-delete hash in inventory.")

        # The current old binary is linked as prior run output evidence. It is
        # itself a cleanup candidate and remains PLANNED pending queue checks.
        for output_name in ("lmp", "liblammps.a"):
            candidate_path = str(PurePosixPath(BUILD_ROOT) / output_name)
            if candidate_path not in candidate_artifact_ids:
                raise ValueError(f"Expected historical build output missing from cleanup set: {candidate_path}")
            add_run_artifact(con, candidate_artifact_ids[candidate_path], "HISTORICAL_BUILD_OUTPUT", "OUTPUT",
                             notes="Existing build output evidenced by pre-delete hash manifest; not independently validated as a scientific result.")

        # Register KokkosKernels as a retained dependency-tree input and stage
        # its move outside build without modifying either path.
        tree_path_id = build_path_ids[KOKKOS_CURRENT]
        tree_manifest_artifact_id = evidence_artifact_ids[KOKKOS_MANIFEST.name]
        kokkos_tree_id = ensure_artifact(
            con, project_id=PROJECT_ID, artifact_key="LMP-KOKKOSKERNELS-TREE-" + KOKKOS_GIT_COMMIT,
            artifact_kind="DEPENDENCY_TREE", semantic_role="Pinned KokkosKernels source dependency tree",
            owner_status="THIRD_PARTY", sha256=None, byte_size=None, retention_class="REPRODUCIBLE_INPUT",
            notes=f"Git commit {KOKKOS_GIT_COMMIT}; 5,689 files, 911 directories, 3 symlinks; 235,748,445 logical B / 356,342,784 allocated B. Per-file hashes are in the separately registered tree manifest.",
            scientific_status="LINEAGE_SUPPORTED",
        )
        ensure_location(con, kokkos_tree_id, tree_path_id, build_observed)
        destination_pid = path_id(con, ROOT_HOME_ID, HOME_ROOT, KOKKOS_DESTINATION, build_observed)
        ensure_location(con, kokkos_tree_id, destination_pid, build_observed, "PLANNED")
        add_run_artifact(con, kokkos_tree_id, "KOKKOSKERNELS_SOURCE_TREE", "INPUT", required=1,
                         notes="Pinned dependency source; preserve/move outside build before cleanup. Destination presence must be verified later.")
        add_run_artifact(con, tree_manifest_artifact_id, "KOKKOSKERNELS_HASH_MANIFEST", "PROVENANCE", required=1)

        input_manifest_id = evidence_artifact_ids[INPUT_MANIFEST.name]
        script_id = evidence_artifact_ids[REBUILD_SCRIPT.name]
        source_manifest_id = evidence_artifact_ids[SOURCE_CHANGES.name]
        add_run_artifact(con, input_manifest_id, "BUILD_INPUT_MANIFEST", "INPUT", required=1)
        add_run_artifact(con, source_manifest_id, "DIRTY_SOURCE_HASH_MANIFEST", "INPUT", required=1)
        add_run_artifact(con, script_id, "SLURM_REBUILD_SCRIPT", "PROVENANCE", required=1)
        for relpath, art_id in source_artifact_ids.items():
            add_run_artifact(con, art_id, "SOURCE_CHANGE:" + relpath, "INPUT", required=1,
                             notes="Exact local dirty/untracked source path tied to the recorded base Git commit.")

        # The repo-26 Git import is a prerequisite; bind the build to that exact
        # commit without treating the upstream branch as merge authorization.
        repo_row = con.execute("SELECT repository_id FROM git_repository WHERE local_path=?", (GIT_REPO_PATH,)).fetchone()
        if not repo_row:
            raise RuntimeError("Repo 26 must be imported before this build cleanup stage")
        repo_id = repo_row[0]
        repo_snapshot = con.execute(
            "SELECT 1 FROM git_repository_snapshot WHERE repository_id=? AND head_sha=?",
            (repo_id, GIT_HEAD),
        ).fetchone()
        if not repo_snapshot:
            raise RuntimeError("Repo 26 snapshot at the exact source base commit must be imported first")
        con.execute(
            """INSERT INTO run_git_binding(run_id,repository_id,commit_sha,binding_role,evidence_artifact_id,provenance_status)
               VALUES (?,?,?,?,?,'LINEAGE_SUPPORTED')""",
            (RUN_ID, repo_id, GIT_HEAD, "SOURCE_BASE_COMMIT", source_manifest_id),
        )
        con.execute(
            """INSERT INTO reproduction_recipe
               (run_id,recipe_version,exact_command,working_directory,executable_version,
                environment_description,module_list,script_artifact_id,scheduler_script_artifact_id,
                input_manifest_artifact_id,recipe_status,validation_note)
               VALUES (?,?,?,?,?,?,?,?,?,?, 'DOCUMENTED',?)""",
            (RUN_ID, RECIPE_VERSION, f"sbatch {REMOTE_SCRIPT_EXPECTED}", SOURCE_ROOT,
             "CMake 3.27.7; GCC 9.3.0; OpenMPI 3.1.6; CUDA 11.1.74",
             "RelWithDebInfo; MPI+OpenMP; static LAMMPS; Kokkos CUDA for Volta 70; BLAS KokkosKernels. Absolute tool paths and exact flags are recorded in the Slurm script and input manifest.",
             "No modules loaded by the script; absolute toolchain paths are used.", script_id, script_id, input_manifest_id,
             "Not rerun or runtime-tested. Before deletion, verify the script exists at the submit path, KokkosKernels is preserved at the planned sibling path with all file hashes, and the live Slurm queue is clear. The source Git clone is shallow and customized; no upstream merge/ref update is permitted by this recipe."),
        )

        batch_evidence_id = evidence_artifact_ids[CLEANUP_SUMMARY.name]
        # Catalog cleanup_batch.expected_bytes follows logical deleted size;
        # allocated-byte totals remain in the inventory and receipt evidence.
        con.execute(
            """INSERT INTO cleanup_batch
               (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,
                expected_bytes,evidence_artifact_id)
               VALUES (?,?,?,?,?,'PLANNED',?,?,?)""",
            (CLEANUP_BATCH_ID, captured, ROOT_HOME_ID, "DELETE_HASHED_REPRODUCIBLE_BUILD_INTERMEDIATES",
             "Stage deletion of exactly 1,876 generated files under the old build tree; preserve and relocate the pinned KokkosKernels source tree.",
             "No files are deleted or moved by this script. Execute only after recipe/input/artifact review, script destination verification, Kokkos tree relocation/hash verification, live-queue-clear evidence, and a fresh pre-delete identity check.",
             data["counts"]["cleanup_candidate_logical_bytes"], batch_evidence_id),
        )

        for row in data["candidates"]:
            absolute = row["path"]
            artifact_id = candidate_artifact_ids[absolute]
            relative = str(PurePosixPath(absolute).relative_to(PurePosixPath(HOME_ROOT)))
            con.execute(
                """INSERT INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,
                    pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes)
                   VALUES (?,?,'DELETE',?,'CLEARED',?,?,?,'PLANNED',?)""",
                (CLEANUP_BATCH_ID, artifact_id,
                 "Generated CMake/compiler output in old LAMMPS build tree; exact hash/size recorded and linked to documented fresh-build recipe.",
                 row["sha256"], int(row["size_bytes"]), int(row["mtime_ns"]),
                 f"Path={relative}; allocated_bytes={row['allocated_bytes']}; deletion remains PLANNED pending live-queue and Kokkos relocation checks."),
            )
        con.execute(
            """INSERT INTO cleanup_item
               (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,
                pre_delete_size_bytes,deletion_state,notes)
               VALUES (?,?,'ARCHIVE','Preserve/move the 5,689-file pinned KokkosKernels tree outside build.','NOT_REQUIRED',NULL,'PLANNED',?)""",
            (CLEANUP_BATCH_ID, kokkos_tree_id,
             f"Planned destination: {KOKKOS_DESTINATION}; source tree remains at {KOKKOS_CURRENT} until a separately reviewed move. Verify all 5,689 file hashes from the retained manifest after relocation."),
        )

        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Foreign-key violations after staging: {violations[:5]}")
        con.commit()
        print(json.dumps({"status": "STAGED_NO_FILES_MOVED_OR_DELETED", "run_id": RUN_ID,
                          "cleanup_batch_id": CLEANUP_BATCH_ID, "counts": data["counts"]}, indent=2))
    except Exception:
        if con.in_transaction:
            con.rollback()
        raise
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DB_DEFAULT, help="Catalog path, only opened with --apply")
    parser.add_argument("--apply", action="store_true", help="Append inventory, run/recipe, and PLANNED cleanup rows after importer lock is released")
    args = parser.parse_args()
    data = prepare_inputs()
    print(json.dumps({"mode": "APPLY_REQUESTED" if args.apply else "LOCAL_VALIDATION_ONLY",
                      "catalog_opened": False,
                      "evidence": [{"path": str(s["path"].resolve()), "sha256": s["sha256"]} for s in data["source_specs"]],
                      "counts": data["counts"],
                      "cleanup_effect": "No files are moved or deleted; candidate items remain PLANNED.",
                      "source_status": "The 407 modified/untracked source paths are preserved as 388 full-file hashes and 19 link-target payload hashes."},
                     indent=2, ensure_ascii=False))
    if args.apply:
        if not args.catalog.is_file():
            raise SystemExit(f"Catalog not found: {args.catalog}")
        apply_catalog(args.catalog, data)


if __name__ == "__main__":
    main()
