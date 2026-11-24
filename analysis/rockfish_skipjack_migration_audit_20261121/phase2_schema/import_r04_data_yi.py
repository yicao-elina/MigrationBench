#!/usr/bin/env python3
"""Import the exact composite current inventory for /data/pclancy3/yi.

The non-Conda subtree comes from the complete R04 no-follow scan. The .conda
path set comes from the already reconciled R01/R02/R03 evidence and exact
cleanup receipts; it is not rescanned here. The composite snapshot is complete
for path coverage but is not an atomic full-tree restat.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
YI_DIR = AUDIT / "phase2_data/yi_full_inventory_20261123_r04"
R04_STATUS = YI_DIR / "yi_nonconda_metadata_r04_status.json"
R04_RAW_LEDGER = YI_DIR / "yi_nonconda_metadata_r04.jsonl.gz"
R04_MANIFEST = YI_DIR / "yi_current_nonconda_metadata_manifest_r04.jsonl.gz"
R04_RECON = YI_DIR / "yi_coverage_reconciliation_r04.json"
R03_DIR = AUDIT / "phase2_data/data_refresh_20261123_r03"
R03_RECON = R03_DIR / "conda_coverage_reconciliation_r03.json"
R01_BASELINE = AUDIT / "phase2_data/data_path_inventory_20261122.jsonl.gz"
R02_PARTIAL = AUDIT / "phase2_data/data_refresh_20261123/data_pclancy3_refresh_20261123.jsonl.gz"
COMPOSITE_OUTPUT = YI_DIR / "yi_current_full_metadata_composite_r04.jsonl.gz"
ROOT_ID = "RF-DATA-PCLANCY3"
ROOT_PATH = "/data/pclancy3"
YI_ROOT = ROOT_PATH + "/yi"
SNAPSHOT_ID = "RF-DATA-PCLANCY3-YI-R04-COMPOSITE-20261123T064026Z"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iso(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_analyzer():
    path = R03_DIR / "analyze_conda_coverage_r03.py"
    spec = importlib.util.spec_from_file_location("analyze_conda_coverage_r03", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the R03 Conda reconciler")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_composite_manifest() -> tuple[str, dict]:
    r04_status = json.loads(R04_STATUS.read_text(encoding="utf-8"))
    r04_report = json.loads(R04_RECON.read_text(encoding="utf-8"))
    r03_report = json.loads(R03_RECON.read_text(encoding="utf-8"))
    nonconda = r04_report["r04_nonconda_scan"]
    conda = r03_report["coverage_comparison"]
    if r04_status.get("status") != "COMPLETE" or r04_status.get("remote_exit_code") != 0:
        raise RuntimeError("R04 non-Conda scan is not complete")
    if r04_status.get("error_records") != 0 or not r04_status.get("scan_complete_record_present"):
        raise RuntimeError("R04 non-Conda scan has errors or lacks completion marker")
    if r04_report["current_union"]["status"] != "COMPLETE_RECONCILED":
        raise RuntimeError("R04/R03 union is not marked complete")
    if r03_report.get("coverage_status") != "COMPLETE_RECONCILED_FROM_FULL_BASELINE_INCREMENTAL_SCANS_AND_EXACT_CLEANUP_RECEIPTS":
        raise RuntimeError("R03 Conda coverage is not fully reconciled")
    if (int(conda.get("uncovered_current_paths", -1)) != 0 or
            int(conda.get("unexpected_paths", -1)) != 0 or
            int(conda.get("receipt_deleted_paths_still_observed", -1)) != 0):
        raise RuntimeError("R03 Conda path reconciliation has a gap")

    analyzer = load_analyzer()
    removed, _ = analyzer.load_deleted_paths()
    current, _, _, _ = analyzer.load_current_scans()
    final_conda = {path: item for path, item in current.items() if path not in removed}
    expected_conda = int(r03_report["coverage_comparison"]["expected_current_paths"])
    if len(final_conda) != expected_conda:
        raise RuntimeError(f"R03 Conda current path count mismatch: {len(final_conda)} != {expected_conda}")

    manifest_sha = hashlib.sha256()
    total = {"files": 0, "directories": 0, "symlinks": 0, "other": 0,
             "logical_bytes": 0, "allocated_file_bytes": 0, "allocated_all_path_bytes": 0,
             "paths": 0, "nonconda_paths": 0, "conda_paths": 0}
    YI_DIR.mkdir(parents=True, exist_ok=True)
    with COMPOSITE_OUTPUT.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", compresslevel=6, mtime=0) as gz:
            def emit(path: str, item: dict, region: str) -> None:
                if path != YI_ROOT and not path.startswith(YI_ROOT + "/"):
                    raise RuntimeError("path escaped exact yi scope: " + path)
                entry_type = item.get("type")
                if entry_type not in ("file", "directory", "symlink", "special"):
                    raise RuntimeError("unsupported inventory type at " + path)
                row = {
                    "path": path,
                    "type": entry_type,
                    "uid": item.get("uid"),
                    "gid": item.get("gid"),
                    "owner": item.get("owner"),
                    "group": item.get("group"),
                    "mode_octal": item.get("mode_octal"),
                    "size_bytes": int(item.get("size_bytes") or 0),
                    "allocated_bytes": int(item.get("allocated_bytes") or 0),
                    "mtime_ns": item.get("mtime_ns"),
                    "device": item.get("device"),
                    "inode": item.get("inode"),
                    "nlink": item.get("nlink"),
                    "symlink_target": item.get("symlink_target"),
                    "region": region,
                }
                line = (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                gz.write(line)
                manifest_sha.update(line)
                total["paths"] += 1
                total[region + "_paths"] += 1
                total[{"file": "files", "directory": "directories", "symlink": "symlinks", "special": "other"}[entry_type]] += 1
                total["allocated_all_path_bytes"] += row["allocated_bytes"]
                if entry_type == "file":
                    total["logical_bytes"] += row["size_bytes"]
                    total["allocated_file_bytes"] += row["allocated_bytes"]

            for absolute, item in sorted(final_conda.items()):
                emit(absolute, item, "conda")

            seen_nonconda = set()
            with gzip.open(R04_MANIFEST, "rt", encoding="utf-8") as stream:
                for line in stream:
                    item = json.loads(line)
                    absolute = item["path"]
                    if absolute == YI_ROOT + "/.conda" or absolute.startswith(YI_ROOT + "/.conda/"):
                        raise RuntimeError("R04 non-Conda manifest includes excluded .conda path")
                    if absolute in seen_nonconda:
                        raise RuntimeError("duplicate R04 non-Conda path: " + absolute)
                    seen_nonconda.add(absolute)
                    emit(absolute, item, "nonconda")
            if len(seen_nonconda) != int(nonconda["entry_count"]):
                raise RuntimeError("R04 non-Conda manifest entry count differs from reconciled report")
    if total["paths"] != int(r04_report["current_union"]["paths"]):
        raise RuntimeError("composite manifest path count differs from R04/R03 union")
    current_conda = r03_report["current_summary"]
    expected_counts = {
        "files": int(nonconda["counts"]["file"]) + int(current_conda["regular_files"]),
        "directories": int(nonconda["counts"]["directory"]) + int(current_conda["directories"]),
        "symlinks": int(nonconda["counts"]["symlink"]) + int(current_conda["symlinks"]),
    }
    if any(total[key] != value for key, value in expected_counts.items()):
        raise RuntimeError(f"composite manifest type counts differ from source summaries: {total} != {expected_counts}")
    if total["allocated_all_path_bytes"] != int(r04_report["current_union"]["allocated_all_path_bytes"]):
        raise RuntimeError("composite allocated path total differs from R04/R03 reconciliation")
    return digest(COMPOSITE_OUTPUT), {"uncompressed_manifest_sha256": manifest_sha.hexdigest(),
                                     "counts": total, "r04_status": r04_status,
                                     "r04_report": r04_report, "r03_report": r03_report}


def register_source(con: sqlite3.Connection, path: Path, key_prefix: str,
                    kind: str, captured: str, as_of: str, note: str) -> int:
    path = path.resolve(strict=True)
    sha = digest(path)
    key = f"{key_prefix}:{sha[:16]}"
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if prior and tuple(prior) != (str(path), sha):
        raise RuntimeError("source key already identifies different bytes: " + key)
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (key, str(path), kind, sha, captured, as_of,
         "Exact local evidence file; SHA-256 identifies these bytes.", note),
    )
    return con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0]


def main() -> None:
    manifest_sha, evidence = make_composite_manifest()
    r04 = evidence["r04_report"]
    r03 = evidence["r03_report"]
    finish = iso(evidence["r04_status"]["finished_local_utc"])
    started = "2026-11-23T01:23:18Z"  # R02 Conda refresh began before R03 completion.
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    size = COMPOSITE_OUTPUT.stat().st_size

    con = sqlite3.connect(DB, timeout=600)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=600000")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute("SELECT scoped_path,filesystem_name,mount_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
        if not root or tuple(root) != (ROOT_PATH, "data", "/data"):
            raise RuntimeError("expected /data/pclancy3 root is missing or changed")
        prior = con.execute("SELECT root_id,manifest_sha256 FROM inventory_snapshot WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()
        if prior:
            if prior[0] != ROOT_ID or prior[1] != manifest_sha:
                raise RuntimeError("composite snapshot ID maps to different evidence")
            existing = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()[0]
            con.commit()
            print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "ALREADY_IMPORTED",
                              "observations": existing, "manifest_sha256": manifest_sha}, sort_keys=True))
            return

        source_paths = [
            (COMPOSITE_OUTPUT, "rockfish_yi_composite_manifest_r04", "ROCKFISH_YI_COMPOSITE_METADATA_JSONL_GZIP", finish,
             "Composite path manifest: complete R04 non-Conda scan joined to R03 reconciled .conda coverage."),
            (R04_MANIFEST, "rockfish_yi_nonconda_manifest_r04", "ROCKFISH_YI_R04_NONCONDA_METADATA_JSONL_GZIP", finish,
             "Current complete metadata-only manifest for exact /data/pclancy3/yi outside .conda."),
            (R04_RAW_LEDGER, "rockfish_yi_nonconda_raw_ledger_r04", "ROCKFISH_YI_R04_NONCONDA_RAW_JSONL_GZIP", finish,
             "Raw completed R04 no-follow scan stream for /data/pclancy3/yi outside .conda."),
            (R04_STATUS, "rockfish_yi_nonconda_status_r04", "ROCKFISH_YI_R04_SCAN_STATUS_JSON", finish,
             "R04 non-Conda scan completion status; zero traversal errors."),
            (R04_RECON, "rockfish_yi_reconciliation_r04", "ROCKFISH_YI_R04_RECONCILIATION_JSON", finish,
             "Exact R01/R04 non-Conda diff and R03 .conda path-set reconciliation summary."),
            (R03_RECON, "rockfish_yi_conda_reconciliation_r03", "ROCKFISH_YI_R03_CONDA_RECONCILIATION_JSON", finish,
             "Complete exact .conda path-set reconciliation against R01, R02/R03 and cleanup receipts."),
            (R01_BASELINE, "rockfish_data_yi_baseline_r01", "ROCKFISH_YI_R01_BASELINE_JSONL_GZIP", "2026-11-22T18:33:34Z",
             "Full prior baseline used for .conda path reconciliation; never treated as current on its own."),
            (R02_PARTIAL, "rockfish_data_yi_partial_r02", "ROCKFISH_YI_R02_PARTIAL_JSONL_GZIP", "2026-11-23T01:23:18Z",
             "Partial refresh contributes positive .conda observations only; absence is not used as deletion evidence."),
        ]
        source_ids = []
        for path, prefix, kind, as_of, note in source_paths:
            source_ids.append(register_source(con, path, prefix, kind, captured, as_of, note))
        for path in sorted(R03_DIR.glob("conda_env_*_r03.status.json")):
            source_ids.append(register_source(con, path, "rockfish_yi_conda_r03_status:" + path.stem,
                "ROCKFISH_YI_R03_CONDA_ENV_STATUS_JSON", captured, finish,
                "One of four complete per-environment metadata scans completing the .conda path union."))
        for path in sorted(R03_DIR.glob("conda_env_*_r03.jsonl.gz")):
            source_ids.append(register_source(con, path, "rockfish_yi_conda_r03_ledger:" + path.stem,
                "ROCKFISH_YI_R03_CONDA_ENV_JSONL_GZIP", captured, finish,
                "Raw metadata-only scan for one of four previously missing .conda environment roots."))
        for path in (AUDIT / "phase2_data/conda_index_cache_removed_manifest_20261122.tsv",
                     AUDIT / "phase2_data/conda_index_cache_removed_verification_20261122.txt",
                     AUDIT / "phase2_data/conda_package_cleanup_execution_receipt_20261122.json"):
            source_ids.append(register_source(con, path, "rockfish_yi_conda_cleanup_receipt:" + path.name,
                "ROCKFISH_YI_CONDA_CLEANUP_RECEIPT", captured, "2026-11-22T00:00:00Z",
                "Exact path-level cleanup receipt used to reconcile removed .conda paths."))

        # Persist a staging table so the large observation import uses bounded memory.
        con.execute("DROP TABLE IF EXISTS temp.yi_inventory_stage")
        con.execute("""CREATE TEMP TABLE yi_inventory_stage (
            relative_path TEXT PRIMARY KEY,
            entry_type TEXT NOT NULL,
            uid INTEGER, gid INTEGER, mode_octal TEXT,
            logical_bytes INTEGER, allocated_bytes INTEGER, mtime_ns INTEGER,
            inode TEXT, device TEXT, symlink_target TEXT, link_count INTEGER,
            observation_status TEXT NOT NULL
        ) WITHOUT ROWID""")
        batch = []
        with gzip.open(COMPOSITE_OUTPUT, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                absolute = PurePosixPath(row["path"])
                try:
                    rel = str(absolute.relative_to(PurePosixPath(ROOT_PATH)))
                except ValueError as exc:
                    raise RuntimeError("composite path escaped /data/pclancy3: " + row["path"]) from exc
                kind = {"file": "FILE", "directory": "DIRECTORY", "symlink": "SYMLINK", "special": "OTHER"}[row["type"]]
                batch.append((rel, kind, row.get("uid"), row.get("gid"), row.get("mode_octal"),
                              row["size_bytes"] if kind == "FILE" else None,
                              row["allocated_bytes"], row.get("mtime_ns"),
                              str(row["inode"]) if row.get("inode") is not None else None,
                              str(row["device"]) if row.get("device") is not None else None,
                              row.get("symlink_target"),
                              int(row["nlink"]) if row.get("nlink") is not None else None,
                              "PRESENT"))
                if len(batch) >= 10000:
                    con.executemany("INSERT INTO yi_inventory_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            con.executemany("INSERT INTO yi_inventory_stage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        counts = dict(con.execute("""SELECT entry_type,COUNT(*) FROM yi_inventory_stage GROUP BY entry_type""").fetchall())
        actual_paths = con.execute("SELECT COUNT(*) FROM yi_inventory_stage").fetchone()[0]
        if actual_paths != int(r04["current_union"]["paths"]):
            raise RuntimeError(f"staged path count mismatch: {actual_paths}")
        logical = con.execute("SELECT COALESCE(SUM(logical_bytes),0) FROM yi_inventory_stage WHERE entry_type='FILE'").fetchone()[0]
        allocated = con.execute("SELECT COALESCE(SUM(allocated_bytes),0) FROM yi_inventory_stage WHERE entry_type='FILE'").fetchone()[0]
        if logical != int(r04["current_union"]["regular_file_logical_bytes"]):
            raise RuntimeError("staged logical-byte total differs from R04/R03 reconciliation")
        if allocated != int(r04["current_union"]["regular_file_allocated_bytes"]):
            raise RuntimeError("staged regular-file allocation differs from R04/R03 reconciliation")

        note = (
            "Complete path coverage for /data/pclancy3/yi as a composite, non-atomic inventory. "
            "R04 non-.conda metadata was freshly scanned with zero errors; .conda uses complete R01/R02/R03 "
            "path-set reconciliation and exact cleanup receipts, not a fresh full restat. .conda owner/UID/GID "
            "freshness is partial where only R01 fields existed. No payloads were read or hashed. "
            "Quota counters are not localized by this snapshot."
        )
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES(?,?,?,?,?,'r04-composite-v1','COMPLETE',?, ?, ?,0,?,?,?,?,?)""",
            (SNAPSHOT_ID, ROOT_ID, started, finish,
             "Composite exact union of current R04 non-Conda metadata and reconciled R03 .conda path set; no-follow, no payload access.",
             counts.get("FILE", 0), counts.get("DIRECTORY", 0), counts.get("SYMLINK", 0),
             logical, allocated, str(COMPOSITE_OUTPUT.resolve()), manifest_sha, note),
        )
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) SELECT ?,relative_path,? FROM yi_inventory_stage",
                    (ROOT_ID, finish))
        con.execute(
            """INSERT INTO inventory_observation
               (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,
                mtime_ns,inode,device,symlink_target,hash_scope,observation_status,link_count)
               SELECT ?,p.path_id,s.entry_type,s.uid,s.gid,s.mode_octal,s.logical_bytes,s.allocated_bytes,
                      s.mtime_ns,s.inode,s.device,s.symlink_target,'NONE',s.observation_status,s.link_count
               FROM yi_inventory_stage AS s
               JOIN storage_path AS p ON p.root_id=? AND p.relative_path=s.relative_path""",
            (SNAPSHOT_ID, ROOT_ID),
        )
        composite_id = con.execute("SELECT source_id FROM registry_source WHERE source_path=? AND sha256=? ORDER BY source_id DESC LIMIT 1",
                                   (str(COMPOSITE_OUTPUT.resolve()), digest(COMPOSITE_OUTPUT))).fetchone()[0]
        source_ids.append(composite_id)
        for source_id in sorted(set(source_ids)):
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'inventory_source',?,'INVENTORY_SNAPSHOT','' || ?,'CONTRIBUTES_TO_COMPOSITE',?)""",
                (source_id, str(source_id), SNAPSHOT_ID,
                 "Evidence input to the complete path-coverage composite; source-specific freshness is stated in its notes."),
            )
        stored = con.execute("SELECT COUNT(*) FROM inventory_observation WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone()[0]
        if stored != actual_paths:
            raise RuntimeError(f"stored observations differ from staged count: {stored} != {actual_paths}")
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign-key violations while importing yi composite: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"snapshot_id": SNAPSHOT_ID, "status": "IMPORTED_COMPLETE_PATH_COVERAGE",
                      "paths": actual_paths, "files": counts.get("FILE", 0),
                      "directories": counts.get("DIRECTORY", 0), "symlinks": counts.get("SYMLINK", 0),
                      "logical_bytes": logical, "allocated_file_bytes": allocated,
                      "manifest_path": str(COMPOSITE_OUTPUT), "manifest_bytes": size,
                      "manifest_sha256": manifest_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
