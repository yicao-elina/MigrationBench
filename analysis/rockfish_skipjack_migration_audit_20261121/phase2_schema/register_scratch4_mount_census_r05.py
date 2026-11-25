#!/usr/bin/env python3
"""Register aggregate R05/R06 evidence for accessible outer /scratch4 roots.

The scans retain per-root completion receipts and selected large-file/name
observations, but not a complete per-path ledger. This importer therefore
links evidence to the mount root without misrepresenting it as a complete
inventory_snapshot.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
SCRATCH = AUDIT / "phase2_scratch4"
ROOT_ID = "RF-SCRATCH4-FS"
R05 = SCRATCH / "scratch4_accessible_roots_r05_20261123"
UID = SCRATCH / "scratch4_uid3901_outer_roots_r06_20261123"
ROOT_PROBE_DIR = SCRATCH / "scratch4_mount_root_20261123_r05"
COMBINED = SCRATCH / "scratch4_accessible_roots_r05_r06_20261123"
ROOTS_TOTAL = 58


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def register_source(con: sqlite3.Connection, path: Path, key: str, kind: str,
                    captured: str, note: str) -> int:
    path = path.resolve(strict=True)
    digest = sha256(path)
    source_key = f"{key}:{digest[:16]}"
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
    if prior and tuple(prior) != (str(path), digest):
        raise RuntimeError("registry source collision: " + source_key)
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (source_key, str(path), kind, digest, captured, captured,
         "Exact local evidence bytes identified by SHA-256; remote file payloads were not read.", note),
    )
    return con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]


def main() -> None:
    uid_summary_path = UID / "uid3901_roots_summary_r06.json"
    orchestrated_path = R05 / "one_root_batches/orchestrated_roots_summary_r05.json"
    uid_summary = json.loads(uid_summary_path.read_text(encoding="utf-8"))
    orchestrated = json.loads(orchestrated_path.read_text(encoding="utf-8"))
    if int(uid_summary.get("roots_total", -1)) != ROOTS_TOTAL:
        raise RuntimeError("UID target-root scan does not cover all 58 roots")
    attempts = uid_summary.get("completed_attempts", [])
    if len(attempts) != ROOTS_TOTAL:
        raise RuntimeError(f"UID scan incomplete: {len(attempts)}/{ROOTS_TOTAL}")
    if sum(int(row.get("uid3901_match_count", 0)) for row in attempts) != 0:
        raise RuntimeError("UID 3901 matches need separate path-level review; refusing aggregate-only registration")
    if len(orchestrated.get("attempted_roots", [])) != 38:
        raise RuntimeError("recursive R05 scan summary does not include all 38 non-empty roots")
    if uid_summary.get("target_uid") != 3901:
        raise RuntimeError("unexpected UID verification target")

    root_probe = ROOT_PROBE_DIR / "scratch4_mount_root_children_20261123_r05.jsonl"
    root_status = ROOT_PROBE_DIR / "scratch4_mount_root_probe_20261123_r05.status.json"
    review = SCRATCH / "SCRATCH4_MOUNT_ROOT_R05_REVIEW_20261123.md"
    scope_report = SCRATCH / "SCRATCH4_PCLANCY3_SCOPE_RECONCILIATION_20261123.md"
    scanner = SCRATCH / "inventory_accessible_scratch4_roots_r05.py"
    orchestrator_script = SCRATCH / "orchestrate_accessible_scratch4_root_batches_r05.py"
    uid_script = SCRATCH / "check_scratch4_uid3901_r06.py"
    combined_report = COMBINED / "SCRATCH4_ACCESSIBLE_ROOTS_R05_R06_REPORT_20261123.md"
    combined_tsv = COMBINED / "scratch4_accessible_roots_r05_r06_summary.tsv"
    combined_json = COMBINED / "scratch4_accessible_roots_r05_r06_summary.json"
    core = [root_probe, root_status, review, scope_report, scanner, orchestrator_script,
            uid_summary_path, uid_script, orchestrated_path,
            R05 / "accessible_roots_metadata_r05.jsonl",
            R05 / "accessible_roots_metadata_r05.status.json"]
    for path in core:
        if not path.exists():
            if path == uid_script:
                continue
            raise FileNotFoundError(path)
    for item in sorted(R05.rglob("accessible_roots_metadata_r05.jsonl")) + sorted(R05.rglob("accessible_roots_metadata_r05.status.json")):
        if item.exists():
            core.append(item)
    for directory in sorted(UID.glob("*_r06")):
        for name in ("receipt.json", "uid3901_paths.txt", "find_stderr.txt"):
            item = directory / name
            if item.exists():
                core.append(item)
    for path in (combined_report, combined_tsv, combined_json):
        if path.exists():
            core.append(path)

    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=600)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=600000")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute("SELECT scoped_path,filesystem_name,mount_path FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
        if root != ("/scratch4", "scratch4", "/scratch4"):
            raise RuntimeError("expected /scratch4 storage root is missing or changed")
        for index, path in enumerate(dict.fromkeys(core)):
            if path == review:
                kind = "SCRATCH4_MOUNT_ROOT_CENSUS_REPORT"
            elif path == scope_report:
                kind = "SCRATCH4_PROJECT_SCOPE_RECONCILIATION"
            elif path == combined_report:
                kind = "SCRATCH4_OUTER_ROOT_CENSUS_REPORT"
            elif path == combined_tsv:
                kind = "SCRATCH4_OUTER_ROOT_CENSUS_MATRIX"
            elif path == combined_json:
                kind = "SCRATCH4_OUTER_ROOT_CENSUS_MACHINE_SUMMARY"
            elif path == uid_summary_path:
                kind = "SCRATCH4_UID3901_ROOT_SCAN_SUMMARY"
            elif path.name == "receipt.json":
                kind = "SCRATCH4_UID3901_ROOT_SCAN_RECEIPT"
            elif path.name == "uid3901_paths.txt":
                kind = "SCRATCH4_UID3901_ROOT_MATCH_OUTPUT"
            elif path.name == "find_stderr.txt":
                kind = "SCRATCH4_UID3901_ROOT_SCAN_STDERR"
            elif path.name == "orchestrated_roots_summary_r05.json":
                kind = "SCRATCH4_MOUNT_ROOT_SCAN_SUMMARY"
            elif path.name == "accessible_roots_metadata_r05.jsonl":
                kind = "SCRATCH4_MOUNT_ROOT_METADATA_LEDGER"
            elif path.name == "accessible_roots_metadata_r05.status.json":
                kind = "SCRATCH4_MOUNT_ROOT_SCAN_STATUS"
            elif path == root_probe:
                kind = "SCRATCH4_MOUNT_ROOT_ACCESS_CAPTURE"
            elif path == root_status:
                kind = "SCRATCH4_MOUNT_ROOT_ACCESS_STATUS"
            else:
                kind = "SCRATCH4_MOUNT_ROOT_SCAN_METHOD"
            source_id = register_source(con, path, "rockfish_scratch4_outer_r05:" + path.stem + f":{index}",
                kind, captured,
                "Bounded metadata-only R05/R06 evidence for accessible other-owner /scratch4 roots; incomplete roots remain partial, no payload read/hash, and no cleanup decision follows from the scan.")
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'storage_scope',?,'STORAGE_ROOT',?,'OBSERVES_SCOPE',?)""",
                (source_id, ROOT_ID, ROOT_ID, "Aggregate accessible-root evidence linked to /scratch4; it is not a complete per-path inventory snapshot."),
            )
        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("foreign-key check failed: " + repr(violations[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"status": "REGISTERED_AGGREGATE_EVIDENCE_ONLY", "root_id": ROOT_ID,
        "uid_roots": len(attempts), "uid_match_count": 0,
        "uid_status_counts": {state: sum(row.get("status") == state for row in attempts)
                              for state in sorted({row.get("status") for row in attempts})},
        "recursive_nonempty_roots": len(orchestrated["attempted_roots"]),
        "registered_source_files": len(set(core))}, sort_keys=True))


if __name__ == "__main__":
    main()
