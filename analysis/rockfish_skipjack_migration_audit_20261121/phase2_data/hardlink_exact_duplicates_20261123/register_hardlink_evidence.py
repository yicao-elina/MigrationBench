#!/usr/bin/env python3
"""Register hardlink preflight/execution captures against duplicate groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parents[2]
DB = AUDIT / "provenance_catalog.sqlite"
ROOT_ID = "RF-DATA-YI"
ROOT_PATH = "/data/pclancy3/yi"
OPERATION_SCRIPT = Path(__file__).resolve().with_name("guarded_hardlink_duplicates.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", required=True, type=Path)
    p.add_argument("--execution", type=Path)
    p.add_argument("--apply", action="store_true", help="write verified evidence into the catalog")
    args = p.parse_args()
    preflight_path = args.preflight.resolve(strict=True)
    execution_path = args.execution.resolve(strict=True) if args.execution else None
    pre = json.loads(preflight_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path else None
    if pre.get("mode") != "preflight" or len(pre.get("actions", [])) != 5:
        raise SystemExit("expected five-item read-only hardlink preflight")
    if any(a.get("state") != "PREFLIGHT_PASSED" or a.get("preflight_problems") for a in pre["actions"]):
        raise SystemExit("preflight did not pass for every proposed pair")
    if any(a.get("queue_overlaps") for a in pre["actions"]):
        raise SystemExit("preflight has active/unknown queue overlap")
    if execution:
        if execution.get("mode") != "execute" or len(execution.get("actions", [])) != 5:
            raise SystemExit("execution receipt does not cover all five requested pairs")
        if any(a.get("state") != "HARDLINKED_VERIFIED" for a in execution["actions"]):
            raise SystemExit("execution receipt contains a pair without post-link verification")

    expected = {a["group"]: a for a in pre["actions"]}
    if execution and {a["group"] for a in execution["actions"]} != set(expected):
        raise SystemExit("execution receipt group set differs from preflight")
    if not args.apply:
        print(json.dumps({"state": "PREFLIGHT_ONLY", "groups": sorted(expected),
                          "preflight_sha256": sha(preflight_path),
                          "execution_sha256": sha(execution_path) if execution_path else None}, sort_keys=True))
        return

    con = sqlite3.connect(DB, timeout=120)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        group_ids: dict[str, int] = {}
        for label, record in expected.items():
            path = record["pre_link"]["a"]["path"]
            if not path.startswith(ROOT_PATH + "/"):
                raise ValueError("candidate path outside expected user root: " + path)
            rel = path[len(ROOT_PATH) + 1:]
            rows = con.execute(
                """SELECT DISTINCT a.artifact_id,a.sha256,a.byte_size,dg.duplicate_group_id
                   FROM storage_path sp JOIN artifact_location al USING(path_id)
                   JOIN artifact a USING(artifact_id) JOIN duplicate_member dm USING(artifact_id)
                   JOIN duplicate_group dg USING(duplicate_group_id)
                   WHERE sp.root_id=? AND sp.relative_path=?""",
                (ROOT_ID, rel),
            ).fetchall()
            ids = {int(r["duplicate_group_id"]) for r in rows}
            if len(ids) != 1:
                raise ValueError(f"{label} path does not map to exactly one duplicate group: {ids}")
            group_id = ids.pop()
            if len(rows) != 1 or rows[0]["sha256"] != record["pre_link"]["a"]["sha256"] or rows[0]["byte_size"] != record["pre_link"]["a"]["stat"]["size_bytes"]:
                raise ValueError(f"{label} canonical artifact differs from pre-link path/hash/size")
            members = con.execute(
                """SELECT count(*) FROM duplicate_member dm JOIN artifact_location al USING(artifact_id)
                   WHERE dm.duplicate_group_id=? AND al.availability='PRESENT'""",
                (group_id,),
            ).fetchone()[0]
            if members != 2:
                raise ValueError(f"{label} duplicate group does not have two present members")
            group_ids[label] = group_id

        def register(path: Path, capture: dict, kind: str, prefix: str, note: str) -> int:
            digest = sha(path)
            key = prefix + ":" + digest[:20]
            captured = capture.get("captured_utc") or now_utc()
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (key, str(path), kind, digest, captured, captured,
                 "User-authorized exact duplicate resolution; Rockfish full-content hashes and per-path stat evidence captured before any link replacement.",
                 note),
            )
            row = con.execute("SELECT source_id,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
            if not row or row["sha256"] != digest:
                raise ValueError("registered registry-source hash mismatch: " + key)
            return int(row["source_id"])

        pre_source = register(
            preflight_path, pre, "DUPLICATE_HARDLINK_PREFLIGHT_CAPTURE_JSON",
            "duplicate_hardlink_preflight",
            "Five exact-hash pairs; complete per-path pre-link stat, ACL and xattr records; all-user queue snapshot before each candidate; quota-before capture. Historical mtime/ctime differences are retained here.",
        )
        execution_source = None
        if execution and execution_path:
            execution_source = register(
                execution_path, execution, "DUPLICATE_HARDLINK_EXECUTION_RECEIPT_JSON",
                "duplicate_hardlink_execution",
                "Atomic same-filesystem path-preserving hardlink operations with fresh all-user queue/stat/hash guards and post-link inode/nlink/hash verification; records quota-after capture.",
            )
        operation_digest = sha(OPERATION_SCRIPT)
        operation_key = "duplicate_hardlink_guarded_script:" + operation_digest[:20]
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
               VALUES(?,?,'HARDLINK_DEDUP_GUARDED_SCRIPT',?,?,?,?,?,?)""",
            (operation_key, str(OPERATION_SCRIPT), operation_digest, now_utc(), pre.get("captured_utc") or now_utc(), 1,
             "Method source for the user-authorized exact duplicate hardlink operation.",
             "Read-only preflight by default; explicit --execute; per-pair all-user queue, exact hash and inode identity, ACL/xattr/stat guards; atomic temporary hardlink and replace; post-link hash/inode/nlink verification."),
        )
        script_row = con.execute("SELECT source_id,sha256 FROM registry_source WHERE source_key=?", (operation_key,)).fetchone()
        if not script_row or script_row["sha256"] != operation_digest:
            raise ValueError("operation script registry-source hash mismatch")
        script_source = int(script_row["source_id"])
        for label, group_id in group_ids.items():
            con.execute(
                """INSERT OR IGNORE INTO duplicate_group_evidence
                   (duplicate_group_id,source_id,evidence_role,notes) VALUES(?,?,'INODE_EVIDENCE',?)""",
                (group_id, pre_source, "Pre-link distinct inode identity and path-specific timestamps recorded before any action."),
            )
            con.execute(
                """INSERT OR IGNORE INTO duplicate_group_evidence
                   (duplicate_group_id,source_id,evidence_role,notes) VALUES(?,?,'REVIEW',?)""",
                (group_id, script_source, "Guarded hardlink method and exact path allowlist used for this resolution."),
            )
            if execution_source:
                con.execute(
                    """INSERT OR IGNORE INTO duplicate_group_evidence
                       (duplicate_group_id,source_id,evidence_role,notes) VALUES(?,?,'INODE_EVIDENCE',?)""",
                    (group_id, execution_source, "Post-link common inode, nlink=2, exact SHA-256, and quota bracket recorded; both pathnames retained."),
                )
                current = con.execute("SELECT rationale FROM duplicate_group WHERE duplicate_group_id=?", (group_id,)).fetchone()
                note = "Verified hardlink resolution: both path names retained on one inode; pre-link mtime/ctime differences remain preserved as registry evidence."
                rationale = current[0] or ""
                if note not in rationale:
                    rationale = (rationale + " " + note).strip()
                con.execute("UPDATE duplicate_group SET disposition='KEEP_BOTH',rationale=? WHERE duplicate_group_id=?", (rationale, group_id))

        fk_errors = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk_errors:
            raise ValueError("foreign-key errors during hardlink evidence registration")
        con.commit()
        print(json.dumps({"state": "REGISTERED", "groups": group_ids, "preflight_source_id": pre_source,
                          "execution_source_id": execution_source, "foreign_key_errors": 0}, sort_keys=True))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
