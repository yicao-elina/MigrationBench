#!/usr/bin/env python3
"""Reconcile former-collaborator duplicate records after verified deletion.

This updates SQL notes only. It never accesses Rockfish or deletes files.
Each candidate group must match a complete immutable receipt, a completed
cleanup batch, and a complete current handoff scan before its status is
rewritten.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
ADDENDUM = AUDIT / "phase2_data/former_collab_cleanup_execution_addendum_20261123.md"
VICTOR_RECEIPT = AUDIT / "phase2_data/victor_duplicate_delete_receipt_20261122.json"
YUQI_RECEIPT = AUDIT / "phase2_data/yuqi_pseudopotential_delete_receipt_20261122.jsonl"
HANDOFF = AUDIT / "phase2_data/yi_refresh_20261123/handoff_roots_20261123.jsonl"
EXPECTED = {
    "/data/pclancy3/for-yuqi/pseudo/": "RF-DATA-FOR-YUQI-PSEUDO-DUPLICATES-20261122",
    "/data/pclancy3/for-victor/formation_energy/": "RF-DATA-FOR-VICTOR-FE-DUPLICATES-20261122",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_and_validate_receipts() -> dict[str, dict]:
    victor = json.loads(VICTOR_RECEIPT.read_text())
    victor_rows = victor.get("deleted", [])
    if (victor.get("status") != "DELETED_AND_VERIFIED" or
            victor.get("queue_status") != "CHECKED" or
            victor.get("queue_overlap_count") != 0 or
            victor.get("verified_count") != 9 or victor.get("errors") or
            len(victor_rows) != 9 or
            not all(row.get("unlinked") is True and row.get("sha256") for row in victor_rows)):
        raise RuntimeError("Victor receipt is not a complete zero-overlap verified cleanup")
    if (sum(int(row["logical_bytes"]) for row in victor_rows) != victor["verified_logical_bytes"] or
            sum(int(row["allocated_bytes"]) for row in victor_rows) != victor["verified_allocated_bytes"]):
        raise RuntimeError("Victor receipt totals do not reconcile")

    yuqi_rows = [json.loads(line) for line in YUQI_RECEIPT.read_text().splitlines() if line.strip()]
    preflight = [row for row in yuqi_rows if row.get("event") == "preflight_passed"]
    deleted = [row for row in yuqi_rows if row.get("event") == "deleted_verified"]
    complete = [row for row in yuqi_rows if row.get("event") == "complete"]
    if (len(preflight) != 1 or preflight[0].get("execute") is not True or
            preflight[0].get("queue_overlaps") != 0 or len(deleted) != 65 or
            len(complete) != 1 or complete[0].get("file_count") != 65 or
            not all(row.get("verified_missing") is True and row.get("sha256") for row in deleted)):
        raise RuntimeError("Yuqi receipt is not a complete zero-overlap verified cleanup")
    yuqi_logical = sum(int(row["size_bytes"]) for row in deleted)
    yuqi_allocated = sum(int(row["allocated_bytes"]) for row in deleted)
    if (yuqi_logical != complete[0].get("logical_bytes") or
            yuqi_allocated != complete[0].get("allocated_bytes")):
        raise RuntimeError("Yuqi receipt totals do not reconcile")

    handoff_roots: dict[str, list[dict]] = {"/data/pclancy3/for-yuqi": [],
                                             "/data/pclancy3/for-victor": []}
    completed: set[str] = set()
    for line in HANDOFF.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        root = row.get("root")
        if row.get("record") == "entry" and root in handoff_roots:
            handoff_roots[root].append(row)
        elif row.get("record") == "root_complete" and root in handoff_roots:
            if int(row.get("error_count", -1)) != 0:
                raise RuntimeError("handoff root is not error-free: " + root)
            completed.add(root)
    if set(handoff_roots) != completed:
        raise RuntimeError("latest handoff refresh lacks complete markers for both roots")
    expected_victor = {
        ("/data/pclancy3/for-victor/formation_energy", "dir", 4096),
        ("/data/pclancy3/for-victor/formation_energy/Ef_new.py", "file", 2331),
        ("/data/pclancy3/for-victor/cyqe", "symlink", 22),
    }
    seen_victor = {(row["path"], row["type"], int(row["size"]) if row.get("size") is not None else None)
                   for row in handoff_roots["/data/pclancy3/for-victor"]}
    expected_yuqi = {("/data/pclancy3/for-yuqi/pseudo", "dir", 8192)}
    seen_yuqi = {(row["path"], row["type"], int(row["size"]) if row.get("size") is not None else None)
                 for row in handoff_roots["/data/pclancy3/for-yuqi"]}
    if seen_victor != expected_victor or seen_yuqi != expected_yuqi:
        raise RuntimeError("current handoff roots differ from the verified retained path sets")

    return {
        "RF-DATA-FOR-YUQI-PSEUDO-DUPLICATES-20261122": {
            "receipt": YUQI_RECEIPT, "receipt_sha256": digest(YUQI_RECEIPT),
            "count": len(deleted), "logical": yuqi_logical, "allocated": yuqi_allocated,
            "timestamp": complete[0].get("utc"),
        },
        "RF-DATA-FOR-VICTOR-FE-DUPLICATES-20261122": {
            "receipt": VICTOR_RECEIPT, "receipt_sha256": digest(VICTOR_RECEIPT),
            "count": len(victor_rows), "logical": victor["verified_logical_bytes"],
            "allocated": victor["verified_allocated_bytes"], "timestamp": victor.get("timestamp_utc"),
        },
    }


def main() -> None:
    evidence = read_and_validate_receipts()
    addendum = ADDENDUM.resolve(strict=True)
    addendum_source_key = "audit_deliverable:" + addendum.relative_to(AUDIT.parents[1]).as_posix() + ":" + digest(addendum)[:16]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=180)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    changes: dict[str, int] = {}
    try:
        con.execute("BEGIN IMMEDIATE")
        source_row = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (addendum_source_key,)).fetchone()
        if not source_row:
            raise RuntimeError("execution addendum must be registered before SQL outcome reconciliation")
        addendum_source_id = source_row[0]
        for batch_id, info in evidence.items():
            batch = con.execute("SELECT status,actual_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (batch_id,)).fetchone()
            if not batch or batch[0] != "COMPLETED" or batch[1] != info["logical"]:
                raise RuntimeError("completed cleanup batch/totals differ from immutable receipt: " + batch_id)
            item_rows = con.execute(
                """SELECT ci.artifact_id,ci.deletion_state,ci.disposition,ci.pre_delete_sha256,
                          ci.pre_delete_size_bytes,ci.notes,a.sha256,a.byte_size
                   FROM cleanup_item ci JOIN artifact a USING(artifact_id)
                   WHERE ci.cleanup_batch_id=?""", (batch_id,)
            ).fetchall()
            if (len(item_rows) != info["count"] or
                    any(row[1] != "VERIFIED" or row[2] != "DELETE" or row[3] != row[6] or row[4] != row[7]
                        for row in item_rows)):
                raise RuntimeError("SQL cleanup items differ from immutable receipt: " + batch_id)
            if sum(int(row[4] or 0) for row in item_rows) != info["logical"]:
                raise RuntimeError("SQL cleanup-item logical total differs from immutable receipt: " + batch_id)
            receipt = info["receipt"].resolve(strict=True)
            receipt_rel = receipt.relative_to(AUDIT.parents[1]).as_posix()
            for group_id, group_rationale in con.execute(
                "SELECT duplicate_group_id,rationale FROM duplicate_group WHERE disposition='DELETE_COPY'"
            ).fetchall():
                members = con.execute(
                    """SELECT dm.canonical_member,a.artifact_id,a.artifact_key,a.sha256,a.byte_size,
                              al.availability,sr.scoped_path,sp.relative_path,ci.cleanup_batch_id,
                              ci.deletion_state,ci.disposition
                       FROM duplicate_member dm JOIN artifact a USING(artifact_id)
                       LEFT JOIN artifact_location al USING(artifact_id)
                       LEFT JOIN storage_path sp USING(path_id)
                       LEFT JOIN storage_root sr USING(root_id)
                       LEFT JOIN cleanup_item ci USING(artifact_id)
                       WHERE dm.duplicate_group_id=?""", (group_id,)
                ).fetchall()
                if len(members) != 2:
                    continue
                canonical = next((row for row in members if row[0] == 1), None)
                copy = next((row for row in members if row[0] == 0), None)
                if not canonical or not copy:
                    continue
                copy_path = str(Path(copy[6]) / copy[7]) if copy[7] else str(copy[6])
                expected_batch = next((bid for prefix, bid in EXPECTED.items() if copy_path.startswith(prefix)), None)
                if expected_batch != batch_id:
                    continue
                if (canonical[3] is None or canonical[3] != copy[3] or canonical[4] != copy[4] or
                        canonical[5] != "PRESENT" or copy[5] != "REMOVED" or
                        copy[8] != batch_id or copy[9] != "VERIFIED" or copy[10] != "DELETE"):
                    raise RuntimeError("duplicate pair state does not match verified cleanup: " + str(group_id))
                new_rationale = (
                    "Verified duplicate copy removed under " + batch_id + "; canonical path retained at " +
                    (str(Path(canonical[6]) / canonical[7]) if canonical[7] else str(canonical[6])) +
                    "; both retained catalog SHA-256 values match (" + canonical[3] +"). " +
                    "The exact cleanup receipt records the copy missing after deletion; the execution addendum " +
                    "also confirms the zero-overlap queue preflight and current handoff scan."
                )
                con.execute("UPDATE duplicate_group SET rationale=? WHERE duplicate_group_id=?", (new_rationale, group_id))
                con.execute(
                    """INSERT OR IGNORE INTO registry_entity_ref
                       (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                       VALUES(?,'cleanup_execution_addendum',?,'DUPLICATE_GROUP',?,'RECONCILES_STATUS',?)""",
                    (addendum_source_id, str(group_id), str(group_id), "Verified completed duplicate-copy removal; receipt=" + receipt_rel + "; sha256=" + info["receipt_sha256"]),
                )
                changes[batch_id] = changes.get(batch_id, 0) + 1
            outcome_note = (
                "Outcome reconciled on 2026-11-23 from execution addendum " +
                addendum.relative_to(AUDIT.parents[1]).as_posix() + "; exact receipt=" + receipt_rel +
                "; receipt_sha256=" + info["receipt_sha256"] +
                "; deletion_state=VERIFIED, artifact_location=REMOVED."
            )
            for (artifact_id,) in con.execute(
                "SELECT artifact_id FROM cleanup_item WHERE cleanup_batch_id=?", (batch_id,)
            ):
                con.execute(
                    "UPDATE cleanup_item SET notes=CASE WHEN instr(notes,?)>0 THEN notes ELSE trim(notes || ' ' || ?) END WHERE cleanup_batch_id=? AND artifact_id=?",
                    (outcome_note, outcome_note, batch_id, artifact_id),
                )
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'cleanup_execution_addendum',?,'CLEANUP_BATCH',?,'RECONCILES_STATUS',?)""",
                (addendum_source_id, batch_id, batch_id, "Verified outcome for " + str(info["count"]) + " exact duplicates; receipt=" + receipt_rel),
            )
        expected_groups = {
            "RF-DATA-FOR-YUQI-PSEUDO-DUPLICATES-20261122": 65,
            "RF-DATA-FOR-VICTOR-FE-DUPLICATES-20261122": 9,
        }
        if changes != expected_groups:
            raise RuntimeError("not all expected duplicate groups were reconciled: " + repr(changes))
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign-key check failed: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"batches_reconciled": changes, "cleanup_items_updated": sum(info["count"] for info in evidence.values()),
                      "addendum_sha256": digest(addendum), "foreign_key_violations": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
