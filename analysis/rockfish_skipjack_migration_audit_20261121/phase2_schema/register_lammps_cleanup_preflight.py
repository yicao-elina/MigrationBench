#!/usr/bin/env python3
"""Register a passing no-delete LAMMPS preflight and advance its SQL batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
EXPECTED_BATCH = "RF-LAMMPS-BUILD-DELETE-20261122-R01"
EXPECTED_COUNT = 1876
EXPECTED_LOGICAL = 4052888613
EXPECTED_ALLOCATED = 4090057216
RUN_ID = "RF-LAMMPS-FLARE-BUILD-B620C52-20261122-R01"
LOCAL_ROOT_ID = "RF-LOCAL-LAMMPS-AUDIT"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-key", default="lammps_build_cleanup_preflight_20261122_r02")
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    events = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = next((row for row in events if row.get("event") == "preflight_passed"), None)
    complete = next((row for row in events if row.get("event") == "complete"), None)
    if not passed or not complete or complete.get("deleted") is not False:
        raise SystemExit("Expected a passing read-only receipt with no deletions")
    expected = (EXPECTED_COUNT, EXPECTED_LOGICAL, EXPECTED_ALLOCATED)
    actual = (passed.get("candidate_count"), passed.get("logical_bytes"), passed.get("allocated_bytes"))
    if actual != expected or passed.get("queue_overlaps") != 0 or passed.get("action") != "NO_DELETE":
        raise SystemExit(f"Preflight evidence differs from reviewed batch: {actual}")
    observed = datetime.fromisoformat(passed["utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    observed_utc = observed.isoformat(timespec="seconds").replace("+00:00", "Z")
    digest = sha256(receipt)
    local_root = AUDIT / "phase2_data"
    relative = receipt.relative_to(local_root).as_posix()

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute(
            "SELECT status,expected_bytes,source_root_id FROM cleanup_batch WHERE cleanup_batch_id=?",
            (EXPECTED_BATCH,),
        ).fetchone()
        if not batch or batch[1] != EXPECTED_LOGICAL or batch[2] != "RF-DATA-YCAO73-HOME":
            raise RuntimeError("Cleanup batch identity or expected logical bytes do not match")
        counts = con.execute(
            "SELECT deletion_state,count(*) FROM cleanup_item WHERE cleanup_batch_id=? GROUP BY deletion_state",
            (EXPECTED_BATCH,),
        ).fetchall()
        state_counts = {state: count for state, count in counts}
        if state_counts not in ({"PLANNED": EXPECTED_COUNT, "VERIFIED": 1},
                                {"EVIDENCE_READY": EXPECTED_COUNT, "VERIFIED": 1}):
            raise RuntimeError(f"Unexpected cleanup item state counts: {state_counts}")

        prior_source = con.execute(
            "SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (args.source_key,)
        ).fetchone()
        expected_source = (str(receipt), digest)
        if prior_source and prior_source != expected_source:
            raise RuntimeError("Preflight source key already points to different evidence")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES(?,?,'LAMMPS_BUILD_PREFLIGHT_RECEIPT',?,?,?,1,?)""",
            (args.source_key, str(receipt), digest, observed_utc, observed_utc,
             "Read-only Rockfish preflight; exact candidate manifest, per-file hashes, preserved dependency tree, and Slurm queue checked."),
        )
        con.execute(
            "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
            (LOCAL_ROOT_ID, relative, observed_utc),
        )
        path_id = con.execute(
            "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT_ID, relative)
        ).fetchone()[0]
        project_id = con.execute(
            "SELECT project_id FROM storage_root WHERE root_id=?", (EXPECTED_BATCH and batch[2],)
        ).fetchone()[0]
        artifact_key = "LMP-EVIDENCE:" + receipt.name
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                scientific_status,retention_class,notes)
               VALUES(?,?,'BUILD_CLEANUP_PREFLIGHT','Read-only pre-delete guard receipt','USER_CONFIRMED',?,?,'VERIFIED','PERMANENT_SOURCE',?)""",
            (project_id, artifact_key, digest, receipt.stat().st_size,
             "Passing zero-overlap preflight; no remote files removed by this receipt."),
        )
        artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()[0]
        existing_artifact = con.execute(
            "SELECT sha256 FROM artifact WHERE artifact_id=?", (artifact_id,)
        ).fetchone()[0]
        if existing_artifact != digest:
            raise RuntimeError("Preflight artifact key already has a different hash")
        con.execute(
            """INSERT OR IGNORE INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES(?,?,'PRESENT',?,?)""",
            (artifact_id, path_id, observed_utc, observed_utc),
        )
        con.execute(
            """INSERT OR IGNORE INTO run_artifact
               (run_id,artifact_id,role,direction,required_for_reproduction,notes)
               VALUES(?,?,'cleanup preflight','PROVENANCE',0,?)""",
            (RUN_ID, artifact_id, "Guard receipt for the planned removal of reproducible build outputs."),
        )
        con.execute(
            "UPDATE cleanup_item SET deletion_state='EVIDENCE_READY' WHERE cleanup_batch_id=? AND disposition='DELETE' AND deletion_state='PLANNED'",
            (EXPECTED_BATCH,),
        )
        con.execute(
            "UPDATE cleanup_batch SET status='EVIDENCE_READY',evidence_artifact_id=? WHERE cleanup_batch_id=? AND status='PLANNED'",
            (artifact_id, EXPECTED_BATCH),
        )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed while registering preflight")
        con.commit()
        print(json.dumps({"batch": EXPECTED_BATCH, "status": "EVIDENCE_READY", "artifact_id": artifact_id,
                          "receipt_sha256": digest, "candidate_count": EXPECTED_COUNT,
                          "logical_bytes": EXPECTED_LOGICAL, "allocated_bytes": EXPECTED_ALLOCATED}, sort_keys=True))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
