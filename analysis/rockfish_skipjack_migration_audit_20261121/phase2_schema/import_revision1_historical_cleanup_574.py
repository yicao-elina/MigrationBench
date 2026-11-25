#!/usr/bin/env python3
"""Import the historical 574-file Revision 1 restart cleanup into the catalog.

The unified cleanup ledger, TSV path manifest, merge receipt, and post-delete
verification JSON are opened as read-only evidence sources. This importer writes
only to provenance_catalog.sqlite and is idempotent per source hash. It does not
read/hash restart payloads, invent file modes/allocated bytes, or infer run links
from directory/legacy run labels.

This importer must be run only after schema migration 015 is applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


AUDIT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = AUDIT_DIR / "provenance_catalog.sqlite"
DEFAULT_LEDGER = AUDIT_DIR / "cleanup" / "cleanup_audit.sqlite"
EVIDENCE_DIR = AUDIT_DIR / "phase2_data" / "revision1_historical_restart_cleanup_574_20261123"
DEFAULT_MASTER_SNAPSHOT = EVIDENCE_DIR / "cleanup_audit_master_snapshot_20261123.sqlite"
DEFAULT_MANIFEST = EVIDENCE_DIR / "removed_574_wfc_mix_paths_20261122.tsv"
DEFAULT_VERIFY = EVIDENCE_DIR / "current_rockfish_verification_20261123.json"
DEFAULT_MERGE_RECEIPT = EVIDENCE_DIR / "legacy_cleanup_ledger_merge_receipt.json"

ACTION_ID = "remove_stale_wfc_mix_restarts_20261122_v1"
EXPECTED_COUNT = 574
EXPECTED_BYTES = 1_536_705_954_176
ROOT_ID = "RF-SCRATCH16-YI"
SCOPED_ROOT = "/scratch16/pclancy3/yi"
PROJECT_ID = "MIGRATIONBENCH"
BATCH_ID = "RF-SCRATCH16-YI-REV1-574-RESTART-CLEANUP-20261122-V1"
ARTIFACT_PREFIX = "ROCKFISH:RF-SCRATCH16-YI:"
MANIFEST_ARTIFACT_KEY = "LOCAL_AUDIT:REV1-574-RESTART-CLEANUP-PATH-MANIFEST-20261123"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    return con


def registry_source(
    target: sqlite3.Connection,
    *,
    source_key: str,
    source_path: Path,
    registry_kind: str,
    digest: str,
    captured: str,
    as_of: str | None,
    immutable: bool,
    authority_note: str,
    notes: str,
) -> int:
    row = target.execute(
        "SELECT source_id,source_path,registry_kind,sha256,as_of_utc,immutable_snapshot FROM registry_source WHERE source_key=?",
        (source_key,),
    ).fetchone()
    expected = (str(source_path.resolve()), registry_kind, digest, as_of, int(immutable))
    if row:
        actual = (row["source_path"], row["registry_kind"], row["sha256"], row["as_of_utc"], row["immutable_snapshot"])
        if actual != expected:
            raise ValueError(f"Existing registry source conflicts for {source_key}: {actual} != {expected}")
        return int(row["source_id"])
    cur = target.execute(
        """INSERT INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            source_key,
            str(source_path.resolve()),
            registry_kind,
            digest,
            captured,
            as_of,
            int(immutable),
            authority_note,
            notes,
        ),
    )
    return int(cur.lastrowid)


def validate_source_rows(
    ledger: sqlite3.Connection,
    original_snapshot: sqlite3.Connection,
    manifest_path: Path,
    verify_path: Path,
):
    action = ledger.execute("SELECT * FROM cleanup_action WHERE action_id=?", (ACTION_ID,)).fetchone()
    if not action:
        raise ValueError(f"Source ledger does not contain {ACTION_ID}")
    rows = ledger.execute(
        "SELECT * FROM cleanup_artifact WHERE action_id=? ORDER BY artifact_path", (ACTION_ID,)
    ).fetchall()
    original_action = original_snapshot.execute(
        "SELECT * FROM cleanup_action WHERE action_id=?", (ACTION_ID,)
    ).fetchone()
    original_rows = original_snapshot.execute(
        "SELECT * FROM cleanup_artifact WHERE action_id=? ORDER BY artifact_path", (ACTION_ID,)
    ).fetchall()
    if not original_action or len(original_rows) != EXPECTED_COUNT:
        raise ValueError("Original master snapshot does not contain the complete historical action")
    action_columns = (
        "action_id",
        "created_utc",
        "source_root",
        "rule",
        "rationale",
        "status",
        "expected_bytes",
        "actual_bytes",
        "completed_utc",
    )
    if any(str(action[col]) != str(original_action[col]) for col in action_columns):
        raise ValueError("Unified cleanup action metadata differs from the original master snapshot")
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} source paths, found {len(rows)}")
    total = sum(int(row["size_bytes"]) for row in rows)
    if total != EXPECTED_BYTES or action["expected_bytes"] != EXPECTED_BYTES or action["actual_bytes"] != EXPECTED_BYTES:
        raise ValueError(f"Unexpected logical-byte totals: rows={total}, action={tuple(action)}")
    if action["status"] != "COMPLETED" or not action["completed_utc"]:
        raise ValueError("The historical source action is not marked COMPLETED")
    if sum(row["pre_delete_sha256"] is not None for row in rows) != 0:
        raise ValueError("Unexpected content hash: this historical receipt is metadata-only")
    if any(row["file_mode"] not in (None, "not_recorded", "") for row in rows):
        raise ValueError("Unexpected populated file mode in historical receipt")
    if any(row["status"] != "DELETED" for row in rows):
        raise ValueError("Not every source path is marked DELETED")
    if len({row["slurm_job_id"] for row in rows}) != 76:
        raise ValueError("Expected exactly 76 distinct historical Slurm job IDs")
    if any(not row["artifact_path"].startswith(SCOPED_ROOT + "/") for row in rows):
        raise ValueError("A source path is outside the expected scoped root")

    with manifest_path.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f, delimiter="\t"))
    if len(manifest_rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} manifest paths, found {len(manifest_rows)}")
    manifest_by_path = {row["artifact_path"]: row for row in manifest_rows}
    if len(manifest_by_path) != EXPECTED_COUNT:
        raise ValueError("Manifest contains duplicate paths")
    source_by_path = {row["artifact_path"]: row for row in rows}
    original_by_path = {row["artifact_path"]: row for row in original_rows}
    if set(manifest_by_path) != set(source_by_path) or set(manifest_by_path) != set(original_by_path):
        raise ValueError("Unified ledger, original master snapshot, and path manifest paths differ")
    stable_columns = (
        "action_id",
        "run_id",
        "slurm_job_id",
        "artifact_path",
        "role",
        "size_bytes",
        "mtime_ns",
        "file_mode",
        "pre_delete_sha256",
        "status",
    )
    for path, source_row in source_by_path.items():
        record = manifest_by_path[path]
        original_row = original_by_path[path]
        normalize = lambda value: "" if value is None else str(value)
        for col in stable_columns:
            if normalize(record[col]) != normalize(source_row[col]):
                raise ValueError(f"Source/manifest mismatch for {path}: {col}")
            if normalize(record[col]) != normalize(original_row[col]):
                raise ValueError(f"Master snapshot/manifest mismatch for {path}: {col}")
        if str(record["artifact_id"]) != str(original_row["artifact_id"]):
            raise ValueError(f"Manifest legacy artifact_id differs from master snapshot for {path}")
        if record.get("pre_delete_sha256", "") not in ("", None):
            raise ValueError(f"Manifest has an unexpected payload hash for {path}")
        if record.get("file_mode") not in ("not_recorded", ""):
            raise ValueError(f"Manifest has an unexpected file mode for {path}")

    verification = json.loads(verify_path.read_text(encoding="utf-8"))
    manifest_digest = file_sha256(manifest_path)
    referenced_manifest_sha256 = verification.get("path_manifest_sha256")
    if referenced_manifest_sha256 is None:
        referenced_manifest_sha256 = verification.get("local_execution", {}).get("path_manifest_sha256")
    if referenced_manifest_sha256 != manifest_digest:
        raise ValueError("Post-delete verification refers to a different path manifest")
    if (
        verification.get("verified") is not True
        or verification.get("expected_path_count") != EXPECTED_COUNT
        or verification.get("missing_path_count") != EXPECTED_COUNT
        or verification.get("present_path_count") != 0
        or verification.get("path_error_count") != 0
    ):
        raise ValueError("Post-delete verification does not confirm all 574 paths missing")
    if verification.get("queue", {}).get("root_overlap_count") != 0:
        raise ValueError("Post-delete verification reports active queue overlap")
    return action, rows, verification, manifest_digest, original_rows


def ensure_migration_015(target: sqlite3.Connection) -> None:
    migration = target.execute(
        "SELECT migration_file FROM schema_migration WHERE version=15"
    ).fetchone()
    allocated_col = next(
        (r for r in target.execute("PRAGMA table_info(cleanup_execution_receipt)") if r["name"] == "actual_allocated_bytes"),
        None,
    )
    if not migration or migration["migration_file"] != "015_cleanup_receipt_allocated_bytes_nullable.sql":
        raise RuntimeError("Apply schema migration 015 before importing this receipt")
    if not allocated_col or allocated_col["notnull"]:
        raise RuntimeError("cleanup_execution_receipt.actual_allocated_bytes must be nullable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--master-snapshot", type=Path, default=DEFAULT_MASTER_SNAPSHOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFY)
    parser.add_argument("--merge-receipt", type=Path, default=DEFAULT_MERGE_RECEIPT)
    args = parser.parse_args()

    target_path = args.target.resolve()
    ledger_path = args.ledger.resolve()
    master_snapshot_path = args.master_snapshot.resolve()
    manifest_path = args.manifest.resolve()
    verification_path = args.verification.resolve()
    merge_receipt_path = args.merge_receipt.resolve()
    for path in (target_path, ledger_path, master_snapshot_path, manifest_path, verification_path, merge_receipt_path):
        if not path.is_file():
            raise SystemExit(f"Required file does not exist: {path}")

    wal_path = Path(str(ledger_path) + "-wal")
    if wal_path.exists() and wal_path.stat().st_size:
        raise SystemExit(
            "The unified cleanup ledger has a non-empty WAL. Create/checkpoint an immutable snapshot before importing so the registered hash covers the complete source state."
        )

    ledger_stat_before = ledger_path.stat()
    ledger = readonly_connection(ledger_path)
    ledger.execute("BEGIN")
    original_snapshot = readonly_connection(master_snapshot_path)
    original_snapshot.execute("BEGIN")
    target = sqlite3.connect(target_path, timeout=120)
    target.row_factory = sqlite3.Row
    target.execute("PRAGMA foreign_keys=ON")
    try:
        action, source_rows, verification, manifest_digest, original_rows = validate_source_rows(
            ledger, original_snapshot, manifest_path, verification_path
        )
        merge_receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
        if (
            merge_receipt.get("action_id") != ACTION_ID
            or merge_receipt.get("final_path_count") != EXPECTED_COUNT
            or merge_receipt.get("merged_path_count") != EXPECTED_COUNT
            or merge_receipt.get("integrity_check") != "ok"
        ):
            raise ValueError("Legacy cleanup-ledger merge receipt is not complete")

        ledger_hash = file_sha256(ledger_path)
        master_hash = file_sha256(master_snapshot_path)
        verify_hash = file_sha256(verification_path)
        merge_hash = file_sha256(merge_receipt_path)
        if merge_receipt.get("source_snapshot_sha256") != master_hash:
            raise ValueError("Original master snapshot hash differs from the merge receipt")
        ledger_stat_after = ledger_path.stat()
        if (ledger_stat_before.st_ino, ledger_stat_before.st_size, ledger_stat_before.st_mtime_ns) != (
            ledger_stat_after.st_ino,
            ledger_stat_after.st_size,
            ledger_stat_after.st_mtime_ns,
        ):
            raise RuntimeError("Unified cleanup ledger changed while the read-only import snapshot was being captured")
        imported_at = utc_now()
        verify_as_of = verification["verified_utc"]
        merge_as_of = merge_receipt["merged_utc"]
        ledger_as_of = action["completed_utc"].replace("+00:00", "Z")

        target.execute("BEGIN IMMEDIATE")
        ensure_migration_015(target)
        root = target.execute(
            "SELECT * FROM storage_root WHERE root_id=?", (ROOT_ID,)
        ).fetchone()
        if not root or root["scoped_path"] != SCOPED_ROOT:
            raise ValueError(f"Unexpected or missing source root mapping: {root}")
        if target.execute("SELECT 1 FROM project WHERE project_id=?", (PROJECT_ID,)).fetchone() is None:
            raise ValueError(f"Expected project {PROJECT_ID} is absent")

        ledger_source = registry_source(
            target,
            source_key=f"REV1-574-CLEANUP-LEDGER-{ledger_hash[:16]}",
            source_path=ledger_path,
            registry_kind="LEGACY_CLEANUP_DATABASE_SNAPSHOT",
            digest=ledger_hash,
            captured=imported_at,
            as_of=ledger_as_of,
            immutable=False,
            authority_note="Read-only source: the current unified cleanup ledger; this importer does not merge or rewrite its 574 action/path rows.",
            notes=f"Contains the already-merged historical action {ACTION_ID}; source file SHA-256={ledger_hash}. Payload SHA, mode, and allocated-byte observations were not recorded.",
        )
        manifest_source = registry_source(
            target,
            source_key=f"REV1-574-CLEANUP-MANIFEST-{manifest_digest[:16]}",
            source_path=manifest_path,
            registry_kind="CLEANUP_PATH_MANIFEST",
            digest=manifest_digest,
            captured=imported_at,
            as_of=ledger_as_of,
            immutable=True,
            authority_note="Metadata-only list of 574 removed restart paths with size, mtime, role, and exact legacy job id.",
            notes="No restart payload hashes, no actual file modes, and no allocated-byte measurements are present.",
        )
        verify_source = registry_source(
            target,
            source_key=f"REV1-574-CLEANUP-POSTVERIFY-{verify_hash[:16]}",
            source_path=verification_path,
            registry_kind="CLEANUP_POSTDELETE_VERIFICATION_AND_QUEUE_SUMMARY",
            digest=verify_hash,
            captured=imported_at,
            as_of=verify_as_of,
            immutable=True,
            authority_note="Rockfish post-delete verification: all 574 manifest paths absent; accompanying queue summary reports zero root overlaps (post-delete only).",
            notes=f"Verification refers to manifest SHA-256={manifest_digest}; post-delete queue summary reports {verification['queue']['total_jobs']} jobs and zero root overlaps; raw queue stdout is represented by digest {verification['queue']['raw_stdout_sha256']}. The source action rationale separately records no active workdir overlap at deletion time.",
        )
        merge_source = registry_source(
            target,
            source_key=f"REV1-574-CLEANUP-LEDGER-MERGE-{merge_hash[:16]}",
            source_path=merge_receipt_path,
            registry_kind="CLEANUP_LEDGER_MERGE_RECEIPT",
            digest=merge_hash,
            captured=imported_at,
            as_of=merge_as_of,
            immutable=True,
            authority_note="Confirms the old cleanup action and 574 paths were merged into the unified cleanup ledger before this catalog import.",
            notes="This importer reads the resulting unified ledger and does not repeat that merge.",
        )
        master_source = registry_source(
            target,
            source_key=f"REV1-574-CLEANUP-MASTER-SNAPSHOT-{master_hash[:16]}",
            source_path=master_snapshot_path,
            registry_kind="LEGACY_CLEANUP_MASTER_SNAPSHOT",
            digest=master_hash,
            captured=imported_at,
            as_of=None,
            immutable=True,
            authority_note="Original master-worktree cleanup database snapshot; preserves the pre-merge cleanup_artifact primary keys referenced by the 574-path TSV.",
            notes="Primary keys in this source are provenance identifiers only. The unified cleanup ledger reinserted these paths with different cleanup_artifact IDs; registry_entity_ref stores the old-to-catalog artifact crosswalk.",
        )

        # A compact immutable manifest artifact is the cleanup batch's evidence pointer.
        evidence = target.execute(
            "SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?",
            (MANIFEST_ARTIFACT_KEY,),
        ).fetchone()
        manifest_size = manifest_path.stat().st_size
        if evidence:
            if evidence["sha256"] != manifest_digest or evidence["byte_size"] != manifest_size:
                raise ValueError("Existing manifest evidence artifact conflicts")
            evidence_artifact_id = int(evidence["artifact_id"])
        else:
            cur = target.execute(
                """INSERT INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,
                    byte_size,scientific_status,retention_class,notes)
                   VALUES (?,?, 'MANIFEST','HISTORICAL_CLEANUP_PATH_MANIFEST','USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL','PERMANENT_SOURCE',?)""",
                (
                    PROJECT_ID,
                    MANIFEST_ARTIFACT_KEY,
                    manifest_digest,
                    manifest_size,
                    f"Local audit evidence for {ACTION_ID}; hash identifies this TSV manifest only, not restart-file contents. Linked source IDs: ledger={ledger_source}, manifest={manifest_source}, postverify={verify_source}, merge={merge_source}.",
                ),
            )
            evidence_artifact_id = int(cur.lastrowid)

        # The main database stores one path/location artifact per deleted file.
        batch = target.execute(
            "SELECT * FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)
        ).fetchone()
        expected_batch = (
            ROOT_ID,
            action["rule"],
            action["rationale"],
            "COMPLETED",
            EXPECTED_BYTES,
            EXPECTED_BYTES,
            action["completed_utc"].replace("+00:00", "Z"),
            evidence_artifact_id,
        )
        if batch:
            actual_batch = (
                batch["source_root_id"],
                batch["policy"],
                batch["rationale"],
                batch["status"],
                batch["expected_bytes"],
                batch["actual_bytes"],
                batch["completed_utc"].replace("+00:00", "Z") if batch["completed_utc"] else None,
                batch["evidence_artifact_id"],
            )
            if actual_batch != expected_batch:
                raise ValueError(f"Existing cleanup batch conflicts: {actual_batch} != {expected_batch}")
        else:
            target.execute(
                """INSERT INTO cleanup_batch
                   (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,
                    authorization_note,expected_bytes,actual_bytes,completed_utc,evidence_artifact_id)
                   VALUES (?,?,?,?,?,'COMPLETED',?,?,?,?,?)""",
                (
                    BATCH_ID,
                    action["created_utc"].replace("+00:00", "Z"),
                    ROOT_ID,
                    action["rule"],
                    action["rationale"],
                    action["rationale"],
                    EXPECTED_BYTES,
                    EXPECTED_BYTES,
                    action["completed_utc"].replace("+00:00", "Z"),
                    evidence_artifact_id,
                ),
            )

        original_by_path = {str(row["artifact_path"]): row for row in original_rows}
        unique_job_ids = sorted({str(row["slurm_job_id"]) for row in source_rows})
        job_to_runs: dict[str, list[str]] = {}
        job_to_obs: dict[str, list[int]] = {}
        for job_id in unique_job_ids:
            matches = target.execute(
                """SELECT run_id,scheduler_observation_id
                   FROM scheduler_job_source_observation WHERE job_id=?
                   ORDER BY run_id,scheduler_observation_id""",
                (job_id,),
            ).fetchall()
            runs = sorted({str(row["run_id"]) for row in matches})
            job_to_runs[job_id] = runs
            job_to_obs[job_id] = [int(row["scheduler_observation_id"]) for row in matches]
        exact_linked_jobs = sum(len(job_to_runs[job]) == 1 for job in unique_job_ids)
        unmatched_jobs = sum(len(job_to_runs[job]) == 0 for job in unique_job_ids)
        ambiguous_jobs = sum(len(job_to_runs[job]) > 1 for job in unique_job_ids)

        # Fail closed on partial prior imports; a rerun either verifies the whole receipt or inserts it whole.
        existing_item_count = target.execute(
            "SELECT COUNT(*) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH_ID,)
        ).fetchone()[0]
        if existing_item_count not in (0, EXPECTED_COUNT):
            raise ValueError(f"Partial cleanup-item import detected: {existing_item_count}/{EXPECTED_COUNT}")

        added_paths = 0
        added_artifacts = 0
        added_locations = 0
        added_items = 0
        added_run_links = 0
        for row in source_rows:
            path = str(row["artifact_path"])
            relative = path[len(SCOPED_ROOT) + 1 :]
            path_row = target.execute(
                "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
                (ROOT_ID, relative),
            ).fetchone()
            if path_row:
                path_id = int(path_row["path_id"])
            else:
                cur = target.execute(
                    "INSERT INTO storage_path(root_id,relative_path,first_seen_utc) VALUES (?,?,?)",
                    (ROOT_ID, relative, imported_at),
                )
                path_id = int(cur.lastrowid)
                added_paths += 1

            artifact_key = ARTIFACT_PREFIX + relative
            artifact = target.execute(
                "SELECT * FROM artifact WHERE artifact_key=?", (artifact_key,)
            ).fetchone()
            artifact_notes = (
                f"Historical deleted QE restart from {ACTION_ID}; metadata-only record. "
                f"Legacy run label={row['run_id']}; exact Slurm job id={row['slurm_job_id']}; "
                "content SHA-256 absent, mode unrecorded, allocated bytes unmeasured. "
                f"Post-delete absence verified at {verification['verified_utc']}."
            )
            if artifact:
                expected = (PROJECT_ID, str(row["role"]), "USER_CONFIRMED", None, int(row["size_bytes"]), "PROVENANCE_PARTIAL", "TEMPORARY")
                actual = (artifact["project_id"], artifact["semantic_role"], artifact["owner_status"], artifact["sha256"], artifact["byte_size"], artifact["scientific_status"], artifact["retention_class"])
                if actual != expected:
                    raise ValueError(f"Existing artifact metadata conflict for {artifact_key}: {actual} != {expected}")
                artifact_id = int(artifact["artifact_id"])
            else:
                cur = target.execute(
                    """INSERT INTO artifact
                       (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,
                        byte_size,scientific_status,retention_class,notes)
                       VALUES (?,?, 'FILE',?,'USER_CONFIRMED',NULL,?,'PROVENANCE_PARTIAL','TEMPORARY',?)""",
                    (PROJECT_ID, artifact_key, row["role"], int(row["size_bytes"]), artifact_notes),
                )
                artifact_id = int(cur.lastrowid)
                added_artifacts += 1

            original_artifact_id = int(original_by_path[path]["artifact_id"])
            original_entity_key = f"{ACTION_ID}:{original_artifact_id}"
            old_id_ref = target.execute(
                """SELECT external_ref_id,source_id,entity_type,entity_key,catalog_entity_type,
                          catalog_entity_key,relationship
                   FROM registry_entity_ref
                   WHERE source_id=? AND entity_type='cleanup_artifact' AND entity_key=?
                     AND catalog_entity_type='artifact' AND catalog_entity_key=?
                     AND relationship='REIDENTIFIED_AS'""",
                (master_source, original_entity_key, artifact_key),
            ).fetchone()
            if not old_id_ref:
                # If the old ID is already mapped elsewhere, do not silently create an inconsistent crosswalk.
                conflicting_ref = target.execute(
                    """SELECT catalog_entity_key FROM registry_entity_ref
                       WHERE source_id=? AND entity_type='cleanup_artifact' AND entity_key=?""",
                    (master_source, original_entity_key),
                ).fetchone()
                if conflicting_ref:
                    raise ValueError(f"Original artifact ID {original_entity_key} already maps elsewhere")
                target.execute(
                    """INSERT INTO registry_entity_ref
                       (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                       VALUES (?,'cleanup_artifact',?,'artifact',?,'REIDENTIFIED_AS',?)""",
                    (
                        master_source,
                        original_entity_key,
                        artifact_key,
                        f"Original cleanup_artifact primary key {original_artifact_id}; unified cleanup ledger row key is {row['artifact_id']}; stable path={path}.",
                    ),
                )
            elif int(old_id_ref["source_id"]) != master_source:
                raise ValueError(f"Original artifact ID crosswalk source differs for {path}")

            locations = target.execute(
                "SELECT * FROM artifact_location WHERE path_id=?", (path_id,)
            ).fetchall()
            this_location = next((loc for loc in locations if int(loc["artifact_id"]) == artifact_id), None)
            conflicting_locations = [loc for loc in locations if int(loc["artifact_id"]) != artifact_id]
            if conflicting_locations:
                raise ValueError(f"A different artifact already occupies catalog path {path}")
            if this_location:
                if this_location["availability"] != "REMOVED":
                    raise ValueError(f"Existing location is not REMOVED for {path}")
            else:
                target.execute(
                    """INSERT INTO artifact_location
                       (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                       VALUES (?,?,'REMOVED',NULL,?)""",
                    (artifact_id, path_id, verification["verified_utc"]),
                )
                added_locations += 1

            run_id = job_to_runs[str(row["slurm_job_id"])][0] if len(job_to_runs[str(row["slurm_job_id"])]) == 1 else None
            if run_id:
                link = target.execute(
                    "SELECT role,direction,required_for_reproduction FROM run_artifact WHERE run_id=? AND artifact_id=?",
                    (run_id, artifact_id),
                ).fetchone()
                if link:
                    if link["role"] != row["role"] or link["direction"] != "INTERMEDIATE" or link["required_for_reproduction"] != 0:
                        raise ValueError(f"Existing run-artifact link conflicts for {artifact_key}")
                else:
                    obs_ids = ",".join(map(str, job_to_obs[str(row["slurm_job_id"])]))
                    target.execute(
                        """INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes)
                           VALUES (?,?,?,'INTERMEDIATE',0,?)""",
                        (
                            run_id,
                            artifact_id,
                            row["role"],
                            f"Exact job_id match to scheduler_job_source_observation; job_id={row['slurm_job_id']}; observation_ids={obs_ids}. Legacy run label is retained as source metadata only.",
                        ),
                    )
                    added_run_links += 1

            existing_item = target.execute(
                "SELECT * FROM cleanup_item WHERE cleanup_batch_id=? AND artifact_id=?",
                (BATCH_ID, artifact_id),
            ).fetchone()
            expected_item = (
                "DELETE",
                "CLEARED",
                None,
                int(row["size_bytes"]),
                int(row["mtime_ns"]),
                "VERIFIED",
                action["completed_utc"].replace("+00:00", "Z"),
                int(row["size_bytes"]),
                ledger_source,
                "cleanup_artifact",
                str(row["artifact_id"]),
            )
            if existing_item:
                actual_item = (
                    existing_item["disposition"],
                    existing_item["owner_clearance"],
                    existing_item["pre_delete_sha256"],
                    existing_item["pre_delete_size_bytes"],
                    existing_item["pre_delete_mtime_ns"],
                    existing_item["deletion_state"],
                    existing_item["deleted_utc"].replace("+00:00", "Z") if existing_item["deleted_utc"] else None,
                    existing_item["actual_bytes"],
                    existing_item["legacy_cleanup_source_id"],
                    existing_item["legacy_cleanup_table"],
                    existing_item["legacy_cleanup_row_key"],
                )
                if actual_item != expected_item:
                    raise ValueError(f"Existing cleanup item conflicts for {artifact_key}: {actual_item} != {expected_item}")
            else:
                target.execute(
                    """INSERT INTO cleanup_item
                       (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,
                        pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,
                        deleted_utc,actual_bytes,legacy_cleanup_source_id,legacy_cleanup_table,
                        legacy_cleanup_row_key,notes)
                       VALUES (?,?, 'DELETE',?,'CLEARED',NULL,?,?, 'VERIFIED',?,?,?,?,?,?)""",
                    (
                        BATCH_ID,
                        artifact_id,
                        action["rationale"],
                        int(row["size_bytes"]),
                        int(row["mtime_ns"]),
                        action["completed_utc"].replace("+00:00", "Z"),
                        int(row["size_bytes"]),
                        ledger_source,
                        "cleanup_artifact",
                        str(row["artifact_id"]),
                        "Metadata-only legacy receipt: pre_delete_sha256=NULL; file mode not recorded; per-file allocated bytes unknown. Current verification confirms path absent.",
                    ),
                )
                added_items += 1

        # Preserve the unmapped-job cases explicitly without guessing from the legacy run label.
        notes = (
            f"Source cleanup ledger action={ACTION_ID}; paths verified absent by post-delete check. "
            f"Exact scheduler job_id mapping: {exact_linked_jobs} of {len(unique_job_ids)} jobs mapped to one workflow_run; "
            f"{unmatched_jobs} unmatched and {ambiguous_jobs} mapped to multiple workflow_run IDs, with no inferred run link. "
            "No payload hashes, file modes, or allocated bytes were recorded. The accompanying queue summary is post-delete; the historical source action rationale records the pre-delete no-overlap check. No quota bracket is inferred from unrelated observations."
        )
        receipt = target.execute(
            "SELECT * FROM cleanup_execution_receipt WHERE cleanup_batch_id=? AND receipt_source_id=?",
            (BATCH_ID, ledger_source),
        ).fetchone()
        if receipt:
            expected_receipt = (
                action["completed_utc"].replace("+00:00", "Z"),
                "COMPLETED",
                EXPECTED_COUNT,
                EXPECTED_BYTES,
                None,
            )
            actual_receipt = (
                receipt["observed_utc"].replace("+00:00", "Z"),
                receipt["execution_state"],
                receipt["deleted_file_count"],
                receipt["actual_logical_bytes"],
                receipt["actual_allocated_bytes"],
            )
            if actual_receipt != expected_receipt:
                raise ValueError(f"Existing cleanup receipt conflicts: {actual_receipt} != {expected_receipt}")
        else:
            target.execute(
                """INSERT INTO cleanup_execution_receipt
                   (cleanup_batch_id,receipt_source_id,observed_utc,execution_state,deleted_file_count,
                    actual_logical_bytes,actual_allocated_bytes,quota_before_observation_id,
                    quota_after_observation_id,queue_snapshot_source_id,notes)
                   VALUES (?,?,?,'COMPLETED',?,?,NULL,NULL,NULL,?,?)""",
                (
                    BATCH_ID,
                    ledger_source,
                    action["completed_utc"].replace("+00:00", "Z"),
                    EXPECTED_COUNT,
                    EXPECTED_BYTES,
                    verify_source,
                    notes,
                ),
            )

        # Evidence bindings are stored in registry_source notes/relationships and batch receipt.
        # The merge receipt source is intentionally attached to the same stable receipt context.
        target.execute(
            "UPDATE cleanup_batch SET authorization_note=? WHERE cleanup_batch_id=? AND authorization_note=?",
            (
                action["rationale"] + f" Evidence sources: cleanup ledger #{ledger_source}, path manifest #{manifest_source}, post-delete verification #{verify_source}, prior unified-ledger merge receipt #{merge_source}.",
                BATCH_ID,
                action["rationale"],
            ),
        )

        # Validate the complete result before committing; no source database is ever attached for writes.
        final_items = target.execute(
            "SELECT COUNT(*),SUM(pre_delete_size_bytes),SUM(actual_bytes),SUM(pre_delete_sha256 IS NOT NULL) FROM cleanup_item WHERE cleanup_batch_id=?",
            (BATCH_ID,),
        ).fetchone()
        if (final_items[0], final_items[1], final_items[2], final_items[3]) != (
            EXPECTED_COUNT,
            EXPECTED_BYTES,
            EXPECTED_BYTES,
            0,
        ):
            raise ValueError(f"Imported item totals are inconsistent: {tuple(final_items)}")
        if target.execute(
            "SELECT COUNT(*) FROM cleanup_execution_receipt WHERE cleanup_batch_id=? AND actual_allocated_bytes IS NOT NULL",
            (BATCH_ID,),
        ).fetchone()[0]:
            raise ValueError("Historical allocated-byte receipt must remain NULL")
        fk_errors = list(target.execute("PRAGMA foreign_key_check"))
        if fk_errors:
            raise ValueError(f"Foreign-key errors before commit: {len(fk_errors)}")
        target.commit()
        print(json.dumps({
            "action_id": ACTION_ID,
            "target_batch_id": BATCH_ID,
            "source_ledger_sha256": ledger_hash,
            "manifest_sha256": manifest_digest,
            "postdelete_verification_sha256": verify_hash,
            "merge_receipt_sha256": merge_hash,
            "source_is_read_only": True,
            "source_path_count": len(source_rows),
            "logical_bytes": EXPECTED_BYTES,
            "actual_allocated_bytes": None,
            "payload_hash_count": 0,
            "file_mode_count": 0,
            "unique_job_ids": len(unique_job_ids),
            "jobs_exactly_linked": exact_linked_jobs,
            "jobs_unmatched_no_guess": unmatched_jobs,
            "jobs_ambiguous_no_guess": ambiguous_jobs,
            "added_storage_paths": added_paths,
            "added_artifacts": added_artifacts,
            "added_locations": added_locations,
            "added_cleanup_items": added_items,
            "added_run_artifact_links": added_run_links,
            "foreign_key_errors": 0,
        }, indent=2))
    except Exception:
        target.rollback()
        raise
    finally:
        ledger.close()
        target.close()


if __name__ == "__main__":
    main()
