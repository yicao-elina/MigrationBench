#!/usr/bin/env python3
"""Stage the coordinate-backed QE restart cleanup in provenance_catalog.sqlite.

The script is read-only unless --apply is supplied. It records exact candidate
metadata, retained coordinate evidence, scheduler context, and held rows before
the guarded Rockfish executor is allowed to unlink anything. Restart payloads
are intentionally not hashed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
REVIEW = AUDIT / "phase2_data/restart_coordinate_audit_20261123T1044Z"
DB = AUDIT / "provenance_catalog.sqlite"
MANIFEST = REVIEW / "eligible_restart_deletion_manifest_20261123T1129Z.json"
DISPOSITIONS = REVIEW / "candidate_restart_disposition_20261123T1129Z.tsv"
CROSSWALK = REVIEW / "run_coordinate_crosswalk_20261123T1129Z.tsv"
PREFLIGHT = REVIEW / "guarded_preflight_latest_20261123.jsonl"
BATCH = "RF-YI-QE-RESTART-CLEANUP-20261123"
LOCAL_ROOT_ID = "LOCAL-MAIN"
ROOT_IDS = {
    "/scratch16/pclancy3/yi": "RF-SCRATCH16-YI",
    "/data/pclancy3/yi": "RF-DATA-YI",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def root_for(path: str) -> tuple[str, str]:
    for prefix, root_id in ROOT_IDS.items():
        if path.startswith(prefix + "/"):
            return root_id, prefix
    raise ValueError("path outside the approved Yi roots: " + path)


def relative_to(path: str, prefix: str) -> str:
    return path[len(prefix):].lstrip("/")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def clean_state(value: str) -> str:
    value = (value or "").upper()
    for prefix, result in (("COMPLETED", "COMPLETED"), ("FAILED", "FAILED"),
                           ("TIMEOUT", "TIMEOUT"), ("CANCELLED", "CANCELLED"),
                           ("RUNNING", "RUNNING"), ("PENDING", "PENDING")):
        if prefix in value:
            return result
    return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the staged cleanup and evidence rows")
    parser.add_argument("--mark-executing", action="store_true", help="move a staged batch to EXECUTING before the remote unlink")
    args = parser.parse_args()

    if args.mark_executing and not args.apply:
        raise SystemExit("--mark-executing requires --apply")

    for path in (MANIFEST, DISPOSITIONS, CROSSWALK, PREFLIGHT):
        path.resolve(strict=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    disposition_rows = read_tsv(DISPOSITIONS)
    crosswalk_rows = read_tsv(CROSSWALK)
    preflight_events = [json.loads(line) for line in PREFLIGHT.read_text(encoding="utf-8").splitlines() if line.strip()]
    preflight = preflight_events[-1]
    files = manifest["files"]
    by_path = {row["path"]: row for row in files}
    eligible = [row for row in disposition_rows if row["disposition"] == "DELETE-ELIGIBLE"]
    held = [row for row in disposition_rows if row["disposition"] != "DELETE-ELIGIBLE"]
    if len(files) != 7581 or len(by_path) != len(files):
        raise SystemExit("eligible manifest must contain 7,581 unique exact paths")
    if len(disposition_rows) != 7589 or len(eligible) != 7581 or len(held) != 8:
        raise SystemExit("candidate disposition counts differ from the reviewed scope")
    if set(by_path) != {row["path"] for row in eligible}:
        raise SystemExit("eligible manifest paths differ from DELETE-ELIGIBLE decisions")
    total = sum(int(row["identity"]["size_bytes"]) for row in files)
    allocated = sum(int(row["identity"]["allocated_bytes"]) for row in files)
    if total != 1225323826928 or allocated != 1225535520768:
        raise SystemExit("eligible manifest byte totals differ from the reviewed scope")
    if preflight.get("status") != "PREFLIGHT_PASS_NO_DELETE" or preflight.get("execute") is not False:
        raise SystemExit("latest guarded preflight did not pass in read-only mode")
    if preflight.get("candidate_count") != len(files) or preflight.get("candidate_logical_bytes") != total:
        raise SystemExit("preflight candidate totals differ from the exact manifest")
    if preflight.get("candidate_roots") != 46 or preflight.get("run_root_queue_overlaps") != 0:
        raise SystemExit("preflight queue/root evidence does not pass")
    if preflight.get("restart_payloads_opened_or_hashed") is not False:
        raise SystemExit("restart payloads must not be opened or hashed")
    if len(crosswalk_rows) != 48:
        raise SystemExit("run crosswalk must contain 46 eligible and 2 held roots")

    if args.mark_executing:
        con = sqlite3.connect(DB, timeout=180)
        con.execute("PRAGMA busy_timeout=180000")
        try:
            con.execute("BEGIN IMMEDIATE")
            batch = con.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
            if not batch or batch[0] != "EVIDENCE_READY" or batch[1] != 1225323826928:
                raise RuntimeError("cleanup batch is not staged with the reviewed byte total")
            counts = con.execute("SELECT disposition,deletion_state,count(*) FROM cleanup_item WHERE cleanup_batch_id=? GROUP BY disposition,deletion_state", (BATCH,)).fetchall()
            if sorted(tuple(row) for row in counts) != [("DELETE", "EVIDENCE_READY", 7581), ("HOLD", "HELD", 8)]:
                raise RuntimeError("cleanup item states do not match the staged manifest")
            con.execute("UPDATE cleanup_batch SET status='EXECUTING' WHERE cleanup_batch_id=?", (BATCH,))
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        print(json.dumps({"batch_id": BATCH, "status": "EXECUTING", "eligible_count": 7581, "held_count": 8, "logical_bytes": 1225323826928}, sort_keys=True))
        return

    counts = Counter(row["file_class"] for row in files)
    summary = {
        "batch_id": BATCH,
        "eligible_file_count": len(files),
        "held_file_count": len(held),
        "eligible_logical_bytes": total,
        "eligible_allocated_bytes": allocated,
        "run_root_count": len({row["run_root"] for row in files}),
        "class_counts": dict(counts),
        "read_only_preflight": preflight,
        "apply": args.apply,
    }
    if not args.apply:
        print(json.dumps(summary, sort_keys=True))
        return

    observed = utc_now()
    local_repo = AUDIT.parents[1]
    con = sqlite3.connect(DB, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        con.execute("BEGIN IMMEDIATE")
        project = con.execute("SELECT project_id FROM project WHERE project_id='MIGRATIONBENCH'").fetchone()
        if not project:
            raise RuntimeError("MIGRATIONBENCH project row is missing")
        local_root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT_ID,)).fetchone()
        if not local_root:
            raise RuntimeError("LOCAL-MAIN storage root is missing")

        def add_local_evidence(path: Path, role: str, kind: str, note: str) -> int:
            path = path.resolve(strict=True)
            digest = sha256(path)
            key = f"qe_restart_coordinate_cleanup:{role}:{digest[:20]}"
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (key, str(path), kind, digest, observed, manifest["queue_snapshot_utc"],
                 "User-authorized QE restart cleanup with preserved coordinate evidence.", note),
            )
            source = con.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
            if not source or source["source_path"] != str(path) or source["sha256"] != digest:
                raise RuntimeError("registry source identity conflict: " + role)
            artifact_key = f"RF-YI-RESTART-CLEANUP-EVIDENCE:{role}:{digest[:20]}"
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                   VALUES('MIGRATIONBENCH',?,? ,?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PUBLICATION_EVIDENCE',?)""",
                (artifact_key, kind, role, digest, path.stat().st_size, note),
            )
            art = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if not art or art["sha256"] != digest or art["byte_size"] != path.stat().st_size:
                raise RuntimeError("evidence artifact identity conflict: " + role)
            relative = path.relative_to(Path(local_root["scoped_path"])).as_posix()
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (LOCAL_ROOT_ID, relative, observed))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT_ID, relative)).fetchone()[0]
            con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                        (art["artifact_id"], path_id, observed, observed))
            con.execute("INSERT OR IGNORE INTO registry_entity_ref(source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes) VALUES(?,'CLEANUP_BATCH',?,'CLEANUP_BATCH',?,'EVIDENCE_FOR',?)",
                        (source["source_id"], BATCH, BATCH, note))
            return int(art["artifact_id"])

        evidence = {}
        evidence["manifest"] = add_local_evidence(MANIFEST, "ELIGIBLE_MANIFEST", "QE_RESTART_DELETE_MANIFEST_JSON",
            "Exact 7,581-path eligible manifest with current stat identity and retained XML/NEB path hashes; restart payloads not hashed.")
        evidence["dispositions"] = add_local_evidence(DISPOSITIONS, "CANDIDATE_DISPOSITIONS", "QE_RESTART_DISPOSITION_TSV",
            "All 7,589 candidates, including 8 held zero-byte placeholders; 48 WorkDir-mismatch mixes are eligible with an explicit warning.")
        evidence["crosswalk"] = add_local_evidence(CROSSWALK, "RUN_COORDINATE_CROSSWALK", "QE_RESTART_RUN_COORDINATE_CROSSWALK_TSV",
            "Per-root input/log/job crosswalk, XML hashes and complete NEB image-path status.")
        evidence["report"] = add_local_evidence(REVIEW / "RESTART_COORDINATE_AUDIT_20261123T1129Z.md", "COORDINATE_AUDIT_REPORT", "QE_RESTART_COORDINATE_AUDIT_REPORT",
            "Narrative of eligibility, holds, SCF limitations, and scheduler path mismatch.")
        evidence["executor"] = add_local_evidence(REVIEW / "guarded_delete_exact_restart_manifest.py", "GUARDED_DELETE_EXECUTOR", "QE_RESTART_GUARDED_DELETE_EXECUTOR",
            "No-follow exact-path executor, per-root fresh all-user queue checks, and post-delete coordinate verification.")
        evidence["preflight"] = add_local_evidence(PREFLIGHT, "READ_ONLY_PREFLIGHT", "QE_RESTART_READ_ONLY_PREFLIGHT_JSONL",
            "Fresh 11:51Z read-only preflight passed for 7,581 files and 46 roots with zero queue overlap.")
        evidence["queue"] = add_local_evidence(Path(manifest["queue_snapshot_path"]), "ALL_USER_QUEUE_SNAPSHOT", "SLURM_ALL_USER_QUEUE_SNAPSHOT",
            "Private raw all-user queue capture; local catalog only, not committed to GitHub or copied into a public repository.")
        evidence["sacct"] = add_local_evidence(Path(manifest["sacct_snapshot_path"]), "RESTART_SACCT_CROSSWALK", "SLURM_SACCT_RESTART_CROSSWALK",
            "Job/WorkDir/submit-line observations used to link candidates; timestamps retained as raw site output.")

        batch = con.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        if batch:
            if batch["status"] not in ("EVIDENCE_READY", "EXECUTING") or batch["expected_bytes"] != total:
                raise RuntimeError("existing cleanup batch conflicts with the reviewed candidate set")
        else:
            con.execute(
                """INSERT INTO cleanup_batch
                   (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,expected_bytes,evidence_artifact_id)
                   VALUES(?,?,NULL,?,?, 'EVIDENCE_READY',?,?,?)""",
                (BATCH, observed, "DELETE_QE_RESTARTS_AFTER_COORDINATE_AND_QUEUE_GUARDS",
                 "Remove exact QE WFC and mixing restart payloads after retaining parseable same-run coordinates and complete NEB image paths where applicable.",
                 "User explicitly directed deletion of WFC and mixing restart files when the latest coordinate checkpoint or complete intermediate-image path remains. Restart payload hashes are omitted by direction; 48 non-NEB mixes have a postdated same-prefix XML but unresolved sacct WorkDir linkage, recorded as a provenance warning. Eight zero-byte placeholders remain held.", total, evidence["manifest"]),
            )

        cross_by_root = {row["run_root"]: row for row in crosswalk_rows}
        all_rows = {row["path"]: row for row in disposition_rows}
        run_by_root = {}
        for root in sorted({row["run_root"] for row in disposition_rows}):
            root_id, prefix = root_for(root + "/sentinel")
            run_id = "RF-QE-RESTART-ROOT-" + hashlib.sha256(root.encode()).hexdigest()[:20]
            run_by_root[root] = (run_id, root_id, prefix)
            cross = cross_by_root.get(root, {})
            workflow = "QE_NEB" if cross.get("latest_path_path") else "QE_RELAX_OR_SCF"
            state = clean_state(cross.get("sacct_state_workdir_submitline", ""))
            notes = ("Synthetic run-group key for exact restart-candidate root; source run identity and convergence are not inferred from filenames. "
                     f"slurm_job_ids={cross.get('slurm_job_ids','')}; sacct={cross.get('sacct_state_workdir_submitline','')}; "
                     f"queue_overlap={cross.get('active_queue_overlap','')}; candidate_disposition={cross.get('candidate_disposition','')}")
            con.execute(
                """INSERT OR IGNORE INTO workflow_run
                   (run_id,project_id,run_label,workflow_type,method,system_key,source_root_id,source_relative_path,scheduler_state,scientific_status,notes)
                   VALUES(?,'MIGRATIONBENCH',?,?,'Quantum ESPRESSO','Sb2Te3',?,?,?,'PROVENANCE_PARTIAL',?)""",
                (run_id, relative_to(root, prefix), workflow, root_id, relative_to(root, prefix), state, notes),
            )
            existing = con.execute("SELECT source_root_id,source_relative_path FROM workflow_run WHERE run_id=?", (run_id,)).fetchone()
            if not existing or existing["source_root_id"] != root_id or existing["source_relative_path"] != relative_to(root, prefix):
                raise RuntimeError("workflow run root identity conflict: " + root)

        snapshots = {}
        for prefix, root_id in ROOT_IDS.items():
            subset = [row for row in disposition_rows if row["path"].startswith(prefix + "/")]
            sid = "RF-YI-QE-RESTART-CANDIDATES-" + root_id + "-20261123T1129Z"
            logical = sum(int(row["current_size_bytes"]) for row in subset)
            allocation = sum(int(row["current_allocated_bytes"]) for row in subset)
            con.execute(
                """INSERT OR IGNORE INTO inventory_snapshot
                   (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,files_seen,unreadable_count,logical_bytes,allocated_bytes,manifest_path,manifest_sha256,notes)
                   VALUES(?,?,?,?,?,'restart-coordinate-audit-v1','COMPLETE',?,0,?,?,?,?,?)""",
                (sid, root_id, manifest["candidate_snapshot_utc"], manifest["final_candidate_lstat_utc"],
                 "Exact-name QE WFC/mix candidate metadata capture and identity refresh; XML/NEB path payload hashes only; no restart payload hashes.",
                 len(subset), logical, allocation, str(DISPOSITIONS), sha256(DISPOSITIONS),
                 "Targeted candidate inventory, not a full-root scan. Queue gates and coordinate evidence are in cleanup_candidate_context."),
            )
            snapshots[root_id] = sid

        path_artifacts = {}
        def add_preserved_evidence(path: str, digest: str, role: str, run_id: str, root_id: str, prefix: str) -> int:
            key_tuple = (path, digest, role)
            if key_tuple in path_artifacts:
                artifact_id = path_artifacts[key_tuple]
            else:
                akey = "RF-YI-QE-PRESERVED-EVIDENCE:" + hashlib.sha256((path + "\0" + role).encode()).hexdigest()[:24]
                con.execute(
                    """INSERT OR IGNORE INTO artifact
                       (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                       VALUES('MIGRATIONBENCH',?,'QE_COORDINATE_EVIDENCE',?,'USER_CONFIRMED',?,NULL,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
                    (akey, role, digest, "Preserved coordinate evidence; exact content SHA-256 verified by guarded cleanup preflight. Scientific convergence is assessed separately."),
                )
                aid = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (akey,)).fetchone()
                if not aid or aid["sha256"] != digest:
                    raise RuntimeError("preserved evidence hash identity conflict: " + path)
                rel = relative_to(path, prefix)
                con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (root_id, rel, observed))
                pid = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, rel)).fetchone()[0]
                con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)", (aid["artifact_id"], pid, manifest["final_candidate_lstat_utc"], manifest["final_candidate_lstat_utc"]))
                artifact_id = int(aid["artifact_id"])
                path_artifacts[key_tuple] = artifact_id
            con.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'OUTPUT',1,?)",
                        (run_id, artifact_id, role, "Retained coordinate artifact authorizing WFC/mix restart removal."))
            return artifact_id

        for row in disposition_rows:
            path = row["path"]
            root_id, prefix = root_for(path)
            run_id, _, _ = run_by_root[row["run_root"]]
            identity = by_path.get(path, {}).get("identity")
            if identity is None:
                identity = {"size_bytes": int(row["current_size_bytes"]), "allocated_bytes": int(row["current_allocated_bytes"]),
                            "mode": row["current_mode"], "uid": int(row["current_uid"]), "gid": int(row["current_gid"]),
                            "device": int(row["current_device"]), "inode": int(row["current_inode"]), "nlink": int(row["current_nlink"]),
                            "mtime_ns": int(row["current_mtime_ns"])}
            if identity["size_bytes"] != int(row["current_size_bytes"]) or identity["inode"] != int(row["current_inode"]):
                raise RuntimeError("manifest/disposition identity mismatch: " + path)
            rel = relative_to(path, prefix)
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (root_id, rel, observed))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, rel)).fetchone()[0]
            sid = snapshots[root_id]
            con.execute(
                """INSERT OR IGNORE INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,sha256,hash_scope,observation_status,link_count)
                   VALUES(?,?,'FILE',?,?,?,?,?,?,?,?,NULL,'NONE','PRESENT',?)""",
                (sid, path_id, identity["uid"], identity["gid"], identity["mode"], identity["size_bytes"], identity["allocated_bytes"], identity["mtime_ns"], str(identity["inode"]), str(identity["device"]), identity["nlink"]),
            )
            class_role = {"QE_WFC_N": "QE_WFC_RESTART", "QE_WFC_DAT": "QE_WFC_RESTART", "QE_MIX_N": "QE_MIXING_RESTART"}[row["file_class"]]
            artifact_key = "RF-YI-QE-RESTART:" + hashlib.sha256(path.encode()).hexdigest()[:24]
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                   VALUES('MIGRATIONBENCH',?,'QE_RESTART_FILE',?,'USER_CONFIRMED',NULL,?,'PROVENANCE_PARTIAL','TEMPORARY',?)""",
                (artifact_key, class_role, identity["size_bytes"],
                 "Metadata-only identity; restart payload SHA-256 intentionally not collected. Coordinates/path evidence is retained and separately hashed."),
            )
            artifact = con.execute("SELECT artifact_id,byte_size,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if not artifact or artifact["byte_size"] != identity["size_bytes"] or artifact["sha256"] is not None:
                raise RuntimeError("restart artifact identity conflict: " + path)
            con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                        (artifact["artifact_id"], path_id, manifest["final_candidate_lstat_utc"], manifest["final_candidate_lstat_utc"]))
            disposition = "DELETE" if row["disposition"] == "DELETE-ELIGIBLE" else "HOLD"
            deletion_state = "EVIDENCE_READY" if disposition == "DELETE" else "HELD"
            con.execute(
                """INSERT OR IGNORE INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes)
                   VALUES(?,?,?,?,?,NULL,?,?,?,?)""",
                (BATCH, artifact["artifact_id"], disposition, row["disposition_reason"], "CLEARED",
                 identity["size_bytes"], identity["mtime_ns"], deletion_state,
                 "Exact stat identity staged before unlink. No payload hash. Eight zero-byte placeholders remain held."),
            )
            con.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'INTERMEDIATE',0,?)",
                        (run_id, artifact["artifact_id"], class_role, "Temporary restart removed only after coordinate evidence verification."))

            coord_path = row["direct_coordinate_xml_path"]
            coord_hash = row["direct_coordinate_xml_sha256"]
            if not coord_path or not coord_hash:
                raise RuntimeError("candidate has no hashed coordinate XML evidence: " + path)
            coord_artifact = add_preserved_evidence(coord_path, coord_hash, "QE_COORDINATE_XML", run_id, root_id, prefix)
            neb_path = row["latest_path_trajectory_path"]
            neb_hash = row["latest_path_sha256"]
            if neb_path:
                if not neb_hash:
                    raise RuntimeError("NEB path lacks a content hash: " + path)
                neb_artifact = add_preserved_evidence(neb_path, neb_hash, "QE_NEB_PATH_TRAJECTORY", run_id, root_id, prefix)
            else:
                neb_artifact = None
            manifest_paths = cross_by_root[row["run_root"]].get("manifest_paths", "") or ""
            latest_parse = row["latest_path_parse"] or ""
            complete = int(bool(neb_path and latest_parse == "OK" and row["latest_path_image_ids"]))
            queue_utc = manifest["queue_snapshot_utc"].replace("+00:00", "Z")
            con.execute(
                """INSERT OR IGNORE INTO cleanup_candidate_context
                   (cleanup_batch_id,artifact_id,run_id,candidate_kind,manifest_paths,manifest_present,manifest_restart_required_flag,manifest_restart_required_basis,input_paths,output_log_paths,coordinate_evidence_path,coordinate_evidence_sha256,coordinate_xml_parse,coordinate_nat,coordinate_evidence_kind,neb_path_evidence,neb_image_ids,neb_complete_image_coverage,source_queue_capture_utc,source_queue_status,source_queue_workdir_overlap,disposition_reason,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (BATCH, artifact["artifact_id"], run_id, row["file_class"], manifest_paths, int(bool(manifest_paths)),
                 row["manifest_wfc_restart_required_flag"] or "NOT_STATED", row["triage_prior_reason"] or "",
                 row["input_paths"], row["output_log_paths"], coord_path, coord_hash, row["direct_xml_parse"],
                 int(row["xml_natom"]) if row["xml_natom"].isdigit() else None, "QE_XML_ATOMIC_POSITIONS",
                 neb_path or "", row["latest_path_image_ids"] or "", complete, queue_utc,
                 "FRESH_ALL_USER_QUEUE_ZERO_OVERLAP", 0, row["disposition_reason"],
                 "same-prefix coordinate_artifact_id=" + str(coord_artifact) + ("; neb_path_artifact_id=" + str(neb_artifact) if neb_artifact else "") +
                 "; slurm_job_ids=" + row["slurm_job_ids"] + "; workdir_provenance=" + row["workdir_provenance"]),
            )

        # Associate the preserved audit evidence with the batch and make reruns idempotent.
        existing_items = con.execute("SELECT disposition,deletion_state,pre_delete_size_bytes FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH,)).fetchall()
        if len(existing_items) != 7589:
            raise RuntimeError("staged cleanup item count does not equal 7,589")
        if con.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE' AND deletion_state='EVIDENCE_READY'", (BATCH,)).fetchone()[0] != 7581:
            raise RuntimeError("staged eligible cleanup item count differs")
        if con.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='HOLD' AND deletion_state='HELD'", (BATCH,)).fetchone()[0] != 8:
            raise RuntimeError("staged held cleanup item count differs")
        if con.execute("SELECT coalesce(sum(pre_delete_size_bytes),0) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE'", (BATCH,)).fetchone()[0] != total:
            raise RuntimeError("staged cleanup byte total differs")
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign-key check failed while staging cleanup evidence")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    summary["staged"] = True
    summary["source_sha256"] = {str(path): sha256(path) for path in (MANIFEST, DISPOSITIONS, CROSSWALK, PREFLIGHT)}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
