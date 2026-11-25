#!/usr/bin/env python3
"""Stage exact Revision 1 QE restart metadata and coordinate lineage in the catalog."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
REVIEW = AUDIT / "phase2_data/revision1_restart_audit_20261123T081721Z"
MANIFEST = REVIEW / "candidate_files.tsv"
CROSSWALK = REVIEW / "run_group_crosswalk.tsv"
SUMMARY = REVIEW / "audit_summary.json"
BATCH_ID = "RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123"
ROOT_ID = "RF-SCRATCH16-YI"
REMOTE_REPO = Path("/scratch16/pclancy3/yi/MigrationBench")
ROOT_PATH = Path("/scratch16/pclancy3/yi")
RELEASE_ID = "MB-REVISION1-SUBMITTED-20261110"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def state(value: str) -> str:
    text = (value or "").upper().split()[0]
    if text.startswith("COMPLETED"):
        return "COMPLETED"
    if text.startswith("CANCELLED"):
        return "CANCELLED"
    if text.startswith("TIMEOUT"):
        return "TIMEOUT"
    if text.startswith("RUNNING"):
        return "RUNNING"
    if text.startswith("PENDING"):
        return "PENDING"
    if text in ("FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"):
        return "FAILED"
    return "UNKNOWN"


def relative_to_yi(path_text: str) -> str:
    path = Path(path_text)
    try:
        return path.relative_to(ROOT_PATH).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Rockfish evidence path is outside {ROOT_PATH}: {path_text}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-utc", default=utc_now())
    parser.add_argument("--queue-source-utc", default="2026-11-23T08:28:02Z")
    args = parser.parse_args()
    for path in (MANIFEST, CROSSWALK, SUMMARY):
        path.resolve(strict=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    source_observed_utc = summary["queue_capture"]["captured_utc"]
    candidates = read_tsv(MANIFEST)
    groups = read_tsv(CROSSWALK)
    if len(candidates) != 4298 or len(groups) != 113:
        raise SystemExit("candidate/group counts differ from the reviewed scope")
    if len({x["absolute_path"] for x in candidates}) != len(candidates):
        raise SystemExit("candidate paths are not unique")
    by_path = {x["absolute_path"]: x for x in candidates}
    logical = sum(int(x["size_bytes"]) for x in candidates)
    allocated = sum(int(x["allocated_bytes"]) for x in candidates)
    if logical != int(summary["candidate_bytes"]) or allocated != int(summary["allocated_bytes"]):
        raise SystemExit("pre-delete hash totals differ from the audited metadata totals")
    if logical != 1036297176856 or allocated != 1036429164544:
        raise SystemExit("metadata totals differ from the reviewed scope")

    db = sqlite3.connect(DB, timeout=180)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=180000")
    try:
        if db.execute("SELECT count(*) FROM schema_migration WHERE version=12").fetchone()[0] == 0:
            db.executescript((AUDIT / "phase2_schema/012_revision1_release_and_scheduler_evidence.sql").read_text(encoding="utf-8"))
        root = db.execute("SELECT scoped_path,project_id FROM storage_root WHERE root_id=?", (ROOT_ID,)).fetchone()
        if root != ("/scratch16/pclancy3/yi", None) and (not root or root[0] != "/scratch16/pclancy3/yi"):
            raise SystemExit(f"unexpected scratch16 storage root: {tuple(root) if root else None}")
        if db.execute("SELECT count(*) FROM project WHERE project_id='MIGRATIONBENCH'").fetchone()[0] != 1:
            raise SystemExit("MIGRATIONBENCH project row is missing")

        def evidence(path: Path, kind: str, role: str, note: str, remote_path: str | None = None) -> tuple[int, int]:
            path = path.resolve(strict=True)
            digest = sha256(path)
            rel = path.relative_to(AUDIT.parents[1]).as_posix()
            source_key = f"rev1_restart:{role}:{digest[:20]}"
            db.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (source_key, str(path), kind, digest, utc_now(), args.observed_utc,
                 "User-authorized QE WFC/mixing restart cleanup; pre-delete evidence is immutable.", note),
            )
            source = db.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
            if not source or source[1] != str(path) or source[2] != digest:
                raise RuntimeError("registry source identity conflict: " + source_key)
            artifact_key = f"RF-REV1-CLEANUP-EVIDENCE:{role}:{digest[:20]}"
            db.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                   VALUES('MIGRATIONBENCH',?,?,?,?,?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
                (artifact_key, "CLEANUP_AUDIT_EVIDENCE", role, "USER_CONFIRMED", digest, path.stat().st_size, note),
            )
            art = db.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if not art or art[1] != digest or art[2] != path.stat().st_size:
                raise RuntimeError("audit artifact identity conflict: " + artifact_key)
            destination = remote_path or ("MigrationBench/" + rel)
            db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (ROOT_ID, destination, utc_now()))
            path_id = db.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, destination)).fetchone()[0]
            db.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)""",
                (art[0], path_id, utc_now(), utc_now()),
            )
            db.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,'CLEANUP_BATCH',?,'CLEANUP_BATCH',?,'EVIDENCE_FOR',?)""",
                (source[0], BATCH_ID, BATCH_ID, note),
            )
            return int(source[0]), int(art[0])

        src_manifest, art_manifest = evidence(MANIFEST, "QE_RESTART_CANDIDATE_MANIFEST", "CANDIDATE_FILES_TSV", "Exact path, run, identity, coordinate, scheduler, and disposition crosswalk.")
        src_crosswalk, art_crosswalk = evidence(CROSSWALK, "QE_RESTART_RUN_GROUP_CROSSWALK", "RUN_GROUP_CROSSWALK_TSV", "113 job/run groups including coordinate and complete NEB image-path coverage.")
        src_summary, art_summary = evidence(SUMMARY, "QE_RESTART_AUDIT_SUMMARY", "AUDIT_SUMMARY_JSON", "Counts and explicit limitations for the current target root.")
        executor = AUDIT / "phase2_data/delete_revision1_restart_candidates_remote.py"
        src_executor, art_executor = evidence(executor, "QE_RESTART_METADATA_DELETE_EXECUTOR", "METADATA_DELETE_EXECUTOR", "Exact-path metadata checks, fresh all-user queue guard, no-follow unlink, and per-file absence receipts. Restart payload hashes are intentionally omitted.")

        snapshot_id = "RF-SCRATCH16-YI-REV1-RESTARTS-20261123T081721Z"
        db.execute(
            """INSERT OR IGNORE INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,files_seen,unreadable_count,logical_bytes,allocated_bytes,manifest_path,manifest_sha256,notes)
               VALUES(?,?,?,?,?,'revision1-restart-audit-v1','COMPLETE',?,0,?,?,?,?,?)""",
            (snapshot_id, ROOT_ID, "2026-11-23T08:17:21Z", source_observed_utc,
             "Targeted QE WFC/mixing path discovery, per-file metadata only, parseable retained QE XML coordinate join, Slurm/queue and NEB .path crosswalk; no restart payload hashes.",
             4298, logical, allocated, str(MANIFEST), sha256(MANIFEST),
             "COMPLETE metadata inventory for the exact restart-candidate set in revision1_migrationbench_runs; no restart payload SHA-256 pass; not a whole-/scratch16 inventory. The 574 formerly >500 MB paths are a separate historical disappearance gap."),
        )
        prior_batch = db.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()
        if prior_batch and tuple(prior_batch) != ("EVIDENCE_READY", logical):
            raise SystemExit(f"existing cleanup batch is not an idempotent evidence-ready rerun: {tuple(prior_batch)}")
        if not prior_batch:
            db.execute(
                """INSERT INTO cleanup_batch
                   (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,expected_bytes,evidence_artifact_id)
                   VALUES(?,?,?,?,?,'EVIDENCE_READY',?,?,?)""",
                (BATCH_ID, utc_now(), ROOT_ID,
                 "DELETE_QE_WFC_AND_MIX_AFTER_COORDINATE_EVIDENCE_METADATA_QUEUE_GUARDS",
                 "User explicitly directed that WFC and related large QE restart files be removed; the latest usable coordinates/path trajectories and the normal input/output evidence are retained.",
                 "User clarification: delete the remaining 3,817 WFC and 481 mixing restart files; coordinate XML is sufficient for migration and a fresh restart. No manifest contains an explicit WFC keep requirement; 7 groups lack manifests and are individually documented. Restart payload SHA-256 is intentionally not collected; deletion is guarded by exact inventoried metadata, retained coordinate evidence, and a fresh all-user queue snapshot.",
                 logical, art_manifest),
            )

        group_by_name = {x["run_group"]: x for x in groups}
        run_ids = {}
        for group in groups:
            run_id = "RF-MB-REV1-RESTART-JOB-" + group["run_job_id"]
            run_ids[group["run_group"]] = run_id
            input_lines = " ".join(json.dumps(group.get("input_key_lines", ""), ensure_ascii=False).split())
            lower = input_lines.lower()
            workflow = "QE_NEB" if "'neb'" in lower or '"neb"' in lower else ("QE_RELAX" if "'relax'" in lower or '"relax"' in lower else "QE_SCF_OR_UNKNOWN")
            notes = ("Exact sacct fields retained in scheduler_job_source_observation; cluster timestamps are left raw because sacct output omits timezone. "
                     "Coordinate XML proves atomic positions only, not scientific convergence. " +
                     f"sacct_state={group.get('slurm_state','')}; exit={group.get('slurm_exit_code','')}; workdir={group.get('slurm_workdir','')}")
            db.execute(
                """INSERT OR IGNORE INTO workflow_run
                   (run_id,project_id,run_label,workflow_type,method,system_key,source_root_id,source_relative_path,scheduler_state,scientific_status,notes)
                   VALUES(?,'MIGRATIONBENCH',?,?, 'Quantum ESPRESSO', ?, ?, ?, ?,'PROVENANCE_PARTIAL',?)""",
                (run_id, group["run_group"], workflow, group.get("slurm_job_name") or "", ROOT_ID,
                 "revision1_migrationbench_runs/" + group["run_group"], state(group.get("slurm_state", "")), notes),
            )
            observed = summary["sacct_capture"]["captured_utc"]
            db.execute(
                """INSERT OR IGNORE INTO scheduler_job_source_observation
                   (run_id,cluster_name,job_id,job_name,sacct_state,exit_code,submit_time_raw,start_time_raw,end_time_raw,workdir,observed_utc,source_artifact_id,notes)
                   VALUES(?,'Rockfish',?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, group["run_job_id"], group.get("slurm_job_name", ""), group.get("slurm_state", ""),
                 group.get("slurm_exit_code", ""), group.get("slurm_submit", ""), group.get("slurm_start", ""),
                 group.get("slurm_end", ""), group.get("slurm_workdir", ""), observed, art_crosswalk,
                 "Raw sacct timestamp strings preserved without an invented timezone; captured status is scheduler history, not a convergence assertion."),
            )

        # Register exact restart files and every coordinate / NEB path artifact used to justify deletion.
        coordinate_artifacts = {}
        neb_artifacts = {}
        for row in candidates:
            p = row["absolute_path"]
            hv = by_path[p]
            for key in ("device", "inode", "uid", "gid", "mode", "nlink", "size_bytes", "allocated_bytes", "mtime_ns"):
                expected = int(row[key]) if key not in ("mode",) else row[key]
                actual = int(hv[key]) if key != "mode" else hv[key]
                if actual != expected:
                    raise RuntimeError(f"manifest metadata mismatch {key}: {p}")
            candidate_rel = relative_to_yi(p)
            db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (ROOT_ID, candidate_rel, utc_now()))
            path_id = db.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, candidate_rel)).fetchone()[0]
            group_run = run_ids[row["run_group"]]
            kind = row["candidate_kind"]
            role = "QE_WFC_RESTART" if kind == "QE_WFC" else "QE_MIXING_RESTART"
            akey = "RF-REV1-QE-RESTART:" + hashlib.sha256(p.encode("utf-8")).hexdigest()[:24]
            db.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                   VALUES('MIGRATIONBENCH',?,'QE_RESTART_FILE',?,'USER_CONFIRMED',NULL,?,'PROVENANCE_PARTIAL','TEMPORARY',?)""",
                (akey, role, int(hv["size_bytes"]),
                 "METADATA_ONLY: no restart payload SHA-256 collected by user direction; inventoried identity and size recorded. Latest parsed QE atomic positions remain at " + row["coordinate_evidence_path"] + ". User authorizes deletion; restart is not needed for migration/fresh-start reproduction."),
            )
            aid, digest, byte_size = db.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (akey,)).fetchone()
            if digest is not None or byte_size != int(hv["size_bytes"]):
                raise RuntimeError("candidate artifact key collision: " + p)
            db.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)""",
                (aid, path_id, source_observed_utc, source_observed_utc),
            )
            db.execute(
                """INSERT OR IGNORE INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,sha256,hash_scope,observation_status,link_count)
                   VALUES(?,?,'FILE',?,?,?,?,?,?,?,?,NULL,'NONE','PRESENT',?)""",
                (snapshot_id, path_id, int(hv["uid"]), int(hv["gid"]), hv["mode"], int(hv["size_bytes"]),
                 int(hv["allocated_bytes"]), int(hv["mtime_ns"]), str(hv["inode"]), str(hv["device"]), int(hv["nlink"])),
            )
            db.execute(
                """INSERT OR IGNORE INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes)
                   VALUES(?,?,'DELETE',?,'CLEARED',NULL,?,?,'EVIDENCE_READY',?)""",
                (BATCH_ID, aid,
                 row["disposition_reason"] + " Coordinate status: " + row["coordinate_xml_parse"] + "; atoms=" + row["coordinate_nat"] + "; evidence=" + row["coordinate_evidence_path"],
                 int(hv["size_bytes"]), int(hv["mtime_ns"]),
                 "METADATA_ONLY: payload SHA-256 intentionally omitted. Stable identity checked again by the deleter; fresh all-user queue guard runs immediately before deletion."),
            )
            db.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,'INTERMEDIATE',0,?)""",
                (group_run, aid, role, "Temporary restart payload; deleted only after coordinate evidence is retained."),
            )
            db.execute(
                """INSERT OR IGNORE INTO cleanup_candidate_context
                   (cleanup_batch_id,artifact_id,run_id,candidate_kind,manifest_paths,manifest_present,manifest_restart_required_flag,manifest_restart_required_basis,input_paths,output_log_paths,coordinate_evidence_path,coordinate_evidence_sha256,coordinate_xml_parse,coordinate_nat,coordinate_evidence_kind,neb_path_evidence,neb_image_ids,neb_complete_image_coverage,source_queue_capture_utc,source_queue_status,source_queue_workdir_overlap,disposition_reason,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (BATCH_ID, aid, group_run, kind, row["manifest_paths"], int(row["manifest_present"] == "True"),
                 row["manifest_restart_required_flag"], row["manifest_restart_required_basis"], row["input_paths"],
                 row["output_log_paths"], row["coordinate_evidence_path"], row["coordinate_evidence_sha256"],
                 row["coordinate_xml_parse"], int(row["coordinate_nat"] or 0) or None, row["coordinate_evidence_kind"],
                 row["neb_path_evidence_paths_hashes"], row["neb_image_ids"], int(row["neb_complete_image_coverage"] == "True"),
                 row["queue_capture_utc"], row["queue_parse_status"], int(row["live_queue_workdir_overlap"] == "True"),
                 row["disposition_reason"], "Historical queue capture only; the destructive executor must take a fresh queue snapshot."),
            )

            coord_key = (row["coordinate_evidence_path"], row["coordinate_evidence_sha256"])
            if coord_key not in coordinate_artifacts:
                cpath = relative_to_yi(coord_key[0])
                db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (ROOT_ID, cpath, utc_now()))
                cpath_id = db.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (ROOT_ID, cpath)).fetchone()[0]
                ckey = "RF-REV1-COORDINATE-XML:" + hashlib.sha256(coord_key[0].encode("utf-8")).hexdigest()[:24]
                db.execute(
                    """INSERT OR IGNORE INTO artifact
                       (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,scientific_status,retention_class,notes)
                       VALUES('MIGRATIONBENCH',?,'QE_COORDINATE_XML','ATOMIC_POSITION_EVIDENCE','USER_CONFIRMED',?,'PROVENANCE_PARTIAL','REPRODUCIBLE_INTERMEDIATE',?)""",
                    (ckey, coord_key[1], "Parsed QE XML atomic_positions; coordinate recovery only, no convergence or acceptance assertion."),
                )
                caid, csha = db.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (ckey,)).fetchone()
                if csha != coord_key[1]: raise RuntimeError("coordinate XML hash collision: " + coord_key[0])
                db.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)", (caid,cpath_id,source_observed_utc,source_observed_utc))
                coordinate_artifacts[coord_key] = caid
            caid = coordinate_artifacts[coord_key]
            db.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,'ATOMIC_COORDINATE_XML','OUTPUT',1,?)", (group_run, caid, "Retained coordinate evidence used to justify restart deletion; parse status="+row["coordinate_xml_parse"]+"; nat="+row["coordinate_nat"]+"."))

            for path_info in json.loads(row["neb_path_evidence_paths_hashes"] or "[]"):
                nkey = (path_info["path"], path_info["sha256"])
                if nkey not in neb_artifacts:
                    nrel = relative_to_yi(nkey[0])
                    db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (ROOT_ID,nrel,utc_now()))
                    npath_id = db.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",(ROOT_ID,nrel)).fetchone()[0]
                    akey2 = "RF-REV1-NEB-PATH:"+hashlib.sha256(nkey[0].encode("utf-8")).hexdigest()[:24]
                    db.execute("""INSERT OR IGNORE INTO artifact
                       (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                       VALUES('MIGRATIONBENCH',?,'QE_NEB_PATH_FILE','ALL_IMAGE_COORDINATE_TRAJECTORY','USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL','COMPACT_RESULT',?)""",
                       (akey2,nkey[1],int(path_info["size"]),"Parsed path file with image ids "+json.dumps(path_info.get("image_ids",[]))+"; coordinate trajectory only."))
                    naid,nsha,nbytes=db.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?",(akey2,)).fetchone()
                    if nsha!=nkey[1] or nbytes!=int(path_info["size"]): raise RuntimeError("NEB path identity conflict: "+nkey[0])
                    db.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",(naid,npath_id,source_observed_utc,source_observed_utc))
                    neb_artifacts[nkey]=naid
                naid=neb_artifacts[nkey]
                role_n="NEB_COORDINATE_PATH_"+hashlib.sha256(nkey[0].encode("utf-8")).hexdigest()[:8]
                db.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'OUTPUT',1,?)",(group_run,naid,role_n,"Retained QE NEB path coordinates; parsed image ids="+json.dumps(path_info.get("image_ids",[]))+"."))

        # Index source receipt materials in the active catalog and bind them to the cleanup batch.
        db.execute("UPDATE cleanup_batch SET status='EVIDENCE_READY',evidence_artifact_id=?,expected_bytes=? WHERE cleanup_batch_id=?", (art_manifest,logical,BATCH_ID))
        if db.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE' AND deletion_state='EVIDENCE_READY'",(BATCH_ID,)).fetchone()[0]!=4298:
            raise RuntimeError("SQL evidence-ready cleanup item count is not exactly 4,298")
        if db.execute("SELECT count(*) FROM cleanup_candidate_context WHERE cleanup_batch_id=?",(BATCH_ID,)).fetchone()[0]!=4298:
            raise RuntimeError("SQL cleanup context count is incomplete")
        if db.execute("SELECT count(*) FROM workflow_run WHERE run_id LIKE 'RF-MB-REV1-RESTART-JOB-%'").fetchone()[0] < 113:
            raise RuntimeError("SQL workflow run count is incomplete")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign_key_check failed during import")
        db.commit()
        print(json.dumps({"cleanup_batch_id":BATCH_ID,"state":"EVIDENCE_READY_NO_DELETION",
                          "candidate_count":4298,"logical_bytes":logical,"allocated_bytes":allocated,
                          "run_groups":len(groups),"coordinate_xml_artifacts":len(coordinate_artifacts),
                          "neb_path_artifacts":len(neb_artifacts),"hash_scope":"METADATA_ONLY_NO_RESTART_CONTENT_SHA256",
                          "queue_capture_utc":args.queue_source_utc,"deletion_performed":False},sort_keys=True))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
