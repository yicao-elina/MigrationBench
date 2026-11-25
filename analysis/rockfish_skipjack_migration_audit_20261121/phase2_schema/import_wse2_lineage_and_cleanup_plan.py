#!/usr/bin/env python3
"""Load WSe2 run lineage and stage (but do not execute) WFC cleanup."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE.parent
ROOT_DIR = AUDIT_DIR.parents[1]
DB = AUDIT_DIR / "provenance_catalog.sqlite"
S4_DIR = AUDIT_DIR / "phase2_scratch4"
RUNS_TSV = S4_DIR / "wse2_run_lineage.tsv"
CANDIDATES_TSV = S4_DIR / "candidate_wfc_manifest.tsv"
MANIFEST_JSON = S4_DIR / "WSE2_REPRODUCTION_MANIFEST.json"
EVIDENCE_DIR = S4_DIR / "sbatch_scripts"
SOURCE_ROOTS = {
    "RF-SCRATCH4-YI": PurePosixPath("/scratch4/pclancy3/yi"),
    "RF-HOME-YI": PurePosixPath("/home/ycao73"),
    "LOCAL-MAIN": PurePosixPath(str(ROOT_DIR)),
}
OBSERVED_UTC = "2026-11-22T18:21:53Z"
BATCH_ID = "RF-S4-WSE2-WFC-CLEANUP-20261122"
RECIPE_VERSION = "curated-rerun-2026-11-22-v1"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def remote_key(root_id: str, absolute_path: str) -> str:
    relative = str(PurePosixPath(absolute_path).relative_to(SOURCE_ROOTS[root_id]))
    return f"{root_id}:{relative}"


def get_remote_artifact(connection: sqlite3.Connection, root_id: str, absolute_path: str) -> tuple[int, int]:
    relative = str(PurePosixPath(absolute_path).relative_to(SOURCE_ROOTS[root_id]))
    row = connection.execute(
        """SELECT a.artifact_id,p.path_id FROM artifact a
           JOIN artifact_location l USING(artifact_id)
           JOIN storage_path p USING(path_id)
           WHERE a.artifact_key=? AND p.root_id=? AND p.relative_path=?""",
        (remote_key(root_id, absolute_path), root_id, relative),
    ).fetchone()
    if not row:
        raise RuntimeError(f"Path missing from imported inventory: {absolute_path}")
    return int(row[0]), int(row[1])


def ensure_local_artifact(connection: sqlite3.Connection, absolute_path: Path, role: str,
                          retention: str, observed: str, local_snapshot_id: str,
                          share_artifact_id: int | None = None) -> int:
    absolute = absolute_path.resolve()
    root = SOURCE_ROOTS["LOCAL-MAIN"]
    relative = str(PurePosixPath(str(absolute)).relative_to(root))
    key = f"LOCAL-MAIN:{relative}"
    stat_result = absolute.stat()
    digest = sha256(absolute)
    connection.execute(
        "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)",
        (relative, observed),
    )
    path_id = connection.execute(
        "SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?",
        (relative,),
    ).fetchone()[0]
    mode = f"0{stat.S_IMODE(stat_result.st_mode):04o}"
    connection.execute(
        """INSERT OR IGNORE INTO inventory_observation
           (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,
            mtime_ns,inode,device,sha256,hash_scope,observation_status,link_count)
           VALUES(?,?, 'FILE',NULL,NULL,?,?,?,?,?,?,?,'FULL_FILE','PRESENT',?)""",
        (local_snapshot_id, path_id, mode, stat_result.st_size,
         getattr(stat_result, "st_blocks", 0) * 512, stat_result.st_mtime_ns,
         str(stat_result.st_ino), str(stat_result.st_dev), digest,
         stat_result.st_nlink),
    )
    if share_artifact_id is None:
        connection.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                scientific_status,retention_class,notes)
               VALUES('WSE2',?,'FILE',?,'USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL',?,?)""",
            (key, role, digest, stat_result.st_size, retention,
             "Local audit evidence; full-file SHA256 captured by targeted audit."),
        )
        artifact_id = connection.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()[0]
    else:
        artifact_id = share_artifact_id
        existing_hash = connection.execute("SELECT sha256 FROM artifact WHERE artifact_id=?", (artifact_id,)).fetchone()[0]
        if existing_hash != digest:
            raise RuntimeError(f"Local copy hash differs from its Rockfish counterpart: {absolute}")
    connection.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""",
        (artifact_id, path_id, observed, observed),
    )
    return int(artifact_id)


def set_remote_artifact(connection: sqlite3.Connection, artifact_id: int, digest: str,
                        size: int | None, role: str, retention: str,
                        owner_status: str = "UNKNOWN") -> None:
    prior = connection.execute("SELECT sha256 FROM artifact WHERE artifact_id=?", (artifact_id,)).fetchone()[0]
    if prior and prior != digest:
        raise RuntimeError(f"Conflicting content hash for artifact_id={artifact_id}")
    connection.execute(
        """UPDATE artifact SET sha256=?, byte_size=COALESCE(byte_size,?), semantic_role=?,
                  retention_class=?, owner_status=? WHERE artifact_id=?""",
        (digest, size, role, retention, owner_status, artifact_id),
    )


def add_run_artifact(connection: sqlite3.Connection, run_id: str, artifact_id: int,
                      role: str, direction: str, required: bool = False) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO run_artifact
           (run_id,artifact_id,role,direction,required_for_reproduction)
           VALUES(?,?,?,?,?)""",
        (run_id, artifact_id, role, direction, int(required)),
    )


def main() -> None:
    runs = read_tsv(RUNS_TSV)
    candidates = read_tsv(CANDIDATES_TSV)
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    if not runs or len(candidates) != 8:
        raise SystemExit("Expected three WSe2 runs and exactly eight WFC candidates.")
    observed = now_utc()
    source_manifest_hash = sha256(MANIFEST_JSON)

    with sqlite3.connect(DB) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("SELECT COUNT(*) FROM workflow_run WHERE run_id LIKE 'RF-SCRATCH4-WSE2-%'").fetchone()[0]:
            raise SystemExit("WSe2 runs already exist in the catalog; refusing duplicate import.")
        if connection.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone():
            raise SystemExit("Cleanup batch already exists; refusing duplicate import.")

        connection.execute(
            """INSERT OR IGNORE INTO campaign
               (campaign_id,project_id,campaign_name,start_date,end_date,status,notes)
               VALUES('WSE2_VC_RELAX_202412','WSE2','December 2024 WSe2 QE variable-cell relaxations',
                      '2024-12-20','2024-12-21','COMPLETE',?)""",
            ("Historical run identity is date-bounded because Slurm numeric job IDs were reused in 2026.",),
        )

        connection.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
               VALUES('LOCAL-WSE2-EVIDENCE-20261122','LOCAL-MAIN',?,?,?,?,'PARTIAL',4,0,0,0,NULL,NULL,?,?,?)""",
            (observed, observed, "Targeted stat and SHA-256 of local WSe2 audit evidence", "python-stat+sha256",
             str(MANIFEST_JSON.resolve()), source_manifest_hash,
             "Partial local snapshot for three verified submitted-script copies plus the curated reproduction record."),
        )
        connection.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES('wse2_curated_reproduction_manifest_20261122',?,'CURATED_REPRODUCTION_MANIFEST',?,?,?,1,?)""",
            (str(MANIFEST_JSON.resolve()), source_manifest_hash, observed, observed,
             "Newly curated reproduction record; explicitly not an original historical run manifest."),
        )
        manifest_artifact_id = ensure_local_artifact(
            connection, MANIFEST_JSON, "REPRODUCTION_MANIFEST", "PERMANENT_SOURCE",
            observed, "LOCAL-WSE2-EVIDENCE-20261122",
        )

        run_artifacts: dict[str, dict[str, int]] = {}
        candidate_ids: dict[str, tuple[int, int]] = {}
        candidates_by_run: dict[str, list[dict[str, str]]] = {}
        run_id_by_directory = {row["run_directory"]: row["run_id"] for row in runs}
        for candidate in candidates:
            mapped_run_id = run_id_by_directory.get(candidate["run_path"])
            if not mapped_run_id:
                raise RuntimeError(f"Candidate run path has no run lineage row: {candidate['run_path']}")
            candidates_by_run.setdefault(mapped_run_id, []).append(candidate)

        run_total_bytes = 0
        for row in runs:
            run_id = row["run_id"]
            run_dir = row["run_directory"]
            job_start = row["start_utc"] + "Z" if not row["start_utc"].endswith("Z") else row["start_utc"]
            job_end = row["end_utc"] + "Z" if not row["end_utc"].endswith("Z") else row["end_utc"]
            connection.execute(
                """INSERT INTO workflow_run
                   (run_id,project_id,campaign_id,run_label,workflow_type,method,system_key,
                    source_root_id,source_relative_path,scheduler_state,scientific_status,start_utc,end_utc,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, "WSE2", "WSE2_VC_RELAX_202412", row["run_id"], "QE_VC_RELAX",
                 "Quantum ESPRESSO 7.2 / vc-relax", f"WSe2_n{row['atom_count']}",
                 "RF-SCRATCH4-YI", str(PurePosixPath(run_dir).relative_to(SOURCE_ROOTS["RF-SCRATCH4-YI"])),
                 "COMPLETED", "PROVENANCE_PARTIAL", job_start, job_end,
                 "Run and output lineage are supported by hashes and Slurm/QE evidence; paper-claim and formal materials acceptance are unresolved."),
            )
            connection.execute(
                """INSERT INTO slurm_job
                   (run_id,cluster_name,job_id,array_task_id,submitted_or_started_utc,job_name,partition_name,account_name)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (run_id, "Rockfish", row["job_id"], "", job_start, "WSe2-2d", "parallel", "pclancy3"),
            )
            connection.execute(
                """INSERT INTO slurm_state_observation
                   (slurm_job_key,observed_utc,state,exit_code,start_utc,end_utc,source_command,raw_record_path,notes)
                   SELECT slurm_job_key,?,?,?,?,?,?,?,? FROM slurm_job
                   WHERE cluster_name='Rockfish' AND job_id=? AND submitted_or_started_utc=?""",
                (row["scheduler_state_observed_utc"], row["scheduler_state"], row["exit_code"],
                 job_start, job_end, "Date-bounded sacct query for this 2024 job instance; do not join by ID alone.",
                 str((S4_DIR / "quota_and_slurm_candidates.txt").resolve()),
                 "Job ID was reused in 2026 for unrelated jobs; historical state is date-bounded." ,
                 row["job_id"], job_start),
            )

            items = {
                "input": (row["input_path"], row["input_sha256"], int(row["input_bytes"]), "QE_INPUT", "REPRODUCIBLE_INPUT", "INPUT"),
                "script": (row["launch_script_path"], row["launch_script_sha256"], None, "SLURM_SUBMISSION_SCRIPT", "REPRODUCIBLE_INPUT", "INPUT"),
                "slurm_log": (row["slurm_log_path"], row["slurm_log_sha256"], None, "SLURM_LOG", "PUBLICATION_EVIDENCE", "LOG"),
                "qe_output": (row["qe_output_path"], row["qe_output_sha256"], int(row["qe_output_bytes"]), "QE_TEXT_OUTPUT", "COMPACT_RESULT", "OUTPUT"),
                "prefix_xml": (row["prefix_xml_path"], row["prefix_xml_sha256"], None, "QE_XML_OUTPUT", "COMPACT_RESULT", "OUTPUT"),
                "save_xml": (row["save_data_xml_path"], row["save_data_xml_sha256"], None, "QE_XML_OUTPUT", "COMPACT_RESULT", "OUTPUT"),
                "charge_density": (row["charge_density_path"], row["charge_density_sha256"], int(row["charge_density_bytes"]), "QE_CHARGE_DENSITY", "COMPACT_RESULT", "OUTPUT"),
                "pseudo_w": (row["pseudo_w_path"], row["pseudo_w_sha256"], int(row["pseudo_w_bytes"]), "PSEUDOPOTENTIAL", "PERMANENT_SOURCE", "INPUT"),
                "pseudo_se": (row["pseudo_se_path"], row["pseudo_se_sha256"], int(row["pseudo_se_bytes"]), "PSEUDOPOTENTIAL", "PERMANENT_SOURCE", "INPUT"),
            }
            artifact_ids: dict[str, int] = {}
            for label, (path, digest, size, role, retention, direction) in items.items():
                root_id = "RF-HOME-YI" if label.startswith("pseudo") else "RF-SCRATCH4-YI"
                if root_id == "RF-HOME-YI":
                    relative = str(PurePosixPath(path).relative_to(SOURCE_ROOTS[root_id]))
                    connection.execute(
                        "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
                        (root_id, relative, observed),
                    )
                    path_id = connection.execute(
                        "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
                        (root_id, relative),
                    ).fetchone()[0]
                    key = remote_key(root_id, path)
                    connection.execute(
                        """INSERT OR IGNORE INTO artifact
                           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                            scientific_status,retention_class,notes)
                           VALUES('WSE2',?,'FILE',?,'UNKNOWN',?,?,'PROVENANCE_PARTIAL',?,?)""",
                        (key, role, digest, size, retention,
                         "Referenced as an exact QE input dependency; current availability recorded by the run-level audit."),
                    )
                    artifact_id = connection.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()[0]
                    connection.execute(
                        """INSERT OR IGNORE INTO artifact_location
                           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                           VALUES(?,?,'PRESENT',?,?)""",
                        (artifact_id, path_id, OBSERVED_UTC, OBSERVED_UTC),
                    )
                else:
                    artifact_id, _ = get_remote_artifact(connection, root_id, path)
                    set_remote_artifact(connection, artifact_id, digest, size, role, retention)
                artifact_ids[label] = artifact_id
                add_run_artifact(connection, run_id, artifact_id, label, direction,
                                 required=(label in {"input", "script", "pseudo_w", "pseudo_se"}))

            local_script = S4_DIR / row["launch_script_local_copy"]
            local_script_artifact = ensure_local_artifact(
                connection, local_script, "SLURM_SUBMISSION_SCRIPT", "REPRODUCIBLE_INPUT",
                observed, "LOCAL-WSE2-EVIDENCE-20261122", share_artifact_id=artifact_ids["script"],
            )
            add_run_artifact(connection, run_id, local_script_artifact, "local_verified_script_copy", "PROVENANCE", True)
            add_run_artifact(connection, run_id, manifest_artifact_id, "curated_reproduction_manifest", "PROVENANCE", True)

            wfc_ids: list[int] = []
            for candidate in candidates_by_run[run_id]:
                artifact_id, _ = get_remote_artifact(connection, "RF-SCRATCH4-YI", candidate["artifact_path"])
                set_remote_artifact(connection, artifact_id, candidate["pre_delete_sha256"],
                                     int(candidate["size_bytes"]), "QE_WFC_RESTART",
                                     "REPRODUCIBLE_INTERMEDIATE", "USER_CONFIRMED")
                add_run_artifact(connection, run_id, artifact_id, "qe_wfc_restart", "INTERMEDIATE")
                candidate_ids[candidate["candidate_id"]] = (artifact_id, int(candidate["size_bytes"]))
                wfc_ids.append(artifact_id)
                run_total_bytes += int(candidate["size_bytes"])

            main_output_id = artifact_ids["qe_output"]
            for parent_label in ("input", "script", "pseudo_w", "pseudo_se"):
                connection.execute(
                    """INSERT INTO derivation_edge
                       (parent_artifact_id,child_artifact_id,run_id,operation,tool_name,tool_version,
                        script_artifact_id,parameter_json,provenance_status,notes)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (artifact_ids[parent_label], main_output_id, run_id, "QE vc-relax",
                     "Quantum ESPRESSO", "7.2", artifact_ids["script"],
                     json.dumps({"ntasks": 48, "calculation": "vc-relax", "input": row["input_path"]}),
                     "LINEAGE_SUPPORTED", "Input, launch script, QE output and pseudopotential hashes are recorded."),
                )

            staging = (
                f"RUN_DIR=$(mktemp -d -p /scratch4/pclancy3/yi/3.WSe2 'repro-{run_id}-XXXXXXXX')\n"
                f"cp {json.dumps(row['input_path'])} \"$RUN_DIR/\"\n"
                f"cp {json.dumps(row['launch_script_path'])} \"$RUN_DIR/\"\n"
                "cd \"$RUN_DIR\"\n"
                "sbatch sbatch-relax+nscf+bands.slurm"
            )
            connection.execute(
                """INSERT INTO reproduction_recipe
                   (run_id,recipe_version,exact_command,working_directory,executable_version,
                    environment_description,module_list,script_artifact_id,scheduler_script_artifact_id,
                    input_manifest_artifact_id,recipe_status,validation_note)
                   VALUES(?,?,?,?,?,?,?,?,?,?,'DOCUMENTED',?)""",
                (run_id, RECIPE_VERSION, staging, "/scratch4/pclancy3/yi/3.WSe2/",
                 "QE 7.2 at /data/apps/extern/qe/7.2/bin/pw.x; 48 MPI ranks",
                 "Login shell; module reset; source /data/apps/go.sh; `ml intel`; `ml qe/7.2`. Shared go.sh was not hashed; bitwise identity is not asserted.",
                 "intel; qe/7.2", artifact_ids["script"], artifact_ids["script"],
                 manifest_artifact_id,
                 "The existing output says the initial wavefunctions were randomized atomic wfcs. The input uses outdir='./' and omits restart_mode/startingwfc; rerun in a fresh directory. NSCF and bands commands are commented out."),
            )
            recipe_id = connection.execute(
                "SELECT recipe_id FROM reproduction_recipe WHERE run_id=? AND recipe_version=?",
                (run_id, RECIPE_VERSION),
            ).fetchone()[0]
            for input_label in ("input", "pseudo_w", "pseudo_se"):
                connection.execute(
                    "INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",
                    (recipe_id, artifact_ids[input_label], input_label),
                )

            markers = row["output_acceptance_markers"]
            for gate_id, status, description in (
                ("slurm_exit_0", "PASS" if row["exit_code"] == "0:0" else "FAIL", "Date-bounded Slurm exit code equals 0:0."),
                ("qe_job_done", "PASS" if "JOB DONE" in markers else "FAIL", "QE text output ends with JOB DONE."),
                ("qe_scf_converged", "PASS" if "scf cycles" in markers else "UNKNOWN", "Parsed output marker for SCF completion."),
                ("qe_bfgs_converged", "PASS" if "bfgs converged" in markers else "UNKNOWN", "Parsed output marker for BFGS convergence; does not confer paper acceptance."),
            ):
                connection.execute(
                    """INSERT INTO quality_gate(run_id,gate_id,status,checked_utc,evidence_artifact_id,rule_description)
                       VALUES(?,?,?,?,?,?)""",
                    (run_id, gate_id, status, OBSERVED_UTC, artifact_ids["qe_output"], description),
                )

            dup_id = None
            if row["prefix_xml_sha256"] == row["save_data_xml_sha256"]:
                connection.execute(
                    """INSERT INTO duplicate_group(detection_run_id,match_method,verification_status,disposition,rationale)
                       VALUES(?, 'SHA256','VERIFIED_IDENTICAL','KEEP_BOTH',?)""",
                    (f"{run_id}-xml-pair", "The two QE XML paths have matching SHA-256 but remain at both paths because the existing calculation tree and parsers may refer to either."),
                )
                dup_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
                connection.execute("INSERT INTO duplicate_member(duplicate_group_id,artifact_id,canonical_member) VALUES(?,?,1)", (dup_id, artifact_ids["prefix_xml"]))
                connection.execute("INSERT INTO duplicate_member(duplicate_group_id,artifact_id,canonical_member) VALUES(?,?,0)", (dup_id, artifact_ids["save_xml"]))

            run_artifacts[run_id] = artifact_ids

        if run_total_bytes != sum(int(c["size_bytes"]) for c in candidates):
            raise RuntimeError("WFC candidate byte total changed during import")

        connection.execute(
            """INSERT INTO cleanup_batch
               (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,
                authorization_note,expected_bytes,evidence_artifact_id)
               VALUES(?,?,?,?,?,'EVIDENCE_READY',?,?,?)""",
            (BATCH_ID, observed, "RF-SCRATCH4-YI", "DELETE_REPRODUCIBLE_QE_WFC_RESTARTS",
             "QE WFC restart state is regenerable from preserved input, exact submitted script, pseudopotentials, QE runtime reference, Slurm lineage, output evidence, and a curated fresh-directory reproduction recipe.",
             "Explicit user authorization: remove reproducible WFC intermediate files after registering paths, hashes, and rerun method. No files have been deleted yet.",
             run_total_bytes, manifest_artifact_id),
        )
        for candidate in candidates:
            artifact_id, size = candidate_ids[candidate["candidate_id"]]
            connection.execute(
                """INSERT INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,
                    pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes)
                   VALUES(?,?,'DELETE',?,'CLEARED',?,?,?,'EVIDENCE_READY',?)""",
                (BATCH_ID, artifact_id,
                 "QE wavefunction restart; fresh input defaults produced randomized atomic wavefunctions in the historical run, and the existing complete output tree plus exact rerun recipe are retained.",
                 candidate["pre_delete_sha256"], size, int(candidate["mtime_epoch_ns"]),
                 "UID/GID 3901/3001 and mode/size verified immediately before SQL staging; deletion requires a final preflight hash and queue check."),
            )
        connection.commit()

    with sqlite3.connect(DB) as connection:
        print("WSe2 runs:", connection.execute("SELECT COUNT(*) FROM workflow_run WHERE campaign_id='WSE2_VC_RELAX_202412'").fetchone()[0])
        print("WFC artifacts:", connection.execute("SELECT COUNT(*) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()[0])
        print("WFC bytes planned:", connection.execute("SELECT SUM(pre_delete_size_bytes) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()[0])
        print("Cleanup batch:", connection.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone())
        print("WFC deletion gate rows:", connection.execute("""SELECT COUNT(*) FROM cleanup_item c JOIN artifact a USING(artifact_id)
                     WHERE c.cleanup_batch_id=? AND c.disposition='DELETE' AND c.owner_clearance='CLEARED'
                       AND a.retention_class='REPRODUCIBLE_INTERMEDIATE' AND a.owner_status='USER_CONFIRMED'""", (BATCH_ID,)).fetchone()[0])


if __name__ == "__main__":
    main()
