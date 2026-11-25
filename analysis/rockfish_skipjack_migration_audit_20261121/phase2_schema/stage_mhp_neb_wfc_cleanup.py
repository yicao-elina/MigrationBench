#!/usr/bin/env python3
"""Register MHP NEB evidence and stage only its reproducible WFC files."""

import csv
import hashlib
import json
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
DATA = AUDIT / "phase2_data"
MANIFEST = DATA / "MHP_NEUTRAL_NEB_REPRODUCTION_MANIFEST_20261122.json"
HASHES = DATA / "mhp_490_wfc_predelete_sha256_remote_20261122.jsonl"
METADATA = DATA / "mhp_wfc_and_result_artifact_metadata_20261122.tsv"
DECISION = DATA / "mhp_wfc_cleanup_decision_20261122.md"
SACCT = DATA / "mhp_10_wfc_sacct_recheck_20261122.txt"
QUEUE = DATA / "mhp_490_wfc_preflight_queue_20261122.txt"
ROOT_ID = "RF-DATA-MHP-DEFECT"
ROOT_PATH = PurePosixPath("/data/pclancy3/MHP_defect")
HOME_ID = "RF-HOME-YI"
HOME_PATH = PurePosixPath("/home/ycao73")
RUN_ID = "RF-MHP-NEB222-NEUTRAL-10-30"
CAMPAIGN_ID = "MHP_NEUTRAL_NEB_202501"
BATCH_ID = "RF-DATA-MHP-NEB-WFC-CLEANUP-20261122"
RECIPE_VERSION = "curated-fresh-start-2026-11-22-v1"


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def register_local_artifact(con, path, observed, semantic_role, retention="PERMANENT_SOURCE"):
    path = path.resolve()
    root = Path(con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()[0])
    relative = str(path.relative_to(root))
    st = path.stat()
    digest = sha256(path)
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)", (relative, observed))
    path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (relative,)).fetchone()[0]
    key = "LOCAL-MAIN:" + relative
    con.execute("""INSERT OR IGNORE INTO artifact
        (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
        VALUES('MHP_DEFECT',?,'FILE',?,'USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL',?,?)""",
        (key, semantic_role, digest, st.st_size, retention, "Local immutable audit evidence; SHA-256 captured during catalog import."))
    aid = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()[0]
    con.execute("UPDATE artifact SET sha256=?,byte_size=?,owner_status='USER_CONFIRMED',semantic_role=?,retention_class=? WHERE artifact_id=?", (digest, st.st_size, semantic_role, retention, aid))
    con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)", (aid, path_id, observed, observed))
    return aid


def register_remote_artifact(con, root_id, root_path, absolute_path, observed, project_id,
                             semantic_role, retention, size=None, digest=None, owner="USER_CONFIRMED"):
    absolute = PurePosixPath(absolute_path)
    relative = str(absolute.relative_to(root_path))
    path_row = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, relative)).fetchone()
    if not path_row:
        raise RuntimeError("Path absent from imported inventory: " + absolute_path)
    path_id = int(path_row[0])
    key = root_id + ":" + relative
    prior = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (key,)).fetchone()
    if prior and prior[1] and digest and prior[1] != digest:
        raise RuntimeError("Conflicting SHA-256 for " + absolute_path)
    con.execute("""INSERT OR IGNORE INTO artifact
        (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
        VALUES(?,?,'FILE',?,?,?,?,?,?,?)""",
        (project_id, key, semantic_role, owner, digest, size, "PROVENANCE_PARTIAL", retention,
         "Cataloged from imported metadata inventory; no scientific acceptance is implied."))
    aid = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()[0]
    con.execute("UPDATE artifact SET project_id=COALESCE(project_id,?),sha256=COALESCE(?,sha256),byte_size=COALESCE(?,byte_size),semantic_role=?,owner_status=?,retention_class=? WHERE artifact_id=?",
                (project_id, digest, size, semantic_role, owner, retention, aid))
    con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)", (aid, path_id, observed, observed))
    return int(aid)


def main():
    observed = now_utc()
    manifest = json.loads(MANIFEST.read_text())
    hash_rows = [json.loads(line) for line in HASHES.read_text().splitlines() if line.strip()]
    hashes = [row for row in hash_rows if row.get("record") == "file"]
    if manifest["wfc_file_count"] != 490 or len(hashes) != 490 or manifest["wfc_logical_bytes"] != 26235975200:
        raise SystemExit("WFC manifest/hash rows are incomplete or totals differ.")
    if any(row.get("record") == "error" for row in hash_rows):
        raise SystemExit("Remote WFC hash run contains an error record.")
    if not (Path(DATA / "mhp_490_wfc_preflight_queue_20261122.txt").exists() and Path(DATA / "mhp_10_wfc_queue_workdir_recheck_20261122.txt").exists()):
        raise SystemExit("Queue evidence is missing.")
    queue_text = QUEUE.read_text() + (DATA / "mhp_10_wfc_queue_workdir_recheck_20261122.txt").read_text()
    if "/data/pclancy3/MHP_defect/yi/neb222-neutral-10-30" in queue_text or "/scratch16/pclancy3/yi/defects_scripts/2-0.neutral/2.neb/10-30" in queue_text:
        raise SystemExit("A queued Slurm workdir overlaps the MHP run.")
    meta = read_tsv(METADATA)
    wfc_meta = [row for row in meta if row["KIND"] in ("parallel_wfc_shard", "neb_save_wfc1")]
    result_meta = [row for row in meta if row["KIND"] == "result_evidence"]
    hash_by_path = {row["path"]: row for row in hashes}
    if len(wfc_meta) != 490 or len(result_meta) != 40:
        raise SystemExit("WFC/result metadata counts differ from reviewed inventory.")
    for row in wfc_meta:
        hashed = hash_by_path.get(row["PATH"])
        if not hashed or int(row["SIZE_BYTES"]) != hashed["size"] or int(row["MTIME_NS"]) != hashed["mtime_ns"]:
            raise SystemExit("WFC pre-delete hash does not match inventory row: " + row["PATH"])

    manifest_hash = sha256(MANIFEST)
    hashes_hash = sha256(HASHES)
    with sqlite3.connect(DB, timeout=120) as con:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=120000")
        if con.execute("SELECT 1 FROM workflow_run WHERE run_id=?", (RUN_ID,)).fetchone():
            raise SystemExit("MHP workflow run already exists.")
        if con.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone():
            raise SystemExit("MHP cleanup batch already exists.")
        con.execute("BEGIN IMMEDIATE")
        con.execute("INSERT OR IGNORE INTO campaign(campaign_id,project_id,campaign_name,start_date,end_date,status,notes) VALUES(?,?,?,?,?,'FAILED',?)",
                    (CAMPAIGN_ID, "MHP_DEFECT", "Neutral 10-image NEB, 30 October input set", "2025-01-18", "2025-01-22",
                     "Two date-bounded Slurm attempts share one workdir; exact WFC-producing attempt unresolved. Existing NEB output is incomplete and not converged."))
        con.execute("INSERT OR IGNORE INTO registry_source(source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes) VALUES(?,?,?,?,?,?,1,?,?)",
                    ("mhp_neutral_neb_reproduction_manifest_20261122", str(MANIFEST.resolve()), "CURATED_REPRODUCTION_MANIFEST", manifest_hash, observed, observed,
                     "Newly curated reproduction plan; not an original run manifest.", "Fresh-start rerun recipe. The historical attempt linkage remains partial."))
        con.execute("INSERT OR IGNORE INTO registry_source(source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes) VALUES(?,?,?,?,?,?,1,?,?)",
                    ("mhp_neutral_neb_wfc_predelete_hashes_20261122", str(HASHES.resolve()), "FULL_FILE_SHA256_PREDELETE_AUDIT", hashes_hash, observed, observed,
                     "Read-only Rockfish lstat and SHA-256 of 490 WFC files; all metadata matched the preexisting inventory.", "All 490 files were UID 3901/GID 3001, mode 0644, nlink 1; current queue contained no MHP workdir match."))

        manifest_artifact = register_local_artifact(con, MANIFEST, observed, "CURATED_REPRODUCTION_MANIFEST")
        hash_evidence_artifact = register_local_artifact(con, HASHES, observed, "WFC_PREDELETE_SHA256_MANIFEST")
        decision_artifact = register_local_artifact(con, DECISION, observed, "WFC_CLEANUP_DECISION_NOTE")
        sacct_artifact = register_local_artifact(con, SACCT, observed, "DATE_BOUNDED_SLURM_ACCOUNTING_EVIDENCE")
        con.execute("""INSERT INTO workflow_run
            (run_id,project_id,campaign_id,run_label,workflow_type,method,system_key,source_root_id,source_relative_path,
             scheduler_state,scientific_status,start_utc,end_utc,notes)
            VALUES(?,?,?,?,?,?,?,?,?,'TIMEOUT','FAILURE_MODE',?,?,?)""",
            (RUN_ID, "MHP_DEFECT", CAMPAIGN_ID, "neb222-neutral-10-30", "QE_NEB", "Quantum ESPRESSO 7.2 NEB", "neutral_10_image_defect",
             ROOT_ID, "yi/neb222-neutral-10-30", "2025-01-18T19:55:21Z", "2025-01-22T00:02:24Z",
             "One catalog run directory with two date-bounded submissions. WFC-generation job cannot be attributed. neb.out truncates after iteration 103 started; last complete max force 0.181623 eV/Å exceeds 0.001 eV/Å. No validated converged NEB result."))

        attempts = [
            ("20074366", "2025-01-18T19:55:21Z", "2025-01-18T19:55:26Z", "COMPLETED", "0:0", 5, "Saved Slurm log contains MPI_Abort despite scheduler exit 0:0."),
            ("20080482", "2025-01-19T00:02:19Z", "2025-01-22T00:02:24Z", "TIMEOUT", "0:0", 259205, "Saved Slurm log confirms cancellation at the 72-hour time limit."),
        ]
        for job_id, start, end, state, code, elapsed, notes in attempts:
            con.execute("INSERT INTO slurm_job(run_id,cluster_name,job_id,array_task_id,submitted_or_started_utc,job_name,partition_name,account_name) VALUES(?,?,?,?,?,?,?,?)",
                        (RUN_ID, "Rockfish", job_id, "", start, "charg-bulk_211", "parallel", "pclancy3"))
            job_key = con.execute("SELECT slurm_job_key FROM slurm_job WHERE cluster_name='Rockfish' AND job_id=? AND submitted_or_started_utc=?", (job_id, start)).fetchone()[0]
            con.execute("INSERT INTO slurm_state_observation(slurm_job_key,observed_utc,state,exit_code,start_utc,end_utc,elapsed_seconds,source_command,raw_record_path,notes) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (job_key, observed, state, code, start, end, elapsed, "Date-bounded sacct for 2025-01-17..2025-01-23; job IDs can be reused.", str(SACCT.resolve()), notes))

        # Artifact paths are first looked up in the imported immutable MHP inventory.
        direct = {}
        core_defs = [
            ("neb.in", "QE_NEB_INPUT", "REPRODUCIBLE_INPUT", "INPUT", True),
            ("pw_1.in", "QE_ENDPOINT_INPUT", "REPRODUCIBLE_INPUT", "INPUT", False),
            ("pw_2.in", "QE_ENDPOINT_INPUT", "REPRODUCIBLE_INPUT", "INPUT", False),
            ("sbatch-cif_relax.slurm", "SLURM_SUBMISSION_SCRIPT", "REPRODUCIBLE_INPUT", "INPUT", True),
            ("neb.out", "QE_NEB_OUTPUT", "PUBLICATION_EVIDENCE", "OUTPUT", True),
            ("slurm.20074366", "SLURM_LOG", "PUBLICATION_EVIDENCE", "LOG", False),
            ("slurm.20080482", "SLURM_LOG", "PUBLICATION_EVIDENCE", "LOG", False),
            ("neb.path", "NEB_PATH_TABLE", "PUBLICATION_EVIDENCE", "OUTPUT", False),
            ("neb.xyz", "NEB_XYZ_TRAJECTORY", "PUBLICATION_EVIDENCE", "OUTPUT", False),
            ("neb.axsf", "NEB_ANIMATED_STRUCTURE", "PUBLICATION_EVIDENCE", "OUTPUT", False),
            ("neb.crd", "NEB_COORDINATE_OUTPUT", "PUBLICATION_EVIDENCE", "OUTPUT", False),
            ("neb.int", "NEB_INTERMEDIATE_OUTPUT", "PUBLICATION_EVIDENCE", "OUTPUT", False),
            ("neb.dat", "NEB_DATA_OUTPUT", "PUBLICATION_EVIDENCE", "OUTPUT", False),
        ]
        main_hashes = {}
        remote_text = (DATA / "mhp_source_run_comparison_20261122.txt").read_text()
        current_root = None
        for line in remote_text.splitlines():
            if line.startswith("ROOT "):
                current_root = line.split()[1]
            if current_root == "/data/pclancy3/MHP_defect/yi/neb222-neutral-10-30" and line.startswith("FILE "):
                parts = line.split()
                if len(parts) >= 8:
                    main_hashes[parts[1]] = {"size": int(parts[2]), "sha": parts[-1]}
        for basename, semantic, retention, direction, required in core_defs:
            absolute = str(PurePosixPath("/data/pclancy3/MHP_defect/yi/neb222-neutral-10-30") / basename)
            h = main_hashes.get(basename)
            aid = register_remote_artifact(con, ROOT_ID, ROOT_PATH, absolute, observed, "MHP_DEFECT", semantic, retention,
                                           h["size"] if h else None, h["sha"] if h else None)
            direct[basename] = aid
            con.execute("INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,?,?,?)",
                        (RUN_ID, aid, semantic.lower(), direction, int(required), "Exact path retained in the imported MHP inventory."))
            if h:
                con.execute("INSERT OR IGNORE INTO registry_source(source_key,source_path,registry_kind,sha256,captured_utc,immutable_snapshot,authority_note) VALUES(?,?,?,?,?,1,?)",
                            ("mhp_source_sha256_" + basename.replace(".", "_"), absolute, "TARGETED_REMOTE_SHA256", h["sha"], observed, "Full-file SHA-256 recorded by the read-only MHP evidence pass."))

        result_by_path = {}
        for row in result_meta:
            path = row["PATH"]
            if path.endswith("/PW.out"):
                role, retention = "QE_IMAGE_OUTPUT_LOG", "PUBLICATION_EVIDENCE"
            elif path.endswith("charge-density.dat"):
                role, retention = "QE_CHARGE_DENSITY", "PUBLICATION_EVIDENCE"
            elif path.endswith("data-file-schema.xml") or path.endswith("neb.xml"):
                role, retention = "QE_XML_OUTPUT", "PUBLICATION_EVIDENCE"
            else:
                role, retention = "QE_RESULT_EVIDENCE", "PUBLICATION_EVIDENCE"
            aid = register_remote_artifact(con, ROOT_ID, ROOT_PATH, path, observed, "MHP_DEFECT", role, retention,
                                           int(row["SIZE_BYTES"]), None)
            result_by_path[path] = aid
            con.execute("INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'OUTPUT',0,?)",
                        (RUN_ID, aid, role.lower(), "Per-image SCF/XML/charge evidence; not evidence that the full NEB converged."))

        # Also retain the precisely hashed/home-linked QE pseudopotentials.
        pseudo_hashes = [
            ("/home/ycao73/pseudo/br_pbe_v1.4.uspp.F.UPF", "611dc0ac2804aed2e5b8a3c138f272b2a4e131009aad5279454170f948176951"),
            ("/home/ycao73/pseudo/cs_pbe_v1.uspp.F.UPF", "5039c3123828de0c626459254b67b6637225866dd81d5ee1d855e26f99d3ac2c"),
            ("/home/ycao73/pseudo/pb_pbe_v1.uspp.F.UPF", "4b8527e7e8c6da86173825f1c5ab9cfeeb04525e46e5a5c5d0d13bf3059ceeeb"),
        ]
        pseudo_ids = []
        for path, digest in pseudo_hashes:
            aid = register_remote_artifact(con, HOME_ID, HOME_PATH, path, observed, "MHP_DEFECT", "QE_PSEUDOPOTENTIAL", "PERMANENT_SOURCE", None, digest)
            pseudo_ids.append(aid)
            con.execute("INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'INPUT',1,?)",
                        (RUN_ID, aid, "pseudopotential", "~/pseudo dependency resolved to this hashed canonical file."))

        for entry in hashes:
            absolute = entry["path"]
            meta_row = next(row for row in wfc_meta if row["PATH"] == absolute)
            aid = register_remote_artifact(con, ROOT_ID, ROOT_PATH, absolute, observed, "MHP_DEFECT", "QE_WFC_RESTART",
                                           "REPRODUCIBLE_INTERMEDIATE", entry["size"], entry["sha256"])
            con.execute("UPDATE artifact SET owner_status='USER_CONFIRMED',scientific_status='PROVENANCE_PARTIAL',retention_class='REPRODUCIBLE_INTERMEDIATE',notes=? WHERE artifact_id=?",
                        ("UID/GID/mode/size/mtime/nlink and SHA-256 verified during this cleanup audit. Regenerable by the documented fresh-start recipe; this removes continuation state for the unfinished historical run.", aid))
            con.execute("INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'INTERMEDIATE',0,?)",
                        (RUN_ID, aid, "qe_wfc_restart", "WFC-producing job attribution unresolved; input explicitly restarts from scratch."))

        for input_id in [direct["neb.in"], direct["sbatch-cif_relax.slurm"]] + pseudo_ids:
            con.execute("INSERT INTO derivation_edge(parent_artifact_id,child_artifact_id,run_id,operation,tool_name,tool_version,script_artifact_id,parameter_json,provenance_status,notes) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (input_id, direct["neb.out"], RUN_ID, "QE 10-image NEB", "Quantum ESPRESSO", "7.2", direct["sbatch-cif_relax.slurm"],
                         json.dumps({"mpi_ranks":48,"num_of_images":10,"restart_mode":"from_scratch"}), "LINEAGE_SUPPORTED",
                         "Reproduction linkage is input/script backed; historical WFC-producing job attribution is unresolved and convergence failed."))

        con.execute("INSERT INTO reproduction_recipe(run_id,recipe_version,exact_command,working_directory,executable_version,environment_description,module_list,script_artifact_id,scheduler_script_artifact_id,input_manifest_artifact_id,recipe_status,validation_note) VALUES(?,?,?,?,?,?,?,?,?,?,'DOCUMENTED',?)",
                    (RUN_ID, RECIPE_VERSION,
                     "RUN_DIR=$(mktemp -d /scratch16/pclancy3/yi/defects_scripts/2-0.neutral/2.neb/10-30/mhp-rerun-XXXXXXXX)\n"
                     "cp /data/pclancy3/MHP_defect/yi/neb222-neutral-10-30/neb.in /data/pclancy3/MHP_defect/yi/neb222-neutral-10-30/pw_1.in /data/pclancy3/MHP_defect/yi/neb222-neutral-10-30/pw_2.in /data/pclancy3/MHP_defect/yi/neb222-neutral-10-30/sbatch-cif_relax.slurm \"$RUN_DIR/\"\n"
                     "cd \"$RUN_DIR\"\nmkdir -p out\nsbatch sbatch-cif_relax.slurm",
                     "/scratch16/pclancy3/yi/defects_scripts/2-0.neutral/2.neb/10-30/<fresh-mhp-rerun>",
                     "Quantum ESPRESSO 7.2; /data/apps/extern/qe/7.2/bin/neb.x; 48 MPI ranks",
                     "The saved Slurm script runs module reset; source /data/apps/go.sh; ml intel; ml qe/7.2. It launches mpirun -np 48. Runtime and pseudopotential paths must remain available with recorded versions/hashes.",
                     "intel; qe/7.2", direct["sbatch-cif_relax.slurm"], direct["sbatch-cif_relax.slurm"], manifest_artifact,
                     "Curated fresh-start recipe. Original per-run manifest absent. Current NEB output is incomplete (last complete force 0.181623 eV/Å vs 0.001 threshold). Rerun starts in a new empty out directory and does not continue the archived electronic state."))
        recipe_id = con.execute("SELECT recipe_id FROM reproduction_recipe WHERE run_id=? AND recipe_version=?", (RUN_ID, RECIPE_VERSION)).fetchone()[0]
        for aid, role in [(direct["neb.in"], "neb_input")] + [(aid, "pseudopotential") for aid in pseudo_ids]:
            con.execute("INSERT INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)", (recipe_id, aid, role))

        con.execute("INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'PROVENANCE',1,?)",
                    (RUN_ID, manifest_artifact, "curated_reproduction_manifest", "Newly authored record of preserved input/script/output paths and fresh-start method."))
        con.execute("INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'PROVENANCE',1,?)",
                    (RUN_ID, decision_artifact, "cleanup_decision_note", "Records why the WFC files can be removed while failure evidence remains."))
        con.execute("INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'PROVENANCE',1,?)",
                    (RUN_ID, hash_evidence_artifact, "predelete_wfc_hash_evidence", "Full SHA-256/metadata preflight for all 490 candidate files."))
        con.execute("INSERT INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'PROVENANCE',0,?)",
                    (RUN_ID, sacct_artifact, "date_bounded_slurm_accounting", "Two job instances recorded with dates to avoid 2026 ID reuse."))

        for gate_id, status, rule, notes in [
            ("neb_force_below_threshold", "FAIL", "Final complete NEB force must be <= 0.001 eV/Å.", "Last complete iteration 102 max force was 0.181623 eV/Å; iteration 103 is truncated."),
            ("neb_output_complete", "FAIL", "Final NEB output contains complete terminal state and convergence summary.", "neb.out is 3,324 lines and truncates during image 10 SCF."),
            ("publication_acceptance", "UNKNOWN", "A validated claim or publication record maps to this run.", "No matching DFT-NEB calculation/artifact/trajectory rows or paper-claim mapping were found."),
        ]:
            con.execute("INSERT INTO quality_gate(run_id,gate_id,status,checked_utc,evidence_artifact_id,rule_description,notes) VALUES(?,?,?,?,?,?,?)",
                        (RUN_ID, gate_id, status, observed, direct["neb.out"], rule, notes))

        total_logical = sum(entry["size"] for entry in hashes)
        if total_logical != 26235975200:
            raise RuntimeError("Unexpected WFC logical total: %d" % total_logical)
        con.execute("INSERT INTO cleanup_batch(cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,expected_bytes,evidence_artifact_id) VALUES(?,?,?,?,?,'EVIDENCE_READY',?,?,?)",
                    (BATCH_ID, observed, ROOT_ID, "DELETE_REPRODUCIBLE_QE_NEB_WFC_RESTARTS",
                     "490 owned WFC restarts are regenerated by an exact fresh-start QE 7.2 NEB recipe from retained input, Slurm script, pseudopotentials and outputs. The historical result is an incomplete failure mode; XML/charge/PW/log/path evidence remains.",
                     "Explicit user authorization: delete only reproducible temporary WFC files after SQL path/hash/recipe evidence. All 490 files were hashed and the active queue showed no workdir overlap.", total_logical, manifest_artifact))
        for entry in hashes:
            aid = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (ROOT_ID + ":" + str(PurePosixPath(entry["path"]).relative_to(ROOT_PATH)),)).fetchone()[0]
            con.execute("INSERT INTO cleanup_item(cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes) VALUES(?,?,'DELETE',?,'CLEARED',?,?,?,'EVIDENCE_READY',?)",
                        (BATCH_ID, aid, "Reproducible QE WFC restart; neb.in uses restart_mode=from_scratch and the documented clean-directory recipe regenerates electronic states.",
                         entry["sha256"], entry["size"], entry["mtime_ns"],
                         "Pre-delete lstat and SHA-256 matched the imported MHP inventory. Current run's unresolved continuation state is intentionally discarded; incomplete result evidence is retained."))
        con.commit()

    with sqlite3.connect(DB) as con:
        print(json.dumps({
            "run_id": RUN_ID,
            "cleanup_batch": con.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone(),
            "cleanup_items": con.execute("SELECT COUNT(*) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()[0],
            "wfc_logical_bytes": con.execute("SELECT SUM(pre_delete_size_bytes) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()[0],
            "repro_guard_eligible": con.execute("""SELECT COUNT(*) FROM cleanup_item c JOIN artifact a USING(artifact_id)
                JOIN run_artifact ra USING(artifact_id) JOIN reproduction_recipe rr ON rr.run_id=ra.run_id
                WHERE c.cleanup_batch_id=? AND a.retention_class='REPRODUCIBLE_INTERMEDIATE' AND ra.direction='INTERMEDIATE'
                AND rr.run_id=? AND rr.recipe_status='DOCUMENTED'""", (BATCH_ID, RUN_ID)).fetchone()[0]
        }, sort_keys=True))


if __name__ == "__main__":
    main()
