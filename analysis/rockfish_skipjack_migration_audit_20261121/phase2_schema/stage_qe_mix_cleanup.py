#!/usr/bin/env python3
"""Register QE mix lineage and stage the user-authorized, guarded cleanup batch."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
DATA = AUDIT / "phase2_data" / "yi_refresh_20261123"
MANIFEST = DATA / "qe_mix_reproduction_manifest_20261123.json"
LINEAGE = DATA / "qe_mix_file_lineage_20261123.tsv"
HASH_LEDGER = DATA / "qe_mix_retained_artifact_hashes_20261123.jsonl"
SACCT_UTC = DATA / "qe_candidate_sacct_utc_20261123.psv"
RECIPE_REVIEW = DATA / "QE_MIX_RECIPE_INDEPENDENT_AUDIT_20261123.md"
ROOT_ID = "RF-DATA-YI"
HOME_ROOT_ID = "RF-HOME-YI"
APP_ROOT_ID = "RF-DATA-FS"
CAMPAIGN_ID = "MBREG-CR-IN-GAP-RELAX-202507"
BATCH_ID = "RF-DATA-YI-CR-IN-GAP-MIX-CLEANUP-20261123"
OBSERVED_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

EVIDENCE_FILES = [
    ("qe_mix_reproduction_manifest_20261123.json", "QE_MIX_REPRODUCTION_MANIFEST"),
    ("prepare_qe_mix_fresh_start.py", "QE_FRESH_START_PREPARER"),
    ("qe_mix_file_lineage_20261123.tsv", "QE_MIX_PATH_HASH_RUN_JOB_LINEAGE"),
    ("qe_mix_retained_artifact_hashes_20261123.jsonl", "QE_RETAINED_ARTIFACT_SHA256_LEDGER"),
    ("candidate_targeted_lstat_20261123.jsonl", "QE_CANDIDATE_LSTAT_LEDGER"),
    ("recipe_evidence_20261123.jsonl", "QE_INPUT_SCRIPT_AND_LOG_EVIDENCE"),
    ("qe_mix_fresh_start_recipe_proposal_20261123.md", "QE_FRESH_START_RECIPE"),
    ("QE_MIX_RECIPE_INDEPENDENT_AUDIT_20261123.md", "QE_INDEPENDENT_RECIPE_REVIEW"),
    ("qe_mix_external_reference_check_20261123.md", "QE_EXTERNAL_REFERENCE_CHECK"),
    ("qe_candidate_sacct_20261123.psv", "SACCT_RAW_TIMEZONE_UNSPECIFIED"),
    ("qe_candidate_sacct_utc_20261123.psv", "SACCT_UTC_CAPTURE"),
    ("qe_binary_identity_20261123.txt", "QE_BINARY_IDENTITY"),
    ("QE_BINARY_AND_SACCT_TIME_AUDIT_20261123.md", "QE_BINARY_AND_TIMEZONE_AUDIT"),
    ("qe_sacct_timezone_check_20261123.txt", "SACCT_TIMEZONE_CHECK"),
    ("YI_BOUNDED_REFRESH_REPORT_20261123.md", "QE_BOUNDED_REFRESH_REPORT"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def register_local_evidence(con, path: Path, kind: str):
    path = path.resolve(strict=True)
    relative = path.relative_to(REPO).as_posix()
    digest = sha256(path)
    size = path.stat().st_size
    suffix = digest[:16]
    source_key = "qe_mix_evidence:" + relative + ":" + suffix
    artifact_key = "QE-MIX-AUDIT:" + relative + ":" + suffix
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (source_key, str(path), kind, digest, OBSERVED_UTC, OBSERVED_UTC,
         "Locally captured QE run/cleanup audit evidence; hash identifies this exact version.",
         "Supports run lineage, clean-start recipe review, or cleanup decision."),
    )
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
    if not prior or (prior[0], prior[1]) != (str(path), digest):
        raise RuntimeError("Local evidence source identity conflict: " + source_key)
    source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES('ROCKFISH_STORAGE_AUDIT',?,?,?,?,?,?,'PROVENANCE_PARTIAL','PERMANENT_SOURCE',?)""",
        (artifact_key, kind, "QE mix lineage/cleanup evidence", "USER_CONFIRMED", digest, size,
         "Local evidence file for the 2026-11-23 QE restart cleanup review."),
    )
    artifact = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if artifact[1:] != (digest, size):
        raise RuntimeError("Local evidence artifact identity conflict: " + artifact_key)
    root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()
    if not root or Path(root[0]).resolve() != REPO.resolve():
        raise RuntimeError("LOCAL-MAIN must resolve to the unified MigrationBench worktree")
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)",
                (relative, OBSERVED_UTC))
    path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (relative,)).fetchone()[0]
    con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                (artifact[0], path_id, OBSERVED_UTC, OBSERVED_UTC))
    return int(artifact[0]), int(source_id), source_key


def ensure_storage_path(con, root_id: str, absolute: str) -> int:
    root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (root_id,)).fetchone()
    if not root:
        raise RuntimeError("Missing storage root: " + root_id)
    root_path = PurePosixPath(root[0])
    abs_path = PurePosixPath(absolute)
    try:
        relative = str(abs_path.relative_to(root_path))
    except ValueError:
        raise RuntimeError("Path is outside storage root %s: %s" % (root_id, absolute))
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
                (root_id, relative, OBSERVED_UTC))
    return int(con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, relative)).fetchone()[0])


def register_remote_artifact(con, root_id: str, absolute: str, *, project_id: str, kind: str,
                             role: str, digest: str, size: int, retention: str,
                             science_status: str, owner: str = "USER_CONFIRMED", notes: str = "") -> int:
    path_id = ensure_storage_path(con, root_id, absolute)
    root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (root_id,)).fetchone()[0]
    relative = str(PurePosixPath(absolute).relative_to(PurePosixPath(root)))
    artifact_key = root_id + ":" + relative
    prior = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if prior and prior[1] is not None and prior[1] != digest:
        raise RuntimeError("Catalog SHA conflicts with current evidence: " + absolute)
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (project_id, artifact_key, kind, role, owner, digest, size, science_status, retention, notes),
    )
    row = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if row[1] not in (None, digest) or row[2] not in (None, size):
        raise RuntimeError("Catalog artifact size/SHA conflict: " + absolute)
    con.execute(
        """UPDATE artifact SET project_id=COALESCE(project_id,?),artifact_kind=?,semantic_role=?,
           owner_status=?,sha256=?,byte_size=?,scientific_status=?,retention_class=?,notes=?
           WHERE artifact_id=?""",
        (project_id, kind, role, owner, digest, size, science_status, retention, notes, row[0]),
    )
    con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                (row[0], path_id, OBSERVED_UTC, OBSERVED_UTC))
    return int(row[0])


def add_artifact_location(con, artifact_id: int, root_id: str, absolute: str):
    path_id = ensure_storage_path(con, root_id, absolute)
    con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                (artifact_id, path_id, OBSERVED_UTC, OBSERVED_UTC))


def add_run_artifact(con, run_id: str, artifact_id: int, role: str, direction: str,
                     required: bool, notes: str = ""):
    con.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,?,?,?)",
                (run_id, artifact_id, role, direction, int(required), notes))


def add_derivation(con, parent: int, child: int, run_id: str, operation: str,
                   tool: str, version: str, script: int, params: dict, notes: str):
    exists = con.execute(
        """SELECT 1 FROM derivation_edge WHERE parent_artifact_id=? AND child_artifact_id=?
           AND run_id=? AND operation=? AND script_artifact_id=?""",
        (parent, child, run_id, operation, script),
    ).fetchone()
    if exists:
        return
    con.execute(
        """INSERT INTO derivation_edge(parent_artifact_id,child_artifact_id,run_id,operation,
           tool_name,tool_version,script_artifact_id,parameter_json,provenance_status,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (parent, child, run_id, operation, tool, version, script,
         json.dumps(params, sort_keys=True), "LINEAGE_SUPPORTED", notes),
    )


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["mix_file_count"] != 12 or manifest["mix_logical_bytes"] != 5352688896 or manifest["mix_allocated_bytes"] != 5352980480:
        raise SystemExit("QE mix manifest counts/totals differ from reviewed evidence")
    mixes = [item for run in manifest["runs"] for item in run["mix_intermediates"]]
    if len(mixes) != 12 or len({x["path"] for x in mixes}) != 12:
        raise SystemExit("QE mix path set is incomplete or duplicated")
    if not all(x["uid"] == 3901 and x["gid"] == 3001 and x["nlink"] == 1 and x["stable_during_read"] for x in mixes):
        raise SystemExit("QE mix owner/link/hash-stability evidence does not pass")
    if sha256(DATA / "qe_mix_retained_artifact_hashes_20261123.jsonl") != manifest["source_hash_ledger_sha256"]:
        raise SystemExit("retained artifact hash ledger changed since independent audit")
    helper_sha = sha256(DATA / "prepare_qe_mix_fresh_start.py")
    if manifest.get("fresh_start_helper", {}).get("sha256") != helper_sha:
        raise SystemExit("fresh-start helper hash differs from the reviewed manifest")
    for name, _ in EVIDENCE_FILES:
        if not (DATA / name).is_file():
            raise SystemExit("required evidence is missing: " + name)
    with LINEAGE.open(newline="", encoding="utf-8") as stream:
        lineage_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(lineage_rows) != 12:
        raise SystemExit("lineage TSV must contain exactly 12 QE mix records")
    lineage_by_path = {row["path"]: row for row in lineage_rows}
    manifest_by_path = {row["path"]: row for row in mixes}
    if set(lineage_by_path) != set(manifest_by_path):
        raise SystemExit("lineage TSV paths differ from the reviewed reproduction manifest")
    for path, row in lineage_by_path.items():
        if row["sha256"] != manifest_by_path[path]["sha256"] or int(row["logical_bytes"]) != manifest_by_path[path]["size_bytes"]:
            raise SystemExit("lineage TSV hash/size differs from manifest: " + path)
    retained_hashes = {row["path"]: row for row in read_jsonl(HASH_LEDGER) if row.get("record_type") == "file_hash"}
    sacct_rows = {}
    for line in SACCT_UTC.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split("|")
        if len(fields) == 6 and "." not in fields[0]:
            sacct_rows[fields[0]] = fields
    run_ids = {run["run_id"] for run in manifest["runs"]}
    if len(run_ids) != 3:
        raise SystemExit("expected three unique QE run IDs")

    con = sqlite3.connect(DB, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        if con.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone():
            raise RuntimeError("Cleanup batch already exists; refusing a second stage")
        if any(con.execute("SELECT 1 FROM workflow_run WHERE run_id=?", (run_id,)).fetchone() for run_id in run_ids):
            raise RuntimeError("One of the QE workflow runs already exists; audit before staging")
        if not con.execute("SELECT 1 FROM project WHERE project_id='MIGRATIONBENCH'").fetchone():
            raise RuntimeError("MigrationBench project is absent")
        if not con.execute("SELECT 1 FROM project WHERE project_id='QE_PSEUDOPOTENTIALS'").fetchone():
            raise RuntimeError("QE pseudopotential project is absent")
        con.execute("BEGIN IMMEDIATE")

        evidence = {}
        for name, kind in EVIDENCE_FILES:
            aid, sid, source_key = register_local_evidence(con, DATA / name, kind)
            evidence[name] = {"artifact_id": aid, "source_id": sid, "source_key": source_key}

        manifest_aid = evidence[MANIFEST.name]["artifact_id"]
        helper_aid = evidence["prepare_qe_mix_fresh_start.py"]["artifact_id"]
        review_aid = evidence[RECIPE_REVIEW.name]["artifact_id"]
        sacct_aid = evidence[SACCT_UTC.name]["artifact_id"]
        binary_artifacts = {}
        for binary_name, binary in manifest["qe_binary_identity"].items():
            binary_artifacts[binary_name] = register_remote_artifact(
                con, APP_ROOT_ID, binary["path"], project_id="HPC_ENVIRONMENTS",
                kind="SOFTWARE_EXECUTABLE", role="Quantum ESPRESSO 7.2 " + binary_name,
                digest=binary["sha256"], size=int(binary["size_bytes"]), retention="PERMANENT_SOURCE",
                science_status="LINEAGE_SUPPORTED", owner="THIRD_PARTY",
                notes="Rockfish shared QE installation; exact executable hash/path read-only verified. Never treated as user cleanup candidate.",
            )
        campaign_notes = (
            "Three July 2025 Cr-in-gap Sb2Te3 QE relaxations with distinct spin/charge inputs. "
            "Each PW stage hit max_seconds before JOB DONE; each PP stage failed wrong nx; Slurm accounting is COMPLETED 0:0 despite MPI_Abort in stdout."
        )
        con.execute(
            """INSERT OR IGNORE INTO campaign
               (campaign_id,project_id,campaign_name,source_campaign_key,start_date,end_date,status,notes)
               VALUES(?,?,?,?,?,?,'FAILED',?)""",
            (CAMPAIGN_ID, "MIGRATIONBENCH", "Cr-in-gap Sb2Te3 spin/charge QE relaxations, July 2025",
             "Rockfish:/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/2.Cr-in-gap",
             "2025-07-10", "2025-07-16", campaign_notes),
        )
        campaign = con.execute("SELECT project_id,source_campaign_key FROM campaign WHERE campaign_id=?", (CAMPAIGN_ID,)).fetchone()
        if not campaign or campaign[0] != "MIGRATIONBENCH":
            raise RuntimeError("Campaign identity conflict")

        # Exact retained source and result files, including three SG15 UPFs.
        run_artifacts = {}
        pseudo_artifacts = {}
        for run in manifest["runs"]:
            run_id = run["run_id"]
            run_artifacts[run_id] = {}
            for item in run["retained_artifacts"]:
                role = item["role"]
                if role.startswith("pseudopotential_"):
                    symbol = role.split("_", 1)[1]
                    content_key = (symbol, item["sha256"])
                    if content_key not in pseudo_artifacts:
                        aid = register_remote_artifact(
                            con, ROOT_ID, item["path"], project_id="QE_PSEUDOPOTENTIALS",
                            kind="PSEUDOPOTENTIAL", role="SG15 v1.2 PBE " + symbol,
                            digest=item["sha256"], size=int(item["size_bytes"]),
                            retention="PERMANENT_SOURCE", science_status="LINEAGE_SUPPORTED",
                            notes="Exact QE input dependency. Content SHA is shared by /data and /home aliases; do not count aliases as separate bytes.",
                        )
                        pseudo_artifacts[content_key] = aid
                    aid = pseudo_artifacts[content_key]
                    if item["path"].startswith("/home/"):
                        add_artifact_location(con, aid, HOME_ROOT_ID, item["path"])
                        if item.get("resolved_path"):
                            add_artifact_location(con, aid, ROOT_ID, item["resolved_path"])
                    else:
                        add_artifact_location(con, aid, ROOT_ID, item["path"])
                    run_artifacts[run_id][role] = aid
                    continue
                semantic = {
                    "pw_input": ("QE_INPUT", "PW relaxation input", "REPRODUCIBLE_INPUT", "LINEAGE_SUPPORTED"),
                    "pp_input": ("QE_POSTPROCESS_INPUT", "PP post-processing input", "REPRODUCIBLE_INPUT", "LINEAGE_SUPPORTED"),
                    "scheduler_script": ("SLURM_SUBMISSION_SCRIPT", "Historical QE submission script", "REPRODUCIBLE_INPUT", "LINEAGE_SUPPORTED"),
                    "pw_output_log": ("QE_OUTPUT_LOG", "Time-limited PW relaxation log", "PUBLICATION_EVIDENCE", "FAILURE_MODE"),
                    "failed_pp_output_log": ("QE_OUTPUT_LOG", "Failed PP post-processing log", "PUBLICATION_EVIDENCE", "FAILURE_MODE"),
                    "time_limited_pw_save_xml": ("QE_XML_OUTPUT", "Time-limited PW save XML", "PUBLICATION_EVIDENCE", "FAILURE_MODE"),
                }[role]
                aid = register_remote_artifact(
                    con, ROOT_ID, item["path"], project_id="MIGRATIONBENCH",
                    kind=semantic[0], role=semantic[1], digest=item["sha256"], size=int(item["size_bytes"]),
                    retention=semantic[2], science_status=semantic[3],
                    notes="Exact path/hash captured with stable pre/post fstat and O_NOFOLLOW. See the linked QE run and known partial/failure gates.",
                )
                run_artifacts[run_id][role] = aid

            run_artifacts[run_id]["pw_binary"] = binary_artifacts["pw.x"]
            run_artifacts[run_id]["pp_binary"] = binary_artifacts["pp.x"]

            scheduler_row = next(x for x in run["retained_artifacts"] if x["role"] == "scheduler_script")
            scheduler_log_path = scheduler_row["path"].rsplit("/", 1)[0] + "/slurm." + run["scheduler_job"]["job_id"]
            scheduler_log_evidence = retained_hashes.get(scheduler_log_path)
            if not scheduler_log_evidence:
                raise RuntimeError("Hashed Slurm output is missing: " + scheduler_log_path)
            run_artifacts[run_id]["scheduler_log"] = register_remote_artifact(
                con, ROOT_ID, scheduler_log_path,
                project_id="MIGRATIONBENCH", kind="SLURM_OUTPUT_LOG", role="Historical Slurm stdout/stderr",
                digest=scheduler_log_evidence["sha256"], size=int(scheduler_log_evidence["size_bytes"]),
                retention="PUBLICATION_EVIDENCE", science_status="FAILURE_MODE",
                notes="Slurm stdout contains EDT timestamps and MPI_Abort during PP; no claim of successful end-to-end job.",
            )
            if not isinstance(run_artifacts[run_id]["scheduler_log"], int):
                raise RuntimeError("Unexpected scheduler log artifact id")

            job = run["scheduler_job"]
            fields = sacct_rows.get(job["job_id"])
            if not fields or fields[2] != "COMPLETED" or fields[3] != "0:0":
                raise RuntimeError("UTC sacct record missing or differs for job " + job["job_id"])
            start_utc, end_utc = job["start_utc"], job["end_utc"]
            if fields[4] + "Z" != start_utc or fields[5] + "Z" != end_utc:
                raise RuntimeError("UTC sacct timestamps differ from the run manifest for job " + job["job_id"])
            workdir = run["workdir"]
            relative = str(PurePosixPath(workdir).relative_to(PurePosixPath("/data/pclancy3/yi")))
            con.execute(
                """INSERT INTO workflow_run
                   (run_id,project_id,campaign_id,run_label,workflow_type,method,system_key,
                    source_root_id,source_relative_path,scheduler_state,scientific_status,
                    start_utc,end_utc,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,'COMPLETED','FAILURE_MODE',?,?,?)""",
                (run_id, "MIGRATIONBENCH", CAMPAIGN_ID, run["run_key"], "QE_PW_RELAX",
                 "Quantum ESPRESSO 7.2 pw.x relaxation", "Sb2Te3-Cr-in-gap-" + run["run_key"],
                 ROOT_ID, relative, start_utc, end_utc,
                 "QE log reports Maximum CPU time exceeded before JOB DONE; Slurm stdout later reports MPI_Abort in pp.x. This is a time-limited partial/failure-mode run, not a converged result."),
            )
            con.execute(
                """INSERT INTO slurm_job(run_id,cluster_name,job_id,array_task_id,submitted_or_started_utc,
                   job_name,partition_name,account_name) VALUES(?,?,?,?,?,?,?,?)""",
                (run_id, "Rockfish", job["job_id"], "", start_utc, job["job_name"], job["partition"], job["account"]),
            )
            job_key = int(con.execute("SELECT slurm_job_key FROM slurm_job WHERE cluster_name='Rockfish' AND job_id=? AND array_task_id='' AND submitted_or_started_utc=?",
                                      (job["job_id"], start_utc)).fetchone()[0])
            start_dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
            elapsed = int((end_dt - start_dt).total_seconds())
            con.execute(
                """INSERT INTO slurm_state_observation
                   (slurm_job_key,observed_utc,state,exit_code,start_utc,end_utc,elapsed_seconds,
                    source_command,raw_record_path,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (job_key, OBSERVED_UTC, "COMPLETED", "0:0", start_utc, end_utc, elapsed,
                 "TZ=UTC SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf /cm/shared/apps/slurm/current/bin/sacct --starttime=2025-07-10T00:00:00 --endtime=2025-07-17T00:00:00 --jobs=7104884,7105613,7167643 --format=JobIDRaw,JobName,State,ExitCode,Start,End -P",
                 str(SACCT_UTC.resolve()),
                 "UTC recapture. Earlier timezone-naive PSV is preserved separately and is not treated as UTC. Scheduler exit status does not override the PW time limit or PP MPI_Abort."),
            )

            # Exact recipe is reviewable, pinned, and explicitly not yet run.
            script_id = helper_aid
            scheduler_script_id = run_artifacts[run_id]["scheduler_script"]
            manifest_id = manifest_aid
            exact_command = (
                "On Rockfish, copy prepare_qe_mix_fresh_start.py from the local audit bundle. Run: "
                "python3 prepare_qe_mix_fresh_start.py " + run["run_key"] + ". The helper verifies exact input, Slurm script, pp.in, SG15 pseudopotential and pw.x hashes, "
                "then prepares a fresh PW-only work directory under /scratch16/pclancy3/yi without submitting. Review the generated input/script; then manually run "
                "cd <printed-directory> && sbatch sbatch-relax-pw-fresh.slurm. Do not run PP as part of this recipe; the historical PP output is a known wrong-nx failure."
            )
            environment = (
                "Rockfish; module reset; source /data/apps/go.sh; ml intel; ml qe/7.2; "
                "pw.x=/data/apps/extern/qe/7.2/bin/pw.x SHA256=e893e4835199422ac09218709bab451b6d47620fdfaa873d135c582da028cb81; "
                "original Slurm resources retained; mpirun -np 4."
            )
            con.execute(
                """INSERT INTO reproduction_recipe
                   (run_id,recipe_version,exact_command,working_directory,executable_version,
                    environment_description,module_list,script_artifact_id,scheduler_script_artifact_id,
                    input_manifest_artifact_id,recipe_status,last_reproduced_utc,validation_note)
                   VALUES(?,?,?,?,?,?,?,?,?,?,'REVIEWED',NULL,?)""",
                (run_id, "fresh-start-pw-only-2026-11-23-v1", exact_command,
                 "/scratch16/pclancy3/yi/<helper-created-run-directory>", "Quantum ESPRESSO 7.2; pw.x SHA256 e893e483...",
                 environment + " Fresh outdir under scratch16; original time cap max_seconds=250000 remains, so a rerun may again stop before convergence.",
                 "intel; qe/7.2", script_id, scheduler_script_id, manifest_id,
                 "Independent review completed. Recipe has not been run or submitted; it starts a new trajectory and is not an exact continuation of the time-limited historical state."),
            )
            recipe_id = int(con.execute("SELECT recipe_id FROM reproduction_recipe WHERE run_id=? AND recipe_version=?",
                                        (run_id, "fresh-start-pw-only-2026-11-23-v1")).fetchone()[0])
            for role in ("pw_input", "scheduler_script"):
                add_run_artifact(con, run_id, run_artifacts[run_id][role], role, "INPUT", True,
                                 "Exact source file/hash used by the reviewed fresh-start method.")
            add_run_artifact(con, run_id, run_artifacts[run_id]["pw_binary"], "qe_7_2_pw_x", "ENVIRONMENT", True,
                             "The reviewed clean-start method pins the executable path and SHA-256.")
            add_run_artifact(con, run_id, run_artifacts[run_id]["pp_binary"], "qe_7_2_pp_x", "ENVIRONMENT", False,
                             "Historical batch script invoked pp.x; it failed wrong nx. The fresh PW-only recipe does not run PP.")
            for role in ("pp_input", "pw_output_log", "failed_pp_output_log", "time_limited_pw_save_xml", "scheduler_log"):
                direction = "OUTPUT" if role in ("pw_output_log", "failed_pp_output_log", "time_limited_pw_save_xml") else ("LOG" if role == "scheduler_log" else "PROVENANCE")
                add_run_artifact(con, run_id, run_artifacts[run_id][role], role, direction, False,
                                 "Retained historical evidence; the PP log records wrong nx and the PW output is time-limited.")
            for item in run["retained_artifacts"]:
                if item["role"].startswith("pseudopotential_"):
                    aid = run_artifacts[run_id][item["role"]]
                    add_run_artifact(con, run_id, aid, item["role"], "INPUT", True,
                                     "Exact content hash; source path as named by this input, with canonical aliases recorded as artifact locations.")
                    con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",
                                (recipe_id, aid, item["role"]))
                    add_derivation(con, aid, run_artifacts[run_id]["time_limited_pw_save_xml"], run_id,
                                   "QE pw.x relaxation", "Quantum ESPRESSO", "7.2", scheduler_script_id,
                                   {"input": run["workdir"] + "/relax.in", "pseudo": item["path"], "job_id": job["job_id"]},
                                   "Input/pseudopotential/job/output link is supported by exact hashes, Slurm accounting and preserved output evidence; convergence failed.")
                elif item["role"] == "pw_input":
                    add_run_artifact(con, run_id, run_artifacts[run_id][item["role"]], "pw_input", "INPUT", True,
                                     "Exact retained pw.x input; hash and path verified.")
                    con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",
                                (recipe_id, run_artifacts[run_id][item["role"]], "pw_input"))
                    for out_role in ("pw_output_log", "time_limited_pw_save_xml"):
                        add_derivation(con, run_artifacts[run_id]["pw_input"], run_artifacts[run_id][out_role], run_id,
                                       "QE pw.x relaxation", "Quantum ESPRESSO", "7.2", scheduler_script_id,
                                       {"restart_mode_source": run["input_restart_mode"], "clean_restart_mode": "from_scratch",
                                        "max_seconds": 250000, "outdir": "./out_fresh/", "job_id": job["job_id"]},
                                       "Fresh-start recipe is documented but unrun; historical output stopped at max_seconds and is not converged.")
                        add_derivation(con, run_artifacts[run_id]["pw_binary"], run_artifacts[run_id][out_role], run_id,
                                       "QE pw.x relaxation", "Quantum ESPRESSO", "7.2", scheduler_script_id,
                                       {"job_id": job["job_id"], "binary_sha256": manifest["qe_binary_identity"]["pw.x"]["sha256"]},
                                       "Exact executable identity and preserved log/XML are linked; the PW run was time-limited.")
                    add_derivation(con, run_artifacts[run_id]["pp_input"], run_artifacts[run_id]["failed_pp_output_log"], run_id,
                                   "QE pp.x post-processing attempt", "Quantum ESPRESSO", "7.2", scheduler_script_id,
                                   {"job_id": job["job_id"], "known_error": "chdens wrong nx"},
                                   "The preserved pp.out is failure evidence, not a successful derived result.")
                    add_derivation(con, run_artifacts[run_id]["time_limited_pw_save_xml"], run_artifacts[run_id]["failed_pp_output_log"], run_id,
                                   "QE pp.x post-processing attempt", "Quantum ESPRESSO", "7.2", scheduler_script_id,
                                   {"job_id": job["job_id"], "known_error": "chdens wrong nx"},
                                   "The XML save was the pp.x input state; pp.out records a failed post-processing derivative.")
                    add_derivation(con, run_artifacts[run_id]["pp_binary"], run_artifacts[run_id]["failed_pp_output_log"], run_id,
                                   "QE pp.x post-processing attempt", "Quantum ESPRESSO", "7.2", scheduler_script_id,
                                   {"job_id": job["job_id"], "binary_sha256": manifest["qe_binary_identity"]["pp.x"]["sha256"]},
                                   "The executable hash and failed PP log are linked; no successful post-processing is claimed.")
            # Register the historical mix state separately as disposable only under the user-approved fresh-start policy.
            for item in run["mix_intermediates"]:
                mix_aid = register_remote_artifact(
                    con, ROOT_ID, item["path"], project_id="MIGRATIONBENCH", kind="QE_MIX_RESTART",
                    role="QE mixing-history restart intermediate", digest=item["sha256"], size=int(item["size_bytes"]),
                    retention="REPRODUCIBLE_INTERMEDIATE", science_status="PROVENANCE_PARTIAL",
                    notes="Exact path/hash and stable lstat captured. Internal QE continuation state; not required by the reviewed fresh-start recipe. Deleting it loses continuation of the exact partial historical run.",
                )
                run_artifacts[run_id][item["path"]] = mix_aid
                add_run_artifact(con, run_id, mix_aid, "qe_mix_restart_intermediate", "INTERMEDIATE", False,
                                 "Internal QE mixing history; regenerable only by a new fresh-start run, not an exact historical continuation.")
                add_derivation(con, run_artifacts[run_id]["pw_input"], mix_aid, run_id,
                               "QE pw.x mixing-history write", "Quantum ESPRESSO", "7.2", scheduler_script_id,
                               {"job_id": job["job_id"], "restart_mode_source": run["input_restart_mode"]},
                               "Path, owner, hash and historical run/job are linked; file is not needed by the fresh-start recipe.")
                add_derivation(con, run_artifacts[run_id]["pw_binary"], mix_aid, run_id,
                               "QE pw.x mixing-history write", "Quantum ESPRESSO", "7.2", scheduler_script_id,
                               {"job_id": job["job_id"], "binary_sha256": manifest["qe_binary_identity"]["pw.x"]["sha256"]},
                               "Exact QE executable identity is linked to the historical internal restart output.")

            # The source script's scheduler status is kept separate from scientific gates.
            gate_specs = [
                ("slurm_accounting_exit", "PASS", "Slurm reports COMPLETED with ExitCode 0:0.", "Scheduler status only; it is not proof that PW converged or PP succeeded.", run_artifacts[run_id]["scheduler_log"]),
                ("pw_relaxation_converged", "FAIL", "The ionic relaxation reaches its convergence gates before any time limit.", "relax.out reports Maximum CPU time exceeded before JOB DONE; classify the saved XML as a time-limited partial result.", run_artifacts[run_id]["pw_output_log"]),
                ("postprocessing_success", "FAIL", "pp.x exits without an error and produces validated postprocessing output.", "pp.out reports Error in routine chdens (1): wrong nx; Slurm stdout records MPI_Abort.", run_artifacts[run_id]["failed_pp_output_log"]),
                ("batch_script_exit_propagation", "FAIL", "The batch script propagates a failed pp.x exit to the scheduler.", "The script ends with date and does not enable failure propagation; sacct reports 0:0 despite MPI_Abort in Slurm stdout.", run_artifacts[run_id]["scheduler_log"]),
                ("fresh_start_recipe_execution", "NOT_RUN", "Run the reviewed fresh-start PW recipe and inspect its new outputs.", "Recipe helper and command are recorded; no fresh job was submitted.", manifest_aid),
            ]
            for gate_id, status, rule, notes, evidence_aid in gate_specs:
                con.execute(
                    """INSERT INTO quality_gate(run_id,gate_id,status,checked_utc,evidence_artifact_id,rule_description,notes)
                       VALUES(?,?,?,?,?,?,?)""",
                    (run_id, gate_id, status, OBSERVED_UTC, evidence_aid, rule, notes),
                )
            for source_name in ("qe_mix_file_lineage_20261123.tsv", "qe_mix_reproduction_manifest_20261123.json",
                                "qe_candidate_sacct_utc_20261123.psv", "QE_MIX_RECIPE_INDEPENDENT_AUDIT_20261123.md"):
                source_id = evidence[source_name]["source_id"]
                con.execute(
                    """INSERT OR IGNORE INTO registry_entity_ref
                       (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                       VALUES(?,?,?,?,?,'EVIDENCE_FOR',?)""",
                    (source_id, "qe_mix_run", run["run_key"], "WORKFLOW_RUN", run_id,
                     "Exact run/job/input/output lineage and quality boundary."),
                )

        decision_reason = (
            "User-authorized fresh-start cleanup after exact run/input/script/output/job/pseudopotential/hash registration. "
            "The QE mix files are internal restart state and no explicit reference was found in 184 readable text files under the calculation subtree. "
            "The wider /data/pclancy3/yi text scan did not complete. Deletion loses the continuation point for time-limited partial runs; the user requested fresh-start reruns instead."
        )
        con.execute(
            """INSERT INTO cleanup_batch
               (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,expected_bytes,evidence_artifact_id)
               VALUES(?,?,?,?,?,'EVIDENCE_READY',?,?,?)""",
            (BATCH_ID, OBSERVED_UTC, ROOT_ID, "DELETE_USER_AUTHORIZED_FRESH_START_QE_MIX_RESTARTS",
             "Remove only the 12 exact-hash QE mixing-history files after registering fresh-start recipes, retained partial outputs, and scheduler/PP failure gates.",
             "Explicit user authorization in the active request: reproducible intermediate restart files may be removed once path and rerun method are registered. Does not authorize exact continuation loss silently; this consequence is recorded in the manifest and report.",
             5352688896, manifest_aid),
        )
        for item in mixes:
            aid_row = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?",
                                  (ROOT_ID + ":" + str(PurePosixPath(item["path"]).relative_to(PurePosixPath("/data/pclancy3/yi"))),)).fetchone()
            if not aid_row:
                raise RuntimeError("Staged QE mix artifact is missing: " + item["path"])
            con.execute(
                """INSERT INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,pre_delete_sha256,
                    pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes)
                   VALUES(?,?,'DELETE',?,'CLEARED',?,?,?,'EVIDENCE_READY',?)""",
                (BATCH_ID, aid_row[0], decision_reason, item["sha256"], int(item["size_bytes"]), int(item["mtime_ns"]),
                 "Predelete identity: uid=%s gid=%s mode=%s nlink=%s dev=%s inode=%s allocated=%s; stable O_NOFOLLOW SHA-256. Fresh queue and immediate hash guard are still mandatory." %
                 (item["uid"], item["gid"], item["mode_octal"], item["nlink"], item["device"], item["inode"], item["allocated_bytes"]))
            )

        # Tie the cleanup manifest and review records directly to the staged batch.
        for name in (MANIFEST.name, "qe_mix_external_reference_check_20261123.md", RECIPE_REVIEW.name,
                     "qe_mix_file_lineage_20261123.tsv", "qe_mix_retained_artifact_hashes_20261123.jsonl"):
            source_id = evidence[name]["source_id"]
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,?,?,?,?,'EVIDENCE_FOR',?)""",
                (source_id, "qe_cleanup_batch", BATCH_ID, "CLEANUP_BATCH", BATCH_ID,
                 "Evidence required to stage and guard the exact 12-file cleanup batch."),
            )
        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("Foreign key check failed: " + repr(violations[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect(DB) as check:
        batch = check.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()
        item_count = check.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()[0]
        run_count = check.execute("SELECT count(*) FROM workflow_run WHERE run_id IN (%s)" % ",".join("?" for _ in run_ids), tuple(run_ids)).fetchone()[0]
        mix_count = item_count
        print(json.dumps({"campaign_id": CAMPAIGN_ID, "runs": run_count, "mix_artifacts": mix_count,
                          "cleanup_batch": batch, "cleanup_items": item_count,
                          "expected_logical_bytes": 5352688896,
                          "foreign_key_violations": len(check.execute("PRAGMA foreign_key_check").fetchall())}, default=list, sort_keys=True))


if __name__ == "__main__":
    main()
