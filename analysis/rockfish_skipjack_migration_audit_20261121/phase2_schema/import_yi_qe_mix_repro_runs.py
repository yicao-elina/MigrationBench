#!/usr/bin/env python3
"""Register exact QE mix run provenance and stage its guarded cleanup in the catalog."""

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
DATA = AUDIT / "phase2_data"
REVIEW = DATA / "yi_mix_repro_review_20261123"
RUN_MANIFEST = REVIEW / "qe_run_reproduction_inputs_manifest_20261123.json"
CANDIDATE_MANIFEST = REVIEW / "yi_qe_mix_reproduction_manifest_20261123.json"
HASH_LEDGER = REVIEW / "mix_candidate_predelete_identity_hashes_20261123.jsonl"
PREFLIGHT_STATUS = REVIEW / "mix_candidate_identity_preflight_20261123.status.json"
CURRENT_DEPS = REVIEW / "current_dependency_identity_20261123.json"
MATRIX = REVIEW / "mix_candidate_reproducibility_matrix.tsv"
SACCT_TSV = REVIEW / "sacct_successful_jobs.tsv"
SACCT_WORKDIR_TSV = REVIEW / "sacct_successful_jobs_with_workdir.tsv"
SMALL_EVIDENCE = REVIEW / "small_evidence_sha256.tsv"
ROOT_ID = "RF-DATA-YI"
HOME_ROOT_ID = "RF-HOME-YI"
DATA_FS_ROOT_ID = "RF-DATA-FS"
QE731_ROOT_ID = "RF-SCRATCH16-JCOMBAR1-QE731-EMR"
CAMPAIGNS = {
    "benchmark-441": ("MBREG-CR-SB2TE3-SIZE-BENCH-202506", "Cr–Sb₂Te₃ QE size-effect benchmark, June 2025", "cr_sb2te3_size_effect_benchmark_june_2025"),
    "edge-move2": ("MBREG-CR-SB2TE3-EDGE-MOVES-202506", "Cr–Sb₂Te₃ edge movement QE relaxations, June 2025", "cr_sb2te3_edge_move_qe_june_2025"),
    "edge-move-up": ("MBREG-CR-SB2TE3-EDGE-MOVES-202506", "Cr–Sb₂Te₃ edge movement QE relaxations, June 2025", "cr_sb2te3_edge_move_qe_june_2025"),
}
RUN_IDS = {
    "benchmark-441": "RF-MB-CR-SB2TE3-BENCHMARK-441-J6567243",
    "edge-move2": "RF-MB-CR-SB2TE3-EDGE-MOVE2-J5990677",
    "edge-move-up": "RF-MB-CR-SB2TE3-EDGE-MOVE-UP-J5614979",
}
JOB_RAW = REVIEW / "sacct_successful_jobs_20250601_20250701.attempt4.raw.txt"
CAPTURE_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_source(con: sqlite3.Connection, path: Path, kind: str, *, notes: str = "") -> tuple[int, int]:
    path = path.resolve(strict=True)
    rel = path.relative_to(REPO).as_posix()
    digest = sha256(path)
    size = path.stat().st_size
    key = f"yi_qe_mix_r04_evidence:{rel}:{digest[:16]}"
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (key, str(path), kind, digest, CAPTURE_UTC, CAPTURE_UTC,
         "Exact local capture or review artifact for QE mix-run provenance and cleanup.", notes),
    )
    row = con.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if not row or row[1:] != (str(path), digest):
        raise RuntimeError("Evidence source key identity conflict: " + key)
    source_id = int(row[0])

    rel_root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()
    if not rel_root or Path(rel_root[0]).resolve() != REPO.resolve():
        raise RuntimeError("LOCAL-MAIN does not resolve to the unified main worktree")
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)", (rel, CAPTURE_UTC))
    path_id = int(con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (rel,)).fetchone()[0])
    artifact_key = f"RF-AUDIT:{rel}:{digest[:16]}"
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES('ROCKFISH_STORAGE_AUDIT',?,'AUDIT_EVIDENCE',?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
        (artifact_key, kind, digest, size, notes or "Local immutable audit evidence with exact content hash."),
    )
    artifact = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if not artifact or artifact[1:] != (digest, size):
        raise RuntimeError("Evidence artifact identity conflict: " + artifact_key)
    aid = int(artifact[0])
    con.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""",
        (aid, path_id, CAPTURE_UTC, CAPTURE_UTC),
    )
    con.execute("UPDATE artifact_location SET availability='PRESENT',last_verified_utc=? WHERE artifact_id=? AND path_id=?", (CAPTURE_UTC, aid, path_id))
    return source_id, aid


def ensure_dependency_root(con: sqlite3.Connection) -> None:
    path = "/scratch16/jcombar1/QE731/2023/qe-emr"
    row = con.execute("SELECT root_id,cluster_name,filesystem_name,mount_path,status FROM storage_root WHERE scoped_path=?", (path,)).fetchone()
    expected = (QE731_ROOT_ID, "Rockfish", "scratch16", "/scratch16", "READ_ONLY")
    if row:
        if tuple(row) != expected:
            raise RuntimeError("Existing QE731 dependency storage root conflicts: " + repr(tuple(row)))
        return
    if con.execute("SELECT 1 FROM storage_root WHERE root_id=?", (QE731_ROOT_ID,)).fetchone():
        raise RuntimeError("QE731 dependency root ID is already used for another path")
    con.execute(
        """INSERT INTO storage_root
           (root_id,cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,
            scope_kind,project_id,status,notes)
           VALUES(?,?,?,?,?,?,'UNKNOWN','OTHER','HPC_ENVIRONMENTS','READ_ONLY',?)""",
        (QE731_ROOT_ID, "Rockfish", "scratch16", "/scratch16", path, "jcombar1",
         "Dependency-only pointer to the exact shared QE executable used by job 6567243; no recursive inventory or cleanup scope is implied."),
    )


def root_for_path(con: sqlite3.Connection, absolute: str) -> tuple[str, str]:
    path = PurePosixPath(absolute)
    roots = con.execute("SELECT root_id,scoped_path FROM storage_root WHERE status IN ('ACTIVE','READ_ONLY')").fetchall()
    matches = []
    for root_id, scoped in roots:
        root_path = PurePosixPath(scoped)
        try:
            relative = str(path.relative_to(root_path))
        except ValueError:
            continue
        matches.append((len(str(root_path)), root_id, relative))
    if not matches:
        raise RuntimeError("No storage root covers dependency/artifact path: " + absolute)
    _, root_id, relative = max(matches)
    return str(root_id), relative


def register_remote_artifact(con: sqlite3.Connection, absolute: str, *, project_id: str, kind: str,
                             role: str, digest: str, size: int, retention: str,
                             science: str, owner: str, notes: str) -> int:
    root_id, relative = root_for_path(con, absolute)
    root_path = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (root_id,)).fetchone()[0]
    path_id_row = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, relative)).fetchone()
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (root_id, relative, CAPTURE_UTC))
    path_id = int(path_id_row[0]) if path_id_row else int(con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, relative)).fetchone()[0])
    artifact_key = root_id + ":" + relative
    prior = con.execute("SELECT artifact_id,project_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if prior and prior[2] not in (None, digest):
        raise RuntimeError("Catalog SHA-256 conflicts with freshly verified remote file: " + absolute)
    if prior and prior[3] not in (None, size):
        raise RuntimeError("Catalog size conflicts with freshly verified remote file: " + absolute)
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (project_id, artifact_key, kind, role, owner, digest, size, science, retention, notes),
    )
    row = con.execute("SELECT artifact_id,project_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if not row or row[2] not in (None, digest) or row[3] not in (None, size):
        raise RuntimeError("Remote artifact catalog identity conflict: " + absolute)
    aid = int(row[0])
    # Preserve an established project assignment; fill only an absent classification.
    con.execute(
        """UPDATE artifact SET project_id=COALESCE(project_id,?),sha256=COALESCE(sha256,?),
           byte_size=COALESCE(byte_size,?),owner_status=CASE WHEN owner_status='UNKNOWN' THEN ? ELSE owner_status END,
           retention_class=CASE WHEN retention_class='UNCLASSIFIED' THEN ? ELSE retention_class END,
           scientific_status=CASE WHEN scientific_status='UNKNOWN' THEN ? ELSE scientific_status END
           WHERE artifact_id=?""",
        (project_id, digest, size, owner, retention, science, aid),
    )
    con.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""",
        (aid, path_id, CAPTURE_UTC, CAPTURE_UTC),
    )
    con.execute("UPDATE artifact_location SET availability='PRESENT',last_verified_utc=? WHERE artifact_id=? AND path_id=?", (CAPTURE_UTC, aid, path_id))
    return aid


def add_run_artifact(con: sqlite3.Connection, run_id: str, aid: int, role: str, direction: str,
                     required: bool = False, notes: str = "") -> None:
    con.execute(
        """INSERT OR IGNORE INTO run_artifact
           (run_id,artifact_id,role,direction,required_for_reproduction,notes)
           VALUES(?,?,?,?,?,?)""",
        (run_id, aid, role, direction, int(required), notes),
    )


def add_recipe_input(con: sqlite3.Connection, recipe_id: int, aid: int, role: str) -> None:
    con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)", (recipe_id, aid, role))


def add_ref(con: sqlite3.Connection, source_id: int, entity_type: str, entity_key: str,
            catalog_type: str, catalog_key: str, notes: str) -> None:
    con.execute(
        """INSERT OR IGNORE INTO registry_entity_ref
           (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
           VALUES(?,?,?,?,?,'EVIDENCE_FOR',?)""",
        (source_id, entity_type, entity_key, catalog_type, catalog_key, notes),
    )


def main() -> None:
    candidate_manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_STATUS.read_text(encoding="utf-8"))
    if preflight.get("remote_ok") is not True or preflight.get("remote_candidate_count") != 432:
        raise SystemExit("candidate SHA/fstat preflight did not pass")
    if preflight.get("manifest_sha256") != sha256(CANDIDATE_MANIFEST):
        raise SystemExit("candidate manifest changed after its read-only remote preflight")
    if preflight.get("ledger_sha256") != sha256(HASH_LEDGER):
        raise SystemExit("predelete identity/hash ledger changed")
    if run_manifest.get("source_candidate_manifest_sha256") != sha256(CANDIDATE_MANIFEST):
        raise SystemExit("run recipe manifest does not bind the current candidate manifest")
    if len(run_manifest.get("runs", [])) != 3:
        raise SystemExit("expected exactly three reviewed run recipes")

    current_by_path = {row["path"]: row for row in load_jsonl(HASH_LEDGER)}
    candidate_items = {item["path"]: (group, item) for group in candidate_manifest["groups"] for item in group["items"]}
    if len(current_by_path) != 432 or set(current_by_path) != set(candidate_items):
        raise SystemExit("candidate path set differs between inventory manifest and predelete hash ledger")
    for path, pre in current_by_path.items():
        group, inv = candidate_items[path]
        for field in ("device", "inode", "uid", "gid", "mode_octal", "nlink", "size_bytes", "allocated_bytes", "mtime_ns"):
            if pre[field] != inv[field]:
                raise SystemExit("preflight identity differs from R04 inventory for " + path + " field " + field)
        if pre["uid"] != 3901 or pre["gid"] != 3001 or pre["nlink"] != 1:
            raise SystemExit("owner/group/hardlink guard failed: " + path)
    delete_paths = {path for path, (group, _) in candidate_items.items() if group["disposition"] == "DELETE"}
    hold_paths = set(candidate_items) - delete_paths
    if (len(delete_paths), len(hold_paths)) != (384, 48):
        raise SystemExit("unexpected delete/hold split")

    small_rows = []
    with SMALL_EVIDENCE.open(newline="", encoding="utf-8") as stream:
        small_rows = list(csv.DictReader(stream, delimiter="\t"))
    current_deps = json.loads(CURRENT_DEPS.read_text(encoding="utf-8"))
    dep_by_path = {row["path"]: row for row in current_deps["items"]}
    if current_deps.get("sha256_mismatches"):
        raise SystemExit("retained input/script/output/runtime hash mismatch")
    by_group_role: dict[str, dict[str, list[dict]]] = {}
    for row in small_rows:
        dep = dep_by_path.get(row["absolute_path"])
        if not dep:
            continue
        if int(row["size_bytes"]) != dep["size_bytes"] or row["sha256"] != dep["sha256"]:
            raise SystemExit("retained input/script/output SHA mismatch: " + row["absolute_path"])
        by_group_role.setdefault(row["run_group"], {}).setdefault(row["evidence_role"], []).append(row)
    sacct = {}
    with SACCT_WORKDIR_TSV.open(newline="", encoding="utf-8") as stream:
        sacct = {row["job_id"]: row for row in csv.DictReader(stream, delimiter="\t")}
    if {run["group_id"] for run in run_manifest["runs"]} != set(RUN_IDS):
        raise SystemExit("run manifest group IDs differ from the SQL run mapping")

    con = sqlite3.connect(DB, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        if con.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id='RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123'").fetchone():
            raise RuntimeError("QE mix R04 cleanup batch already exists; inspect its receipts before rerunning importer")
        existing_runs = [run_id for run_id in RUN_IDS.values() if con.execute("SELECT 1 FROM workflow_run WHERE run_id=?", (run_id,)).fetchone()]
        if existing_runs:
            raise RuntimeError("QE mix R04 run IDs already exist; inspect catalog before re-import: " + repr(existing_runs))
        if not con.execute("SELECT 1 FROM project WHERE project_id='MIGRATIONBENCH'").fetchone():
            raise RuntimeError("MIGRATIONBENCH project row is absent")
        con.execute("BEGIN IMMEDIATE")
        ensure_dependency_root(con)

        evidence_specs = [
            (CANDIDATE_MANIFEST, "QE_MIX_R04_CANDIDATE_AND_IDENTITY_MANIFEST", "Exact four-group candidate manifest tied to complete R04 inventory."),
            (RUN_MANIFEST, "QE_RUN_RERUN_INPUTS_MANIFEST", "Run input, scheduler script, runtime, pseudopotential, and reviewed fresh-directory launch evidence."),
            (HASH_LEDGER, "QE_MIX_PREDELETE_SHA256_LEDGER", "Current identity and full SHA-256 for all 432 mix candidates; no file was modified during this pass."),
            (PREFLIGHT_STATUS, "QE_MIX_IDENTITY_PREFLIGHT_STATUS", "Read-only remote preflight receipt for exact path set, inode, owner, permissions, size, allocation, mtime, and SHA."),
            (CURRENT_DEPS, "QE_RUN_CURRENT_DEPENDENCY_HASHES", "Current hashes for retained input, scripts, logs, outputs, pseudopotentials, and QE executables."),
            (MATRIX, "QE_MIX_REPRODUCIBILITY_MATRIX", "Four-group matching matrix; raw-discovery candidate remains partial because WorkDir differs."),
            (REVIEW / "rerun_procedure_matrix.tsv", "QE_MIX_RERUN_PROCEDURE_MATRIX", "Exact scripts, submit directory, output paths, runtime and fresh-start procedure review."),
            (REVIEW / "small_evidence_sha256.tsv", "QE_MIX_SMALL_EVIDENCE_SHA256", "Hashes for exact retained inputs/scripts/job logs/outputs/pseudopotentials and external dependencies."),
            (REVIEW / "MIX_REPRODUCIBILITY_REVIEW.md", "QE_MIX_REPRODUCIBILITY_REVIEW", "Independent mapping of candidate paths to successful QE inputs/scripts/outputs and scheduler workdirs."),
            (REVIEW / "RERUN_PROCEDURE_REVIEW.md", "QE_MIX_RERUN_PROCEDURE_REVIEW", "Reviewed clean-directory rerun method, without executing a rerun."),
            (REVIEW / "MIX_EXTERNAL_DEPENDENCY_REVIEW.md", "QE_MIX_EXTERNAL_DEPENDENCY_REVIEW", "Exact external pseudopotential and executable dependency review."),
            (REVIEW / "runtime_dependency_inventory.tsv", "QE_MIX_RUNTIME_DEPENDENCY_INVENTORY", "Exact dependency paths, resolved binary target, ownership and scope boundary."),
            (REVIEW / "sacct_successful_jobs.tsv", "QE_MIX_SACCT_SUCCESSFUL_JOBS_UTC", "UTC scheduler state, exit codes, starts and ends for four candidate jobs."),
            (SACCT_WORKDIR_TSV, "QE_MIX_SACCT_WORKDIR_SUBMITLINE", "Date-bounded sacct WorkDir, SubmitLine and state records."),
            (REVIEW / "sacct_successful_jobs_workdir_submitline.json", "QE_MIX_SACCT_WORKDIR_SUBMITLINE_RAW", "Raw JSON capture of successful job workdirs and submit lines."),
            (DATA / "yi_full_inventory_20261123_r04" / "R04_REPORT.md", "DATA_YI_R04_SCAN_REPORT", "Complete accessible subtree inventory and R01/R03/R04 reconciliation report."),
            (DATA / "yi_full_inventory_20261123_r04" / "yi_current_full_metadata_composite_r04.jsonl.gz", "DATA_YI_R04_COMPOSITE_METADATA", "Composite path metadata stream for R03 conda and R04 nonconda inventory."),
        ]
        evidence = {}
        for path, kind, notes in evidence_specs:
            sid, aid = evidence_source(con, path, kind, notes=notes)
            evidence[path.name] = {"source_id": sid, "artifact_id": aid, "path": str(path.resolve())}
        candidate_manifest_aid = evidence[CANDIDATE_MANIFEST.name]["artifact_id"]
        run_manifest_aid = evidence[RUN_MANIFEST.name]["artifact_id"]
        candidate_hash_aid = evidence[HASH_LEDGER.name]["artifact_id"]

        dep_artifacts: dict[str, int] = {}
        for row in current_deps["items"]:
            path = row["path"]
            if path == "/scratch16/jcombar1/QE731/2023/qe-emr/bin/pw.x":
                continue
            if path == "/data/apps/go.sh":
                project, kind, semantic, retention, science, owner, note = (
                    "HPC_ENVIRONMENTS", "HPC_MODULE_BOOTSTRAP", "Rockfish shared module bootstrap script",
                    "PERMANENT_SOURCE", "LINEAGE_SUPPORTED", "THIRD_PARTY", "Exact shared Rockfish environment bootstrap, current SHA-256 captured; no cleanup scope implied.")
            elif path.endswith("/pw.x"):
                project, kind, semantic, retention, science, owner, note = (
                    "HPC_ENVIRONMENTS", "SOFTWARE_EXECUTABLE", "Quantum ESPRESSO pw.x runtime",
                    "PERMANENT_SOURCE", "LINEAGE_SUPPORTED", "THIRD_PARTY", "Exact shared QE executable identity; the binary is outside user cleanup scope.")
            else:
                continue
            aid = register_remote_artifact(con, path, project_id=project, kind=kind, role=semantic,
                                           digest=row["sha256"], size=int(row["size_bytes"]), retention=retention,
                                           science=science, owner=owner, notes=note)
            dep_artifacts[path] = aid
        # Record the symlink path used in the benchmark PATH as a second location for its resolved executable.
        qe_emr_target = "/scratch16/jcombar1/QE731/2023/qe-emr/PW/src/pw.x"
        qe_emr_link = "/scratch16/jcombar1/QE731/2023/qe-emr/bin/pw.x"
        qe_emr_aid = dep_artifacts[qe_emr_target]
        link_root, link_rel = root_for_path(con, qe_emr_link)
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (link_root, link_rel, CAPTURE_UTC))
        link_path_id = int(con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (link_root, link_rel)).fetchone()[0])
        con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)", (qe_emr_aid, link_path_id, CAPTURE_UTC, CAPTURE_UTC))

        campaigns = {}
        for group_id, (campaign_id, campaign_name, source_key) in CAMPAIGNS.items():
            if campaign_id in campaigns:
                continue
            notes = "Historical QE jobs completed in Slurm, but no publication-quality acceptance is inferred. Each recipe has exact retained inputs, scheduler script, outputs, runtime and pseudopotential identities."
            con.execute(
                """INSERT OR IGNORE INTO campaign
                   (campaign_id,project_id,campaign_name,source_campaign_key,start_date,end_date,status,notes)
                   VALUES(?,'MIGRATIONBENCH',?,?,?,?,'UNKNOWN',?)""",
                (campaign_id, campaign_name, "Rockfish:" + source_key, "2025-06-18", "2025-06-27", notes),
            )
            campaign = con.execute("SELECT project_id,source_campaign_key FROM campaign WHERE campaign_id=?", (campaign_id,)).fetchone()
            if not campaign or campaign[0] != "MIGRATIONBENCH" or campaign[1] != "Rockfish:" + source_key:
                raise RuntimeError("Campaign identity conflict: " + campaign_id)
            campaigns[campaign_id] = True

        group_to_mix_aids: dict[str, list[int]] = {}
        run_artifacts: dict[str, dict[str, int]] = {}
        recipe_rows: dict[str, dict] = {}
        for run in run_manifest["runs"]:
            group_id = run["group_id"]
            run_id = run["run_id"]
            if run_id != RUN_IDS[group_id]:
                raise RuntimeError("Unexpected run ID for group " + group_id)
            job = run["slurm_job"]
            if job["state"] != "COMPLETED" or job["exit_code"] != "0:0":
                raise RuntimeError("A reviewed QE job is not scheduler-successful: " + job["job_id"])
            source_rows = sacct[job["job_id"]]
            for key, expected in (("state", job["state"]), ("exit_code", job["exit_code"]), ("workdir", job["workdir"]), ("submit_line", job["submit_line"])):
                if source_rows[key] != expected:
                    raise RuntimeError("sacct WorkDir/submit/status mismatch for job " + job["job_id"] + " field " + key)
            if source_rows["start_utc"] + "Z" != job["start_utc"] or source_rows["end_utc"] + "Z" != job["end_utc"]:
                raise RuntimeError("UTC sacct job times mismatch for " + job["job_id"])
            campaign_id = CAMPAIGNS[group_id][0]
            workdir_relative = str(PurePosixPath(job["workdir"]).relative_to(PurePosixPath("/data/pclancy3/yi")))
            workflow_type = "QE_VC_RELAX" if group_id == "benchmark-441" else "QE_PW_RELAX"
            system_key = {
                "benchmark-441": "Sb2Te3_Cr_size_effect_4x4x1_PBE",
                "edge-move2": "Sb2Te3_Cr_octa_edge_move2_PBE",
                "edge-move-up": "Sb2Te3_Cr_octa_edge_move_up_PBE",
            }[group_id]
            con.execute(
                """INSERT INTO workflow_run
                   (run_id,project_id,campaign_id,run_label,workflow_type,method,system_key,
                    source_root_id,source_relative_path,scheduler_state,scientific_status,start_utc,end_utc,notes)
                   VALUES(?,'MIGRATIONBENCH',?,?,?,?,?,'RF-DATA-YI',?,'COMPLETED','PROVENANCE_PARTIAL',?,?,?)""",
                (run_id, campaign_id, group_id, workflow_type, "PBE Quantum ESPRESSO pw.x", system_key,
                 workdir_relative, job["start_utc"], job["end_utc"],
                 "Slurm reported COMPLETED/0:0 and the exact submit directory/script/input/output hashes were verified. Scientific acceptance is not inferred; the recipe starts a clean new work directory and has not been run."),
            )
            con.execute(
                """INSERT INTO slurm_job
                   (run_id,cluster_name,job_id,array_task_id,submitted_or_started_utc,job_name,partition_name,account_name)
                   VALUES(?,'Rockfish',?,'',?,?,?,?)""",
                (run_id, job["job_id"], job["start_utc"], job["job_name"], job["partition"], job["account"]),
            )
            job_key = int(con.execute("SELECT slurm_job_key FROM slurm_job WHERE cluster_name='Rockfish' AND job_id=? AND array_task_id='' AND submitted_or_started_utc=?", (job["job_id"], job["start_utc"])).fetchone()[0])
            start_dt = datetime.fromisoformat(job["start_utc"].replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(job["end_utc"].replace("Z", "+00:00"))
            sacct_command = "TZ=UTC SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf /cm/shared/apps/slurm/current/bin/sacct -X -j 6567243,5990677,5614979,6181656 --starttime=2025-06-01T00:00:00 --endtime=2025-07-01T00:00:00 --format=JobID,JobName,State,ExitCode,Start,End"
            con.execute(
                """INSERT INTO slurm_state_observation
                   (slurm_job_key,observed_utc,state,exit_code,start_utc,end_utc,elapsed_seconds,source_command,raw_record_path,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (job_key, CAPTURE_UTC, "COMPLETED", "0:0", job["start_utc"], job["end_utc"], int((end_dt-start_dt).total_seconds()),
                 sacct_command, str(SACCT_TSV.resolve()),
                 "Date-bounded UTC sacct record and separate WorkDir/SubmitLine capture. Scheduler completion is not scientific acceptance."),
            )

            artifact_map: dict[str, int] = {}
            for label, item, kind, semantic, retention, science, owner in [
                ("input", run["input"], "QE_INPUT", "Retained QE input used by historical job", "REPRODUCIBLE_INPUT", "LINEAGE_SUPPORTED", "USER_CONFIRMED"),
                ("script", run["successful_slurm_script"], "SLURM_SUBMISSION_SCRIPT", "Exact successful historical QE Slurm script", "REPRODUCIBLE_INPUT", "LINEAGE_SUPPORTED", "USER_CONFIRMED"),
                ("log", run["successful_job_log"], "SLURM_OUTPUT_LOG", "Historical Slurm stdout/stderr", "PUBLICATION_EVIDENCE", "PROVENANCE_PARTIAL", "USER_CONFIRMED"),
                ("output", run["completion_output"], "QE_OUTPUT_LOG", "Historical QE relax.out completion evidence", "PUBLICATION_EVIDENCE", "PROVENANCE_PARTIAL", "USER_CONFIRMED"),
            ]:
                aid = register_remote_artifact(con, item["path"], project_id="MIGRATIONBENCH", kind=kind,
                                               role=semantic, digest=item["sha256"], size=int(item["size_bytes"]),
                                               retention=retention, science=science, owner=owner,
                                               notes="Exact path, size and SHA-256 reverified on Rockfish on 2026-11-23." )
                artifact_map[label] = aid
            for index, item in enumerate(run["retained_derived_outputs"]):
                aid = register_remote_artifact(con, item["path"], project_id="MIGRATIONBENCH", kind="RETAINED_DERIVED_STRUCTURE",
                                               role="Retained relax.xyz; derivation is not created by the selected QE Slurm script",
                                               digest=item["sha256"], size=int(item["size_bytes"]), retention="PUBLICATION_EVIDENCE",
                                               science="PROVENANCE_PARTIAL", owner="USER_CONFIRMED",
                                               notes="Exact retained path/hash reverified. The successful QE script only writes relax.out; relax.xyz is evidence with a partial derivation link.")
                artifact_map[f"derived_output_{index}"] = aid

            runtime = run["runtime_executable"]
            runtime_aid = dep_artifacts[runtime["path"]]
            bootstrap_aid = dep_artifacts[run["module_bootstrap"]["path"]]
            artifact_map["qe_binary"] = runtime_aid
            artifact_map["module_bootstrap"] = bootstrap_aid
            pseudo_aids = []
            for pseudo in run["pseudopotentials"]:
                aid = register_remote_artifact(con, pseudo["path"], project_id="QE_PSEUDOPOTENTIALS", kind="PSEUDOPOTENTIAL",
                                               role="Exact pseudopotential read by retained QE input", digest=pseudo["sha256"],
                                               size=int(pseudo["size_bytes"]), retention="PERMANENT_SOURCE",
                                               science="LINEAGE_SUPPORTED", owner="USER_CONFIRMED",
                                               notes="Exact pseudopotential path and SHA-256 independently verified on Rockfish; external dependency is held in the home root.")
                pseudo_aids.append((pseudo, aid))

            for key in RUN_MANIFEST, CANDIDATE_MANIFEST:
                source_id = evidence[key.name]["source_id"]
                add_ref(con, source_id, "qe_mix_run", group_id, "WORKFLOW_RUN", run_id,
                        "Run-to-recipe evidence manifest; exact input/script/runtime IDs and candidate scope are stated separately.")
            for name in ("MIX_REPRODUCIBILITY_REVIEW.md", "RERUN_PROCEDURE_REVIEW.md", "MIX_EXTERNAL_DEPENDENCY_REVIEW.md",
                         "sacct_successful_jobs.tsv", "sacct_successful_jobs_with_workdir.tsv", "small_evidence_sha256.tsv"):
                add_ref(con, evidence[name]["source_id"], "qe_mix_run", group_id, "WORKFLOW_RUN", run_id,
                        "Evidence for exact historical job, run artifacts, runtime dependencies and the reviewed clean-start recipe.")

            for role, aid in (("qe_input", artifact_map["input"]), ("successful_slurm_script", artifact_map["script"]),
                              ("slurm_job_log", artifact_map["log"]), ("qe_output_log", artifact_map["output"]),
                              ("qe_executable", runtime_aid), ("rockfish_module_bootstrap", bootstrap_aid)):
                direction = "INPUT" if role in ("qe_input", "successful_slurm_script") else ("LOG" if role == "slurm_job_log" else ("OUTPUT" if role == "qe_output_log" else "ENVIRONMENT"))
                add_run_artifact(con, run_id, aid, role, direction, role in ("qe_input", "successful_slurm_script", "qe_executable"),
                                 "Exact file used by the historical job or needed to reconstruct the environment.")
            add_run_artifact(con, run_id, run_manifest_aid, "reviewed_rerun_manifest", "PROVENANCE", True,
                             "Hash-pinned run inputs, exact successful Slurm script, runtime, pseudopotentials and clean-directory launch command.")
            for pseudo, aid in pseudo_aids:
                role = "pseudopotential:" + Path(pseudo["path"]).name
                add_run_artifact(con, run_id, aid, role, "INPUT", True, "Exact QE pseudopotential dependency named by the input.")
            for index, item in enumerate(run["retained_derived_outputs"]):
                add_run_artifact(con, run_id, artifact_map[f"derived_output_{index}"], "retained_relax_xyz", "OUTPUT", False,
                                 "Retained structure evidence; exact QE script does not create it, so derivation is partial.")

            executable_version = ("QE7.3.1 path tree; exact binary SHA-256 " + runtime["sha256"] + "; --version not queried"
                                  if group_id == "benchmark-441" else "QE 7.2 shared executable SHA-256 " + runtime["sha256"])
            env_desc = run["environment_description"] + " Module bootstrap /data/apps/go.sh SHA-256=" + run["module_bootstrap"]["sha256"] + "."
            exact_command = (
                "Create a new empty Rockfish work directory; copy only " + run["input"]["path"] + " and " + run["successful_slurm_script"]["path"] +
                " into it; create an empty out/ directory; then run: cd <new-empty-workdir> && mkdir -p out && sbatch " + Path(run["successful_slurm_script"]["path"]).name +
                ". The exact script uses the submit directory as cwd and does not set restart_mode. Historical out/ and .mixN files are not inputs. " +
                "This recipe has been reviewed but not executed."
            )
            con.execute(
                """INSERT INTO reproduction_recipe
                   (run_id,recipe_version,exact_command,working_directory,executable_version,
                    environment_description,module_list,script_artifact_id,scheduler_script_artifact_id,
                    input_manifest_artifact_id,recipe_status,last_reproduced_utc,validation_note)
                   VALUES(?,?,?,?,?,?,?,?,?,?,'REVIEWED',NULL,?)""",
                (run_id, "fresh-workdir-qe-2026-11-23-v1", exact_command, job["workdir"], executable_version,
                 env_desc, ",".join(run["module_list"]), artifact_map["script"], artifact_map["script"], run_manifest_aid,
                 "Independent script/WorkDir/input/output/dependency review completed. No rerun was submitted. Scheduler COMPLETED/0:0 does not imply scientific acceptance."),
            )
            recipe_id = int(con.execute("SELECT recipe_id FROM reproduction_recipe WHERE run_id=? AND recipe_version='fresh-workdir-qe-2026-11-23-v1'", (run_id,)).fetchone()[0])
            for role, aid in (("QE_INPUT", artifact_map["input"]), ("SUCCESSFUL_SLURM_SCRIPT", artifact_map["script"]),
                              ("QE_EXECUTABLE", runtime_aid), ("MODULE_BOOTSTRAP", bootstrap_aid)):
                add_recipe_input(con, recipe_id, aid, role)
            for pseudo, aid in pseudo_aids:
                add_recipe_input(con, recipe_id, aid, "PSEUDOPOTENTIAL:" + Path(pseudo["path"]).name)
            add_run_artifact(con, run_id, run_manifest_aid, "recipe_input_manifest", "PROVENANCE", True,
                             "Contains exact rerun source inputs and reviewed launch recipe; not a claim of exact bitwise reproducibility.")

            # State and scientific quality are separate. Preserve scheduler success without overclaiming result validation.
            con.execute(
                """INSERT INTO quality_gate(run_id,gate_id,status,checked_utc,evidence_artifact_id,rule_description,notes)
                   VALUES(?,?,?,?,?,?,?)""",
                (run_id, "slurm_exit", "PASS", CAPTURE_UTC, artifact_map["log"], "Slurm job completes with exit 0:0.", "Scheduler accounting only; it is not a scientific quality verdict."),
            )
            con.execute(
                """INSERT INTO quality_gate(run_id,gate_id,status,checked_utc,evidence_artifact_id,rule_description,notes)
                   VALUES(?,?,?,?,?,?,?)""",
                (run_id, "scientific_acceptance", "NOT_RUN", CAPTURE_UTC, artifact_map["output"], "Independently validate the physical result and publication suitability.", "This audit catalogs provenance and reproducibility; it does not claim scientific acceptance."),
            )

            group_to_mix_aids[group_id] = []
            group_record = next(g for g in candidate_manifest["groups"] if g["group_id"] == group_id)
            for item in group_record["items"]:
                pre = current_by_path[item["path"]]
                relative = str(PurePosixPath(item["path"]).relative_to(PurePosixPath("/data/pclancy3/yi")))
                mix_aid = register_remote_artifact(con, item["path"], project_id="MIGRATIONBENCH", kind="QE_MIX_RESTART",
                                                   role="QE .mixN mixing-history restart intermediate", digest=pre["sha256"],
                                                   size=int(pre["size_bytes"]), retention="REPRODUCIBLE_INTERMEDIATE",
                                                   science="LINEAGE_SUPPORTED", owner="USER_CONFIRMED",
                                                   notes="Exact R04 path, inode identity, owner, size, mtime and SHA-256 linked to a successful job, input, output, Slurm script and reviewed fresh-start recipe. This internal state is not an input to a fresh rerun.")
                group_to_mix_aids[group_id].append(mix_aid)
                add_run_artifact(con, run_id, mix_aid, "qe_mix_restart_intermediate", "INTERMEDIATE", False,
                                 "Internal QE mixing history; reproducible from the input in a new run, but deleting it discards this exact continuation state.")
            run_artifacts[run_id] = artifact_map

        # Register raw-discovery artifacts without binding them to the mismatched Slurm job.
        raw_group = next(g for g in candidate_manifest["groups"] if g["group_id"] == "raw-discovery-relax-cr")
        raw_aids = []
        for row in small_rows:
            if row["run_group"] != "raw-discovery-relax-cr" or row["evidence_role"] not in {"QE_INPUT", "SUCCESSFUL_SLURM_SCRIPT", "SUCCESSFUL_JOB_LOG", "COMPLETION_OUTPUT"}:
                continue
            dep = dep_by_path[row["absolute_path"]]
            aid = register_remote_artifact(con, row["absolute_path"], project_id="MIGRATIONBENCH", kind=row["evidence_role"],
                                           role="Unbound raw-discovery candidate evidence; sacct WorkDir mismatch", digest=dep["sha256"],
                                           size=int(dep["size_bytes"]), retention="UNCLASSIFIED", science="PROVENANCE_PARTIAL",
                                           owner="USER_CONFIRMED", notes="Current file hash verified, but its path is not proven to belong to job 6181656 because sacct records a different now-missing WorkDir.")
            raw_aids.append(aid)
        raw_mix_aids = []
        for item in raw_group["items"]:
            pre = current_by_path[item["path"]]
            aid = register_remote_artifact(con, item["path"], project_id="MIGRATIONBENCH", kind="QE_MIX_RESTART_CANDIDATE",
                                           role="Unresolved QE .mixN candidate; possible raw-discovery restart", digest=pre["sha256"],
                                           size=int(pre["size_bytes"]), retention="UNCLASSIFIED", science="PROVENANCE_PARTIAL",
                                           owner="USER_CONFIRMED", notes="Hash and path recorded, but not bound to job 6181656 because sacct WorkDir is different. Held pending path reconciliation.")
            raw_mix_aids.append(aid)

        decision = (
            "User-authorized removal of the exact .mixN restart files after the current path, owner, full content hash, historical QE input, successful Slurm script, scheduler job, output and clean-directory rerun recipe are cataloged. The 48 raw-discovery candidates with WorkDir mismatch are explicitly held. Only .mixN files in three reviewed out/ directories are eligible; all other files remain untouched."
        )
        batch_id = "RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123"
        con.execute(
            """INSERT INTO cleanup_batch
               (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,expected_bytes,evidence_artifact_id)
               VALUES(?,?,?,?,?,'EVIDENCE_READY',?,?,?)""",
            (batch_id, CAPTURE_UTC, ROOT_ID, "DELETE_REPRODUCIBLE_QE_MIXN_FRESH_START_ONLY",
             "Stage the three path-matched successful QE run groups only: 384 single-link .mixN restarts (7,500,169,152 logical bytes; 7,518,289,920 allocated). Hold the 48 raw-discovery candidates with a sacct WorkDir mismatch.",
             "Explicit user authorization in this thread to catalog rerun method and SQL path before removing reproducible temporary intermediate files. A rerun was reviewed but not submitted. No push or unrelated cleanup is part of this batch.",
             7_500_169_152, candidate_hash_aid),
        )
        for path, (group, item) in candidate_items.items():
            pre = current_by_path[path]
            mix_aid = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (ROOT_ID + ":" + str(PurePosixPath(path).relative_to(PurePosixPath("/data/pclancy3/yi"))),)).fetchone()
            if not mix_aid:
                raise RuntimeError("Candidate artifact missing from catalog: " + path)
            eligible = group["disposition"] == "DELETE"
            con.execute(
                """INSERT INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,
                    pre_delete_sha256,pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (batch_id, int(mix_aid[0]), "DELETE" if eligible else "HOLD", decision if eligible else
                 "Hold: current candidate output directory is not the sacct WorkDir for job 6181656; timestamps and matching local log/script do not prove path lineage.",
                 "CLEARED" if eligible else "BLOCKED", pre["sha256"], int(pre["size_bytes"]), int(pre["mtime_ns"]),
                 "EVIDENCE_READY" if eligible else "HELD",
                 "Predelete identity: uid=%s gid=%s mode=%s nlink=%s dev=%s inode=%s allocated=%s; full SHA-256 captured with stable O_NOFOLLOW read." %
                 (pre["uid"], pre["gid"], pre["mode_octal"], pre["nlink"], pre["device"], pre["inode"], pre["allocated_bytes"]) +
                 (" Per-file all-user queue and identity/hash recheck is required at execution." if eligible else " No deletion authorization for this path while provenance remains partial.")),
            )

        for run in run_manifest["runs"]:
            run_id = run["run_id"]
            group_id = run["group_id"]
            for path in (RUN_MANIFEST, CANDIDATE_MANIFEST, REVIEW / "MIX_REPRODUCIBILITY_REVIEW.md",
                         REVIEW / "RERUN_PROCEDURE_REVIEW.md", REVIEW / "MIX_EXTERNAL_DEPENDENCY_REVIEW.md",
                         SACCT_TSV, SACCT_WORKDIR_TSV, SMALL_EVIDENCE, CURRENT_DEPS):
                add_ref(con, evidence[path.name]["source_id"], "qe_mix_run", group_id, "WORKFLOW_RUN", run_id,
                        "Primary source for the run's exact input, script, runtime, output, scheduler and reproduction relationships.")
        for path in (CANDIDATE_MANIFEST, HASH_LEDGER, PREFLIGHT_STATUS, RUN_MANIFEST, MATRIX,
                     REVIEW / "MIX_REPRODUCIBILITY_REVIEW.md", REVIEW / "RERUN_PROCEDURE_REVIEW.md",
                     REVIEW / "MIX_EXTERNAL_DEPENDENCY_REVIEW.md", CURRENT_DEPS, SACCT_TSV, SACCT_WORKDIR_TSV):
            add_ref(con, evidence[path.name]["source_id"], "qe_mix_cleanup", batch_id, "CLEANUP_BATCH", batch_id,
                    "Evidence for the exact file set, prior successful jobs, rerun method, ownership, predelete hash and held group.")

        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("Foreign key check failed: " + repr(fk[:10]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=20) as check:
        counts = {
            "runs": check.execute("SELECT count(*) FROM workflow_run WHERE run_id IN (?,?,?)", tuple(RUN_IDS.values())).fetchone()[0],
            "recipes": check.execute("SELECT count(*) FROM reproduction_recipe WHERE run_id IN (?,?,?)", tuple(RUN_IDS.values())).fetchone()[0],
            "slurm_jobs": check.execute("SELECT count(*) FROM slurm_job WHERE run_id IN (?,?,?)", tuple(RUN_IDS.values())).fetchone()[0],
            "mix_artifacts": check.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id='RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123'").fetchone()[0],
            "delete_items": check.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id='RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123' AND disposition='DELETE'").fetchone()[0],
            "held_items": check.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id='RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123' AND disposition='HOLD'").fetchone()[0],
            "delete_logical_bytes": check.execute("SELECT sum(pre_delete_size_bytes) FROM cleanup_item WHERE cleanup_batch_id='RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123' AND disposition='DELETE'").fetchone()[0],
            "foreign_key_violations": len(check.execute("PRAGMA foreign_key_check").fetchall()),
        }
        print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
