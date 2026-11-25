#!/usr/bin/env python3
"""Idempotently catalog two captured QE-NEB runs and their mix intermediates.

This importer only writes local JSON manifests and the local provenance SQLite
catalog. It does not connect to Rockfish, submit a job, or create cleanup rows.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
REVIEW = AUDIT / "phase2_data" / "yi_mix_repro_review_20261123"
RECEIPT = REVIEW / "two_neb_mix_pair_predelete_hashes_20261123.json"
CAPTURE_SCRIPT = REVIEW / "capture_two_neb_mix_pair_predelete_hashes_20261123.py"
STAGING_HELPER = REVIEW / "stage_and_submit_qe_neb_fresh_start.py"
TRIAGE = REVIEW / "REMAINING_QE_MIX_TRIAGE_20261123.md"
MANIFEST_DIR = REVIEW / "fresh_start_recipe_artifacts_20261123T112510Z"
ROOT_DATA = "RF-DATA-YI"
ROOT_HOME = "RF-HOME-YI"
ROOT_FS = "RF-DATA-FS"
LOCAL_ROOT = "LOCAL-MAIN"
CAMPAIGN_ID = "MBREG-CR-SB2TE3-NEB-25090135-25090139"
FRESH_ROOT = "/scratch16/pclancy3/yi/reproducible_neb_fresh_starts"
CAPTURE_UTC = "2026-11-23T11:25:10Z"
RUNS = {
    "25090135": {
        "run_id": "RF-DATA-YI-QE-NEB-25090135",
        "label": "V_ql_shallow-ql_mid_high_primary-primary",
        "dir_name": "V_ql_shallow-ql_mid_high_primary-primary",
        "image4_failed": True,
    },
    "25090139": {
        "run_id": "RF-DATA-YI-QE-NEB-25090139",
        "label": "V_ql_mid_te-ql_deep_primary-primary",
        "dir_name": "V_ql_mid_te-ql_deep_primary-primary",
        "image4_failed": True,
    },
}
STAGE_NAMES = ["neb.in", "neb.dat", "pw_1.in", "pw_2.in", "sb2te3.path", "slurm-neb.slurm"]
REMOTE_ENV = {
    "/data/apps/go.sh": "ROCKFISH_MODULE_BOOTSTRAP",
    "/data/apps/extern/qe/7.2/bin/neb.x": "QE_NEB_EXECUTABLE",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_inputs() -> tuple[dict, dict[str, dict], dict[str, dict], dict[str, dict]]:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("capture_utc") != CAPTURE_UTC:
        raise RuntimeError("The predelete receipt capture time changed")
    if len(receipt.get("files", [])) != 232:
        raise RuntimeError("Expected 232 exact file observations in the receipt")
    run_by_id = {row["run_id"]: row for row in receipt["runs"]}
    if set(run_by_id) != {config["run_id"] for config in RUNS.values()}:
        raise RuntimeError("Receipt run IDs do not match the expected Slurm jobs")
    jobs = {}
    for line in receipt["sacct_probe"]["stdout"].splitlines():
        cells = line.split("|")
        if len(cells) != 10:
            raise RuntimeError("Unexpected raw sacct line: " + line)
        job_id, job_name, user, state, exit_code, submit, start, end, workdir, submitline = cells
        jobs[job_id] = {
            "job_id": job_id, "job_name": job_name, "user": user, "state": state,
            "exit_code": exit_code, "submit_raw": submit, "start_raw": start,
            "end_raw": end, "workdir": workdir, "submitline": submitline,
            "raw_line": line,
        }
    if set(jobs) != set(RUNS):
        raise RuntimeError("sacct receipt does not contain exactly jobs 25090135 and 25090139")
    if any(job["state"] != "COMPLETED" or job["exit_code"] != "0:0" for job in jobs.values()):
        raise RuntimeError("Unexpected scheduler result in the captured sacct receipt")
    if any(job["submitline"] != "sbatch slurm-neb.slurm" for job in jobs.values()):
        raise RuntimeError("Unexpected historical submission command")
    rows_by_run = {}
    for run_id in run_by_id:
        rows = [row for row in receipt["files"] if row.get("run_id") == run_id]
        if len(rows) != 115:
            raise RuntimeError("Expected 115 file records for " + run_id)
        rows_by_run[run_id] = rows
        mix = [row for row in rows if row["role"] == "QE_MIXING_RESTART"]
        if len(mix) != 96:
            raise RuntimeError("Expected 96 .mixN records for " + run_id)
        if any(row.get("hash_scope") != "FULL_FILE" or len(row.get("sha256", "")) != 64 for row in rows):
            raise RuntimeError("Every captured file must have a full-file SHA-256")
        for required in STAGE_NAMES:
            matches = [row for row in rows if PurePosixPath(row["path"]).name == required]
            if len(matches) != 1:
                raise RuntimeError("Expected exactly one captured staging input " + required)
    if receipt["runtime_probe"].get("returncode") != 0:
        raise RuntimeError("QE 7.2 runtime probe did not pass")
    return receipt, run_by_id, rows_by_run, jobs


def root_for_path(absolute: str) -> tuple[str, str]:
    path = PurePosixPath(absolute)
    candidates = [
        ("/data/pclancy3/yi", ROOT_DATA),
        ("/home/ycao73", ROOT_HOME),
        ("/data", ROOT_FS),
    ]
    for prefix, root_id in candidates:
        try:
            return root_id, str(path.relative_to(PurePosixPath(prefix)))
        except ValueError:
            pass
    raise RuntimeError("No registered Rockfish storage root covers " + absolute)


def ensure_storage_path(con: sqlite3.Connection, root_id: str, relative: str, seen_utc: str) -> int:
    con.execute(
        "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
        (root_id, relative, seen_utc),
    )
    row = con.execute(
        "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, relative)
    ).fetchone()
    if not row:
        raise RuntimeError("Failed to register storage path " + root_id + ":" + relative)
    return int(row[0])


def register_remote_artifact(con: sqlite3.Connection, row: dict, *, project_id: str,
                             artifact_kind: str, semantic_role: str, retention: str,
                             science: str, owner: str, notes: str, observed_utc: str) -> int:
    absolute = row["path"]
    root_id, relative = root_for_path(absolute)
    path_id = ensure_storage_path(con, root_id, relative, observed_utc)
    key = root_id + ":" + relative
    digest = row["sha256"]
    size = int(row["size_bytes"])
    prior = con.execute(
        "SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (key,)
    ).fetchone()
    if prior and prior[1] not in (None, digest):
        raise RuntimeError("Predelete capture SHA conflicts with catalog for " + absolute)
    if prior and prior[2] not in (None, size):
        raise RuntimeError("Predelete capture size conflicts with catalog for " + absolute)
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (project_id, key, artifact_kind, semantic_role, owner, digest, size, science, retention, notes),
    )
    found = con.execute(
        "SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (key,)
    ).fetchone()
    if not found or found[1] not in (None, digest) or found[2] not in (None, size):
        raise RuntimeError("Artifact identity conflict for " + absolute)
    aid = int(found[0])
    con.execute(
        """UPDATE artifact SET project_id=COALESCE(project_id,?),artifact_kind=?,semantic_role=?,
           owner_status=?,sha256=?,byte_size=?,scientific_status=?,retention_class=?,notes=?
           WHERE artifact_id=?""",
        (project_id, artifact_kind, semantic_role, owner, digest, size, science, retention, notes, aid),
    )
    con.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""",
        (aid, path_id, observed_utc, observed_utc),
    )
    # A later cataloged deletion/missing state must not be reversed by this old capture.
    con.execute(
        """UPDATE artifact_location SET availability='PRESENT',
           first_verified_utc=COALESCE(first_verified_utc,?),
           last_verified_utc=CASE WHEN last_verified_utc IS NULL OR last_verified_utc<=?
                                 THEN ? ELSE last_verified_utc END
           WHERE artifact_id=? AND path_id=? AND availability='PRESENT'""",
        (observed_utc, observed_utc, observed_utc, aid, path_id),
    )
    return aid


def register_local_artifact(con: sqlite3.Connection, path: Path, *, kind: str, role: str,
                            notes: str, project_id: str = "ROCKFISH_STORAGE_AUDIT",
                            observed_utc: str) -> tuple[int, int]:
    path = path.resolve(strict=True)
    relative = path.relative_to(REPO).as_posix()
    digest = sha256_file(path)
    size = path.stat().st_size
    key = "RF-AUDIT:" + relative + ":" + digest[:16]
    path_id = ensure_storage_path(con, LOCAL_ROOT, relative, observed_utc)
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (project_id, key, kind, role, "USER_CONFIRMED", digest, size,
         "LINEAGE_SUPPORTED", "PERMANENT_SOURCE", notes),
    )
    row = con.execute(
        "SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (key,)
    ).fetchone()
    if not row or row[1:] != (digest, size):
        raise RuntimeError("Local audit evidence identity conflict: " + str(path))
    aid = int(row[0])
    con.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""", (aid, path_id, observed_utc, observed_utc)
    )
    con.execute(
        "UPDATE artifact_location SET availability='PRESENT',last_verified_utc=? WHERE artifact_id=? AND path_id=?",
        (observed_utc, aid, path_id),
    )
    return aid, digest


def build_manifests(receipt: dict, run_by_id: dict, rows_by_run: dict, jobs: dict) -> dict[str, Path]:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    helper_digest = sha256_file(STAGING_HELPER)
    result = {}
    for job_id, config in RUNS.items():
        run_id = config["run_id"]
        run_info = run_by_id[run_id]
        rows = rows_by_run[run_id]
        source_root = run_info["run_root"]
        inputs = [row for row in rows if PurePosixPath(row["path"]).name in STAGE_NAMES]
        if len(inputs) != len(STAGE_NAMES):
            raise RuntimeError("Fresh-start stage set is incomplete for " + run_id)
        pseudo = [row for row in rows if row["role"] == "QE_PSEUDOPOTENTIAL"]
        if len(pseudo) != 3:
            raise RuntimeError("Expected three named UPFs for " + run_id)
        env_rows = {row["path"]: row for row in receipt["files"] if row["path"] in REMOTE_ENV}
        if set(env_rows) != set(REMOTE_ENV):
            raise RuntimeError("Captured QE 7.2 environment file set is incomplete")
        fresh_destination = FRESH_ROOT + "/" + job_id
        manifest_path = (MANIFEST_DIR / (job_id + "_capture_manifest.json")).resolve()
        exact = (
            "python3 " + json.dumps(str(STAGING_HELPER.resolve())) +
            " --manifest " + json.dumps(str(manifest_path)) +
            " --destination " + json.dumps(fresh_destination) + " --submit"
        )
        manifest = {
            "schema": "qe_neb_fresh_start_capture_manifest_v1",
            "capture_utc": CAPTURE_UTC,
            "source_receipt": {"path": str(RECEIPT.resolve()), "sha256": sha256_file(RECEIPT)},
            "run_id": run_id,
            "slurm": jobs[job_id],
            "source_run_root": source_root,
            "fresh_start_destination": fresh_destination,
            "fresh_start_inputs": [
                {"source_path": row["path"], "role": row["role"], "sha256": row["sha256"],
                 "size_bytes": row["size_bytes"], "mtime_ns": row["mtime_ns"]}
                for row in sorted(inputs, key=lambda item: item["path"])
            ],
            "pseudopotentials": [
                {"path": row["path"], "name": PurePosixPath(row["path"]).name,
                 "sha256": row["sha256"], "size_bytes": row["size_bytes"],
                 "mtime_ns": row["mtime_ns"]}
                for row in sorted(pseudo, key=lambda item: item["path"])
            ],
            "module_bootstrap": {"path": "/data/apps/go.sh",
                "sha256": env_rows["/data/apps/go.sh"]["sha256"],
                "size_bytes": env_rows["/data/apps/go.sh"]["size_bytes"]},
            "qe_executable": {"path": "/data/apps/extern/qe/7.2/bin/neb.x",
                "sha256": env_rows["/data/apps/extern/qe/7.2/bin/neb.x"]["sha256"],
                "size_bytes": env_rows["/data/apps/extern/qe/7.2/bin/neb.x"]["size_bytes"]},
            "runtime_files": [
                {key: env_rows[path].get(key) for key in ("path", "role", "sha256", "size_bytes", "allocated_bytes",
                    "mtime_ns", "uid", "gid", "mode_octal", "nlink", "inode", "device", "hash_scope")}
                for path in sorted(env_rows)
            ],
            "runtime_probe": receipt["runtime_probe"],
            "retained_run_evidence": [
                {key: row.get(key) for key in ("path", "role", "sha256", "size_bytes", "allocated_bytes",
                    "mtime_ns", "uid", "gid", "mode_octal", "nlink", "inode", "device", "hash_scope")}
                for row in sorted(rows, key=lambda item: item["path"])
            ],
            "staging_helper": {"path": str(STAGING_HELPER.resolve()), "sha256": helper_digest,
                "manifest_path": str(manifest_path),
                "exact_command": exact,
                "remote_submission_command": "sbatch slurm-neb.slurm"},
            "scientific_status": "FAILURE_MODE",
            "validation_note": (
                "Fresh-start recipe stages only the current captured six-file input set into a new, empty, "
                "job-specific directory under scratch16, then submits slurm-neb.slurm there. Current input "
                "files may postdate the May 2026 historical job. The captured historical neb.out contains "
                "SCF_WARNING and image-4 PW.out did not converge after 400 iterations. This recipe reruns "
                "the current captured input; it does not claim exact reproduction of the historical warning."
            ),
        }
        content = dump_json(manifest)
        if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != content:
            raise RuntimeError("Existing generated manifest differs; refusing to overwrite " + str(manifest_path))
        if not manifest_path.exists():
            manifest_path.write_text(content, encoding="utf-8")
        result[run_id] = manifest_path
    return result


def add_ref(con: sqlite3.Connection, source_id: int, run_id: str, notes: str) -> None:
    con.execute(
        """INSERT OR IGNORE INTO registry_entity_ref
           (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
           VALUES(?,'qe_neb_capture',?,'WORKFLOW_RUN',?,'EVIDENCE_FOR',?)""",
        (source_id, run_id, run_id, notes),
    )


def add_run_artifact(con: sqlite3.Connection, run_id: str, aid: int, role: str,
                     direction: str, required: bool, notes: str) -> None:
    con.execute(
        """INSERT OR IGNORE INTO run_artifact
           (run_id,artifact_id,role,direction,required_for_reproduction,notes)
           VALUES(?,?,?,?,?,?)""",
        (run_id, aid, role, direction, int(required), notes),
    )
    row = con.execute(
        "SELECT direction,required_for_reproduction FROM run_artifact WHERE run_id=? AND artifact_id=? AND role=?",
        (run_id, aid, role),
    ).fetchone()
    if not row or tuple(row) != (direction, int(required)):
        raise RuntimeError("Existing run_artifact relationship conflicts with the reviewed mapping")


def register_source(con: sqlite3.Connection, path: Path, kind: str, notes: str,
                    now: str) -> tuple[int, int, str]:
    aid, digest = register_local_artifact(
        con, path, kind="AUDIT_EVIDENCE", role=kind, notes=notes, observed_utc=now
    )
    rel = path.resolve().relative_to(REPO).as_posix()
    source_key = "two_neb_pair_20261123:" + rel + ":" + digest[:16]
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (source_key, str(path.resolve()), kind, digest, now, CAPTURE_UTC,
         "Hashed local receipt/script/manifest used to register the two May 2026 QE-NEB jobs.", notes),
    )
    source = con.execute(
        "SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)
    ).fetchone()
    if not source or tuple(source[1:]) != (str(path.resolve()), digest):
        raise RuntimeError("Registry source identity conflict for " + str(path))
    return int(source[0]), aid, digest


def semantic_for(row: dict) -> tuple[str, str, str, str, str, str]:
    role = row["role"]
    if role == "QE_MIXING_RESTART":
        return ("QE_RESTART_FILE", "QE_MIXING_RESTART", "REPRODUCIBLE_INTERMEDIATE",
                "LINEAGE_SUPPORTED", "USER_CONFIRMED", "Exact predelete .mixN full-file identity; internal restart state is optional for fresh-start reruns.")
    if role == "QE_PSEUDOPOTENTIAL":
        return ("PSEUDOPOTENTIAL", "QE_PSEUDOPOTENTIAL", "PERMANENT_SOURCE",
                "LINEAGE_SUPPORTED", "USER_CONFIRMED", "Named UPF dependency captured by full-file SHA-256 and metadata.")
    if role in {"QE_NEB_INPUT", "QE_NEB_PROTOCOL", "QE_ENDPOINT_INPUT", "QE_NEB_IMAGE_PATH"}:
        return ("QE_INPUT", role, "REPRODUCIBLE_INPUT", "LINEAGE_SUPPORTED", "USER_CONFIRMED",
                "Current captured QE input required for the reviewed fresh-start recipe; may postdate the historical May job.")
    if role == "SLURM_LAUNCH_SCRIPT":
        return ("SLURM_SCRIPT", role, "REPRODUCIBLE_INPUT", "LINEAGE_SUPPORTED", "USER_CONFIRMED",
                "Exact captured Slurm launch script; the helper submits it from a new isolated directory.")
    if role == "QE_NEB_OUTPUT_LOG":
        return ("QE_OUTPUT_LOG", role, "PUBLICATION_EVIDENCE", "FAILURE_MODE", "USER_CONFIRMED",
                "Historical NEB output contains SCF_WARNING; preserve as failed-result evidence.")
    if role == "QE_IMAGE_OUTPUT_LOG":
        image = PurePosixPath(row["path"]).parent.name
        failed = image == "sb2te3_4"
        return ("QE_OUTPUT_LOG", role, "PUBLICATION_EVIDENCE",
                "FAILURE_MODE" if failed else "LINEAGE_SUPPORTED", "USER_CONFIRMED",
                "Image 4 PW.out failed to converge after 400 iterations." if failed else "Per-image historical QE output log retained as run evidence.")
    if role == "QE_IMAGE_XML":
        return ("QE_COORDINATE_XML", role, "PUBLICATION_EVIDENCE", "LINEAGE_SUPPORTED", "USER_CONFIRMED",
                "Retained parsed QE image coordinate XML, with exact captured SHA-256.")
    if role == "SLURM_JOB_LOG":
        return ("SLURM_JOB_LOG", role, "PUBLICATION_EVIDENCE", "LINEAGE_SUPPORTED", "USER_CONFIRMED",
                "Historical Slurm stdout/stderr retained as execution evidence.")
    raise RuntimeError("Unmapped receipt role: " + str(role))


def main() -> None:
    receipt, run_by_id, rows_by_run, jobs = load_inputs()
    manifest_paths = build_manifests(receipt, run_by_id, rows_by_run, jobs)
    now = utc_now()
    con = sqlite3.connect(DB, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        if not con.execute("SELECT 1 FROM project WHERE project_id='MIGRATIONBENCH'").fetchone():
            raise RuntimeError("MIGRATIONBENCH project row is missing")
        con.execute("BEGIN IMMEDIATE")
        camp = con.execute("SELECT campaign_id,project_id FROM campaign WHERE campaign_id=?", (CAMPAIGN_ID,)).fetchone()
        if camp and tuple(camp) != (CAMPAIGN_ID, "MIGRATIONBENCH"):
            raise RuntimeError("Campaign ID is already bound to another project")
        con.execute(
            """INSERT OR IGNORE INTO campaign
               (campaign_id,project_id,campaign_name,source_campaign_key,start_date,end_date,status,notes)
               VALUES(?,?,?,?,?,?,?,?)""",
            (CAMPAIGN_ID, "MIGRATIONBENCH", "Cr–Sb₂Te₃ paired QE-NEB jobs 25090135/25090139",
             "rockfish:qe-neb-pair:25090135-25090139", "2026-05-22", "2026-05-22", "FAILED",
             "Scheduler jobs completed 0:0; the retained image-4 output shows SCF nonconvergence."),
        )

        # Build immutable local evidence sources and artifacts, including the generated capture manifests.
        evidence_paths = [RECEIPT, CAPTURE_SCRIPT, STAGING_HELPER, TRIAGE]
        evidence_paths.extend(manifest_paths.values())
        source_map = {}
        local_artifact_map = {}
        for path in evidence_paths:
            kind = "QE_NEB_PAIR_CAPTURE_MANIFEST" if path in manifest_paths.values() else (
                "QE_NEB_STAGING_HELPER" if path == STAGING_HELPER else (
                "QE_NEB_CAPTURE_SCRIPT" if path == CAPTURE_SCRIPT else (
                "QE_NEB_CAPTURE_RECEIPT" if path == RECEIPT else "QE_NEB_TRIAGE_REVIEW")))
            notes = "Local SHA-256-pinned evidence for the two-job QE-NEB run catalog and fresh-start recipe."
            source_id, aid, digest = register_source(con, path, kind, notes, now)
            source_map[path] = source_id
            local_artifact_map[path] = aid

        remote_aids: dict[tuple[str, str], int] = {}
        unique_remote: dict[str, dict] = {}
        for row in receipt["files"]:
            prior = unique_remote.get(row["path"])
            if prior and (prior["sha256"], prior["size_bytes"]) != (row["sha256"], row["size_bytes"]):
                raise RuntimeError("Duplicate path has inconsistent receipt identity: " + row["path"])
            unique_remote[row["path"]] = row
        for path, row in unique_remote.items():
            if path in REMOTE_ENV:
                kind = "HPC_ENVIRONMENT_COMPONENT"
                role = "MODULE_BOOTSTRAP" if path.endswith("go.sh") else "QE_NEB_EXECUTABLE"
                retention, science, owner = "PERMANENT_SOURCE", "LINEAGE_SUPPORTED", "SHARED_OWNER_CONFIRMED"
                notes = "Shared Rockfish QE 7.2 runtime dependency captured by full-file SHA-256."
            else:
                kind, role, retention, science, owner, notes = semantic_for(row)
            remote_aids[(path, row["sha256"])] = register_remote_artifact(
                con, row, project_id="MIGRATIONBENCH", artifact_kind=kind, semantic_role=role,
                retention=retention, science=science, owner=owner, notes=notes, observed_utc=CAPTURE_UTC,
            )

        # Preserve the complete captured stat/hash metadata in three scoped inventory snapshots.
        by_root: dict[str, dict[str, dict]] = {ROOT_DATA: {}, ROOT_HOME: {}, ROOT_FS: {}}
        for path, row in unique_remote.items():
            root_id, relative = root_for_path(path)
            by_root[root_id][relative] = row
        for root_id, entries in by_root.items():
            snapshot_id = "QE-NEB-PAIR-CAPTURE-" + root_id + "-20261123T112510Z"
            logical = sum(int(row["size_bytes"]) for row in entries.values())
            allocated = sum(int(row.get("allocated_bytes", row["size_bytes"])) for row in entries.values())
            con.execute(
                """INSERT OR IGNORE INTO inventory_snapshot
                   (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                    files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                    manifest_path,manifest_sha256,notes)
                   VALUES(?,?,?,?,?,'capture_two_neb_mix_pair_predelete_hashes_20261123.py','COMPLETE',?,0,0,0,?,?,?,?,'Exact selected-file predelete capture; not a recursive filesystem inventory.')""",
                (snapshot_id, root_id, CAPTURE_UTC, CAPTURE_UTC, "Stable full-file SHA-256 plus lstat metadata capture",
                 len(entries), logical, allocated, str(RECEIPT.resolve()), sha256_file(RECEIPT)),
            )
            for relative, row in entries.items():
                path_id = ensure_storage_path(con, root_id, relative, CAPTURE_UTC)
                mode = row.get("mode_octal")
                if mode and not mode.startswith("0"):
                    mode = "0" + mode
                con.execute(
                    """INSERT OR IGNORE INTO inventory_observation
                       (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,
                        mtime_ns,inode,device,sha256,hash_scope,observation_status,link_count)
                       VALUES(?,?,'FILE',?,?,?,?,?,?,?,?,?,'FULL_FILE','PRESENT',?)""",
                    (snapshot_id, path_id, row.get("uid"), row.get("gid"), mode, int(row["size_bytes"]),
                     int(row.get("allocated_bytes", row["size_bytes"])), int(row["mtime_ns"]),
                     str(row["inode"]), str(row["device"]), row["sha256"], int(row.get("nlink", 1))),
                )

        # Materialize paired workflow runs and exact historical scheduler observations.
        run_artifacts: dict[str, dict[str, int]] = {}
        manifest_artifacts: dict[str, int] = {}
        for job_id, config in RUNS.items():
            run_id = config["run_id"]
            source = run_by_id[run_id]
            job = jobs[job_id]
            workdir = source["run_root"]
            root_id, relative = root_for_path(workdir + "/marker")
            relative_root = str(PurePosixPath(relative).parent)
            expected_run = (
                "MIGRATIONBENCH", CAMPAIGN_ID, config["label"], "DFT_NEB", "Quantum ESPRESSO 7.2 NEB",
                "Cr-Sb2Te3", root_id, relative_root, "COMPLETED", "FAILURE_MODE",
            )
            con.execute(
                """INSERT OR IGNORE INTO workflow_run
                   (run_id,project_id,campaign_id,run_label,workflow_type,method,system_key,source_root_id,
                    source_relative_path,scheduler_state,scientific_status,start_utc,end_utc,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?)""",
                (run_id, *expected_run,
                 "Historical Slurm job completed with exit 0:0. Scientific status is FAILURE_MODE: neb.out has SCF_WARNING and image-4 PW.out failed after 400 iterations. Current captured inputs may postdate the historical May 2026 job."),
            )
            actual_run = con.execute(
                """SELECT project_id,campaign_id,run_label,workflow_type,method,system_key,source_root_id,
                          source_relative_path,scheduler_state,scientific_status
                   FROM workflow_run WHERE run_id=?""", (run_id,)
            ).fetchone()
            if not actual_run or tuple(actual_run) != expected_run:
                raise RuntimeError("Existing workflow_run identity/status conflicts for " + run_id)

            artifact_map: dict[str, int] = {}
            for row in rows_by_run[run_id]:
                aid = remote_aids[(row["path"], row["sha256"])]
                artifact_map[row["path"]] = aid
                if row["role"] == "QE_MIXING_RESTART":
                    add_run_artifact(con, run_id, aid, "qe_mix_restart_intermediate", "INTERMEDIATE", False,
                        "Optional QE mixing history; fresh-start recipe excludes it and regenerates intermediate state.")
                elif row["role"] in {"QE_NEB_OUTPUT_LOG", "QE_IMAGE_OUTPUT_LOG", "QE_IMAGE_XML"}:
                    add_run_artifact(con, run_id, aid, row["role"].lower(), "OUTPUT", False,
                        "Historical output/evidence; retained and not overwritten by the fresh-start recipe.")
                elif row["role"] == "SLURM_JOB_LOG":
                    add_run_artifact(con, run_id, aid, "slurm_stdout_stderr", "LOG", False,
                        "Historical scheduler stdout/stderr evidence retained unchanged.")
                else:
                    add_run_artifact(con, run_id, aid, row["role"].lower(), "INPUT", True,
                        "Captured current input/script/dependency required by the reviewed fresh-start recipe.")

            for path, env_role in REMOTE_ENV.items():
                aid = remote_aids[(path, unique_remote[path]["sha256"])]
                artifact_map[path] = aid
                add_run_artifact(con, run_id, aid, env_role.lower(), "ENVIRONMENT", True,
                    "Shared captured Rockfish runtime dependency used by the reviewed QE 7.2 recipe.")

            manifest_path = manifest_paths[run_id]
            manifest_aid = local_artifact_map[manifest_path]
            manifest_artifacts[run_id] = manifest_aid
            add_run_artifact(con, run_id, manifest_aid, "run_capture_manifest", "PROVENANCE", True,
                "Local hashed manifest binds exact remote capture paths, content hashes, metadata, runtime, Slurm evidence and fresh-start staging command.")
            for local_path in (RECEIPT, CAPTURE_SCRIPT, STAGING_HELPER, TRIAGE):
                add_run_artifact(con, run_id, local_artifact_map[local_path], local_path.stem.lower(), "PROVENANCE", False,
                    "Local source evidence for the captured run mapping and recipe review.")

            observed = con.execute(
                "SELECT scheduler_observation_id FROM scheduler_job_source_observation WHERE run_id=? AND cluster_name='Rockfish' AND job_id=? AND observed_utc=?",
                (run_id, job_id, CAPTURE_UTC),
            ).fetchone()
            receipt_aid = local_artifact_map[RECEIPT]
            con.execute(
                """INSERT OR IGNORE INTO scheduler_job_source_observation
                   (run_id,cluster_name,job_id,job_name,sacct_state,exit_code,submit_time_raw,start_time_raw,
                    end_time_raw,workdir,observed_utc,source_artifact_id,notes)
                   VALUES(?,'Rockfish',?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, job_id, job["job_name"], job["state"], job["exit_code"], job["submit_raw"],
                 job["start_raw"], job["end_raw"], job["workdir"], CAPTURE_UTC, receipt_aid,
                 "Raw sacct times are preserved exactly as returned without timezone suffix; no normalized Slurm timestamp was inferred. Historical submitline: " + job["submitline"]),
            )
            observation = con.execute(
                "SELECT source_artifact_id,workdir,exit_code FROM scheduler_job_source_observation WHERE run_id=? AND cluster_name='Rockfish' AND job_id=? AND observed_utc=?",
                (run_id, job_id, CAPTURE_UTC),
            ).fetchone()
            if not observation or tuple(observation) != (receipt_aid, job["workdir"], "0:0"):
                raise RuntimeError("Existing scheduler source observation conflicts for " + job_id)

            helper_aid = local_artifact_map[STAGING_HELPER]
            slurm_aid = artifact_map[workdir + "/slurm-neb.slurm"]
            runtime_modules = receipt["runtime_probe"]["stdout"].split("Currently Loaded Modules:", 1)[-1].strip()
            validation = (
                "REVIEWED fresh-start procedure. Exact local entry: " +
                json.loads(manifest_path.read_text(encoding="utf-8"))["staging_helper"]["exact_command"] +
                ". The helper verifies captured input, UPF, slurm script, /data/apps/go.sh and QE 7.2 neb.x SHA-256s; stages only the six current input files into the independent new scratch16 directory; refuses overwrite or source/destination nesting; then invokes sbatch slurm-neb.slurm from that new directory. It does not reuse or overwrite the historical run directory. Current inputs may postdate the May 2026 job. Historical neb.out contains SCF_WARNING; image-4 PW.out failed to converge after 400 iterations. The recipe targets a fresh run from current captured inputs and does not guarantee byte-for-byte reproduction of the historical image-4 warning. No rerun has been submitted."
            )
            exact_command = json.loads(manifest_path.read_text(encoding="utf-8"))["staging_helper"]["exact_command"]
            con.execute(
                """INSERT OR IGNORE INTO reproduction_recipe
                   (run_id,recipe_version,exact_command,working_directory,executable_version,
                    environment_description,module_list,script_artifact_id,scheduler_script_artifact_id,
                    input_manifest_artifact_id,recipe_status,last_reproduced_utc,validation_note)
                   VALUES(?,?,?,?,?,?,?,?,?,?, 'REVIEWED',NULL,?)""",
                (run_id, "fresh-start-stage-submit-2026-11-23-v1", exact_command,
                 FRESH_ROOT + "/" + job_id, "Quantum ESPRESSO 7.2 neb.x SHA-256=" + unique_remote["/data/apps/extern/qe/7.2/bin/neb.x"]["sha256"],
                 "Rockfish; bash login shell; source /data/apps/go.sh; ml intel; ml qe/7.2; submit from a new empty scratch16 directory.",
                 runtime_modules, helper_aid, slurm_aid, manifest_aid, validation),
            )
            recipe = con.execute(
                "SELECT recipe_id,exact_command,recipe_status,script_artifact_id,scheduler_script_artifact_id,input_manifest_artifact_id FROM reproduction_recipe WHERE run_id=? AND recipe_version='fresh-start-stage-submit-2026-11-23-v1'",
                (run_id,),
            ).fetchone()
            expected_recipe = (exact_command, "REVIEWED", helper_aid, slurm_aid, manifest_aid)
            if not recipe or tuple(recipe[1:]) != expected_recipe:
                raise RuntimeError("Existing recipe identity or evidence links conflict for " + run_id)
            recipe_id = int(recipe[0])
            for row in rows_by_run[run_id]:
                role = row["role"]
                if role in {"QE_NEB_INPUT", "QE_NEB_PROTOCOL", "QE_ENDPOINT_INPUT", "QE_NEB_IMAGE_PATH", "QE_PSEUDOPOTENTIAL", "SLURM_LAUNCH_SCRIPT"}:
                    con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",
                                (recipe_id, artifact_map[row["path"]], role))
            for env_path, env_role in REMOTE_ENV.items():
                con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",
                            (recipe_id, artifact_map[env_path], env_role))
            for path, role in ((STAGING_HELPER, "STAGING_HELPER"), (manifest_path, "CAPTURE_MANIFEST")):
                con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",
                            (recipe_id, local_artifact_map[path], role))

            # Scheduler and scientific quality are distinct assertions.
            for gate_id, status, evidence_path, rule, notes in (
                ("slurm_exit", "PASS", workdir + "/slurm." + job_id,
                 "Captured sacct state COMPLETED and exit code 0:0.", "Scheduler accounting only; not scientific acceptance."),
                ("neb_image4_scf", "FAIL", workdir + "/out/sb2te3_4/PW.out",
                 "Image 4 must reach self-consistent convergence within the recorded iteration limit.",
                 "Captured PW.out reports failure to converge after 400 iterations."),
                ("scientific_acceptance", "FAIL", workdir + "/neb.out",
                 "NEB result has no unresolved SCF warnings.",
                 "Captured neb.out contains SCF_WARNING; this run is FAILURE_MODE and not validated as a converged MEP."),
            ):
                con.execute(
                    """INSERT OR IGNORE INTO quality_gate
                       (run_id,gate_id,status,checked_utc,evidence_artifact_id,rule_description,notes)
                       VALUES(?,?,?,?,?,?,?)""",
                    (run_id, gate_id, status, CAPTURE_UTC, artifact_map[evidence_path], rule, notes),
                )
                q = con.execute("SELECT status,evidence_artifact_id FROM quality_gate WHERE run_id=? AND gate_id=?", (run_id, gate_id)).fetchone()
                if not q or tuple(q) != (status, artifact_map[evidence_path]):
                    raise RuntimeError("Existing quality gate conflicts for " + run_id + "/" + gate_id)

            for evidence_path in (RECEIPT, CAPTURE_SCRIPT, STAGING_HELPER, TRIAGE, manifest_path):
                add_ref(con, source_map[evidence_path], run_id,
                        "Hashed local evidence supporting the paired QE-NEB run and reviewed fresh-start reproduction links.")
            run_artifacts[run_id] = artifact_map

        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("Foreign key check failed: " + repr([tuple(row) for row in fk[:20]]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect("file:" + str(DB) + "?mode=ro", uri=True, timeout=30) as check:
        ids = tuple(config["run_id"] for config in RUNS.values())
        placeholders = ",".join("?" for _ in ids)
        counts = {
            "workflow_runs": check.execute("SELECT count(*) FROM workflow_run WHERE run_id IN (" + placeholders + ")", ids).fetchone()[0],
            "recipes": check.execute("SELECT count(*) FROM reproduction_recipe WHERE run_id IN (" + placeholders + ")", ids).fetchone()[0],
            "scheduler_source_observations": check.execute("SELECT count(*) FROM scheduler_job_source_observation WHERE run_id IN (" + placeholders + ")", ids).fetchone()[0],
            "run_artifacts": check.execute("SELECT count(*) FROM run_artifact WHERE run_id IN (" + placeholders + ")", ids).fetchone()[0],
            "mix_intermediate_links": check.execute("SELECT count(*) FROM run_artifact WHERE run_id IN (" + placeholders + ") AND direction='INTERMEDIATE'", ids).fetchone()[0],
            "run_output_links": check.execute("SELECT count(*) FROM run_artifact WHERE run_id IN (" + placeholders + ") AND direction='OUTPUT'", ids).fetchone()[0],
            "recipe_inputs": check.execute("SELECT count(*) FROM recipe_input ri JOIN reproduction_recipe rr USING(recipe_id) WHERE rr.run_id IN (" + placeholders + ")", ids).fetchone()[0],
            "quality_gates": check.execute("SELECT count(*) FROM quality_gate WHERE run_id IN (" + placeholders + ")", ids).fetchone()[0],
            "mix_artifacts": check.execute("SELECT count(*) FROM artifact WHERE artifact_key LIKE 'RF-DATA-YI:%/V_ql_shallow-ql_mid_high_primary-primary/out/sb2te3_4/sb2te3.mix%' OR artifact_key LIKE 'RF-DATA-YI:%/V_ql_mid_te-ql_deep_primary-primary/out/sb2te3_4/sb2te3.mix%'", ()).fetchone()[0],
            "cleanup_batches_created_for_pair": check.execute("SELECT count(*) FROM cleanup_batch WHERE cleanup_batch_id LIKE '%25090135%' OR cleanup_batch_id LIKE '%25090139%'", ()).fetchone()[0],
            "foreign_key_violations": len(check.execute("PRAGMA foreign_key_check").fetchall()),
        }
        print(json.dumps(counts, sort_keys=True))
        for run_id in ids:
            print(json.dumps({"run_id": run_id,
                "run_artifacts_by_direction": check.execute("SELECT direction,count(*) FROM run_artifact WHERE run_id=? GROUP BY direction ORDER BY direction", (run_id,)).fetchall(),
                "recipe": check.execute("SELECT recipe_status,exact_command,script_artifact_id,scheduler_script_artifact_id,input_manifest_artifact_id FROM reproduction_recipe WHERE run_id=?", (run_id,)).fetchone()}, default=list, sort_keys=True))


if __name__ == "__main__":
    main()
