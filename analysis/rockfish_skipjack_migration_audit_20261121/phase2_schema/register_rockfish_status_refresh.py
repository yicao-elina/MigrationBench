#!/usr/bin/env python3
"""Register timestamped all-user queue and root access-boundary captures."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
LOCAL_ROOT = "LOCAL-MAIN"

CAPTURES = [
    (
        AUDIT / "phase2_data/git_ref_update_preflight_receipts_20261123.jsonl",
        "rockfish_git_ref_update_preflight_20261123T023959Z",
        "GIT_REF_UPDATE_PREFLIGHT_RECEIPT",
        "GIT_REF_UPDATE_PREFLIGHT",
        "Local-only Git ref update preflight; all candidate writes were deferred due expired /data group quota and checked-out DynamicSciKG main.",
        "PERMANENT_SOURCE",
        "Records exact before/after refs and worktree status for three Rockfish repositories; no ref or file changed and no push occurred.",
    ),
    (
        AUDIT / "phase2_data/revision1_restart_population_refresh_20261123.txt",
        "rockfish_revision1_restart_population_recheck_20261123T0142Z",
        "TARGETED_RESTART_FILENAME_RECHECK",
        "RESTART_FILE_POPULATION_SNAPSHOT",
        "Current filename/size population for WFC- and mixing-named files in the revision1 run tree.",
        "PERMANENT_SOURCE",
        "Read-only find -xdev metadata aggregation over file sizes and basenames; it classifies candidates only.",
    ),
    (
        AUDIT / "phase2_data/revision1_gt500_refresh_20261123.txt",
        "rockfish_revision1_gt500_recheck_20261123T0141Z",
        "TARGETED_LARGE_FILE_THRESHOLD_RECHECK",
        "LARGE_FILE_THRESHOLD_SNAPSHOT",
        "Read-only threshold refresh of the previously reported revision1 migrationbench run tree.",
        "PERMANENT_SOURCE",
        "Remote find -xdev metadata-only query; no paths are emitted and no file payloads are read.",
    ),
    (
        AUDIT / "phase2_scratch4/scratch4_refresh_20261123/squeue_group_20261123.txt",
        "rockfish_scratch4_group_queue_refresh_20261123T0131Z",
        "SLURM_QUEUE_PCLANCY3_GROUP_CAPTURE",
        "SLURM_QUEUE_SNAPSHOT",
        "Fresh group-account queue query used to exclude live Scratch4 project paths from cleanup candidates.",
        "PERMANENT_SOURCE",
        "First line is the exact remote capture timestamp; remaining rows include job IDs, state, and WorkDir.",
    ),
    (
        AUDIT / "phase2_data/root_owner_boundary_refresh_20261123T0130Z.txt",
        "rockfish_root_owner_boundary_refresh_20261123T0130Z",
        "STORAGE_ROOT_OWNER_BOUNDARY_CAPTURE",
        "STORAGE_OWNER_BOUNDARY_SNAPSHOT",
        "Aggregate immediate-child ownership boundary at /data and /scratch4 for UID 3901 and project GID 3001.",
        "PERMANENT_SOURCE",
        "Read-only immediate-child lstat summary; only project-group paths are emitted, no unrelated owner names.",
    ),
    (
        AUDIT / "phase2_data/rockfish_quota_refresh_20261123T0115Z.txt",
        "rockfish_quota_capture_rockfish_quota_refresh_20261123T0115Z",
        "GPFS_QUOTA_CAPTURE",
        "GPFS_QUOTA_SNAPSHOT",
        "Fresh machine-readable GPFS quota capture for the ycao73 user on /data and pclancy3 group on /data and /scratch4.",
        "PERMANENT_SOURCE",
        "First line is the exact remote capture timestamp; GPFS -Y block counters are in 1 KiB units and file counts are separate.",
    ),
    (
        AUDIT / "phase2_data/current_queue_all_users_refresh_20261123T0115Z.txt",
        "rockfish_current_alluser_queue_refresh_20261123T0115Z",
        "SLURM_QUEUE_ALL_USERS_CAPTURE",
        "SLURM_QUEUE_SNAPSHOT",
        "Fresh all-user queue snapshot used to identify current work-directory overlap under /data and /scratch4.",
        "PERMANENT_SOURCE",
        "First line is the exact remote capture timestamp; remaining rows are job id, user, state, and WorkDir.",
    ),
    (
        AUDIT / "phase2_scratch4/scratch4_refresh_20261123_r03/squeue_allusers_r03.txt",
        "rockfish_current_alluser_queue_r03_20261123T043622Z",
        "SLURM_QUEUE_ALL_USERS_CAPTURE",
        "SLURM_QUEUE_SNAPSHOT",
        "Latest successful all-user queue snapshot used for the scoped R03 inventories; the 05:01 UTC refresh failed at Slurm DNS configuration lookup.",
        "PERMANENT_SOURCE",
        "Capture timestamp is 2026-11-23T04:36:22Z; 1,407 jobs were parsed, including two WorkDirs under /scratch4/pclancy3 and none under /data/pclancy3.",
    ),
    (
        AUDIT / "phase2_data/rockfish_final_quota_20261123T0506Z/quota_capture.txt",
        "rockfish_final_quota_20261123T050248Z",
        "GPFS_QUOTA_CAPTURE",
        "GPFS_QUOTA_SNAPSHOT",
        "Latest successful GPFS quota capture for the /data user quota and /data and /scratch4 group quotas.",
        "PERMANENT_SOURCE",
        "Machine-readable timestamped mmlsquota -Y capture; 1 KiB block units are stored as bytes and group usage is not allocated to individual paths.",
    ),
    (
        AUDIT / "phase2_data/root_permission_refresh_20261123T0115Z.txt",
        "rockfish_root_permission_refresh_20261123T0115Z",
        "STORAGE_ROOT_ACCESS_BOUNDARY_CAPTURE",
        "STORAGE_ACCESS_BOUNDARY_SNAPSHOT",
        "Aggregate immediate-child traversal permissions at /data and /scratch4 and readability of the pclancy3 roots.",
        "PERMANENT_SOURCE",
        "Filename records capture minute 2026-11-23T01:15Z; output stores aggregate counts only and no other-user names.",
    ),
    (
        AUDIT / "phase2_data/rockfish_live_quota_20261123T060413Z/stdout.txt",
        "rockfish_live_gpfs_quota_20261123T060420Z",
        "GPFS_QUOTA_CAPTURE",
        "GPFS_QUOTA_SNAPSHOT",
        "Latest successful timestamped quota read for user ycao73 and group pclancy3 on /data and /scratch4.",
        "PERMANENT_SOURCE",
        "Read-only /usr/lpp/mmfs/bin/mmlsquota -Y output at 2026-11-23T06:04:20Z; group usage is shared and is not path-localized.",
    ),
    (
        AUDIT / "phase2_data/rockfish_live_quota_20261123T060413Z/execution_status.json",
        "rockfish_live_gpfs_quota_status_20261123T060413Z",
        "GPFS_QUOTA_CAPTURE_STATUS",
        "GPFS_QUOTA_CAPTURE_STATUS",
        "Execution status for the latest read-only quota capture.",
        "PERMANENT_SOURCE",
        "Records exact SSH argv, return code, timestamp, and byte counts for the quota capture.",
    ),
    (
        AUDIT / "phase2_data/rockfish_live_queue_20261123T060413Z/stdout.txt",
        "rockfish_live_slurm_queue_allusers_20261123T060428Z",
        "SLURM_QUEUE_ALL_USERS_CAPTURE",
        "SLURM_QUEUE_SNAPSHOT",
        "Latest successful all-user Slurm queue capture used to guard current active work directories.",
        "PERMANENT_SOURCE",
        "Read-only squeue output at 2026-11-23T06:04:28Z; 1,401 job rows, with zero under /data/pclancy3 and two under /scratch4/pclancy3.",
    ),
    (
        AUDIT / "phase2_data/rockfish_live_queue_20261123T060413Z/execution_status.json",
        "rockfish_live_slurm_queue_status_20261123T060413Z",
        "SLURM_QUEUE_CAPTURE_STATUS",
        "SLURM_QUEUE_CAPTURE_STATUS",
        "Execution status for the latest all-user Slurm queue capture.",
        "PERMANENT_SOURCE",
        "Records exact SSH argv, return code, timestamp, and byte counts for the all-user queue capture.",
    ),
    (
        AUDIT / "phase2_data/yi_only_scope_refresh_20261123T062955Z/user_quota.txt",
        "rockfish_yi_scope_user_quota_20261123T062956Z",
        "GPFS_USER_QUOTA_CAPTURE",
        "GPFS_USER_QUOTA_SNAPSHOT",
        "Fresh ycao73 user quota observations for data and scratch4 taken for the two yi subtree cleanup scopes.",
        "PERMANENT_SOURCE",
        "Read-only /usr/lpp/mmfs/bin/mmlsquota -u ycao73 -Y data and scratch4; filesystem totals are not directory-localized.",
    ),
    (
        AUDIT / "phase2_data/yi_only_scope_refresh_20261123T062955Z/user_quota.status.json",
        "rockfish_yi_scope_user_quota_status_20261123T062955Z",
        "GPFS_USER_QUOTA_CAPTURE_STATUS",
        "GPFS_USER_QUOTA_CAPTURE_STATUS",
        "Execution receipt for the user-only data and scratch4 quota capture.",
        "PERMANENT_SOURCE",
        "Exact SSH command, exit status, and capture time; no group or other-user quota rows were requested.",
    ),
    (
        AUDIT / "phase2_data/yi_only_scope_refresh_20261123T062955Z/queue.txt",
        "rockfish_yi_scope_queue_20261123T063005Z",
        "SLURM_QUEUE_TARGETED_CAPTURE",
        "SLURM_QUEUE_SNAPSHOT",
        "Filtered live Slurm queue rows for /data/pclancy3/yi and /scratch4/pclancy3/yi only.",
        "PERMANENT_SOURCE",
        "Read-only squeue query filtered on Rockfish before output; this capture contains no other WorkDir rows.",
    ),
    (
        AUDIT / "phase2_data/yi_only_scope_refresh_20261123T062955Z/queue.status.json",
        "rockfish_yi_scope_queue_status_20261123T062955Z",
        "SLURM_QUEUE_CAPTURE_STATUS",
        "SLURM_QUEUE_CAPTURE_STATUS",
        "Execution receipt for the target-filtered Slurm queue query.",
        "PERMANENT_SOURCE",
        "Exact SSH command, exit status, and byte counts for the queue query filtered to the two yi roots.",
    ),
]

_MOUNT_BOUNDARY_DIR = AUDIT / "phase2_data"
for _path in sorted(_MOUNT_BOUNDARY_DIR.glob("mount_access_boundary_*.json")):
    _status = _path.name.endswith(".status.json")
    _stamp = _path.name.removeprefix("mount_access_boundary_")
    _stamp = _stamp.removesuffix(".status.json" if _status else ".json")
    CAPTURES.append((
        _path,
        "rockfish_mount_access_boundary" + ("_status" if _status else "") + "_" + _stamp,
        "STORAGE_ROOT_ACCESS_BOUNDARY_CAPTURE_STATUS" if _status else "STORAGE_ROOT_ACCESS_BOUNDARY_CAPTURE",
        "STORAGE_ACCESS_BOUNDARY_CAPTURE_STATUS" if _status else "STORAGE_ACCESS_BOUNDARY_SNAPSHOT",
        "Execution status for an immediate-root permission/ownership probe." if _status else "Read-only immediate-root permission and ownership counts for /data and /scratch4.",
        "PERMANENT_SOURCE",
        "Timestamped no-follow mount-root probe; it reads no directory contents beyond immediate child traversal checks and does not authorize cleanup.",
    ))

for capture_dir in sorted((AUDIT / "phase2_data").glob("selected_cleanup_queue_*")):
    if not capture_dir.is_dir():
        continue
    stamp = capture_dir.name.removeprefix("selected_cleanup_queue_")
    CAPTURES.extend([
        (capture_dir / "stdout.json", "rockfish_selected_cleanup_queue_" + stamp,
         "SLURM_QUEUE_SELECTED_ROOTS_CAPTURE", "SLURM_QUEUE_SNAPSHOT",
         "Current all-user queue capture filtered remotely to selected yi and collaborator cleanup roots.",
         "PERMANENT_SOURCE", "Records total parsed all-user jobs, raw queue SHA256, unknown WorkDir count, and only matching WorkDir rows."),
        (capture_dir / "execution_status.json", "rockfish_selected_cleanup_queue_status_" + stamp,
         "SLURM_QUEUE_SELECTED_ROOTS_STATUS", "SLURM_QUEUE_CAPTURE_STATUS",
         "Execution receipt for the selected-root all-user queue capture.",
         "PERMANENT_SOURCE", "Captures SSH exit code, selected roots, timestamp, and exact stdout/stderr hashes."),
        (capture_dir / "stderr.txt", "rockfish_selected_cleanup_queue_stderr_" + stamp,
         "SLURM_QUEUE_SELECTED_ROOTS_STDERR", "SLURM_QUEUE_CAPTURE_STATUS",
         "SSH stderr for the selected-root all-user queue query.",
         "PERMANENT_SOURCE", "Raw SSH stderr bytes; empty on successful capture."),
    ])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture_time(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0][:4].isdigit() and "T" in lines[0]:
        dt = datetime.fromisoformat(lines[0])
    elif lines and lines[0].lstrip().startswith("{"):
        payload = lines[0] if path.suffix == ".jsonl" else path.read_text(encoding="utf-8")
        first = json.loads(payload)
        stamp = (first.get("observed_utc") or first.get("observed_local_utc")
                 or first.get("captured_utc")
                 or first.get("finished_local_utc") or first.get("finished_utc"))
        if stamp is None and isinstance(first.get("captured_epoch"), (int, float)):
            dt = datetime.fromtimestamp(float(first["captured_epoch"]), timezone.utc)
        elif stamp is None:
            raise ValueError("JSON capture has no observation or completion timestamp: " + str(path))
        else:
            try:
                dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    elif path.name in {"stderr.txt", "stderr.json"} and path.parent.name.startswith("selected_cleanup_queue_"):
        stamp = path.parent.name.removeprefix("selected_cleanup_queue_")
        dt = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    else:
        # The permission probe emits only aggregate lines; its filename records the minute.
        stamp = path.stem.rsplit("_", 1)[-1]
        dt = datetime.strptime(stamp, "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=180)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    registered = []
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT,)).fetchone()
        if not root or Path(root[0]).resolve() != REPO.resolve():
            raise RuntimeError("LOCAL-MAIN must resolve to the unified worktree")
        for path, source_key, source_kind, artifact_kind, role, retention, authority in CAPTURES:
            path = path.resolve(strict=True)
            sha = digest(path)
            observed = capture_time(path)
            rel = path.relative_to(REPO).as_posix()
            size = path.stat().st_size
            prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
            if prior and prior != (str(path), sha):
                raise RuntimeError(f"Source key already identifies different bytes: {source_key}")
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                    immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (source_key, str(path), source_kind, sha, observed, observed, authority, role),
            )
            artifact_key = "ROCKFISH-STATUS-CAPTURE:" + source_key
            prior_art = con.execute("SELECT sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if prior_art and prior_art != (sha, size):
                raise RuntimeError(f"Artifact key already identifies different bytes: {artifact_key}")
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                    scientific_status,retention_class,notes)
                   VALUES('ROCKFISH_STORAGE_AUDIT',?,?,?,?,?,?,'UNKNOWN',?,?)""",
                (artifact_key, artifact_kind, role, "USER_CONFIRMED", sha, size, retention,
                 f"Exact raw capture sha256={sha}; observed_utc={observed}."),
            )
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
                        (LOCAL_ROOT, rel, observed))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
                                  (LOCAL_ROOT, rel)).fetchone()[0]
            artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()[0]
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""",
                (artifact_id, path_id, observed, now),
            )
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES((SELECT source_id FROM registry_source WHERE source_key=?),
                          'status_capture',?,'ARTIFACT',?,'CAPTURES',?)""",
                (source_key, rel, artifact_key, authority),
            )
            registered.append((source_key, sha, observed, size))
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError(f"Foreign-key check failed: {fk[:3]}")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    for source_key, sha, observed, size in registered:
        print(f"registered={source_key}\tsha256={sha}\tobserved_utc={observed}\tbytes={size}")


if __name__ == "__main__":
    main()
