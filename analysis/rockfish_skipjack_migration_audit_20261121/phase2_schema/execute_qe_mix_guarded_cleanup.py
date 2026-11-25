#!/usr/bin/env python3
"""Run the exact SQL-staged QE mix cleanup after live scheduler and hash guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
MANIFEST = AUDIT / "phase2_data" / "yi_refresh_20261123" / "qe_mix_reproduction_manifest_20261123.json"
BATCH_ID = "RF-DATA-YI-CR-IN-GAP-MIX-CLEANUP-20261123"
QUOTA_ROOT = "RF-DATA-FS"
REMOTE_PROGRAM = r'''#!/usr/bin/env python3
import datetime, hashlib, json, os, stat, subprocess, sys

manifest = json.loads(__MANIFEST_JSON__)
items = [x for run in manifest["runs"] for x in run["mix_intermediates"]]
expected_paths = {x["path"] for x in items}
if len(items) != 12 or len(expected_paths) != 12:
    raise SystemExit("guard failed: manifest is not exactly 12 unique paths")
if any(not p.startswith("/data/pclancy3/yi/") for p in expected_paths):
    raise SystemExit("guard failed: unexpected path root")

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def emit(event, **fields):
    print(json.dumps({"event": event, "utc": utc_now(), **fields}, sort_keys=True), flush=True)

def queue_snapshot():
    env = os.environ.copy()
    env["SLURM_CONF"] = "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf"
    command = ["/cm/shared/apps/slurm/current/bin/squeue", "-a", "-h", "-o", "%i|%u|%T|%Z"]
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        raise RuntimeError("squeue failed: " + result.stderr[-2000:])
    rows = []
    overlaps = []
    for raw in result.stdout.splitlines():
        fields = raw.split("|", 3)
        if len(fields) != 4 or not fields[0] or not fields[1] or not fields[2]:
            raise RuntimeError("squeue produced an unparseable row: " + raw[:500])
        job_id, user, state, workdir = fields
        workdir = workdir.strip()
        if not workdir or workdir in ("(null)", "Unknown", "N/A"):
            raise RuntimeError("squeue row lacks a work directory; cannot prove zero overlap: " + raw[:500])
        workdir = os.path.normpath(workdir)
        row = {"job_id": job_id, "user": user, "state": state, "workdir": workdir}
        rows.append(row)
        for candidate in expected_paths:
            try:
                shared = os.path.commonpath((candidate, workdir))
            except ValueError:
                continue
            if shared == workdir or shared == candidate:
                overlaps.append({"job_id": job_id, "user": user, "state": state,
                                 "workdir": workdir, "candidate": candidate})
    if overlaps:
        raise RuntimeError("live all-user queue overlaps cleanup paths: " + json.dumps(overlaps, sort_keys=True))
    return rows

def identity_and_hash(item):
    path = item["path"]
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode):
        raise RuntimeError("candidate is not a regular file: " + path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before_fd = os.fstat(fd)
        h = hashlib.sha256()
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after_path = os.lstat(path)
    identity = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
    if any(getattr(before_fd, key) != getattr(after_fd, key) for key in identity):
        raise RuntimeError("candidate changed while hashing: " + path)
    if any(getattr(before_path, key) != getattr(after_path, key) for key in identity):
        raise RuntimeError("candidate path changed while hashing: " + path)
    actual = {
        "device": after_fd.st_dev,
        "inode": after_fd.st_ino,
        "uid": after_fd.st_uid,
        "gid": after_fd.st_gid,
        "mode_octal": format(stat.S_IMODE(after_fd.st_mode), "04o"),
        "nlink": after_fd.st_nlink,
        "size_bytes": after_fd.st_size,
        "mtime_ns": after_fd.st_mtime_ns,
        "allocated_bytes": getattr(after_fd, "st_blocks", 0) * 512,
        "sha256": h.hexdigest(),
    }
    expected = {key: item[key] for key in actual}
    if actual != expected:
        raise RuntimeError("candidate identity/hash mismatch for " + path + "; actual=" + json.dumps(actual, sort_keys=True))
    return actual

try:
    queue_rows = queue_snapshot()
    verified = []
    for item in items:
        verified.append({"path": item["path"], **identity_and_hash(item)})
    emit("preflight_passed", queue_overlaps=0, queue_rows=queue_rows,
         candidate_count=len(verified), candidates=verified,
         queue_command="SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf /cm/shared/apps/slurm/current/bin/squeue -a -h -o '%i|%u|%T|%Z'")
    logical_total = 0
    allocated_total = 0
    for item in items:
        # Query every user again immediately before each unlink to close the queue race window.
        queue_rows = queue_snapshot()
        actual = identity_and_hash(item)
        path = item["path"]
        os.unlink(path)
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("unlink returned but candidate remains present: " + path)
        logical_total += int(item["size_bytes"])
        allocated_total += int(actual["allocated_bytes"])
        emit("deleted_verified", path=path, size_bytes=int(item["size_bytes"]),
             allocated_bytes=int(actual["allocated_bytes"]), sha256=item["sha256"],
             deleted_utc=utc_now(), verified_missing=True,
             queue_overlaps=0, queue_rows=queue_rows)
    emit("complete", file_count=len(items), logical_bytes=logical_total,
         allocated_bytes=allocated_total, queue_overlaps=0)
except Exception as exc:
    emit("guard_or_execution_failed", error=type(exc).__name__ + ": " + str(exc))
    raise SystemExit(2)
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_executor(con: sqlite3.Connection, path: Path, observed: str) -> int:
    path = path.resolve(strict=True)
    relative = path.relative_to(REPO).as_posix()
    digest = sha256(path)
    source_key = "qe_mix_executor:" + relative + ":" + digest[:16]
    artifact_key = "QE-MIX-EXECUTOR:" + relative + ":" + digest[:16]
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
           VALUES(?,?, 'GUARDED_CLEANUP_EXECUTOR',?,?,?,1,?,?)""",
        (source_key, str(path), digest, observed, observed,
         "Exact local executor source for the guarded QE mixing-history cleanup.",
         "Contains all-user queue, exact path/owner/mode/link/inode/mtime/size/block/hash checks and immediate pre-unlink rechecks."),
    )
    source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES('ROCKFISH_STORAGE_AUDIT',?,'GUARDED_CLEANUP_EXECUTOR',?,'USER_CONFIRMED',?,?,'VERIFIED','PERMANENT_SOURCE',?)""",
        (artifact_key, "Guarded QE mix deletion executor", digest, path.stat().st_size,
         "Local Python launcher embeds the remote all-user squeue and immediate file identity/hash deletion guards."),
    )
    artifact_id = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if artifact_id[1:] != (digest, path.stat().st_size):
        raise RuntimeError("executor artifact identity conflict")
    root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()
    if not root or Path(root[0]).resolve() != REPO.resolve():
        raise RuntimeError("LOCAL-MAIN does not resolve to the unified worktree")
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)",
                (relative, observed))
    path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?", (relative,)).fetchone()[0]
    con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                (artifact_id[0], path_id, observed, observed))
    con.execute(
        """INSERT OR IGNORE INTO registry_entity_ref
           (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
           VALUES(?, 'qe_cleanup_executor', ?, 'CLEANUP_BATCH', ?, 'EVIDENCE_FOR', ?)""",
        (source_id, BATCH_ID, BATCH_ID, "Guard implementation for this exact staged 12-file batch."),
    )
    return source_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota-before", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--resume-after-preflight-failure", action="store_true")
    parser.add_argument("--prior-failure", type=Path)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = [x for run in manifest["runs"] for x in run["mix_intermediates"]]
    if len(items) != 12 or len({x["path"] for x in items}) != 12:
        raise SystemExit("manifest must list exactly 12 unique cleanup paths")
    if sha256(AUDIT / "phase2_data" / "yi_refresh_20261123" / "qe_mix_retained_artifact_hashes_20261123.jsonl") != manifest["source_hash_ledger_sha256"]:
        raise SystemExit("retained artifact hash ledger changed")
    quota_before = args.quota_before.resolve(strict=True)
    resume = args.resume_after_preflight_failure
    if resume:
        if not args.prior_failure:
            raise SystemExit("--prior-failure is required when resuming")
        prior_events = [json.loads(line) for line in args.prior_failure.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prior_events) != 1 or prior_events[0].get("event") != "guard_or_execution_failed":
            raise SystemExit("resume requires exactly one recorded guard failure and no deletion events")
        if "unexpected keyword argument 'text'" not in prior_events[0].get("error", ""):
            raise SystemExit("prior failure is not the reviewed Python 3.6 preflight compatibility error")
    if not args.receipt.is_absolute():
        args.receipt = (Path.cwd() / args.receipt).resolve()
    if not args.stderr.is_absolute():
        args.stderr = (Path.cwd() / args.stderr).resolve()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    observed = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute("SELECT status,source_root_id,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()
        expected_status = "EXECUTING" if resume else "EVIDENCE_READY"
        if batch != (expected_status, "RF-DATA-YI", 5352688896):
            raise RuntimeError("cleanup batch is not in the expected evidence-ready state")
        staged = con.execute(
            """SELECT ci.pre_delete_sha256,ci.pre_delete_size_bytes,
                      sr.scoped_path || CASE WHEN sp.relative_path='' THEN '' ELSE '/' || sp.relative_path END
               FROM cleanup_item ci JOIN artifact a USING(artifact_id)
               JOIN artifact_location al USING(artifact_id) JOIN storage_path sp USING(path_id)
               JOIN storage_root sr USING(root_id)
               WHERE ci.cleanup_batch_id=? AND ci.disposition='DELETE' AND ci.deletion_state='EVIDENCE_READY'
                 AND al.availability='PRESENT'""", (BATCH_ID,)
        ).fetchall()
        expected = {x["path"]: (x["sha256"], int(x["size_bytes"])) for x in items}
        actual = {row[2]: (row[0], int(row[1])) for row in staged}
        if actual != expected or len(staged) != 12:
            raise RuntimeError("staged cleanup rows differ from the reviewed manifest")
        quota_source = con.execute("SELECT source_id,sha256 FROM registry_source WHERE source_path=?", (str(quota_before),)).fetchone()
        if not quota_source or quota_source[1] != sha256(quota_before):
            raise RuntimeError("quota-before evidence is not registered with its current hash")
        quota_row = con.execute(
            """SELECT 1 FROM quota_observation WHERE raw_output_path=? AND root_id=?
               AND quota_subject_kind='GROUP' AND quota_subject='pclancy3'""",
            (str(quota_before), QUOTA_ROOT),
        ).fetchone()
        if not quota_row:
            raise RuntimeError("fresh group quota-before observation is missing")
        source_id = register_executor(con, Path(__file__), observed)
        con.execute(
            """INSERT OR IGNORE INTO registry_entity_ref
               (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
               VALUES(?, 'qe_cleanup_quota', ?, 'CLEANUP_BATCH', ?, 'EVIDENCE_FOR', ?)""",
            (quota_source[0], BATCH_ID, BATCH_ID, "Group/user GPFS quota capture immediately before guarded deletion."),
        )
        if not resume:
            con.execute("UPDATE cleanup_batch SET status='EXECUTING' WHERE cleanup_batch_id=? AND status='EVIDENCE_READY'", (BATCH_ID,))
        if con.execute("SELECT status FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()[0] != "EXECUTING":
            raise RuntimeError("cleanup batch is not EXECUTING")
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign-key check failed before remote cleanup")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    remote_source = REMOTE_PROGRAM.replace("__MANIFEST_JSON__", repr(json.dumps(manifest, separators=(",", ":"))))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("wb") as output, args.stderr.open("wb") as error:
        proc = subprocess.run(["ssh", "rockfish", "python3", "-"], input=remote_source.encode("utf-8"), stdout=output, stderr=error)
    if proc.returncode:
        raise SystemExit(f"remote guarded cleanup exited {proc.returncode}; batch remains EXECUTING; inspect {args.receipt} and {args.stderr}")
    events = [json.loads(line) for line in args.receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
    complete = next((event for event in reversed(events) if event.get("event") == "complete"), None)
    if not complete or complete.get("file_count") != 12 or complete.get("logical_bytes") != 5352688896:
        raise SystemExit("remote command returned success without the exact 12-file completion receipt")
    print(json.dumps({"batch": BATCH_ID, "status": "REMOTE_DELETED_AND_VERIFIED", "file_count": complete["file_count"],
                      "logical_bytes": complete["logical_bytes"], "allocated_bytes": complete["allocated_bytes"],
                      "receipt": str(args.receipt), "executor_source_id": source_id}, sort_keys=True))


if __name__ == "__main__":
    main()
