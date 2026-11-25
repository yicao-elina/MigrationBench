#!/usr/bin/env python3
"""Delete only the staged three-group QE mix set after live queue and SHA guards."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import sqlite3
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
REVIEW = AUDIT / "phase2_data/yi_mix_repro_review_20261123"
MANIFEST = REVIEW / "yi_qe_mix_reproduction_manifest_20261123.json"
HASH_LEDGER = REVIEW / "mix_candidate_predelete_identity_hashes_20261123.jsonl"
BATCH_ID = "RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123"

REMOTE = r'''import base64,datetime,hashlib,json,os,stat,subprocess,sys,zlib
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
manifest=payload["manifest"]
ledger={x["path"]:x for x in payload["hash_ledger"]}
groups=manifest["groups"]
items=[(group,item) for group in groups for item in group["items"]]
delete=[(g,x) for g,x in items if g["disposition"]=="DELETE"]
held=[(g,x) for g,x in items if g["disposition"]=="HOLD"]
deletion_dirs=sorted({g["candidate_dir"] for g,_ in delete})
if len(items)!=432 or len({x["path"] for _,x in items})!=432 or len(delete)!=384 or len(held)!=48:
    raise SystemExit("manifest must contain 432 unique paths, with exactly 384 DELETE and 48 HOLD")
if set(ledger)!={x["path"] for _,x in items}:
    raise SystemExit("predelete hash ledger path set differs from candidate manifest")
if any(not x["path"].startswith("/data/pclancy3/yi/") for _,x in items):
    raise SystemExit("unexpected candidate root")
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
def emit(event, **fields): print(json.dumps({"event":event,"utc":utc(),**fields},sort_keys=True),flush=True)
def queue_snapshot():
    env=os.environ.copy();env["SLURM_CONF"]="/cm/shared/apps/slurm/var/etc/slurm/slurm.conf"
    command=["/cm/shared/apps/slurm/current/bin/squeue","-a","-h","-o","%i|%u|%T|%Z"]
    result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode: raise RuntimeError("squeue failed: "+result.stderr.decode("utf-8",errors="replace")[-2000:])
    rows=[]; unknown=0; overlaps=[]; raw=result.stdout.decode("utf-8",errors="replace")
    for line in raw.splitlines():
        fields=line.split("|",3)
        if len(fields)!=4 or not fields[0] or not fields[1] or not fields[2]: unknown+=1;continue
        job,user,state,workdir=fields;workdir=workdir.strip()
        if not workdir or workdir in ("(null)","Unknown","N/A"): unknown+=1;continue
        workdir=os.path.normpath(workdir)
        for candidate in deletion_dirs:
            try: shared=os.path.commonpath((candidate,workdir))
            except ValueError: continue
            if shared==candidate or shared==workdir:
                overlaps.append({"job_id":job,"user":user,"state":state,"workdir":workdir,"candidate":candidate})
    result_row={"job_count":len(raw.splitlines()),"unknown_workdir_rows":unknown,
                "raw_stdout_sha256":hashlib.sha256(result.stdout).hexdigest(),"overlaps":overlaps}
    if unknown: raise RuntimeError("all-user queue has rows with unknown/unparseable WorkDir: "+json.dumps(result_row,sort_keys=True))
    if overlaps: raise RuntimeError("all-user queue overlaps a deletion path: "+json.dumps(result_row,sort_keys=True))
    return result_row
def identity_and_hash(path, expected):
    parent=os.path.dirname(path);name=os.path.basename(path)
    if os.path.realpath(parent)!=parent: raise RuntimeError("candidate parent resolves through a symlink: "+parent)
    pst=os.lstat(parent)
    if not stat.S_ISDIR(pst.st_mode): raise RuntimeError("candidate parent is not a directory: "+parent)
    dirfd=os.open(parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
    try:
        before_path=os.stat(name,dir_fd=dirfd,follow_symlinks=False)
        if not stat.S_ISREG(before_path.st_mode): raise RuntimeError("candidate is not a regular file: "+path)
        fd=os.open(name,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0),dir_fd=dirfd)
        try:
            before=os.fstat(fd);h=hashlib.sha256()
            while True:
                block=os.read(fd,8*1024*1024)
                if not block: break
                h.update(block)
            after=os.fstat(fd)
        finally: os.close(fd)
        after_path=os.stat(name,dir_fd=dirfd,follow_symlinks=False)
    finally: os.close(dirfd)
    keys=("st_dev","st_ino","st_uid","st_gid","st_mode","st_nlink","st_size","st_mtime_ns")
    if any(getattr(before,k)!=getattr(after,k) for k in keys) or any(getattr(before_path,k)!=getattr(after_path,k) for k in keys):
        raise RuntimeError("candidate changed while hashing: "+path)
    actual={"device":after.st_dev,"inode":after.st_ino,"uid":after.st_uid,"gid":after.st_gid,
            "mode_octal":format(stat.S_IMODE(after.st_mode),"04o"),"nlink":after.st_nlink,
            "size_bytes":after.st_size,"allocated_bytes":getattr(after,"st_blocks",0)*512,
            "mtime_ns":after.st_mtime_ns,"sha256":h.hexdigest()}
    if any(actual[k]!=expected[k] for k in actual):
        raise RuntimeError("candidate identity/SHA mismatch: "+path+" actual="+json.dumps(actual,sort_keys=True))
    if actual["uid"]!=3901 or actual["gid"]!=3001 or actual["nlink"]!=1:
        raise RuntimeError("owner/group/link guard failed: "+path)
    return actual
try:
    # Confirm all 432 listed paths still match the R04 metadata before any deletion.
    for group,item in items:
        path=item["path"]; expected=ledger[path]
        if group["disposition"]=="DELETE":
            continue
        # Held files are not hashed again here, but must retain the same complete identity.
        st=os.lstat(path)
        actual={"device":st.st_dev,"inode":st.st_ino,"uid":st.st_uid,"gid":st.st_gid,
                "mode_octal":format(stat.S_IMODE(st.st_mode),"04o"),"nlink":st.st_nlink,
                "size_bytes":st.st_size,"allocated_bytes":getattr(st,"st_blocks",0)*512,
                "mtime_ns":st.st_mtime_ns}
        if not stat.S_ISREG(st.st_mode) or any(actual[k]!=expected[k] for k in actual):
            raise RuntimeError("held candidate changed or disappeared: "+path)
    initial_queue=queue_snapshot()
    emit("batch_preflight_passed", delete_count=len(delete),hold_count=len(held),
         delete_logical_bytes=sum(int(ledger[x["path"]]["size_bytes"]) for _,x in delete),
         delete_allocated_bytes=sum(int(ledger[x["path"]]["allocated_bytes"]) for _,x in delete),
         queue=initial_queue,held_paths_checked=len(held),all_paths_in_authorized_root=True)
    logical_total=0; allocated_total=0; deleted_count=0
    for group,item in sorted(delete,key=lambda pair:pair[1]["path"]):
        path=item["path"]; expected=ledger[path]
        actual=identity_and_hash(path,expected)
        # Recheck the all-user queue after hashing and immediately before unlink.
        current_queue=queue_snapshot()
        parent=os.path.dirname(path);name=os.path.basename(path)
        dirfd=os.open(parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
        try:
            current=os.stat(name,dir_fd=dirfd,follow_symlinks=False)
            if any(getattr(current,k)!=v for k,v in (("st_dev",actual["device"]),("st_ino",actual["inode"]),
                       ("st_uid",actual["uid"]),("st_gid",actual["gid"]),("st_nlink",actual["nlink"]),
                       ("st_mode",int(actual["mode_octal"],8)|(stat.S_IFREG)),
                       ("st_size",actual["size_bytes"]),("st_blocks",actual["allocated_bytes"]//512),
                       ("st_mtime_ns",actual["mtime_ns"]))):
                raise RuntimeError("candidate path identity changed after SHA and queue checks: "+path)
            os.unlink(name,dir_fd=dirfd)
        finally: os.close(dirfd)
        try: os.lstat(path)
        except FileNotFoundError: pass
        else: raise RuntimeError("unlink returned but path remains: "+path)
        logical_total+=actual["size_bytes"];allocated_total+=actual["allocated_bytes"];deleted_count+=1
        emit("deleted_verified",group_id=group["group_id"],path=path,identity=actual,
             verified_missing=True,queue=current_queue,deleted_count=deleted_count)
    emit("batch_complete",delete_count=deleted_count,hold_count=len(held),logical_bytes=logical_total,
         allocated_bytes=allocated_total,queue_overlap_count=0)
except Exception as exc:
    emit("guard_or_execution_failed",deleted_count=locals().get("deleted_count",0),
         error=type(exc).__name__+": "+str(exc))
    raise SystemExit(2)
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota-before", required=True, type=Path)
    parser.add_argument("--queue-capture", required=True, type=Path)
    parser.add_argument("--receipt-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    quota_before = args.quota_before.resolve(strict=True)
    queue_path = args.queue_capture.resolve(strict=True)
    manifest_bytes = MANIFEST.read_bytes()
    hash_bytes = HASH_LEDGER.read_bytes()
    manifest = json.loads(manifest_bytes)
    ledger = [json.loads(line) for line in hash_bytes.splitlines() if line.strip()]
    payload = {"manifest": manifest, "hash_ledger": ledger}
    packed = base64.b64encode(zlib.compress(json.dumps(payload,separators=(",",":")).encode(), 9)).decode("ascii")
    queue_capture = json.loads(queue_path.read_text(encoding="utf-8"))
    observed = queue_capture.get("observed_utc")
    if queue_capture.get("returncode") != 0 or queue_capture.get("unknown_or_unparseable_workdir_rows") != 0 or queue_capture.get("matching_rows"):
        raise SystemExit("latest selected-root queue capture does not pass its precheck")
    captured = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    age_seconds = (datetime.now(timezone.utc) - captured).total_seconds()
    if age_seconds < 0 or age_seconds > 600:
        raise SystemExit(f"selected-root queue capture is too old for cleanup preflight: {age_seconds:.1f}s")
    db = sqlite3.connect(DB, timeout=60)
    try:
        db.execute("PRAGMA busy_timeout=60000")
        row = db.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH_ID,)).fetchone()
        if row != ("EVIDENCE_READY", 7_500_169_152):
            raise SystemExit("SQL batch is not in the expected evidence-ready state")
        if db.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE' AND deletion_state='EVIDENCE_READY'", (BATCH_ID,)).fetchone()[0] != 384:
            raise SystemExit("SQL batch does not contain exactly 384 evidence-ready DELETE items")
        if db.execute("SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='HOLD' AND deletion_state='HELD'", (BATCH_ID,)).fetchone()[0] != 48:
            raise SystemExit("SQL batch does not contain exactly 48 held items")
        db.execute("UPDATE cleanup_batch SET status='EXECUTING' WHERE cleanup_batch_id=? AND status='EVIDENCE_READY'", (BATCH_ID,))
        if db.execute("SELECT changes()").fetchone()[0] != 1:
            raise SystemExit("failed to mark exact cleanup batch as EXECUTING")
        db.commit()
    finally:
        db.close()

    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stdout_path = args.receipt_dir / f"qe_mix_cleanup_remote_{stamp}.stdout.jsonl"
    stderr_path = args.receipt_dir / f"qe_mix_cleanup_remote_{stamp}.stderr.txt"
    status_path = args.receipt_dir / f"qe_mix_cleanup_remote_{stamp}.status.json"
    command = "python3 -c " + shlex.quote(REMOTE) + " " + shlex.quote(packed)
    result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "rockfish", command],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    status = {
        "cleanup_batch_id": BATCH_ID,
        "local_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ssh_returncode": result.returncode,
        "remote_preflight_passed": any(x.get("event") == "batch_preflight_passed" for x in events),
        "remote_batch_complete": any(x.get("event") == "batch_complete" for x in events),
        "verified_deleted_count": sum(x.get("event") == "deleted_verified" and x.get("verified_missing") is True for x in events),
        "receipt_path": str(stdout_path),
        "receipt_sha256": sha256(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256(stderr_path),
        "candidate_manifest_sha256": sha256(MANIFEST),
        "predelete_hash_ledger_sha256": sha256(HASH_LEDGER),
        "quota_before_path": str(quota_before),
        "quota_before_sha256": sha256(quota_before),
        "selected_queue_capture_path": str(queue_path),
        "selected_queue_capture_age_seconds": age_seconds,
        "remote_error": next((x.get("error") for x in events if x.get("event") == "guard_or_execution_failed"), None),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, sort_keys=True))
    if result.returncode or not status["remote_batch_complete"]:
        raise SystemExit("QE mix guarded cleanup did not complete; preserve receipt and run finalizer for exact partial status")


if __name__ == "__main__":
    main()
