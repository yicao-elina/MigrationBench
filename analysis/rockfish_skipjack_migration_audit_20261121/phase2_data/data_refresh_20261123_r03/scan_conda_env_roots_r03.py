#!/usr/bin/env python3
"""Scan selected Rockfish Conda environment roots with isolated completion receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ALLOWED = {
    "automatbench",
    "deepmd2",
    "qe-dev",
    "sklearn-env",
}
REMOTE = r'''import os,sys,json,time,stat,pwd,grp,socket,datetime,traceback
ROOT=__ROOT_JSON__
UID=os.getuid(); GID=os.getgid(); START=time.time()
COUNTS={"entries":0,"directories":0,"regular_files":0,"symlinks":0,"special_files":0,"errors":0,"git_markers":0}
def utc():
 return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
def name(mapping,key):
 try: return mapping(key).pw_name if mapping is pwd.getpwuid else mapping(key).gr_name
 except Exception: return str(key)
def kind(mode):
 if stat.S_ISDIR(mode): return "directory"
 if stat.S_ISREG(mode): return "file"
 if stat.S_ISLNK(mode): return "symlink"
 if stat.S_ISFIFO(mode): return "fifo"
 if stat.S_ISSOCK(mode): return "socket"
 if stat.S_ISCHR(mode): return "char_device"
 if stat.S_ISBLK(mode): return "block_device"
 return "other"
def emit(obj):
 sys.stdout.write(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n"); sys.stdout.flush()
def record(path,st):
 t=kind(st.st_mode); target=None
 if t=="symlink":
  try: target=os.readlink(path)
  except Exception as exc: target="READLINK_ERROR:"+repr(exc)
 return {"record_type":"entry","path":path,"type":t,"uid":st.st_uid,"owner":name(pwd.getpwuid,st.st_uid),"gid":st.st_gid,"group":name(grp.getgrgid,st.st_gid),"mode_octal":format(stat.S_IMODE(st.st_mode),"04o"),"mode_type_octal":format(st.st_mode,"06o"),"size_bytes":st.st_size if t=="file" else 0,"stat_size_bytes":st.st_size,"allocated_bytes":getattr(st,"st_blocks",0)*512,"mtime_ns":getattr(st,"st_mtime_ns",int(st.st_mtime*1000000000)),"device":st.st_dev,"inode":st.st_ino,"nlink":st.st_nlink,"symlink_target":target}
def count(t):
 COUNTS["entries"]+=1
 if t=="directory": COUNTS["directories"]+=1
 elif t=="file": COUNTS["regular_files"]+=1
 elif t=="symlink": COUNTS["symlinks"]+=1
 else: COUNTS["special_files"]+=1
try:
 rootst=os.lstat(ROOT)
 if not stat.S_ISDIR(rootst.st_mode): raise RuntimeError("scan root is not a directory")
 emit({"record_type":"root_started","root":ROOT,"host":socket.gethostname(),"started_utc":utc(),"uid":UID,"gid":GID,"root_identity":record(ROOT,rootst),"metadata_only":True,"follow_symlinks":False})
 stack=[ROOT]
 while stack:
  parent=stack.pop()
  try:
   with os.scandir(parent) as it: entries=sorted(list(it),key=lambda e:e.name)
  except Exception as exc:
   COUNTS["errors"]+=1; emit({"record_type":"read_error","path":parent,"error_type":type(exc).__name__,"error":str(exc)}); continue
  childdirs=[]
  for ent in entries:
   try: st=ent.stat(follow_symlinks=False)
   except Exception as exc:
    COUNTS["errors"]+=1; emit({"record_type":"stat_error","path":ent.path,"parent":parent,"error_type":type(exc).__name__,"error":str(exc)}); continue
   obj=record(ent.path,st); count(obj["type"])
   if os.path.basename(ent.path)==".git": COUNTS["git_markers"]+=1
   emit(obj)
   if obj["type"]=="directory": childdirs.append(ent.path)
  stack.extend(reversed(childdirs))
 emit({"record_type":"root_complete","root":ROOT,"completed_utc":utc(),"completion":"COMPLETED_WITH_ERRORS" if COUNTS["errors"] else "COMPLETE","counts":COUNTS,"elapsed_seconds":time.time()-START})
except Exception as exc:
 emit({"record_type":"fatal_error","root":ROOT,"utc":utc(),"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()}); sys.exit(2)
'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_root(name: str, timeout_s: int) -> dict:
    if name not in ALLOWED:
        raise SystemExit("root is not on the explicit R03 allow-list: " + name)
    remote_root = "/data/pclancy3/yi/.conda/envs/" + name
    program = REMOTE.replace("__ROOT_JSON__", json.dumps(remote_root))
    command = ["ssh", "rockfish", "timeout", "{}s".format(timeout_s), "python3", "-"]
    started = utc_now()
    try:
        result = subprocess.run(command, input=program.encode("utf-8"), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout_s + 30)
        output = result.stdout or b""
        stderr = result.stderr or b""
        exit_code = result.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        stderr = exc.stderr or b""
        exit_code = None
        timed_out = True

    data_path = HERE / ("conda_env_{}_r03.jsonl.gz".format(name))
    stderr_path = HERE / ("conda_env_{}_r03.stderr.txt".format(name))
    status_path = HERE / ("conda_env_{}_r03.status.json".format(name))
    with gzip.open(data_path, "wb", compresslevel=6) as stream:
        stream.write(output)
    stderr_path.write_bytes(stderr)

    lines = output.decode("utf-8", "replace").splitlines()
    final = None
    parse_errors = 0
    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except Exception:
            parse_errors += 1
    if parsed:
        final = parsed[-1]
    complete = bool(final and final.get("record_type") == "root_complete") and not timed_out and exit_code == 0 and parse_errors == 0
    status = {
        "root": remote_root,
        "started_local_utc": started,
        "completed_local_utc": utc_now(),
        "timeout_seconds": timeout_s,
        "timed_out": timed_out,
        "ssh_exit_code": exit_code,
        "output_sha256": digest(output),
        "output_bytes_uncompressed": len(output),
        "archive_sha256": digest(data_path.read_bytes()),
        "stderr_path": str(stderr_path),
        "stderr_bytes": len(stderr),
        "parse_error_count": parse_errors,
        "root_complete_marker": complete,
        "completion_record": final if final and final.get("record_type") == "root_complete" else None,
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", choices=sorted(ALLOWED))
    parser.add_argument("--timeout", type=int, default=300, help="per-root remote timeout in seconds")
    args = parser.parse_args()
    if args.timeout < 30 or args.timeout > 900:
        raise SystemExit("timeout must be from 30 to 900 seconds")
    results = [run_root(root, args.timeout) for root in args.roots]
    cumulative = []
    for root in sorted(ALLOWED):
        status_path = HERE / ("conda_env_{}_r03.status.json".format(root))
        if status_path.is_file():
            cumulative.append(json.loads(status_path.read_text(encoding="utf-8")))
    summary = {"run_utc": utc_now(), "invocation_roots": args.roots, "scan_roots": cumulative,
               "complete_root_count": sum(bool(item["root_complete_marker"]) for item in cumulative),
               "root_count": len(cumulative), "invocation_complete_root_count": sum(bool(item["root_complete_marker"]) for item in results)}
    out = HERE / "conda_env_r03_run_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if summary["complete_root_count"] != summary["root_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
