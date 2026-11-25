#!/usr/bin/env python3
"""Read-only current check for the historical 574-file restart cleanup."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "removed_574_wfc_mix_paths_20261122.tsv"
OUTPUT = HERE / "current_rockfish_verification_20261123.json"
REMOTE = r'''import base64,datetime,hashlib,json,os,re,subprocess,sys,zlib
payload=json.loads(zlib.decompress(base64.b64decode(__PAYLOAD__)))
root="/scratch16/pclancy3/yi/revision1_migrationbench_runs"
paths=payload["paths"]
missing=[];present=[];path_errors=[]
for path in paths:
    if not path.startswith(root+"/"):
        path_errors.append({"path":path,"error":"outside_expected_root"});continue
    try: os.lstat(path)
    except FileNotFoundError: missing.append(path)
    except OSError as e: path_errors.append({"path":path,"error":type(e).__name__+":"+str(e)})
    else: present.append(path)
env=os.environ.copy();env["SLURM_CONF"]="/cm/shared/apps/slurm/var/etc/slurm/slurm.conf"
qcmd=["/cm/shared/apps/slurm/current/bin/squeue","-a","-h","-o","%i|%u|%T|%Z"]
qr=subprocess.run(qcmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
qtext=qr.stdout.decode("utf-8",errors="replace");qunknown=0;overlaps=[]
for line in qtext.splitlines():
    fields=line.split("|",3)
    if len(fields)!=4 or not all(fields[:3]) or not fields[3].strip() or fields[3].strip() in ("(null)","Unknown","N/A"):
        qunknown+=1;continue
    job,user,state,workdir=fields;workdir=os.path.normpath(workdir.strip())
    if workdir==root or workdir.startswith(root+"/"):
        overlaps.append({"job_id":job,"user":user,"state":state,"workdir":workdir})
restart_files=[]
for current,dirs,files in os.walk(root):
    dirs[:]=[d for d in dirs if not os.path.islink(os.path.join(current,d))]
    for name in files:
        lower=name.lower()
        if lower.startswith(("wfc","mix")) or ".wfc" in lower or ".mix" in lower:
            path=os.path.join(current,name)
            try: st=os.lstat(path)
            except OSError as e:
                restart_files.append({"path":path,"error":type(e).__name__+":"+str(e)});continue
            restart_files.append({"path":path,"size_bytes":st.st_size,"allocated_bytes":getattr(st,"st_blocks",0)*512})
result={"verified_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
        "host":"Rockfish","root":root,"expected_path_count":len(paths),"missing_path_count":len(missing),
        "present_path_count":len(present),"path_error_count":len(path_errors),"present_paths":present,
        "path_errors":path_errors,"queue":{"returncode":qr.returncode,"total_jobs":len(qtext.splitlines()),
        "unknown_or_unparseable_workdir_rows":qunknown,"root_overlap_count":len(overlaps),"overlaps":overlaps,
        "raw_stdout_sha256":hashlib.sha256(qr.stdout).hexdigest(),"stderr":qr.stderr.decode("utf-8",errors="replace")},
        "current_wfc_or_mix_name_matches":restart_files,
        "verified":len(paths)==574 and not present and not path_errors and qr.returncode==0 and qunknown==0 and not overlaps and not restart_files}
print(json.dumps(result,sort_keys=True))
if not result["verified"]: raise SystemExit(2)
'''


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    paths = sorted({row["artifact_path"] for row in rows})
    if len(paths) != 574:
        raise SystemExit(f"expected 574 unique paths, found {len(paths)}")
    packed = base64.b64encode(zlib.compress(json.dumps({"paths": paths}, separators=(",", ":")).encode(), 9)).decode("ascii")
    remote = REMOTE.replace("__PAYLOAD__", repr(packed))
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "rockfish", "python3", "-"],
        input=remote,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
    )
    status = {
        "local_captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ssh_returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr": result.stderr,
        "path_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    }
    if result.returncode:
        raise SystemExit(f"remote verification failed ({result.returncode}): {result.stderr[-2000:]} {result.stdout[-2000:]}")
    verification = json.loads(result.stdout)
    verification["local_execution"] = status
    OUTPUT.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "verified": verification["verified"],
                      "expected_path_count": verification["expected_path_count"],
                      "missing_path_count": verification["missing_path_count"],
                      "current_wfc_or_mix_matches": len(verification["current_wfc_or_mix_name_matches"]),
                      "queue_root_overlap_count": verification["queue"]["root_overlap_count"],
                      "observed_utc": verification["verified_utc"]}, sort_keys=True))
    if not verification["verified"]:
        raise SystemExit("historical cleanup paths are not all reconciled; see " + str(OUTPUT))


if __name__ == "__main__":
    main()
