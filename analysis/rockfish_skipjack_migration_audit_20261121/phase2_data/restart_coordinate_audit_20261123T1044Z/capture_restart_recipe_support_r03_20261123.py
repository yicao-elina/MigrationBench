#!/usr/bin/env python3
"""Read-only capture of surviving QE run inputs, launch scripts, and logs.

This records small reproducibility inputs/scripts and metadata for retained
outputs after the exact restart cleanup. It never opens restart payloads,
changes remote paths, or submits a job.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CROSSWALK = HERE / "run_coordinate_crosswalk_20261123T1129Z.tsv"
SACCT = HERE / "sacct_restart_ids_with_submitline_20261123T1114Z.psv"
RUN_CONTEXT = HERE / "run_context_20261123T1112Z.jsonl"
OUT = HERE / "restart_recipe_support_capture_20261123_r03.json"
STATUS = HERE / "restart_recipe_support_capture_20261123_r03.status.json"
STDERR = HERE / "restart_recipe_support_capture_20261123_r03.stderr.txt"


REMOTE = r'''import base64, hashlib, json, os, re, shlex, stat, sys, time
payload = json.loads(base64.b64decode(__CONFIG_B64__))

def digest_file(path, max_bytes, include_text=False):
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            return {"path":path,"exists":True,"entry_type":"NON_REGULAR","mode":oct(stat.S_IMODE(before.st_mode))}
        result = {"path":path,"exists":True,"entry_type":"FILE","size_bytes":before.st_size,
                  "allocated_bytes":getattr(before,"st_blocks",0)*512,"uid":before.st_uid,"gid":before.st_gid,
                  "mode_octal":format(stat.S_IMODE(before.st_mode),"04o"),"mtime_ns":before.st_mtime_ns,
                  "device":before.st_dev,"inode":before.st_ino,"nlink":before.st_nlink}
        if before.st_size <= max_bytes:
            h=hashlib.sha256()
            with open(path,"rb") as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b""): h.update(chunk)
            after=os.lstat(path)
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                result["stable_during_hash"]=False
            else:
                result["stable_during_hash"]=True
                result["sha256"]=h.hexdigest()
                result["hash_scope"]="FULL_FILE"
                if include_text:
                    try:
                        with open(path,"r",encoding="utf-8",errors="replace") as f: result["text"]=f.read(max_bytes+1)
                    except Exception as e: result["text_capture_error"]=type(e).__name__+": "+str(e)
        else:
            result["hash_scope"]="NOT_HASHED_ABOVE_SIZE_LIMIT"
        return result
    except FileNotFoundError:
        return {"path":path,"exists":False}
    except Exception as e:
        return {"path":path,"exists":None,"error":type(e).__name__+": "+str(e)}

def split_paths(value):
    return [x.strip() for x in (value or "").split(" ; ") if x.strip()]

def script_tokens(submit_line, workdir):
    try: tokens=shlex.split(submit_line)
    except Exception: return []
    if not tokens or not os.path.basename(tokens[0]).endswith("sbatch"): return []
    for arg in tokens[1:]:
        if arg.endswith((".slurm",".sbatch",".pbs",".sh")) and not arg.startswith("-"):
            path=arg if os.path.isabs(arg) else os.path.join(workdir,arg)
            return [os.path.normpath(path)]
    return []

def direct_script_candidates(directory):
    found=[]
    try: names=os.listdir(directory)
    except Exception: return found
    for name in names:
        low=name.lower()
        if low.endswith((".slurm",".sbatch",".pbs",".sh")) or low.startswith(("slurm-","job-","job_")) and low.endswith((".sh",".pbs")):
            path=os.path.join(directory,name)
            try:
                if stat.S_ISREG(os.lstat(path).st_mode): found.append(path)
            except OSError: pass
    return sorted(set(found))

def extract_pseudo_refs(input_path):
    refs=[]
    try:
        text=open(input_path,"r",errors="replace").read(2*1024*1024)
    except Exception:
        return refs
    pseudo_dir=None
    m=re.search(r"(?im)^\s*pseudo_dir\s*=\s*['\"]([^'\"]+)['\"]",text)
    if m:
        pseudo_dir=m.group(1)
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.strip().upper().startswith("ATOMIC_SPECIES"):
            for nextline in lines[i+1:]:
                cells=nextline.split()
                if len(cells)<3: break
                if not re.match(r"^[A-Za-z][A-Za-z0-9_+-]*$",cells[0]): break
                name=cells[2]
                base=pseudo_dir or "."
                if not os.path.isabs(base): base=os.path.join(os.path.dirname(input_path),base)
                refs.append(os.path.normpath(os.path.join(base,name)))
            break
    return sorted(set(refs))

result={"captured_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"remote_uid":os.getuid(),"roots":[]}
for root_row in payload:
    root=root_row["run_root"]
    entry={"run_root":root,"scope":root_row["scope"],"crosswalk_candidate_disposition":root_row["candidate_disposition"],
           "crosswalk_job_ids":root_row["slurm_job_ids"],"inputs":[],"logs":[],"manifests":[],"runtime_context":[],
           "scheduler_jobs":[],"script_candidates":[],"pseudo_dependencies":[]}
    try:
        st=os.stat(root); entry["root_accessible"]=True; entry["root_mode_octal"]=format(stat.S_IMODE(st.st_mode),"04o")
    except Exception as e:
        entry["root_accessible"]=False; entry["root_error"]=type(e).__name__+": "+str(e)
        result["roots"].append(entry); continue
    by_workdir={}
    for job in root_row["jobs"]:
        wd=os.path.normpath(job["workdir"])
        entry["scheduler_jobs"].append(job); by_workdir.setdefault(wd,[]).append(job)
    script_paths=set()
    for wd,jobs in by_workdir.items():
        if wd==root or wd.startswith(root+os.sep): script_paths.update(direct_script_candidates(wd))
        for job in jobs: script_paths.update(script_tokens(job.get("submit_line",""),wd))
    script_paths.update(direct_script_candidates(root))
    script_paths.update(root_row.get("script_refs",[]))
    entry["script_candidates"]=[digest_file(p,25*1024*1024,True) for p in sorted(script_paths)]
    input_paths=split_paths(root_row["input_paths"])
    manifest_paths=split_paths(root_row["manifest_paths"])
    output_paths=split_paths(root_row["output_log_paths"])
    entry["inputs"]=[digest_file(p,25*1024*1024,True) for p in input_paths]
    entry["manifests"]=[digest_file(p,25*1024*1024,True) for p in manifest_paths]
    entry["runtime_context"]=[digest_file(p,25*1024*1024) for p in root_row.get("runtime_paths",[])]
    entry["logs"]=[digest_file(p,2*1024*1024) for p in output_paths]
    pseudo=set()
    for inp in input_paths:
        if inp.lower().endswith((".in",".pw",".neb")): pseudo.update(extract_pseudo_refs(inp))
    entry["pseudo_dependencies"]=[digest_file(p,256*1024*1024) for p in sorted(pseudo)]
    entry["job_workdirs"]=[{"workdir":wd,"job_ids":[j["job_id"] for j in jobs]} for wd,jobs in sorted(by_workdir.items())]
    result["roots"].append(entry)
print(json.dumps(result,sort_keys=True,separators=(",",":")))
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with CROSSWALK.open(newline="", encoding="utf-8") as f:
        rows=list(csv.DictReader(f,delimiter="\t"))
    jobs=[]
    for line in SACCT.read_text(encoding="utf-8").splitlines():
        cells=line.split("|")
        if len(cells)<10: continue
        jobs.append({"job_id":cells[0],"job_name":cells[1],"user":cells[2],"state":cells[3],"exit_code":cells[4],
                     "submit_time_raw":cells[5],"start_time_raw":cells[6],"end_time_raw":cells[7],"workdir":cells[8],"submit_line":cells[9]})
    contexts={}
    for line in RUN_CONTEXT.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row=json.loads(line); refs=[]; runtime=[]
        def collect(value, key=""):
            if isinstance(value,dict):
                for k,v in value.items():
                    low=k.lower()
                    if isinstance(v,str) and os.path.isabs(v) and any(x in low for x in ("script","wrapper","launcher","submit")):
                        refs.append(v)
                    collect(v,k)
            elif isinstance(value,list):
                for item in value: collect(item,key)
        for manifest in row.get("manifests",[]):
            try:
                parsed=json.loads(manifest.get("text",""))
                collect(parsed)
            except Exception:
                continue
        for item in row.get("runtime",[]):
            path=item.get("path")
            if path and os.path.isabs(path): runtime.append(path)
        contexts[row["run_root"]]={"script_refs":sorted(set(refs)),"runtime_paths":sorted(set(runtime))}
    config=[]
    for r in rows:
        if not r.get("run_root"): continue
        ids={x.strip() for x in r.get("slurm_job_ids","").split(";") if x.strip()}
        root_jobs=[j for j in jobs if j["job_id"] in ids]
        for raw in [x.strip() for x in r.get("sacct_state_workdir_submitline","").split(" ; ") if x.strip()]:
            fields=raw.split(":",5)
            if len(fields)==6 and fields[0] in ids and fields[5] not in ("NOT_IN_CAPTURE",""):
                captured={"job_id":fields[0],"state":fields[1],"exit_code":fields[2]+":"+fields[3],
                          "workdir":fields[4],"submit_line":fields[5],"source":"crosswalk"}
                if not any(j.get("job_id")==captured["job_id"] and j.get("submit_line")==captured["submit_line"] for j in root_jobs):
                    root_jobs.append(captured)
        context=contexts.get(r["run_root"],{})
        config.append({"scope":r["scope"],"run_root":r["run_root"],"candidate_disposition":r["candidate_disposition"],
                       "slurm_job_ids":r["slurm_job_ids"],"input_paths":r["input_paths"],"output_log_paths":r["output_log_paths"],
                       "manifest_paths":r["manifest_paths"],"jobs":root_jobs,
                       "script_refs":context.get("script_refs",[]),"runtime_paths":context.get("runtime_paths",[])})
    encoded=base64.b64encode(json.dumps(config,separators=(",",":")).encode()).decode()
    remote=REMOTE.replace("__CONFIG_B64__",repr(encoded))
    proc=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=20","rockfish","python3","-"],
                        input=remote,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=1800)
    OUT.write_text(proc.stdout,encoding="utf-8")
    STDERR.write_text(proc.stderr,encoding="utf-8")
    status={"captured_local_utc":datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
            "ssh_returncode":proc.returncode,"output_path":str(OUT),"output_sha256":sha(OUT),"output_bytes":OUT.stat().st_size,
            "stderr_path":str(STDERR),"stderr_sha256":sha(STDERR),"crosswalk_sha256":sha(CROSSWALK),"sacct_sha256":sha(SACCT),
            "capture_script_path":str(Path(__file__).resolve()),"capture_script_sha256":sha(Path(__file__).resolve()),
            "root_count":len(config),"restart_payloads_opened_or_hashed":False}
    try:
        decoded=json.loads(proc.stdout.strip().splitlines()[-1])
        status["remote_capture_utc"]=decoded.get("captured_utc")
        status["roots_returned"]=len(decoded.get("roots",[]))
        status["inaccessible_roots"]=sum(not x.get("root_accessible") for x in decoded.get("roots",[]))
        status["roots_with_script_candidate"]=sum(any(y.get("exists") for y in x.get("script_candidates",[])) for x in decoded.get("roots",[]))
        status["inputs_full_hash"] = sum(y.get("hash_scope")=="FULL_FILE" for x in decoded.get("roots",[]) for y in x.get("inputs",[]))
        status["scripts_full_hash"] = sum(y.get("hash_scope")=="FULL_FILE" for x in decoded.get("roots",[]) for y in x.get("script_candidates",[]))
        status["missing_input_paths"] = sum(not y.get("exists") for x in decoded.get("roots",[]) for y in x.get("inputs",[]))
        status["missing_script_paths"] = sum(not y.get("exists") for x in decoded.get("roots",[]) for y in x.get("script_candidates",[]))
    except Exception as exc:
        status["parse_error"]=repr(exc)
    STATUS.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(status,sort_keys=True))
    if proc.returncode or status.get("roots_returned")!=len(config):
        raise SystemExit("remote support capture failed; inspect status and stderr")


if __name__=="__main__":
    main()
