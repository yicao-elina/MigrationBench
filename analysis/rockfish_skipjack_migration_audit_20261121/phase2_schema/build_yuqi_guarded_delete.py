#!/usr/bin/env python3
"""Build a Python 3.6-compatible single-use remote guard from the hash ledger."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DATA = AUDIT / "phase2_data"
IDENTITY = DATA / "former_collab_file_identity_20261122.tsv"
OUT = DATA / "delete_yuqi_pseudopotentials_guarded.py"

with IDENTITY.open(newline="") as stream:
    rows = [r for r in csv.DictReader(stream, delimiter="\t") if r["path"].startswith("/data/pclancy3/for-yuqi/pseudo/")]
if len(rows) != 65 or any(r["exact_match"] != "True" for r in rows):
    raise SystemExit("identity ledger must contain 65 exact duplicate pairs")
candidates = [
    {"path": r["path"], "size": int(r["size_bytes"]), "mtime_ns": int(r["mtime_ns"]),
     "sha256": r["sha256"], "allocated_bytes": int(r["allocated_bytes"]),
     "canonical_path": r["canonical_path"], "canonical_size": int(r["canonical_size"]),
     "canonical_sha256": r["canonical_sha256"]}
    for r in rows
]

template = r'''import argparse, hashlib, json, os, stat, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/pclancy3/for-yuqi/pseudo")
CANDIDATES = __CANDIDATES__
QUEUE_CMD = ["/cm/shared/apps/slurm/current/bin/squeue", "-a", "-h", "-o", "%i|%u|%T|%Z"]
os.environ["SLURM_CONF"] = "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf"
parser=argparse.ArgumentParser(description="Preflight or remove exact for-yuqi pseudopotential duplicates")
parser.add_argument("--execute", action="store_true", help="delete candidates only after all queue, hash, and metadata checks pass")
args=parser.parse_args()

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def sha256(path):
    h=hashlib.sha256()
    with open(str(path),"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def overlaps(path, root):
    p=Path(path)
    return p == root or p in root.parents or root in p.parents

def queue_overlaps():
    p=subprocess.run(QUEUE_CMD,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    hits=[]
    for line in p.stdout.splitlines():
        fields=line.split("|",3)
        if len(fields)==4 and fields[3].startswith("/") and overlaps(fields[3],ROOT.parent):
            hits.append({"job_id":fields[0],"user":fields[1],"state":fields[2],"workdir":fields[3]})
    return hits

if not ROOT.is_dir() or ROOT.is_symlink():
    raise SystemExit("approved duplicate root missing or symlink")
if queue_overlaps():
    raise SystemExit("live Slurm workdir overlaps duplicate root: "+json.dumps(queue_overlaps()))

checked=[]
for item in CANDIDATES:
    p=Path(item["path"])
    if not str(p).startswith(str(ROOT)+"/") or p.is_symlink() or not p.is_file():
        raise SystemExit("candidate path/type guard failed: "+str(p))
    st=os.lstat(str(p))
    expected=(3901,3001,0o644,1,item["size"],item["mtime_ns"])
    actual=(st.st_uid,st.st_gid,stat.S_IMODE(st.st_mode),st.st_nlink,st.st_size,st.st_mtime_ns)
    if actual != expected:
        raise SystemExit("candidate metadata changed: "+str(p)+" "+repr(actual))
    digest=sha256(p)
    if digest != item["sha256"]:
        raise SystemExit("candidate hash changed: "+str(p))
    canonical=Path(item["canonical_path"])
    if canonical.is_symlink() or not canonical.is_file():
        raise SystemExit("canonical pseudopotential is missing or not a regular file: "+str(canonical))
    cst=os.lstat(str(canonical))
    if cst.st_uid!=3901 or cst.st_size!=item["canonical_size"] or sha256(canonical)!=item["canonical_sha256"] or item["canonical_sha256"]!=digest:
        raise SystemExit("canonical pseudopotential identity/hash changed: "+str(canonical))
    checked.append({"path":str(p),"size_bytes":st.st_size,"allocated_bytes":st.st_blocks*512,
                    "mtime_ns":st.st_mtime_ns,"sha256":digest,"canonical_path":str(canonical)})

if queue_overlaps():
    raise SystemExit("live Slurm workdir appeared during hashing: "+json.dumps(queue_overlaps()))
print(json.dumps({"event":"preflight_passed","utc":utc_now(),"file_count":len(checked),
                  "logical_bytes":sum(x["size_bytes"] for x in checked),
                  "allocated_bytes":sum(x["allocated_bytes"] for x in checked),"queue_overlaps":0,
                  "execute":args.execute},separators=(",",":")),flush=True)

if not args.execute:
    print(json.dumps({"event":"preflight_only","file_count":len(checked),"deleted":False},separators=(",",":")),flush=True)
    raise SystemExit(0)

for item in checked:
    p=Path(item["path"])
    st=os.lstat(str(p))
    if p.is_symlink() or not p.is_file() or (st.st_uid,st.st_gid,st.st_nlink,st.st_size,st.st_mtime_ns)!=(3901,3001,1,item["size_bytes"],item["mtime_ns"]):
        raise SystemExit("immediate pre-unlink identity check failed: "+str(p))
    if sha256(p) != item["sha256"]:
        raise SystemExit("immediate pre-unlink hash check failed: "+str(p))
    if sha256(item["canonical_path"]) != item["sha256"]:
        raise SystemExit("canonical pseudopotential changed immediately before unlink: "+item["canonical_path"])
    p.unlink()
    if p.exists() or p.is_symlink():
        raise SystemExit("post-unlink check failed: "+str(p))
    print(json.dumps({"event":"deleted_verified",**item,"deleted_utc":utc_now(),"verified_missing":True},separators=(",",":")),flush=True)
print(json.dumps({"event":"complete","utc":utc_now(),"file_count":len(checked),
                  "logical_bytes":sum(x["size_bytes"] for x in checked),
                  "allocated_bytes":sum(x["allocated_bytes"] for x in checked)},separators=(",",":")),flush=True)
'''

source = template.replace("__CANDIDATES__", json.dumps(candidates, separators=(",", ":")))
OUT.write_text(source, encoding="utf-8")
print("generated", OUT, "candidates", len(candidates), "bytes", sum(x["size"] for x in candidates))
