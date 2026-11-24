#!/usr/bin/env python3
"""Read-only post-delete verification of the exact QE mix batch and held group."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
REVIEW = AUDIT / "phase2_data/yi_mix_repro_review_20261123"
MANIFEST = REVIEW / "yi_qe_mix_reproduction_manifest_20261123.json"
HASH_LEDGER = REVIEW / "mix_candidate_predelete_identity_hashes_20261123.jsonl"
REMOTE = r'''import base64,datetime,hashlib,json,os,stat,sys,zlib
p=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
items={x["path"]:x for x in p["hash_ledger"]}
deleted=set(p["deleted_paths"])
if len(items)!=432 or not deleted.issubset(items): raise SystemExit("invalid verification manifest")
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
results=[]
for path,expected in sorted(items.items()):
    should_be_missing=path in deleted
    try: st=os.lstat(path)
    except FileNotFoundError:
        results.append({"path":path,"state":"MISSING"})
        continue
    if should_be_missing:
        results.append({"path":path,"state":"UNEXPECTED_PRESENT"})
        continue
    if not stat.S_ISREG(st.st_mode):
        results.append({"path":path,"state":"TYPE_MISMATCH"});continue
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    try:
        before=os.fstat(fd);h=hashlib.sha256()
        while True:
            b=os.read(fd,8*1024*1024)
            if not b:break
            h.update(b)
        after=os.fstat(fd)
    finally:os.close(fd)
    keys=("st_dev","st_ino","st_uid","st_gid","st_mode","st_nlink","st_size","st_mtime_ns")
    stable=all(getattr(before,k)==getattr(after,k) for k in keys)
    actual={"device":after.st_dev,"inode":after.st_ino,"uid":after.st_uid,"gid":after.st_gid,
            "mode_octal":format(stat.S_IMODE(after.st_mode),"04o"),"nlink":after.st_nlink,
            "size_bytes":after.st_size,"allocated_bytes":getattr(after,"st_blocks",0)*512,
            "mtime_ns":after.st_mtime_ns,"sha256":h.hexdigest()}
    same=stable and all(actual[k]==expected[k] for k in actual)
    results.append({"path":path,"state":"PRESENT_VERIFIED" if same else "IDENTITY_OR_SHA_MISMATCH","identity":actual})
summary={"observed_utc":utc(),"candidate_count":len(results),"expected_deleted_count":len(deleted),
         "missing_count":sum(x["state"]=="MISSING" for x in results),
         "present_verified_count":sum(x["state"]=="PRESENT_VERIFIED" for x in results),
         "unexpected_present_count":sum(x["state"]=="UNEXPECTED_PRESENT" for x in results),
         "mismatch_count":sum(x["state"] in ("TYPE_MISMATCH","IDENTITY_OR_SHA_MISMATCH") for x in results),
         "items":results}
print(json.dumps(summary,sort_keys=True))
if summary["missing_count"]!=len(deleted) or summary["present_verified_count"]!=len(items)-len(deleted) or summary["unexpected_present_count"] or summary["mismatch_count"]:
    raise SystemExit(2)
'''


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line in HASH_LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [json.loads(line) for line in args.executor_receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
    deleted = [x["path"] for x in events if x.get("event") == "deleted_verified" and x.get("verified_missing") is True]
    if len(deleted) != len(set(deleted)):
        raise SystemExit("executor receipt has duplicate deletion records")
    expected_delete = {item["path"] for group in manifest["groups"] if group["disposition"] == "DELETE" for item in group["items"]}
    if not set(deleted).issubset(expected_delete):
        raise SystemExit("executor receipt contains paths outside the approved DELETE set")
    payload = {"manifest": manifest, "hash_ledger": ledger, "deleted_paths": deleted}
    packed = base64.b64encode(zlib.compress(json.dumps(payload,separators=(",",":")).encode(),9)).decode()
    command = "python3 -c " + shlex.quote(REMOTE) + " " + shlex.quote(packed)
    result = subprocess.run(["ssh","-T","-o","BatchMode=yes","rockfish",command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=900)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stdout = args.output_dir / f"qe_mix_cleanup_postverify_{stamp}.stdout.json"
    stderr = args.output_dir / f"qe_mix_cleanup_postverify_{stamp}.stderr.txt"
    status_path = args.output_dir / f"qe_mix_cleanup_postverify_{stamp}.status.json"
    stdout.write_text(result.stdout,encoding="utf-8")
    stderr.write_text(result.stderr,encoding="utf-8")
    try: payload_result=json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc: payload_result={"parse_error":repr(exc)}
    status={"ssh_returncode":result.returncode,"verification_path":str(stdout),"verification_sha256":sha(stdout),
            "stderr_path":str(stderr),"stderr_sha256":sha(stderr),"summary":{k:v for k,v in payload_result.items() if k!="items"},
            "candidate_manifest_sha256":sha(MANIFEST),"predelete_hash_ledger_sha256":sha(HASH_LEDGER)}
    status_path.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(status,sort_keys=True))
    if result.returncode: raise SystemExit("post-delete path verification failed")


if __name__ == "__main__":
    main()
