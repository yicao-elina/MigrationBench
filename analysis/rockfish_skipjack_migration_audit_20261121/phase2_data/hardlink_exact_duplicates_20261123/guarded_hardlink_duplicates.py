#!/usr/bin/env python3
"""Capture or execute guarded, path-preserving hardlink deduplication on Rockfish."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
EXPECTED = [
    {
        "group": "DUP-04",
        "a": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/2-naive-fine-tuning/temp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "b": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/2-naive-fine-tuning/temp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "size": 320656778,
        "inode_a": 393620432,
        "inode_b": 438896712,
        "sha256": "1f4c6ec103134aad3039b1c69dd607a1310d6226cc97d2a4b8dc5549e8ed6a86",
    },
    {
        "group": "DUP-05",
        "a": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "b": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "size": 320656648,
        "inode_a": 393620412,
        "inode_b": 679617704,
        "sha256": "e2f72e149940af7fd9016d2e1d58545cbe4734f935d91d9e4dc84c2e7645e454",
    },
    {
        "group": "DUP-06",
        "a": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/1-from-scratch/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "b": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/1-from-scratch/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "size": 320598327,
        "inode_a": 393620439,
        "inode_b": 327963155,
        "sha256": "55d20a704447677096e0be5d9615a0b2e1d144fc74a6a48b30b6b63863a5d99e",
    },
    {
        "group": "DUP-07",
        "a": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/1-from-scratch/old_data/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "b": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/1-from-scratch/old_data/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "size": 320592540,
        "inode_a": 525169501,
        "inode_b": 215786038,
        "sha256": "a63650591ac39a0b34119cbaee74913db86bec82537e61b1a0154b6dcc60e73e",
    },
    {
        "group": "DUP-13",
        "a": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/2-naive-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "b": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/2-naive-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz",
        "size": 128431527,
        "inode_a": 525169471,
        "inode_b": 680116225,
        "sha256": "1da89bba994e3a1037e46d989e5e7dfbf8874aeae8e86d42381d836a11e29cdf",
    },
]

REMOTE = r'''import datetime, hashlib, json, os, stat, subprocess, sys, uuid
cfg = __CONFIG_B64_DECODE__
mode = cfg["mode"]
prefix = "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/"

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def statrow(path):
    s = os.stat(path, follow_symlinks=False)
    return {"device": s.st_dev, "inode": s.st_ino, "uid": s.st_uid, "gid": s.st_gid,
            "mode_octal": format(stat.S_IMODE(s.st_mode), "04o"), "size_bytes": s.st_size,
            "allocated_bytes": getattr(s, "st_blocks", 0) * 512, "nlink": s.st_nlink,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "is_regular": stat.S_ISREG(s.st_mode), "is_symlink": stat.S_ISLNK(s.st_mode)}

def xattrs(path):
    result = {}
    for name in sorted(os.listxattr(path, follow_symlinks=False)):
        result[name] = os.getxattr(path, name, follow_symlinks=False).hex()
    return result

def acl(path):
    p = subprocess.run(["getfacl", "-cp", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode:
        raise RuntimeError("getfacl failed for " + path + ": " + p.stderr.strip())
    return p.stdout.strip()

def queue_snapshot():
    env = os.environ.copy()
    env["SLURM_CONF"] = "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf"
    p = subprocess.run(["/cm/shared/apps/slurm/current/bin/squeue", "-a", "-h", "-o", "%i|%u|%T|%j|%Z"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env, timeout=60)
    if p.returncode:
        raise RuntimeError("all-user squeue failed: " + p.stderr.strip())
    return {"captured_utc": utc(), "raw": p.stdout, "stderr": p.stderr,
            "jobs": [line for line in p.stdout.splitlines() if line.strip()]}

def overlaps(q, item):
    names = {os.path.basename(item["a"]), os.path.basename(item["b"])}
    hits = []
    for line in q["jobs"]:
        cols = line.split("|")
        workdir = cols[4] if len(cols) > 4 else ""
        if not workdir or workdir.lower() in ("unknown", "n/a", "none", "(null)", "unset") or not os.path.isabs(workdir):
            hits.append("UNKNOWN_WORKDIR:" + line)
        elif workdir.startswith(prefix):
            hits.append(line)
    return hits

def capture_pair(item):
    out = {"group": item["group"], "a": item["a"], "b": item["b"]}
    for label in ("a", "b"):
        path = item[label]
        if os.path.islink(path) or not os.path.isfile(path):
            raise RuntimeError("not a regular non-symlink file: " + path)
        s0 = statrow(path)
        d = sha(path)
        x = xattrs(path)
        a = acl(path)
        s1 = statrow(path)
        if s0 != s1:
            raise RuntimeError("file stat changed during hash: " + path)
        out[label] = {"path": path, "stat": s1, "sha256": d, "xattrs": x, "acl": a}
    return out

def validate(item, pair, q):
    a, b = pair["a"], pair["b"]
    problems = []
    for label, expected_inode in (("a", item["inode_a"]), ("b", item["inode_b"])):
        if pair[label]["stat"]["inode"] != expected_inode:
            problems.append(label + ":inode-changed")
    for label in ("a", "b"):
        if pair[label]["stat"]["device"] != 52 or pair[label]["stat"]["size_bytes"] != item["size"]:
            problems.append(label + ":device-or-size-mismatch")
        if pair[label]["sha256"] != item["sha256"]:
            problems.append(label + ":sha256-mismatch")
        if not pair[label]["stat"]["is_regular"] or pair[label]["stat"]["is_symlink"]:
            problems.append(label + ":not-regular")
        if pair[label]["stat"]["nlink"] != 1:
            problems.append(label + ":unexpected-link-count")
    if a["stat"]["device"] != b["stat"]["device"]:
        problems.append("cross-device")
    for key in ("uid", "gid", "mode_octal"):
        if a["stat"][key] != b["stat"][key]:
            problems.append("metadata-diff:" + key)
    for key in ("xattrs", "acl"):
        if a[key] != b[key]:
            problems.append("metadata-diff:" + key)
    hits = overlaps(q, item)
    if hits:
        problems.append("active-queue-overlap")
    return problems, hits

result = {"captured_utc": utc(), "mode": mode, "host": "Rockfish", "actions": [], "errors": []}
try:
    result["quota_before"] = subprocess.run(["/usr/lpp/mmfs/bin/mmlsquota", "-g", "pclancy3", "-Y", "data"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60).__dict__
    for item in cfg["items"]:
        q = queue_snapshot()
        pair = capture_pair(item)
        problems, hits = validate(item, pair, q)
        action = {"group": item["group"], "queue": q, "pre_link": pair,
                  "queue_overlaps": hits, "preflight_problems": problems}
        if mode == "execute" and not problems:
            tmp = item["b"] + ".codex-hardlink-" + str(os.getpid()) + "-" + uuid.uuid4().hex
            try:
                os.link(item["a"], tmp, follow_symlinks=False)
                ts = statrow(tmp)
                if ts["device"] != pair["a"]["stat"]["device"] or ts["inode"] != pair["a"]["stat"]["inode"]:
                    raise RuntimeError("temporary link does not match canonical inode")
                os.replace(tmp, item["b"])
            finally:
                if os.path.lexists(tmp):
                    os.unlink(tmp)
            post_a, post_b = statrow(item["a"]), statrow(item["b"])
            hash_a, hash_b = sha(item["a"]), sha(item["b"])
            if post_a["device"] != post_b["device"] or post_a["inode"] != post_b["inode"] or post_a["nlink"] != 2:
                raise RuntimeError("post-link inode/link-count verification failed for " + item["group"])
            if hash_a != item["sha256"] or hash_b != item["sha256"]:
                raise RuntimeError("post-link SHA-256 verification failed for " + item["group"])
            action["post_link"] = {"a": post_a, "b": post_b, "sha256_a": hash_a, "sha256_b": hash_b,
                                    "allocated_bytes_released_estimate": pair["b"]["stat"]["allocated_bytes"]}
            action["state"] = "HARDLINKED_VERIFIED"
        elif mode == "execute":
            action["state"] = "HELD_PREFLIGHT_FAILED"
        else:
            action["state"] = "PREFLIGHT_PASSED" if not problems else "HELD_PREFLIGHT_FAILED"
        result["actions"].append(action)
    result["quota_after"] = subprocess.run(["/usr/lpp/mmfs/bin/mmlsquota", "-g", "pclancy3", "-Y", "data"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60).__dict__ if mode == "execute" else None
    result["post_queue"] = queue_snapshot() if mode == "execute" else None
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
print(json.dumps(result, sort_keys=True))
if result["errors"] or any(a.get("state") == "HELD_PREFLIGHT_FAILED" for a in result["actions"]):
    sys.exit(2)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true", help="read-only checks and evidence capture")
    group.add_argument("--execute", action="store_true", help="atomically replace duplicate paths with hardlinks")
    args = parser.parse_args()
    mode = "execute" if args.execute else "preflight"
    config_json = json.dumps({"mode": mode, "items": EXPECTED}, separators=(",", ":"))
    remote = REMOTE.replace("__CONFIG_B64_DECODE__", "json.loads(" + repr(config_json) + ")")
    proc = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "rockfish", "python3", "-"],
        input=remote,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=1800,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "execution" if args.execute else "preflight"
    output = OUT_DIR / f"hardlink_{suffix}_{stamp}.json"
    status = OUT_DIR / f"hardlink_{suffix}_{stamp}.status.json"
    output.write_text(proc.stdout, encoding="utf-8")
    status.write_text(
        json.dumps(
            {"captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
             "ssh_returncode": proc.returncode, "stderr": proc.stderr, "output_path": str(output),
             "output_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(), "output_bytes": len(proc.stdout.encode())},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"mode": mode, "ssh_returncode": proc.returncode, "output": str(output),
                      "status": str(status), "stderr": proc.stderr}, sort_keys=True))
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
