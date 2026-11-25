#!/usr/bin/env python3
"""Delete only the eight catalogued WSe2 WFC restart files after live guards."""

import hashlib
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path("/scratch4/pclancy3/yi/3.WSe2/1.pwrelax")
QUEUE_CMD = [
    "/cm/shared/apps/slurm/23.02.2/bin/squeue",
    "-h",
    "-o",
    "%i|%u|%T|%Z",
]
os.environ["SLURM_CONF"] = "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf"

CANDIDATES = [
    ("0.monolayer/WSe2_vc_relax_z.save/wfc2.dat", 143050576, 1734720735167008362, "0f0b268eec01f89bbe409b935c3d8496fdfddbcf00ab69fd693dddbc9099e3a7"),
    ("0.monolayer/WSe2_vc_relax_z.save/wfc1.dat", 143059584, 1734720734316452858, "85accdb40e95a849202932c160b8f2d0a9c8484e656fcafd4dc0f58fe33ec915"),
    ("1/WSe2_vc_relax_z.save/wfc2.dat", 222318940, 1734721573192213463, "c0c50aadb4fbada6f377676f60738b934c41287aebbd7c6d3bd6cdbb717594ee"),
    ("1/WSe2_vc_relax_z.save/wfc1.dat", 222255828, 1734721572556032657, "95e77681c9fc29355ce5c19d8a02ac512d28cc2013cd5753982948adba165453"),
    ("1/WSe2_vc_relax_z.save/wfc3.dat", 222318940, 1734721573893895858, "9c4bbd7051624a142d20c53ffb5e87a579218db4d0c1905f1abafe76df0bb7c4"),
    ("2/WSe2_vc_relax_z.save/wfc2.dat", 1364047632, 1734778037032680271, "b578dcc96faa06274afd50c1e00f450f4447c3054188f83088d57a56fc9ae903"),
    ("2/WSe2_vc_relax_z.save/wfc1.dat", 1364002612, 1734778034787525195, "d3cb0a01b1eaad9cbe5a74b2b10ff3a32ad5a8981730b55fa55da4336114ab00"),
    ("2/WSe2_vc_relax_z.save/wfc3.dat", 1364047632, 1734778039671464256, "7c5796884b5a721e89950d0301d5d3b42a609448140df08f5faff086e8f0c888"),
]
RUN_DIRS = [ROOT / "0.monolayer", ROOT / "1", ROOT / "2"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlapping_jobs() -> List[Dict[str, str]]:
    result = subprocess.run(QUEUE_CMD, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    matches = []
    for line in result.stdout.splitlines():
        fields = line.split("|", 3)
        if len(fields) != 4:
            continue
        job_id, user, state, workdir = fields
        if not workdir.startswith("/"):
            continue
        work = Path(workdir)
        for run_dir in RUN_DIRS:
            if work == run_dir or work in run_dir.parents or run_dir in work.parents:
                matches.append({"job_id": job_id, "user": user, "state": state, "workdir": workdir})
                break
    return matches


def main() -> None:
    if not ROOT.is_dir() or ROOT.is_symlink():
        raise SystemExit(f"Guard failed: root missing or symlink: {ROOT}")
    first_queue = overlapping_jobs()
    if first_queue:
        raise SystemExit("Guard failed: queue overlap: " + json.dumps(first_queue))

    checked = []
    for rel, expected_size, expected_mtime_ns, expected_hash in CANDIDATES:
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"Guard failed: not a regular non-symlink file: {path}")
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise SystemExit(f"Guard failed: resolved path mismatch: {path}")
        if ROOT not in path.resolve(strict=True).parents:
            raise SystemExit(f"Guard failed: path escapes approved root: {path}")
        st = os.lstat(str(path))
        if (st.st_uid, st.st_gid, stat.S_IMODE(st.st_mode), st.st_nlink, st.st_size, st.st_mtime_ns) != (
            3901, 3001, 0o644, 1, expected_size, expected_mtime_ns
        ):
            raise SystemExit(f"Guard failed: metadata changed for {path}: {st}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise SystemExit(f"Guard failed: SHA-256 changed for {path}: {actual_hash}")
        checked.append({"path": str(path), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": actual_hash})

    second_queue = overlapping_jobs()
    if second_queue:
        raise SystemExit("Guard failed after hashing: queue overlap: " + json.dumps(second_queue))

    print(json.dumps({"event": "preflight_passed", "utc": utc_now(), "queue_overlaps": 0, "files": checked}), flush=True)
    removed = []
    for item in checked:
        path = Path(item["path"])
        # Re-check identity immediately before unlink, guarding against path replacement.
        st = os.lstat(str(path))
        if path.is_symlink() or not path.is_file() or st.st_uid != 3901 or st.st_gid != 3001 or st.st_nlink != 1:
            raise SystemExit(f"Guard failed immediately before unlink: {path}")
        if st.st_size != item["size_bytes"] or st.st_mtime_ns != item["mtime_ns"] or sha256(path) != item["sha256"]:
            raise SystemExit(f"Guard failed immediately before unlink: content or metadata changed: {path}")
        path.unlink()
        if path.exists() or path.is_symlink():
            raise SystemExit(f"Post-delete verification failed: {path}")
        removed.append({**item, "deleted_utc": utc_now(), "verified_missing": True})
        print(json.dumps({"event": "deleted_verified", **removed[-1]}), flush=True)
    print(json.dumps({"event": "complete", "utc": utc_now(), "file_count": len(removed), "bytes": sum(x["size_bytes"] for x in removed)}), flush=True)


if __name__ == "__main__":
    main()
