#!/usr/bin/env python3
"""Stage the 37 missing Rockfish endpoint inputs without touching Rockfish."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "rockfish_authoritative_20261004"
REMOTE = "rockfish"
ROOT_V3 = "/scratch16/pclancy3/yi/revision1_migrationbench_inputs/endpoint_dft_refine_s42_20260917_v3"
ROOT_V8 = "/scratch16/pclancy3/yi/revision1_migrationbench_inputs/endpoint_dft_refine_s42_20260917_v8_mixedavid"
ROOT_V7 = "/scratch16/pclancy3/yi/revision1_migrationbench_inputs/endpoint_dft_refine_s42_20260919_v7_2x2x2"

V7 = {"unique_020"}
V8 = {"unique_204"}
IDS = [
    "unique_020", "unique_031", "unique_039", "unique_043", "unique_068",
    "unique_069", "unique_070", "unique_071", "unique_072", "unique_073",
    "unique_077", "unique_078", "unique_080", "unique_093", "unique_096",
    "unique_097", "unique_098", "unique_100", "unique_102", "unique_104",
    "unique_109", "unique_110", "unique_115", "unique_116", "unique_125",
    "unique_126", "unique_129", "unique_143", "unique_145", "unique_149",
    "unique_151", "unique_153", "unique_154", "unique_200", "unique_204",
    "unique_229", "unique_230",
]

def root_for(endpoint: str) -> str:
    if endpoint in V7:
        return ROOT_V7
    if endpoint in V8:
        return ROOT_V8
    return ROOT_V3

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for endpoint in IDS:
        root = root_for(endpoint)
        remote_dir = f"{root}/dft_endpoint_{endpoint}_from_mace_s42"
        remote_input = f"{remote_dir}/relax.in"
        local_dir = DEST / endpoint
        local_dir.mkdir(parents=True, exist_ok=True)
        local_input = local_dir / "relax.in"
        if not local_input.exists() or local_input.stat().st_size == 0:
            subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", f"{REMOTE}:{remote_input}", str(local_input)], check=True)
        rows.append((endpoint, remote_dir, str(local_input), sha256(local_input)))
        print(f"staged {endpoint} {sha256(local_input)}")
    manifest = DEST / "ROCKFISH_SOURCE_MANIFEST.tsv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("endpoint\trockfish_source_dir\tlocal_input\tlocal_sha256\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"wrote {manifest}")

if __name__ == "__main__":
    main()
