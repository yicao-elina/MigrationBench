#!/usr/bin/env python3
"""Collect remote file metadata without copying raw calculation files.

The output is a processed inventory only: raw QE/NEB files remain on Rockfish.
Run from the repository with an SSH config entry named ``rockfish``.
"""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "collaborator_dft_neb" / "rockfish_file_inventory.tsv"
REMOTE_ROOTS = [
    ("/scratch16/pclancy3/victor/1.defects_scripts", 8),
    # Deep recursive scans of these trees are intentionally not the default:
    # nebProject contains very large restart/output subtrees. The next pass
    # expands each accepted calculation directory after this inventory stage.
    ("/scratch16/pclancy3/victor/nebProject", 3),
    ("/scratch16/pclancy3/victor/V-ST", 3),
    ("/scratch16/pclancy3/victor/ST-other", 3),
    ("/scratch16/pclancy3/victor/flare/sb2te3", 3),
]


def main() -> None:
    # Tabs and newlines are excluded by the fixed find format and paths are
    # emitted one per line. Permission errors are retained in stderr for audit.
    remote_script = "set -o pipefail; "
    for root, depth in REMOTE_ROOTS:
        # Metadata-only discovery is restricted to reproducibility-relevant
        # source formats; large binary scratch files are not traversed.
        remote_script += (
            f"find {root!r} -maxdepth {depth} -type f "
            "\\( -name '*.in' -o -name '*.out' -o -name '*.slurm' "
            "-o -name '*.sh' -o -name '*.py' -o -name '*.xyz' "
            "-o -name '*.extxyz' -o -name '*.cif' -o -name '*.dat' "
            "-o -name '*.csv' -o -name '*.json' \\) "
            "-printf '%u\\t%g\\t%s\\t%TY-%Tm-%TdT%TH:%TM:%TS\\t%p\\n'; "
        )
    cmd = ["ssh", "-o", "BatchMode=yes", "rockfish", "sh", "-c", remote_script]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        fh.write(f"# collected_utc={datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"# host=rockfish\n")
        fh.write("owner\tgroup\tsize_bytes\tmtime\tpath\n")
        for line in proc.stdout.splitlines():
            fields = line.split("\t", 4)
            if len(fields) == 5:
                fh.write("\t".join(fields) + "\n")
    print(f"wrote {OUT} with {sum(1 for _ in proc.stdout.splitlines())} metadata rows")
    if proc.stderr:
        print("remote_stderr:", proc.stderr[:2000])


if __name__ == "__main__":
    main()
