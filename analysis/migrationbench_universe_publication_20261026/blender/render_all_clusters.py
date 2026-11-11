#!/usr/bin/env python3
"""Render every cluster representative from the TSV job manifest.

The job manifest is parsed as a real tab-separated file so empty metadata
fields (for example a missing path_id on a reference configuration) cannot
shift the output directory into the wrong column.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
RENDERER = ROOT / "blender" / "render_cluster_environment.py"
JOBS = ROOT / "blender" / "render_jobs.tsv"


def main() -> None:
    with JOBS.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))

    required = {"cluster", "cluster_color", "render_input", "output_dir"}
    for job in jobs:
        missing = sorted(key for key in required if not job.get(key))
        if missing:
            raise ValueError(f"{job.get('cluster', '<unknown>')}: missing {missing}")
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(BLENDER),
            "-b",
            "--python",
            str(RENDERER),
            "--",
            "--input",
            job["render_input"],
            "--output-dir",
            str(output_dir),
            "--frame",
            "0",
            "--width",
            "2800",
            "--height",
            "2800",
            "--samples",
            "96",
            "--engine",
            "cycles",
            "--bond-mode",
            "coordination",
            "--bond-cutoff",
            "3.75",
            "--bond-radius",
            "0.105",
            "--accent-rgb",
            job["cluster_color"],
            "--save-blend",
        ]
        print(f"[render] {job['cluster']} -> {output_dir}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
