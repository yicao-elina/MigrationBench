#!/usr/bin/env python3
"""Capture a compact, hashable runtime and input provenance record."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", action="append", default=[])
    args = parser.parse_args()

    inputs: dict[str, dict[str, str | None]] = {}
    for item in args.input:
        key, _, raw = item.partition("=")
        path = Path(raw)
        inputs[key] = {"path": str(path), "sha256": sha256(path)}
    payload = {
        "schema_version": "runtime_provenance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": str(args.binary),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "git_revision": git_revision(),
        "packages": {
            name: version(name)
            for name in ("ase", "mace-torch", "torch", "numpy", "pandas")
        },
        "inputs": inputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "inputs": len(inputs)}))


if __name__ == "__main__":
    main()
