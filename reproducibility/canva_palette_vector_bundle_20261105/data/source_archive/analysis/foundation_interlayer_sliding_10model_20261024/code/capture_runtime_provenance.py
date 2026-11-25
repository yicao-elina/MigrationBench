#!/usr/bin/env python3
"""Write an immutable, hashable runtime and input provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("ase", "mace-torch", "torch", "numpy", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_input(spec: str) -> dict:
    if "=" not in spec:
        raise ValueError(f"--input must be label=path: {spec}")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).resolve()
    return {"label": label, "path": str(path), "exists": path.is_file(), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", action="append", default=[])
    args = parser.parse_args()
    selected_environment = {
        key: os.environ.get(key)
        for key in (
            "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK",
            "SLURM_PARTITION", "SLURM_ACCOUNT", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
        )
        if os.environ.get(key) is not None
    }
    payload = {
        "schema_version": "foundation_interlayer_sliding_runtime_provenance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "binary": str(args.binary.resolve()),
        "binary_sha256": sha256(args.binary.resolve()),
        "package_versions": package_versions(),
        "selected_environment": selected_environment,
        "inputs": [parse_input(spec) for spec in args.input],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"immutable provenance already exists: {args.output}")
    with args.output.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(args.output, 0o444)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
