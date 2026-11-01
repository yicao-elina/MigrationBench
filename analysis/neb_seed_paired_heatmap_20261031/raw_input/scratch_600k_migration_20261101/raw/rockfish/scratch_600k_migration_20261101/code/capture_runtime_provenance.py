#!/usr/bin/env python3
"""Capture a dependency-light, hash-bound runtime identity for a cluster job."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PACKAGES = (
    "ase",
    "mace-torch",
    "matplotlib",
    "numpy",
    "pandas",
    "pytest",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "torch",
)

IMPORT_NAMES = {
    "mace-torch": "mace",
    "PyYAML": "yaml",
    "scikit-learn": "sklearn",
}

ENV_KEYS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "LOADEDMODULES",
    "OMP_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_MEM_PER_NODE",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_SUBMIT_DIR",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path).expanduser().resolve()


def file_identity(path: Path) -> dict:
    record = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return record


def package_identities(names: list[str]) -> dict[str, dict]:
    result = {}
    for name in names:
        record = {
            "distribution_version": None,
            "module": IMPORT_NAMES.get(name, name.replace("-", "_")),
            "module_version": None,
            "module_file": None,
            "module_file_sha256": None,
            "import_error": None,
        }
        try:
            record["distribution_version"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
        try:
            module = importlib.import_module(record["module"])
            record["module_version"] = getattr(module, "__version__", None)
            module_file = getattr(module, "__file__", None)
            if module_file and Path(module_file).is_file():
                module_path = Path(module_file).resolve()
                record["module_file"] = str(module_path)
                record["module_file_sha256"] = sha256_file(module_path)
        except Exception as exc:  # Runtime provenance must survive optional imports.
            record["import_error"] = f"{type(exc).__name__}: {exc}"
        result[name] = record
    return result


def build_record(inputs: list[tuple[str, Path]], binaries: list[str], packages: list[str]) -> dict:
    binary_records = {}
    for name in binaries:
        resolved = shutil.which(name)
        binary_records[name] = (
            file_identity(Path(resolved).resolve()) if resolved else {"path": None, "exists": False}
        )
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": package_identities(packages),
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        "binaries": binary_records,
        "inputs": {name: file_identity(path) for name, path in inputs},
        "generator": file_identity(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", action="append", default=[], type=named_path)
    parser.add_argument("--binary", action="append", default=[])
    parser.add_argument("--package", action="append", default=[])
    args = parser.parse_args()
    packages = sorted(set(args.package or DEFAULT_PACKAGES))
    record = build_record(args.input, args.binary, packages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runtime_provenance": str(args.output), "inputs": len(args.input)}))


if __name__ == "__main__":
    main()
