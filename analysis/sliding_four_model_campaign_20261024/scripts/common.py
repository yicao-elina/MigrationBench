#!/usr/bin/env python3
"""Small shared helpers for the fixed-path campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml


def load_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    return config, config_path.parent.parent


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | Path, root: Path | None = None) -> dict:
    file_path = Path(path)
    record = {"path": str(file_path), "exists": file_path.is_file()}
    if file_path.is_file():
        record.update({"bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)})
    if root is not None:
        try:
            record["relative_path"] = str(file_path.relative_to(root))
        except ValueError:
            pass
    return record


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_record() -> dict:
    packages = {}
    for name in ("ase", "mace-torch", "numpy", "pandas", "PyYAML", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at_utc": now_utc(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": packages,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CONDA_DEFAULT_ENV",
                "CONDA_PREFIX",
                "CUDA_VISIBLE_DEVICES",
                "LOADEDMODULES",
                "OMP_NUM_THREADS",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
            )
        },
    }


def model(config: dict, model_id: str) -> dict:
    try:
        return config["models"][model_id]
    except KeyError as exc:
        raise KeyError(f"unknown model_id: {model_id}") from exc


def system(config: dict, system_id: str) -> dict:
    try:
        return config["systems"][system_id]
    except KeyError as exc:
        raise KeyError(f"unknown system_id: {system_id}") from exc


def root_path(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

