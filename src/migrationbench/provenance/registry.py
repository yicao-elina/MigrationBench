from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

MANIFEST_NAMES = {
    "runtime_provenance.json",
    "qe_restart_manifest.json",
    "scf_warmup_manifest.json",
    "endpoint_relax_manifest.json",
    "mlff_neb_manifest.json",
    "neb_manifest.json",
    "dataset_manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def iter_runs(raw_root: Path) -> Iterable[dict[str, Any]]:
    for run_dir in sorted(path for path in raw_root.glob("*/*") if path.is_dir()):
        manifests = sorted(
            path for path in run_dir.rglob("*.json") if path.name in MANIFEST_NAMES
        )
        metadata: dict[str, Any] = {}
        for manifest in manifests:
            metadata.update(_read_json(manifest))
        files = sorted(path for path in run_dir.rglob("*") if path.is_file())
        if not files:
            continue
        run_id = str(metadata.get("run_id") or metadata.get("job_name") or run_dir.name)
        job_id = metadata.get("slurm_job_id") or metadata.get("job_id") or ""
        yield {
            "run_id": run_id,
            "source_collection": run_dir.parent.name,
            "relative_path": run_dir.as_posix(),
            "slurm_job_id": str(job_id),
            "manifest_count": len(manifests),
            "file_count": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "tree_sha256": hashlib.sha256(
                "\n".join(f"{path.relative_to(run_dir)} {sha256(path)}" for path in files).encode()
            ).hexdigest(),
        }


def build_registry(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    rows = list(iter_runs(root / "data" / "raw_compact"))
    for row in rows:
        row["relative_path"] = str(Path(row["relative_path"]).relative_to(root))
    output = root / "registry" / "runs.csv"
    fieldnames = [
        "run_id",
        "source_collection",
        "relative_path",
        "slurm_job_id",
        "manifest_count",
        "file_count",
        "bytes",
        "tree_sha256",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd

        pd.DataFrame(rows, columns=fieldnames).to_parquet(
            root / "registry" / "runs.parquet", index=False
        )
    except (ImportError, ModuleNotFoundError):
        pass
    return rows

