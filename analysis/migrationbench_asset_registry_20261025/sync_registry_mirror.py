#!/usr/bin/env python3
"""Stage a validated local registry snapshot into a dedicated Drive folder.

This script intentionally does not edit the native Google Sheet.  It copies
the local source-of-truth artifacts to an immutable timestamped snapshot and
to a small ``current`` mirror directory.  The Sheet is updated separately
through the connected Google Sheets workflow after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from export_registry_data import export_registry


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "registry_local_config.json"
MANIFEST_NAME = "REGISTRY_SYNC_MANIFEST.json"
MIRROR_FILES = (
    "README.md",
    "GOOGLE_SHEETS_LINK.md",
    "migrationbench_asset_registry.sqlite",
    "migrationbench_asset_registry.xlsx",
    "registry_data.json",
    "schema.sql",
    "query_examples.sql",
    "data_quality_report.json",
    "models.csv",
    "model_families.csv",
    "model_groups.csv",
    "model_group_members.csv",
    "model_name_map.csv",
    "model_aliases.csv",
    "tasks.csv",
    "task_groups.csv",
    "task_group_members.csv",
    "task_aliases.csv",
    "task_group_aliases.csv",
    "systems.csv",
    "campaigns.csv",
    "datasets.csv",
    "coverage.csv",
    "assets.csv",
    "gaps.csv",
    "sources.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def destination_from_args(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    override = os.environ.get("MIGRATIONBENCH_DRIVE_SYNC_DIR")
    if override:
        return Path(override).expanduser().resolve()
    config = load_config()
    return Path(str(config["google_drive_file_mirror"])).expanduser().resolve()


def current_manifest() -> dict[str, object]:
    files: list[dict[str, object]] = []
    missing: list[str] = []
    for name in MIRROR_FILES:
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
            continue
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_root": str(ROOT),
        "native_google_sheet_url": load_config().get("native_google_sheet_url"),
        "files": files,
        "missing_files": missing,
    }


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def check() -> int:
    manifest = current_manifest()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 1 if manifest["missing_files"] else 0


def stage(destination: Path) -> int:
    export_registry(ROOT)
    manifest = current_manifest()
    missing = manifest["missing_files"]
    if missing:
        raise SystemExit("Cannot stage: missing local artifacts: " + ", ".join(missing))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = destination / "snapshots" / timestamp
    current = destination / "current"
    snapshot.mkdir(parents=True, exist_ok=False)
    current.mkdir(parents=True, exist_ok=True)

    for entry in manifest["files"]:
        name = str(entry["name"])
        source = ROOT / name
        shutil.copy2(source, snapshot / name)
        temporary = current / f".{name}.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(current / name)

    manifest["snapshot"] = str(snapshot)
    manifest["current_mirror"] = str(current)
    manifest["sync_mode"] = "explicit_stage_only"
    write_json_atomic(snapshot / MANIFEST_NAME, manifest)
    write_json_atomic(current / MANIFEST_NAME, manifest)
    write_json_atomic(ROOT / "last_sync_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="print local artifact hashes without writing")
    parser.add_argument("--stage", action="store_true", help="copy a timestamped snapshot to the Drive mirror")
    parser.add_argument("--destination", help="override the configured Drive mirror directory")
    args = parser.parse_args()
    if args.stage == args.check:
        parser.error("choose exactly one of --check or --stage")
    if args.check:
        return check()
    return stage(destination_from_args(args.destination))


if __name__ == "__main__":
    raise SystemExit(main())
