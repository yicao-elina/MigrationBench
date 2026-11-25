"""Verify foundation-model MD manifests and their referenced file hashes.

This is deliberately dependency-free so it can run before installing the
project environment and can also be used from a local pre-commit check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_dir = root / "manifests"
    manifests = sorted(manifest_dir.glob("*.json"))
    if not manifests:
        raise SystemExit(f"no manifests found in {manifest_dir}")

    for manifest_path in manifests:
        record = json.loads(manifest_path.read_text())
        files = record.get("files")
        if not isinstance(files, list) or not files:
            raise SystemExit(f"{manifest_path}: expected non-empty files list")
        for item in files:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise SystemExit(f"{manifest_path}: each file needs path and sha256")
            target = (root / item["path"]).resolve()
            if root not in target.parents:
                raise SystemExit(f"{manifest_path}: file escapes experiment root: {item['path']}")
            if not target.is_file():
                raise SystemExit(f"{manifest_path}: missing file: {item['path']}")
            actual = sha256(target)
            if actual != item["sha256"]:
                raise SystemExit(
                    f"{manifest_path}: sha256 mismatch for {item['path']}: "
                    f"expected {item['sha256']}, got {actual}"
                )
    print(f"verified {len(manifests)} manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
