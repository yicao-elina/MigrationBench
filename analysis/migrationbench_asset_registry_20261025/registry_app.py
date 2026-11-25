#!/usr/bin/env python3
"""Run the pinned local registry UI from this source-of-truth directory."""

from __future__ import annotations

import marshal
import os
import sys
from pathlib import Path


def _compiled_app() -> Path:
    root = Path(__file__).resolve().parent
    preferred = root / "runtime" / f"registry_app.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
    if preferred.exists():
        return preferred
    candidates = sorted((root / "runtime").glob("registry_app.cpython-*.pyc"))
    if not candidates:
        raise FileNotFoundError("Pinned compiled registry UI is missing from __pycache__.")
    return candidates[-1]


def main() -> None:
    pyc_path = _compiled_app()
    code = marshal.loads(pyc_path.read_bytes()[16:])
    namespace = {
        "__name__": "_migrationbench_registry_app",
        "__file__": str(Path(__file__).resolve()),
        "__package__": None,
        "__cached__": str(pyc_path),
    }
    exec(code, namespace)
    if os.environ.get("MIGRATIONBENCH_PORT"):
        namespace["PORT"] = int(os.environ["MIGRATIONBENCH_PORT"])
    namespace["main"]()


if __name__ == "__main__":
    main()
