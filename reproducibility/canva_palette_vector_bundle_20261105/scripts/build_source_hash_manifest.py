#!/usr/bin/env python3
"""Hash every materialized source-archive file in the Canva bundle."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE / "data/source_archive"
OUT = PACKAGE / "manifests/SOURCE_FILE_HASHES.tsv"


def sha256(path: Path) -> str:
    # Git normalizes CRLF text files in this repository.  Hash text after the
    # same LF normalization so the manifest is stable in the current worktree
    # and after a fresh clone; binary arrays/archives remain byte-exact.
    with path.open("rb") as handle:
        prefix = handle.read(8192)
    digest = hashlib.sha256()
    if b"\x00" in prefix:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    else:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(SOURCE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        rows.append((rel, path.stat().st_size, sha256(path)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        handle.write("relative_path\tbytes\tsha256\tstatus\n")
        for rel, size, digest in rows:
            handle.write(f"{rel}\t{size}\t{digest}\tMATERIALIZED\n")
    print(f"hashed {len(rows)} materialized source files -> {OUT}")


if __name__ == "__main__":
    main()
