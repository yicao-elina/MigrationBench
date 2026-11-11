#!/usr/bin/env python3
"""Copy outside-repo provenance sources into the repository, with a hash manifest.

Reads ``imports/import_plan.tsv`` (columns: source, dest, reason). For a
directory source, every regular file is copied unless it is a cache/junk file
or larger than ``--max-dir-file-mb``; oversized files are not copied but are
listed in ``imports/DEFERRED_LARGE_FILES.tsv`` so they can go to external
storage. A single-file source is always copied if it is below
``--max-single-file-mb``. Existing destination files are never overwritten;
differing ones are reported as conflicts instead.

Outputs (relative to the release package):
  imports/IMPORT_MANIFEST.tsv       one row per copied file (sha256, size)
  imports/DEFERRED_LARGE_FILES.tsv  files left out because of size
  imports/IMPORT_CONFLICTS.tsv      destination exists with different bytes
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

JUNK_NAMES = {".DS_Store", "Thumbs.db"}
JUNK_DIRS = {"__pycache__", "node_modules", ".git", ".ipynb_checkpoints", ".venv"}
MB = 1024 * 1024
SF_DATALESS = 0x40000000  # macOS: file content evicted to iCloud; reading blocks on download


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(source: Path):
    if source.is_file():
        yield source, Path(source.name)
        return
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if any(part in JUNK_DIRS for part in rel.parts):
            continue
        if path.is_symlink() or not path.is_file() or path.name in JUNK_NAMES:
            continue
        yield path, rel


def import_row(row, repo: Path, limits, out):
    source, dest = Path(row["source"]), repo / row["dest"]
    if not source.exists():
        out["missing"].append([str(source), row["dest"], "source missing"])
        return
    single = source.is_file()
    limit = limits["single"] if single else limits["dir"]
    for path, rel in iter_files(source):
        target = dest if single else dest / rel
        stat = path.stat()
        size = stat.st_size
        if getattr(stat, "st_flags", 0) & SF_DATALESS:
            out["dataless"].append([str(path), str(target.relative_to(repo)), size, row["reason"]])
            continue
        if size > limit:
            out["deferred"].append([str(path), str(target.relative_to(repo)), size, row["reason"]])
            continue
        digest = sha256_of(path)
        if target.exists():
            if sha256_of(target) != digest:
                out["conflicts"].append([str(path), str(target.relative_to(repo)), size, digest])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        out["copied"].append([str(path), str(target.relative_to(repo)), size, digest, row["reason"]])


def write_tsv(path: Path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True, help="release package dir")
    parser.add_argument("--max-dir-file-mb", type=float, default=5.0)
    parser.add_argument("--max-single-file-mb", type=float, default=30.0)
    args = parser.parse_args()

    plan = args.package / "imports" / "import_plan.tsv"
    with plan.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    limits = {"dir": args.max_dir_file_mb * MB, "single": args.max_single_file_mb * MB}
    out = {"copied": [], "deferred": [], "conflicts": [], "missing": [], "dataless": []}
    for row in rows:
        import_row(row, args.repo.resolve(), limits, out)

    imports = args.package / "imports"
    write_tsv(imports / "IMPORT_MANIFEST.tsv", ["source", "dest", "bytes", "sha256", "reason"], out["copied"])
    write_tsv(imports / "DEFERRED_LARGE_FILES.tsv", ["source", "intended_dest", "bytes", "reason"], out["deferred"])
    write_tsv(imports / "IMPORT_CONFLICTS.tsv", ["source", "dest", "bytes", "source_sha256"], out["conflicts"])
    write_tsv(imports / "ICLOUD_DATALESS.tsv", ["source", "intended_dest", "bytes", "reason"], out["dataless"])
    for item in out["missing"]:
        print("MISSING:", *item, file=sys.stderr)
    total = sum(r[2] for r in out["copied"])
    print(f"copied {len(out['copied'])} files ({total / MB:.1f} MB); "
          f"deferred {len(out['deferred'])}; iCloud-dataless {len(out['dataless'])}; conflicts {len(out['conflicts'])}; missing {len(out['missing'])}")
    return 1 if out["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
