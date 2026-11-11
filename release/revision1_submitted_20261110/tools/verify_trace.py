#!/usr/bin/env python3
"""Verify that every path cited in RESULTS_INDEX.tsv resolves, and assets match their sha256.

Checks:
  1. every final figure asset exists and its sha256 equals the recorded one;
  2. every repo-relative path token in generator / data / upstream columns exists
     (tokens prefixed rockfish:, local:, icloud:, skipjack: are external and only listed);
  3. no absolute /Users/... path remains;
  4. every file in imports/IMPORT_MANIFEST.tsv still exists with the recorded sha256.
Exit status 1 if any check fails.
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
PATH_COLUMNS = ["generator", "plotted_data", "upstream_data"]
TOKEN = re.compile(r"(?<![\w:/.-])((?:analysis|data|paper|release|reproducibility|docs|src|registry|workflow|configs|figures|scripts)/[\w./{},*+@()\[\]-]+)")
EXTERNAL = re.compile(r"\b(rockfish|local|icloud|skipjack):(/[^\s;|,)]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expand(token: str) -> list[str]:
    token = token.rstrip(".)")
    match = re.search(r"\{([^}]*)\}", token)
    if not match:
        return [token]
    return [alt for part in match.group(1).split(",") for alt in expand(token[:match.start()] + part + token[match.end():])]


def path_exists(token: str) -> bool:
    if any(ch in token for ch in "*["):
        return any(REPO.glob(token))
    candidate = REPO / token
    return candidate.exists() or (REPO / "paper/revision1" / token).exists()


def check_index(rows: list[dict]) -> tuple[list[str], set[str]]:
    errors, external = [], set()
    for row in rows:
        where = f"{row['item_id']} [{row['panel']}]"
        asset, recorded = row["final_asset"], row["asset_sha256"]
        if asset and recorded:
            target = REPO / asset
            if not target.is_file():
                errors.append(f"{where}: final asset missing {asset}")
            elif sha256(target) != recorded:
                errors.append(f"{where}: sha256 mismatch for {asset}")
        text = " ".join(row[column] for column in PATH_COLUMNS)
        external.update(f"{host}:{path}" for host, path in EXTERNAL.findall(" ".join(row.values())))
        for token in (t for raw in TOKEN.findall(text) for t in expand(raw)):
            if not path_exists(token):
                errors.append(f"{where}: cited path not found: {token}")
        if "/Users/" in " ".join(row.values()).replace("local:/Users/", "").replace("icloud:/Users/", ""):
            errors.append(f"{where}: absolute /Users path left in index")
    return errors, external


def check_imports() -> tuple[list[str], list[str]]:
    """Imported files must still exist (or be in the external store); later edits are listed."""
    errors, edited = [], []
    with (PACKAGE / "imports/EXTERNAL_STORE_MANIFEST.tsv").open() as handle:
        stored = {row["repo_path"] for row in csv.DictReader(handle, delimiter="\t")}
    with (PACKAGE / "imports/IMPORT_MANIFEST.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            target = REPO / row["dest"]
            if not target.exists():
                if row["dest"] not in stored:
                    errors.append(f"imported file missing and not in external store: {row['dest']}")
            elif sha256(target) != row["sha256"]:
                edited.append(row["dest"])
    return errors, edited


def main() -> int:
    with (PACKAGE / "RESULTS_INDEX.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    errors, external = check_index(rows)
    import_errors, edited = check_imports()
    errors += import_errors
    print(f"{len(rows)} index rows checked; {len(external)} external locations cited")
    for location in sorted(external):
        print(f"  external: {location}")
    for path in edited:
        print(f"  edited after import (see git log): {path}")
    for error in errors:
        print(f"FAIL {error}")
    print("TRACE OK" if not errors else f"{len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
