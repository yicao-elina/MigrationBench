#!/usr/bin/env python3
r"""Check revision1 manuscript source ledger and figure/table references.

This is a lightweight guardrail for the Overleaf working copy. It verifies that:
  1. every \includegraphics path in the revision1 TeX files exists;
  2. every \input{revision1/tables/...} path exists;
  3. every claim_token in the manuscript source ledger appears in the current
     revision1 manuscript sources or in the referenced table inputs.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


R1 = Path(__file__).resolve().parents[1]
ROOT = R1.parent
TEX_FILES = [R1 / "sn-article-final.tex", R1 / "sn-article-SI.tex"]
LEDGER = R1 / "data_processed" / "manuscript_source_ledger.csv"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_tex_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    errors: list[str] = []
    tex_text = "\n".join(read(p) for p in TEX_FILES)
    normalized_tex_text = tex_text.replace(r"\_", "_")

    for tex in TEX_FILES:
        text = read(tex)
        for raw in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            path = resolve_tex_path(raw)
            if not path.exists():
                errors.append(f"Missing figure referenced by {tex.relative_to(ROOT)}: {raw}")
        for raw in re.findall(r"\\input\{([^}]+)\}", text):
            if raw.startswith("revision1/tables/"):
                path = resolve_tex_path(raw)
                if not path.exists():
                    errors.append(f"Missing table input referenced by {tex.relative_to(ROOT)}: {raw}")
                else:
                    table_text = read(path)
                    tex_text += "\n" + table_text
                    normalized_tex_text += "\n" + table_text.replace(r"\_", "_")

    with LEDGER.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            token = row["claim_token"]
            source = resolve_tex_path(row["source_artifact"])
            if not source.exists():
                errors.append(f"Ledger {row['claim_id']} missing source artifact: {row['source_artifact']}")
            if token and token not in normalized_tex_text:
                errors.append(f"Ledger {row['claim_id']} token not found in manuscript/table text: {token}")

    if errors:
        print("FAIL: manuscript source audit found issues:")
        for err in errors:
            print(f" - {err}")
        return 1

    print("PASS: all ledger tokens and referenced revision1 figures/tables are traceable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
