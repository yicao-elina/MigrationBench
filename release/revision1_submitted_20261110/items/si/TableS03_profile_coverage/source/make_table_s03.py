#!/usr/bin/env python3
"""SI Table S3 (tab:profile_coverage): checkpoints per operator/path behind the Fig. 4 profiles.

The submitted table equals the R02 coverage audit. Fig. 4 itself was rendered from the R05
release, whose scratch groups differ (Known issue 4); the R05 counts are recorded as an
informational check. Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence  # noqa: E402

CURRENT = "paper/revision1/data_processed/current_release"
AUDIT_R02 = f"{CURRENT}/cr_three_path_coverage_audit_R02.tsv"
AUDIT_R05 = f"{CURRENT}/cr_three_path_coverage_audit_R05.tsv"

OPERATORS = [("FIXED_PATH_SINGLE_POINT", "Fixed"), ("SELF_RELAXED_ENDPOINT_ONLY_NEB", "Self")]
PATHS = [("in-gap diffusion", "in gap"), ("gap→QL", "gap to QL"), ("within-QL", "within QL")]
COLUMNS_R02 = ["foundation", "scratch_legacy", "ft_600k", "ft_multit"]   # submitted column order
# R05 has no scratch_legacy group; its current-release Scratch-600K group fills that column.
COLUMNS_R05 = ["foundation", "scratch_600k", "ft_600k", "ft_multit"]


def load_counts(path: Path) -> dict[tuple[str, str, str], int]:
    with path.open(newline="") as handle:
        return {(r["observable"], r["path_class"], r["group"]): int(r["n_models"])
                for r in csv.DictReader(handle, delimiter="\t")}


def build_rows(counts: dict, columns: list[str]) -> list[list[str]]:
    return [[f"{op_label} / {path_label}", *(str(counts[(op, path, group)]) for group in columns)]
            for op, op_label in OPERATORS for path, path_label in PATHS]


def main() -> int:
    ev = Evidence(__file__, "tab:profile_coverage")
    r02 = build_rows(load_counts(ev.input(AUDIT_R02)), COLUMNS_R02)
    ev.compare_table(r02, "sn-article-SI.tex")

    r05 = build_rows(load_counts(ev.input(AUDIT_R05)), COLUMNS_R05)
    header = ["Foundation", "Legacy Scratch (R05: scratch_600k)", "FT-600K", "FT-MultiT"]
    differences = [{"row": a[0], "column": header[i - 1], "R02_submitted": a[i], "R05": b[i]}
                   for a, b in zip(r02, r05) for i in range(1, len(a)) if a[i] != b[i]]
    ev.check("R05 counts (informational, Known issue 4): Fig. 4 uses R05; table submitted from R02",
             True, {"differing_cells": differences, "r05_rows": r05})
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
