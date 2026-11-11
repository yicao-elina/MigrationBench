#!/usr/bin/env python3
"""SI Table S7 (tab:scratch5_path_resolved_descriptive): path-resolved fixed-path errors.

Reimplements the arithmetic of
paper/response_to_reviewers/reviewer_a2_grouped_retraining/build_a5_path_resolved_table.py
(mean +/- sample SD of absolute_error_ev over three seeds per family; MPA0 excluded) without
writing into paper/. Path names and classes come from the path registry; the "(reference path)"
tag follows the registry basis. Results are cross-checked against the generator's manifest.
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_num  # noqa: E402

A2 = "paper/response_to_reviewers/reviewer_a2_grouped_retraining"
NEB_CSV = f"{A2}/data/grouped_neb_barrier_summary.csv"
MANIFEST = f"{A2}/a5_path_resolved_table_manifest.json"
REGISTRY = "paper/revision1/data_processed/path_registry.csv"
PATH_ORDER = ["G1", "G2", "Q1", "Q2", "Q3"]                      # registry names, submitted order
FAMILIES = [("FT-600K", "ft600k_omatft"), ("Scratch-5%", "scratch5_"), ("Scratch", "scratch_grouped")]
REFERENCE_TAG = "reference path"
DECIMALS = 2
TOLERANCE = 1e-9
REGISTRY_TOLERANCE_EV = 1e-5  # registry stores barriers to 6 decimals


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pm(values: list[float]) -> str:
    """Spacing as in the submitted table: '$0.47 \\pm 0.12$'."""
    return f"${fmt_num(statistics.mean(values), DECIMALS)} \\pm {fmt_num(statistics.stdev(values), DECIMALS)}$"


def main() -> int:
    ev = Evidence(__file__, "tab:scratch5_path_resolved_descriptive")
    registry = {r["name"]: r for r in read_csv(ev.input(REGISTRY))}
    neb = read_csv(ev.input(NEB_CSV))
    manifest = json.loads(ev.input(MANIFEST).read_text())["summary"]

    rows, problems = [], []
    for name in PATH_ORDER:
        entry = registry[name]
        path_rows = [r for r in neb if r["pathway"] == entry["audit_id"]]
        references = {float(r["dft_barrier_ev_from_neb_out"]) for r in path_rows}
        reference = min(references)
        if len(references) != 1 or abs(reference - float(entry["qe_forward_barrier_eV"])) > REGISTRY_TOLERANCE_EV:
            problems.append(f"{name}: QE reference {references} vs registry {entry['qe_forward_barrier_eV']}")
        label = entry["class"] + (f" ({REFERENCE_TAG})" if entry["classification_basis"].endswith(REFERENCE_TAG) else "")
        cells = [name, label, fmt_num(reference, DECIMALS)]
        for family, prefix in FAMILIES:
            errors = [float(r["absolute_error_ev"]) for r in path_rows if r["model_id"].startswith(prefix)]
            stored = manifest[entry["audit_id"]]["abs_error_eV_mean_sample_sd"][family]
            if abs(statistics.mean(errors) - stored["mean"]) > TOLERANCE or \
                    abs(statistics.stdev(errors) - stored["sample_sd"]) > TOLERANCE:
                problems.append(f"{name} {family}: recomputed vs manifest {stored}")
            cells.append(pm(errors))
        rows.append(cells)
    ev.check("QE references match path_registry.csv; mean/SD equal a5_path_resolved_table_manifest.json",
             not problems, problems or f"{len(rows)} paths x {len(FAMILIES)} families agree")
    ev.compare_table(rows, "sn-article-SI.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
