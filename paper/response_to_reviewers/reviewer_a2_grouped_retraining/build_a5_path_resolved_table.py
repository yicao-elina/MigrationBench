#!/usr/bin/env python3
"""Build the descriptive A5 path-resolved table from the audited CSV.

The table deliberately retains paths whose DFT reference or pathway class is
not yet publication-grade.  It is therefore a descriptive diagnostic, not a
pooled ranking.  The manifest records the input, script, and output hashes so
that the generated SI insert remains traceable to the raw audit bundle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
PAPER = BUNDLE.parents[1]
REPO = BUNDLE.parents[2]
INPUT = BUNDLE / "data" / "grouped_neb_barrier_summary.csv"
OUTPUT = PAPER / "revision1" / "tables" / "scratch5_path_resolved_descriptive.tex"
MANIFEST = BUNDLE / "a5_path_resolved_table_manifest.json"

FAMILY_ORDER = ["FT-600K", "Scratch-5%", "Scratch"]
FAMILY_PREFIX = {
    "FT-600K": "ft600k_omatft",
    "Scratch-5%": "scratch5_",
    "Scratch": "scratch_",
}
PATH_CLASS = {
    "1-2": "gap-to-QL/into-QL",
    "1-3": "gap-to-QL/into-QL",
    "1-4": "within-gap (reference path)",
    "1-5": "gap-to-QL/into-QL",
    "1-7": "within-gap",
}
PATH_NAME = {"1-2": "Q1", "1-3": "Q2", "1-4": "G1", "1-5": "Q3", "1-7": "G2"}  # see paper/revision1/data_processed/path_registry.csv
PATH_ORDER = ["1-4", "1-7", "1-2", "1-3", "1-5"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_for_model(model: str) -> str:
    if model.startswith(FAMILY_PREFIX["FT-600K"]):
        return "FT-600K"
    if model.startswith(FAMILY_PREFIX["Scratch-5%"]):
        return "Scratch-5%"
    if model.startswith(FAMILY_PREFIX["Scratch"]):
        return "Scratch"
    raise ValueError(f"Unexpected model_id: {model}")


def fmt(values: list[float]) -> str:
    mean = statistics.mean(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} \\pm {spread:.2f}"


def main() -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    dft_reference: dict[str, float] = {}
    with INPUT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = row["pathway"]
            if path not in PATH_ORDER:
                continue
            # MPA0 is a single-seed comparator and is intentionally kept in
            # the raw audit CSV, but is not part of the three-family A5 table.
            if row["model_id"].startswith("ft600k_mpa0"):
                continue
            family = family_for_model(row["model_id"])
            key = (path, family)
            grouped.setdefault(key, []).append(float(row["absolute_error_ev"]))
            dft_reference[path] = float(row["dft_barrier_ev_from_neb_out"])

    missing = [
        (path, family)
        for path in PATH_ORDER
        for family in FAMILY_ORDER
        if (path, family) not in grouped
    ]
    if missing:
        raise RuntimeError(f"Missing path/family rows: {missing}")

    rows = []
    for path in PATH_ORDER:
        values = {
            family: fmt(grouped[(path, family)]) for family in FAMILY_ORDER
        }
        rows.append(
            f"{PATH_NAME[path]} & {PATH_CLASS[path]} & {dft_reference[path]:.2f} & "
            f"${values['FT-600K']}$ & ${values['Scratch-5%']}$ & ${values['Scratch']}$ \\\\"
        )

    table = r"""\begin{table}[htbp]
\centering
\small
\caption{\textbf{Path-resolved fixed-path errors for the grouped models.} Absolute errors (eV), mean $\pm$ sample standard deviation across three independent training initializations for FT-600K, Scratch-5\%, and full Scratch. The QE column gives the forward barrier of the QE image set of each path. Errors are fixed-path errors on identical images, reported per path and not pooled across classes. Path names are defined in Table~\ref{tab:path_registry}.}
\label{tab:scratch5_path_resolved_descriptive}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l p{0.30\linewidth} c c c c}
\toprule
Path & Class & QE barrier & FT-600K & Scratch-5\% & Scratch \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table, encoding="utf-8")

    summary = {}
    for path in PATH_ORDER:
        summary[path] = {
            "dft_candidate_barrier_eV": dft_reference[path],
            "pathway_class": PATH_CLASS[path],
            "n_seeds": {family: len(grouped[(path, family)]) for family in FAMILY_ORDER},
            "abs_error_eV_mean_sample_sd": {
                family: {
                    "mean": statistics.mean(grouped[(path, family)]),
                    "sample_sd": statistics.stdev(grouped[(path, family)]),
                }
                for family in FAMILY_ORDER
            },
        }
    manifest = {
        "status": "PATH_RESOLVED_ERRORS",
        "input": str(INPUT.relative_to(REPO)),
        "input_sha256": sha256(INPUT),
        "script": str(Path(__file__).relative_to(REPO)),
        "output": str(OUTPUT.relative_to(REPO)),
        "output_sha256": sha256(OUTPUT),
        "path_order": PATH_ORDER,
        "families": FAMILY_ORDER,
        "summary": summary,
        "boundary": (
            "Reader-facing path names map to audit IDs in path_registry.csv; errors are "
            "reported per path and are not pooled across classes."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
