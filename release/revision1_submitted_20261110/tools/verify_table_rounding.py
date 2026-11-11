#!/usr/bin/env python3
"""Check that submitted (hand-rounded) LaTeX tables agree with regenerated values.

The SI tables written by ``items/si/_shared_generators/audit_current_figures.py``
were rounded by hand on Overleaf before submission (e.g. ``30.48\\pm3.48`` ->
``30\\pm3``, ``0.063`` -> ``6.3\\times10^{-2}``). This check parses every number
in both versions, row by row, and requires each submitted number to equal the
regenerated one rounded to the precision the submitted number displays.

Usage:
    python3 verify_table_rounding.py --submitted paper/revision1/tables \
        --regenerated items/si/_shared_generators/rerun_output/tables
Exit status 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TABLES = [
    "current_md_horizons.tex", "current_neb_errors.tex", "current_original_space.tex",
    "current_pointwise.tex", "current_shap_fidelity.tex", "current_tsne_sensitivity.tex",
]
NUMBER = re.compile(r"(-?\d+(?:\.\d+)?)(?:\s*\\times\s*10\^\{?(-?\d+)\}?)?")
ROUNDING_SLACK = 1e-9


def table_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or line.startswith("\\"):
            continue
        rows.append([cell.strip() for cell in line[:-2].split("&")])
    return rows[1:]  # drop the header row


def numbers_with_precision(cell: str) -> list[tuple[float, float]]:
    """Return (value, half-unit-of-last-digit) for every number in a cell."""
    found = []
    for mantissa, exponent in NUMBER.findall(cell):
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        scale = 10.0 ** int(exponent) if exponent else 1.0
        found.append((float(mantissa) * scale, 0.5 * 10.0 ** -decimals * scale))
    return found


def compare_table(submitted: Path, regenerated: Path) -> list[str]:
    errors = []
    sub_rows, gen_rows = table_rows(submitted), table_rows(regenerated)
    if len(sub_rows) != len(gen_rows):
        return [f"{submitted.name}: {len(sub_rows)} submitted rows vs {len(gen_rows)} regenerated"]
    for index, (sub_row, gen_row) in enumerate(zip(sub_rows, gen_rows), start=1):
        sub_numbers = [n for cell in sub_row[1:] for n in numbers_with_precision(cell)]
        gen_numbers = [n[0] for cell in gen_row[1:] for n in numbers_with_precision(cell)]
        if len(sub_numbers) != len(gen_numbers):
            errors.append(f"{submitted.name} row {index}: {sub_row} vs {gen_row} (number count)")
            continue
        for (value, half_unit), regenerated_value in zip(sub_numbers, gen_numbers):
            if abs(value - regenerated_value) > half_unit + ROUNDING_SLACK:
                errors.append(f"{submitted.name} row {index}: submitted {value} vs regenerated {regenerated_value}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submitted", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for name in TABLES:
        submitted, regenerated = args.submitted / name, args.regenerated / name
        missing = [str(p) for p in (submitted, regenerated) if not p.is_file()]
        errors = [f"missing {m}" for m in missing] or compare_table(submitted, regenerated)
        print(f"{'OK  ' if not errors else 'FAIL'} {name}")
        failures += errors
    for error in failures:
        print("  " + error)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
