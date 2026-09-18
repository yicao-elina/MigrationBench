#!/usr/bin/env python3
"""Create a QE engine template from a pw.x input by removing atomic positions."""

from __future__ import annotations

import argparse
from pathlib import Path


STOP_CARDS = {
    "ATOMIC_POSITIONS",
    "K_POINTS",
    "CELL_PARAMETERS",
    "ATOMIC_SPECIES",
    "OCCUPATIONS",
    "CONSTRAINTS",
    "ATOMIC_VELOCITIES",
}
SKIP_NAMELISTS = {"&IONS", "&CELL"}


def card_name(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("!") or stripped.startswith("#"):
        return ""
    return stripped.split()[0].upper()


def strip_positions(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped_upper = lines[i].strip().upper()
        if stripped_upper in SKIP_NAMELISTS:
            i += 1
            while i < len(lines):
                if lines[i].strip() == "/":
                    i += 1
                    break
                i += 1
            continue
        name = card_name(lines[i])
        if name == "ATOMIC_POSITIONS":
            i += 1
            while i < len(lines):
                next_name = card_name(lines[i])
                if next_name in STOP_CARDS or lines[i].lstrip().startswith("&"):
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strip_positions(args.input.read_text()))
    print(args.output)


if __name__ == "__main__":
    main()
