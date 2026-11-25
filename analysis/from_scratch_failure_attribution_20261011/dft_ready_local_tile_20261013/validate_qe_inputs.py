from __future__ import annotations

import argparse
import re
from pathlib import Path


PSEUDOS = ("Cr_PBE_FR.SG15v1.2.UPF", "Sb_PBE_FR.SG15v1.2.UPF", "Te_PBE_FR.SG15v1.2.UPF")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--pseudo-dir", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    inputs = sorted((root / "qe_inputs").glob("*.scf.in"))
    if len(inputs) != args.expected:
        raise SystemExit(f"input count gate failed: {len(inputs)} != {args.expected}")
    for pseudo in PSEUDOS:
        if not (Path(args.pseudo_dir) / pseudo).is_file():
            raise SystemExit(f"missing pseudopotential: {pseudo}")
    for path in inputs:
        text = path.read_text()
        match = re.search(r"nat\s*=\s*(\d+)", text)
        if not match or int(match.group(1)) != 120:
            raise SystemExit(f"nat gate failed: {path}")
        if text.count("CELL_PARAMETERS angstrom") != 1 or text.count("ATOMIC_POSITIONS angstrom") != 1:
            raise SystemExit(f"cardinality gate failed: {path}")
        if any(pseudo not in text for pseudo in PSEUDOS):
            raise SystemExit(f"pseudopotential card gate failed: {path}")
    print(f"QE_INPUT_SMOKE_PASS inputs={len(inputs)} nat=120 pbc_cards=present")


if __name__ == "__main__":
    main()
