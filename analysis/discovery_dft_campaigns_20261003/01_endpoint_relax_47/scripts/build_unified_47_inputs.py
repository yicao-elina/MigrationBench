#!/usr/bin/env python3
"""Build a non-magnetic, Discovery-path unified input set for all 47 endpoints."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
READY10 = BASE / "discovery_ready"
STAGED37 = BASE / "rockfish_authoritative_20261004"
OUT = BASE / "discovery_ready_unified"
PSEUDO = "/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/"
IDS10 = ["unique_041", "unique_042", "unique_044", "unique_051", "unique_052", "unique_056", "unique_057", "unique_065", "unique_066", "unique_067"]
IDS37 = [p.name for p in sorted(STAGED37.iterdir()) if p.is_dir()]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def normalize(text: str, endpoint: str) -> str:
    text = text.replace("/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/", PSEUDO)
    text = text.replace("/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/", PSEUDO)
    text = re.sub(r"^\s*nspin\s*=.*\n", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*starting_magnetization\([^\n]*\n", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*tot_magnetization\s*=.*\n", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*prefix\s*=.*$", f"  prefix = 'cr_int_{endpoint}'", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*max_seconds\s*=.*$", "  max_seconds = 171000", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*electron_maxstep\s*=.*$", "  electron_maxstep = 400", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*mixing_beta\s*=.*$", "  mixing_beta = 0.4", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*K_POINTS\s+gamma\s*$", "K_POINTS AUTOMATIC\n2 2 2 1 1 1", text, flags=re.MULTILINE | re.IGNORECASE)
    if "max_seconds" not in text:
        text = text.replace("/\n\n&SYSTEM", "  max_seconds = 171000\n/\n\n&SYSTEM", 1)
    return text

def main() -> None:
    if len(IDS37) != 37:
        raise SystemExit(f"expected 37 staged inputs, found {len(IDS37)}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for endpoint in IDS10 + IDS37:
        if endpoint in IDS10:
            source = READY10 / endpoint / f"relax_{endpoint}.in"
            source_class = "existing_discovery_ready"
        else:
            source = STAGED37 / endpoint / "relax.in"
            source_class = "rockfish_authoritative_staged"
        target_dir = OUT / endpoint
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"relax_{endpoint}.in"
        target.write_text(normalize(source.read_text(encoding="utf-8"), endpoint), encoding="utf-8")
        rows.append((endpoint, source_class, str(source), str(target), sha256(source), sha256(target)))
    manifest = OUT / "UNIFIED_47_MANIFEST.tsv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("endpoint\tsource_class\tsource_input\tunified_input\tsource_sha256\tunified_sha256\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"wrote {len(rows)} unified inputs to {OUT}")

if __name__ == "__main__":
    main()
