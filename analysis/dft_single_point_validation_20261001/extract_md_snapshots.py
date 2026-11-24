"""Extract five high-energy frames without loading a large MD trajectory."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    top = []
    frame = 0
    with args.source.open("r", errors="replace") as f:
        while True:
            first = f.readline()
            if not first:
                break
            first = first.strip()
            if not first:
                continue
            nat = int(first)
            header = f.readline().rstrip("\n")
            match = re.search(r"(?:^| )energy=([^ ]+)", header)
            energy = float(match.group(1)) if match else float("nan")
            lines = [f.readline() for _ in range(nat)]
            if energy == energy:
                item = (energy, frame, header, lines)
                if len(top) < 5:
                    heapq.heappush(top, item)
                elif energy > top[0][0]:
                    heapq.heapreplace(top, item)
            frame += 1
    args.output.mkdir(parents=True, exist_ok=True)
    selected = sorted(top, reverse=True)
    print(json.dumps({"source": str(args.source), "frames": frame, "selected": [(x[0], x[1]) for x in selected]}))
    for rank, (energy, index, header, lines) in enumerate(selected, 1):
        lattice = re.search(r'Lattice="([^"]+)"', header).group(1)
        positions = []
        for line in lines:
            parts = line.split()
            positions.append(f"{parts[0]}  {parts[1]}  {parts[2]}  {parts[3]}\n")
        path = args.output / f"md_multiT_high_energy_rank{rank:02d}_frame{index:06d}.extxyz"
        out_header = (
            f'{len(positions)}\nLattice="{lattice}" '
            f'Properties=species:S:1:pos:R:3 energy={energy:.16g} frame={index} '
            f'source_file="{args.source}" pbc="T T T"\n'
        )
        path.write_text(out_header + "".join(positions))
        print(json.dumps({"rank": rank, "frame": index, "energy_eV": energy, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "natoms": len(positions)}))


if __name__ == "__main__":
    main()
