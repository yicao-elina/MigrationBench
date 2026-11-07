#!/usr/bin/env python3
"""Read-only inventory of real continuation files and exact step coverage."""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
from pathlib import Path

TARGETS = {460000, 461800, 462000, 462200}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def scan(path, include_hash=True):
    frames = 0; first_step = None; last_step = None; found = {}; natoms_seen = set(); props = None; lattice = None; pbc = None
    with path.open("r", errors="strict") as f:
        while True:
            first = f.readline()
            if not first: break
            if not first.strip(): continue
            nat = int(first.strip()); header = f.readline().rstrip("\n"); frames += 1; natoms_seen.add(nat)
            m = re.search(r"(?:^| )step=([-+]?\d+)(?: |$)", header); step = int(m.group(1)) if m else None
            if step is not None:
                first_step = step if first_step is None else min(first_step, step); last_step = step if last_step is None else max(last_step, step)
                if step in TARGETS: found[str(step)] = {"frame_index": frames - 1, "natoms": nat, "header": header}
            if props is None:
                pm = re.search(r"Properties=([^ ]+)", header); props = pm.group(1) if pm else None
                lm = re.search(r'Lattice="([^"]+)"', header); lattice = lm.group(1) if lm else None
                bm = re.search(r'pbc="([^"]+)"', header); pbc = bm.group(1) if bm else None
            for _ in range(nat): f.readline()
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "mtime_epoch": path.stat().st_mtime, "sha256": sha256(path) if include_hash else None, "frames": frames, "first_step": first_step, "last_step": last_step, "target_steps_found": found, "natoms_seen": sorted(natoms_seen), "properties_first_frame": props, "lattice_first_frame": lattice, "pbc_first_frame": pbc, "velocity_field_present_by_schema": bool(props and "momenta" in props)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--no-hash", action="store_true"); a = ap.parse_args()
    paths = sorted(a.root.rglob("*.extxyz")) + sorted(a.root.rglob("*.xyz"))
    paths = [p for p in paths if p.is_file() and "latest_restart" not in p.name]
    result = {"schema_version": "phase1-trajectory-inventory-1.0", "root": str(a.root.resolve()), "target_steps": sorted(TARGETS), "files": [scan(p, include_hash=not a.no_hash) for p in paths]}
    a.output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"files": len(paths), "output": str(a.output)}, indent=2))

if __name__ == "__main__": main()
