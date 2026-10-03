#!/usr/bin/env python3
"""Extract exact fixed-size extxyz frames and record source provenance."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[460000, 461800, 462000, 462200])
    ap.add_argument("--first-step", type=int, default=200200); ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--natoms", type=int, default=2050); args = ap.parse_args()
    source, out = Path(args.source), Path(args.out); out.mkdir(parents=True, exist_ok=False)
    lines_per_frame = args.natoms + 2
    record = {"source": str(source), "source_sha256": sha256(source), "natoms": args.natoms,
              "lines_per_frame": lines_per_frame, "first_saved_step": args.first_step,
              "stride_steps": args.stride, "snapshots": []}
    for step in args.steps:
        delta = step - args.first_step
        if delta < 0 or delta % args.stride: raise ValueError(f"step {step} is not on source grid")
        index = delta // args.stride; start = index * lines_per_frame
        path = out / f"snapshot_{step}.extxyz"
        with source.open() as f:
            for _ in range(start):
                if not f.readline(): raise EOFError(f"missing frame {index}")
            block = [f.readline() for _ in range(lines_per_frame)]
        if len(block) != lines_per_frame or not block[-1]: raise EOFError(f"incomplete frame {step}")
        if not re.search(rf"\bstep={step}\b", block[1]): raise ValueError(f"header step mismatch for {step}")
        path.write_text("".join(block))
        record["snapshots"].append({"step": step, "index": index, "path": str(path), "sha256": sha256(path), "header": block[1].strip()})
    (out / "snapshot_manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__": main()
