#!/usr/bin/env python3
"""Prepare a QE neb.x input for a restartable walltime-bounded run."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PATH_RESTART_RE = re.compile(r"(restart_mode\s*=\s*)'[^']*'", re.IGNORECASE)
CONTROL_BLOCK_RE = re.compile(r"(&CONTROL\b.*?^\s*/)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
MAX_SECONDS_RE = re.compile(r"(^\s*max_seconds\s*=\s*)[-+0-9.EedD]+(\s*,?\s*$)", re.IGNORECASE | re.MULTILINE)


def sha256_file(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_restart_mode(text: str, restart_mode: str) -> str:
    if PATH_RESTART_RE.search(text):
        return PATH_RESTART_RE.sub(rf"\1'{restart_mode}'", text, count=1)
    return text.replace("&PATH", f"&PATH\n  restart_mode = '{restart_mode}',", 1)


def set_max_seconds(text: str, max_seconds: int) -> str:
    def update_control(match):
        block = match.group(1)
        if MAX_SECONDS_RE.search(block):
            return MAX_SECONDS_RE.sub(rf"\g<1>{max_seconds}\2", block, count=1)
        return block.rsplit("/", 1)[0].rstrip() + f"\n  max_seconds = {max_seconds}\n/"

    updated, n = CONTROL_BLOCK_RE.subn(update_control, text, count=1)
    if n == 0:
        raise ValueError("Could not find a QE &CONTROL block in the NEB input.")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--restart-mode", choices=["from_scratch", "restart"], required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--safety-seconds", type=int, required=True)
    parser.add_argument("--parent-run-dir")
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    text = args.input.read_text()
    text = set_restart_mode(text, args.restart_mode)
    text = set_max_seconds(text, args.max_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chain_id": args.chain_id,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "restart_mode": args.restart_mode,
        "max_seconds": args.max_seconds,
        "slurm_walltime": args.walltime,
        "safety_seconds": args.safety_seconds,
        "parent_run_dir": args.parent_run_dir,
        "notes": args.notes,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "restart_mode": args.restart_mode, "max_seconds": args.max_seconds}, indent=2))


if __name__ == "__main__":
    main()
