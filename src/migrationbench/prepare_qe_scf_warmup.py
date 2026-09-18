#!/usr/bin/env python3
"""Prepare a provenance-complete standalone QE SCF warmup for one NEB image."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walltime_seconds(value):
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    hours, minutes, seconds = [int(field) for field in value.split(":")]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def set_namelist_value(text, section, key, value):
    match = re.search(rf"(?ims)^\s*&{re.escape(section)}\s*$.*?^\s*/\s*$", text)
    if not match:
        raise ValueError(f"Missing &{section} namelist")
    block = match.group(0)
    assignment = re.compile(rf"(?im)^\s*{re.escape(key)}\s*=.*$")
    replacement = f"  {key} = {value}"
    if assignment.search(block):
        block = assignment.sub(replacement, block)
    else:
        block = re.sub(r"(?m)^\s*/\s*$", replacement + "\n/", block, count=1)
    return text[: match.start()] + block + text[match.end() :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--image-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--ntasks", type=int, default=4)
    parser.add_argument("--memory", default="180G")
    parser.add_argument("--kpoint-pools", type=int, default=4)
    parser.add_argument("--degauss-Ry", type=float, default=0.01)
    parser.add_argument("--mixing-beta", type=float, default=0.2)
    parser.add_argument("--mixing-mode", default="local-TF")
    parser.add_argument("--mixing-ndim", type=int, default=8)
    parser.add_argument("--electron-maxstep", type=int, default=800)
    parser.add_argument("--conv-thr", default="1.0d-6")
    args = parser.parse_args()

    if args.max_seconds >= walltime_seconds(args.walltime):
        raise ValueError("max_seconds must be below the Slurm walltime")

    source = args.source_pw.resolve()
    text = source.read_text()
    updates = {
        "CONTROL": {
            "calculation": "'scf'",
            "restart_mode": "'from_scratch'",
            "outdir": "'./out/'",
            "max_seconds": str(args.max_seconds),
            "disk_io": "'low'",
        },
        "SYSTEM": {"degauss": f"{args.degauss_Ry:.8f}"},
        "ELECTRONS": {
            "conv_thr": args.conv_thr,
            "electron_maxstep": str(args.electron_maxstep),
            "mixing_beta": f"{args.mixing_beta:.8f}",
            "mixing_mode": repr(args.mixing_mode),
            "mixing_ndim": str(args.mixing_ndim),
            "diagonalization": "'david'",
            "startingpot": "'atomic'",
            "startingwfc": "'atomic+random'",
        },
    }
    for section, values in updates.items():
        for key, value in values.items():
            text = set_namelist_value(text, section, key, value)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "scf.in"
    output.write_text(text)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "standalone_image_scf_warmup_before_qe_neb",
        "path_id": args.path_id,
        "image_index_qe": args.image_index,
        "seed": args.seed,
        "source_pw_input": str(source),
        "source_pw_sha256": sha256_file(source),
        "generated_input": str(output),
        "generated_input_sha256": sha256_file(output),
        "electronic_settings": {
            "degauss_Ry": args.degauss_Ry,
            "mixing_beta": args.mixing_beta,
            "mixing_mode": args.mixing_mode,
            "mixing_ndim": args.mixing_ndim,
            "electron_maxstep": args.electron_maxstep,
            "conv_thr": args.conv_thr,
            "startingpot": "atomic",
            "startingwfc": "atomic+random",
        },
        "resources": {
            "walltime": args.walltime,
            "max_seconds": args.max_seconds,
            "ntasks": args.ntasks,
            "memory": args.memory,
            "kpoint_pools": args.kpoint_pools,
        },
        "acceptance": {
            "qe_job_done": True,
            "scf_converged": True,
            "finite_total_energy": True,
            "restart_artifacts_required": ["charge-density.dat", "data-file-schema.xml"],
            "restart_artifacts_recorded_optional": ["wfc*.dat"],
        },
    }
    manifest_path = out_dir / "scf_warmup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"input": str(output), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
