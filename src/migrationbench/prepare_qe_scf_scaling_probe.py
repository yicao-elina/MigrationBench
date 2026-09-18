#!/usr/bin/env python3
"""Prepare a one-SCF QE CPU-scaling probe without changing the Hamiltonian."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_qe_calculator_identity import identity as calculator_identity
from prepare_qe_scf_warmup import set_namelist_value, walltime_seconds


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="mb_spscale4_s42")
    parser.add_argument("--walltime", default="03:00:00")
    parser.add_argument("--max-seconds", type=int, default=9900)
    parser.add_argument("--ntasks", type=int, default=4)
    parser.add_argument("--memory", default="180G")
    parser.add_argument("--kpoint-pools", type=int, default=4)
    args = parser.parse_args()
    if args.max_seconds >= walltime_seconds(args.walltime):
        raise ValueError("max_seconds must be below Slurm walltime")
    source = args.source_pw.resolve()
    text = source.read_text()
    text = set_namelist_value(text, "CONTROL", "calculation", "'scf'")
    text = set_namelist_value(text, "CONTROL", "restart_mode", "'from_scratch'")
    text = set_namelist_value(text, "CONTROL", "prefix", repr(args.prefix))
    text = set_namelist_value(text, "CONTROL", "outdir", "'./out/'")
    text = set_namelist_value(text, "CONTROL", "max_seconds", str(args.max_seconds))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "scf.in"
    output.write_text(text)
    source_identity = calculator_identity(source, "source")
    probe_identity = calculator_identity(output, "probe")
    if source_identity["calculator_identity"] != probe_identity["calculator_identity"]:
        raise ValueError("Scaling probe changed QE calculator identity")
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "qe_scf_cpu_scaling_diagnostic_not_label",
        "path_id": args.path_id,
        "image_index_qe": 1,
        "seed": args.seed,
        "prefix": args.prefix,
        "source_pw_input": str(source),
        "source_pw_sha256": sha256_file(source),
        "generated_input": str(output),
        "generated_input_sha256": sha256_file(output),
        "calculator_identity": source_identity["calculator_identity"],
        "calculator_identity_unchanged": True,
        "allowed_control_changes": [
            "calculation",
            "restart_mode",
            "prefix",
            "outdir",
            "max_seconds",
        ],
        "label_eligibility": False,
        "resources": {
            "walltime": args.walltime,
            "max_seconds": args.max_seconds,
            "ntasks": args.ntasks,
            "memory": args.memory,
            "kpoint_pools": args.kpoint_pools,
        },
    }
    manifest_path = out_dir / "scf_warmup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "input": str(output),
        "manifest": str(manifest_path),
        "calculator_identity": source_identity["calculator_identity"],
    }, indent=2))


if __name__ == "__main__":
    main()
