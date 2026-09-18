#!/usr/bin/env python3
"""Prepare a production QE NEB input seeded by accepted image-SCF charge densities."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from prepare_qe_scf_warmup import set_namelist_value


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walltime_seconds(value):
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    hours, minutes, seconds = [int(field) for field in value.split(":")]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def select_warmup_jobs(status, path_id, num_images):
    grouped = {}
    for row in status["jobs"]:
        if row["path_id"] == path_id:
            grouped.setdefault(int(row["image_index_qe"]), []).append(row)
    expected = set(range(1, num_images + 1))
    if set(grouped) != expected:
        raise ValueError(f"Warmup image set mismatch: expected {sorted(expected)}, found {sorted(grouped)}")

    def key(row):
        job_id = str(row.get("job_id", ""))
        numeric_job_id = int(job_id) if job_id.isdigit() else -1
        return int(row.get("attempt", 1)), numeric_job_id

    return {index: max(rows, key=key) for index, rows in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-neb-input", type=Path, required=True)
    parser.add_argument("--warmup-status", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walltime", default="48:00:00")
    parser.add_argument("--max-seconds", type=int, default=171000)
    parser.add_argument("--ntasks", type=int, default=4)
    parser.add_argument("--memory", default="180G")
    parser.add_argument("--kpoint-pools", type=int, default=4)
    parser.add_argument("--production-degauss-Ry", type=float, default=0.005)
    args = parser.parse_args()

    if args.max_seconds >= walltime_seconds(args.walltime):
        raise ValueError("max_seconds must be below the Slurm walltime")
    status = json.loads(args.warmup_status.read_text())
    jobs = select_warmup_jobs(status, args.path_id, args.num_images)
    rejected = {
        index: row.get("classification")
        for index, row in jobs.items()
        if row.get("classification") != "accepted_scf_warmstart"
    }
    if rejected:
        raise ValueError(f"Warmup gate failed: {rejected}")

    base = args.base_neb_input.resolve()
    text = base.read_text()
    updates = {
        "PATH": {"restart_mode": "'from_scratch'", "num_of_images": str(args.num_images)},
        "CONTROL": {"max_seconds": str(args.max_seconds), "disk_io": "'low'"},
        "SYSTEM": {"degauss": f"{args.production_degauss_Ry:.8f}"},
        "ELECTRONS": {
            "conv_thr": "1.0d-6",
            "electron_maxstep": "800",
            "mixing_beta": "0.20000000",
            "mixing_mode": "'local-TF'",
            "mixing_ndim": "8",
            "diagonalization": "'david'",
            "startingpot": "'file'",
            "startingwfc": "'atomic+random'",
        },
    }
    for section, values in updates.items():
        for key, value in values.items():
            text = set_namelist_value(text, section, key, value)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    neb_input = out_dir / "neb.in"
    neb_input.write_text(text)
    sources = [
        {
            "image_index_qe": index,
            "warmup_job_id": jobs[index]["job_id"],
            "warmup_attempt": jobs[index].get("attempt", 1),
            "warmup_classification": jobs[index]["classification"],
            "warmup_run_dir": jobs[index]["remote_run_dir"],
            "source_save_dir": jobs[index]["remote_run_dir"].rstrip("/") + "/out/sb2te3.save",
            "copy_policy": "charge_density_xml_pseudopotentials_exclude_wfc",
        }
        for index in sorted(jobs)
    ]
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "production_qe_neb_seeded_by_independent_image_scf_charge_densities",
        "path_id": args.path_id,
        "seed": args.seed,
        "base_neb_input": str(base),
        "base_neb_input_sha256": sha256_file(base),
        "generated_neb_input": str(neb_input),
        "generated_neb_input_sha256": sha256_file(neb_input),
        "warmup_status": str(args.warmup_status.resolve()),
        "warmup_status_sha256": sha256_file(args.warmup_status),
        "warmup_sources": sources,
        "resources": {
            "walltime": args.walltime,
            "max_seconds": args.max_seconds,
            "ntasks": args.ntasks,
            "memory": args.memory,
            "kpoint_pools": args.kpoint_pools,
        },
        "production_settings": {
            "degauss_Ry": args.production_degauss_Ry,
            "startingpot": "file",
            "startingwfc": "atomic+random",
            "warmup_energies_are_labels": False,
            "production_scf_reconvergence_required": True,
        },
    }
    manifest_path = out_dir / "warmstarted_neb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"neb_input": str(neb_input), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
