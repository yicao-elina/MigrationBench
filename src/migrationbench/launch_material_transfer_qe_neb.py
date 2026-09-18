#!/usr/bin/env python3
"""Prepare and optionally submit diagnostic QE NEB for material-transfer paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ase.io import read


PSEUDOS = {
    "Cr": ("51.99610", "cr_pbe_v1.5.uspp.F.UPF"),
    "Mn": ("54.93804", "mn_pbe_v1.5.uspp.F.UPF"),
    "Sb": ("121.76000", "sb_pbe_v1.4.uspp.F.UPF"),
    "Bi": ("208.98040", "bi_pbe_v1.uspp.F.UPF"),
    "Te": ("127.60000", "te_pbe_v1.uspp.F.UPF"),
}

READY_CLASSIFICATIONS = {
    "ready_for_mechanism_clustering",
    "ready_for_dft_preconditioner_review",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(argv, attempts=4):
    for attempt in range(attempts):
        result = subprocess.run(argv, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip()
        transient = result.returncode == 255 or "Connection reset" in result.stderr
        if not transient or attempt == attempts - 1:
            raise RuntimeError(f"Command failed: {' '.join(argv)}\n{result.stderr}")
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def selected_rows(status_paths, branches):
    rows = []
    branch_filter = set(branches or [])
    for status_path in status_paths:
        payload = json.loads(status_path.read_text())
        for row in payload.get("jobs", []):
            if row.get("classification") not in READY_CLASSIFICATIONS:
                continue
            if branch_filter and row.get("branch_id") not in branch_filter:
                continue
            manifest = row.get("manifest")
            if not manifest or not Path(manifest).is_file():
                continue
            rows.append(row)
    by_branch = {}
    for row in rows:
        key = row["branch_id"]
        current = by_branch.get(key)
        current_tether = (current or {}).get("migrant_tether_k_eV_A2", 0) or (current or {}).get("host_tether_k_eV_A2", 0)
        new_tether = row.get("migrant_tether_k_eV_A2", 0) or row.get("host_tether_k_eV_A2", 0)
        if current is None or (current_tether and not new_tether):
            by_branch[key] = row
    return list(by_branch.values())


def local_artifact(manifest_path: Path, recorded_path: str) -> Path:
    recorded = Path(recorded_path)
    if recorded.is_file():
        return recorded.resolve()
    mirrored = manifest_path.parent / recorded.name
    if mirrored.is_file():
        return mirrored.resolve()
    raise FileNotFoundError(f"Cannot locate artifact {recorded_path} from {manifest_path}")


def species_order(symbols, migrant):
    ordered = []
    if migrant in symbols:
        ordered.append(migrant)
    for symbol in ("Cr", "Mn", "Sb", "Bi", "Te"):
        if symbol in symbols and symbol not in ordered:
            ordered.append(symbol)
    for symbol in symbols:
        if symbol not in ordered:
            ordered.append(symbol)
    return ordered


def infer_migrant(symbols, requested):
    if requested in symbols and requested not in {"Sb", "Bi", "Te"}:
        return requested
    candidates = [
        symbol for symbol, count in sorted({symbol: symbols.count(symbol) for symbol in set(symbols)}.items())
        if count == 1 and symbol not in {"Sb", "Bi", "Te"}
    ]
    if len(candidates) == 1:
        return candidates[0]
    return requested


def engine_template(symbols, migrant, prefix, pseudo_dir, max_seconds):
    species = species_order(symbols, migrant)
    missing = [symbol for symbol in species if symbol not in PSEUDOS]
    if missing:
        raise ValueError(f"No pseudopotential mapping for: {', '.join(missing)}")
    species_lines = []
    for symbol in species:
        mass, pseudo = PSEUDOS[symbol]
        species_lines.append(f"  {symbol:2s} {mass:>10s}  {pseudo}")
    system_lines = [
        "&SYSTEM",
        "  vdw_corr = 'DFT-D3',",
        "  degauss = 1.4699723600d-02,",
        "  ecutrho = 400,",
        "  ecutwfc = 50,",
        "  ibrav = 0,",
        f"  nat = {len(symbols)},",
        f"  ntyp = {len(species)},",
        "  occupations = 'smearing',",
        "  smearing = 'cold',",
        "  nspin = 2,",
    ]
    if species and species[0] in {"Cr", "Mn"}:
        system_lines.append("  starting_magnetization(1) = 0.5,")
    system_lines.append("/")
    return "\n".join([
        "&CONTROL",
        "  etot_conv_thr = 1.0000000000d-04,",
        "  forc_conv_thr = 1.0000000000d-03,",
        "  outdir = './out/',",
        f"  prefix = '{prefix}',",
        f"  pseudo_dir = '{pseudo_dir}',",
        "  verbosity = 'high',",
        f"  max_seconds = {max_seconds},",
        "/",
        *system_lines,
        "&ELECTRONS",
        "  conv_thr = 1.0000000000d-06,",
        "  electron_maxstep = 400,",
        "  mixing_beta = 0.4,",
        "/",
        "ATOMIC_SPECIES",
        *species_lines,
        "K_POINTS AUTOMATIC",
        "4 4 1 0 0 0",
        "",
    ])


def build_qe_input(row, args, out_dir):
    mlff_manifest_path = Path(row["manifest"]).resolve()
    mlff_manifest = json.loads(mlff_manifest_path.read_text())
    images_path = local_artifact(mlff_manifest_path, mlff_manifest["outputs"]["images"])
    atoms = read(images_path, index=0)
    symbols = atoms.get_chemical_symbols()
    migrant = infer_migrant(
        symbols,
        row.get("migrant_element") or (row.get("candidate") or {}).get("migrant_element") or "Cr",
    )
    token = hashlib.sha256(row["branch_id"].encode()).hexdigest()[:8]
    prefix = f"mt{token}"
    branch_dir = out_dir / row["branch_id"]
    branch_dir.mkdir(parents=True, exist_ok=True)
    engine_path = branch_dir / "engine_template.in"
    engine_path.write_text(
        engine_template(symbols, migrant, prefix, args.pseudo_dir, args.max_seconds)
    )
    neb_path = branch_dir / "neb.in"
    manifest_path = branch_dir / "neb_manifest.json"
    command([
        args.python,
        "scripts/migrationbench/qe_neb_from_images.py",
        "--images", str(images_path),
        "--engine-template", str(engine_path),
        "--output", str(neb_path),
        "--manifest", str(manifest_path),
        "--path-id", row.get("path_id", "1-7"),
        "--path-family", "material_transfer_ingap",
        "--measurement-purpose", "first_stage_material_transfer_dft_neb_validation",
        "--run-role", "diagnostic",
        "--initialization-role", "mlff_preconditioned",
        "--source-mlff-manifest", str(mlff_manifest_path),
        "--nstep-path", str(args.nstep_path),
        "--max-seconds", str(args.max_seconds),
        "--path-thr", str(args.path_thr),
        "--ds", str(args.ds),
        "--k-min", str(args.k_min),
        "--k-max", str(args.k_max),
    ])
    return {
        "branch_id": row["branch_id"],
        "path_id": row.get("path_id", "1-7"),
        "system_id": row.get("system_id") or row["branch_id"],
        "migrant_element": migrant,
        "source_mlff_job_id": row.get("job_id"),
        "source_mlff_classification": row.get("classification"),
        "source_mlff_manifest": str(mlff_manifest_path),
        "source_mlff_manifest_sha256": sha256_file(mlff_manifest_path),
        "images": str(images_path),
        "images_sha256": sha256_file(images_path),
        "engine_template": str(engine_path.resolve()),
        "engine_template_sha256": sha256_file(engine_path),
        "neb_input": str(neb_path.resolve()),
        "neb_input_sha256": sha256_file(neb_path),
        "neb_manifest": str(manifest_path.resolve()),
        "neb_manifest_sha256": sha256_file(manifest_path),
        "qe_prefix": prefix,
    }


def submit_job(prepared, args):
    token = hashlib.sha256(prepared["branch_id"].encode()).hexdigest()[:6]
    attempt = f"_{args.attempt_label}" if args.attempt_label else ""
    job_name = f"mb_mtq{token}{attempt}_s{args.seed}"
    remote_dir = f"{args.remote_input_root.rstrip('/')}/{job_name}"
    remote_neb = f"{remote_dir}/neb.in"
    remote_manifest = f"{remote_dir}/neb_manifest.json"
    command(["ssh", args.ssh_host, "mkdir", "-p", remote_dir])
    command(["scp", prepared["neb_input"], f"{args.ssh_host}:{remote_neb}"])
    command(["scp", prepared["neb_manifest"], f"{args.ssh_host}:{remote_manifest}"])
    exports = "ALL," + ",".join([
        f"MIGRATIONBENCH_NEB_INPUT={remote_neb}",
        f"MIGRATIONBENCH_QE_EXTRA_ARGS={args.qe_extra_args}",
        f"MIGRATIONBENCH_RUN_ROOT={args.remote_run_root}",
        f"MIGRATIONBENCH_CODE_ROOT={args.remote_code_root}",
    ])
    sbatch = " ".join(shlex.quote(value) for value in (
        "sbatch",
        f"--job-name={job_name}",
        f"--time={args.walltime}",
        f"--ntasks-per-node={args.ntasks}",
        f"--mem={args.memory}",
        f"--export={exports}",
        f"{args.remote_code_root}/scripts/migrationbench/submit_slurm_qe_neb.sh",
    ))
    body = f"""set -e
receipt={shlex.quote(remote_dir + '/submission.receipt')}
if test -s "$receipt"; then cat "$receipt"; exit 0; fi
job_id=$({sbatch} | awk '{{print $NF}}')
tmp="$receipt.tmp.$$"; printf '%s|submitted_new_job\\n' "$job_id" > "$tmp"; mv "$tmp" "$receipt"; cat "$receipt"
"""
    receipt = command([
        "ssh",
        args.ssh_host,
        f"flock -x {shlex.quote(remote_dir + '/submission.lock')} bash -lc {shlex.quote(body)}",
    ])
    job_id, submission_status = receipt.splitlines()[-1].split("|", 1)
    return {
        "job_id": job_id,
        "job_name": job_name,
        "submission_status": submission_status,
        "remote_input_dir": remote_dir,
        "remote_neb_input": remote_neb,
        "remote_neb_manifest": remote_manifest,
        "remote_run_dir": f"{args.remote_run_root.rstrip('/')}/{job_name}_{job_id}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-status", type=Path, action="append", required=True)
    parser.add_argument("--branch", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempt-label", default="")
    parser.add_argument("--pseudo-dir", default="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/vdW-Gap-Engineering/data/pseudo")
    parser.add_argument("--nstep-path", type=int, default=200)
    parser.add_argument("--max-seconds", type=int, default=171000)
    parser.add_argument("--path-thr", type=float, default=0.03)
    parser.add_argument("--ds", type=float, default=0.2)
    parser.add_argument("--k-min", type=float, default=0.1)
    parser.add_argument("--k-max", type=float, default=0.3)
    parser.add_argument("--walltime", default="48:00:00")
    parser.add_argument("--memory", default="180G")
    parser.add_argument("--ntasks", type=int, default=8)
    parser.add_argument("--qe-extra-args", default="-nk 4")
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-input-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_inputs/material_transfer_stage1_qe")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    args = parser.parse_args()

    rows = selected_rows(args.mace_status, args.branch)
    if not rows:
        raise RuntimeError("No MACE rows passed the DFT handoff classification filter")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in rows:
        prepared = build_qe_input(row, args, args.out_dir)
        job = dict(prepared)
        if args.submit:
            job.update(submit_job(prepared, args))
        jobs.append(job)
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "first_stage_material_transfer_DFT_NEB_from_MACE_warm_start",
        "mace_status_files": [str(path.resolve()) for path in args.mace_status],
        "submit_requested": args.submit,
        "resources": {
            "walltime": args.walltime,
            "max_seconds": args.max_seconds,
            "ntasks": args.ntasks,
            "memory": args.memory,
            "qe_extra_args": args.qe_extra_args,
            "kpoints": "automatic 4 4 1 0 0 0",
        },
        "calculator_identity_boundary": {
            "status": "diagnostic_not_final_reference",
            "spin": "collinear nspin=2 with dopant starting magnetization",
            "soc": "not enabled",
            "pseudopotentials": "USPP set from vdW-Gap-Engineering pseudo directory",
            "note": "Do not pool with SG15 historical references without a formal calculator identity gate.",
        },
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"jobs": len(jobs), "submitted": args.submit, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
