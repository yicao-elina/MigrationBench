#!/usr/bin/env python3
"""Prepare fixed-geometry QE SCF pairs for calculator-protocol sensitivity."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_neb_path_topology import read_qe_image  # noqa: E402
from audit_qe_calculator_identity import identity as calculator_identity  # noqa: E402
from parse_qe_neb_path_history import parse_path_file  # noqa: E402
from prepare_qe_scf_warmup import set_namelist_value, walltime_seconds  # noqa: E402


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_namelist_key(text, key):
    return re.sub(
        rf"(?im)^\s*{re.escape(key)}(?:\([^)]*\))?\s*=.*(?:\n|$)", "", text
    )


def replace_positions(text, symbols, positions):
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip().upper().startswith("ATOMIC_POSITIONS"))
    lines[start] = "ATOMIC_POSITIONS angstrom"
    for offset, (symbol, position) in enumerate(zip(symbols, positions), start=1):
        old = lines[start + offset].split()
        if not old or old[0] != symbol:
            raise ValueError(f"Atom order mismatch at atom {offset}: {old[:1]} versus {symbol}")
        lines[start + offset] = "{}  {:.12f}  {:.12f}  {:.12f}".format(symbol, *position)
    return "\n".join(lines) + "\n"


def replace_kpoints(text, specification):
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip().upper().startswith("K_POINTS"))
    end = start + 1
    if "AUTOMATIC" in lines[start].upper():
        end += 1
    if specification["mode"] == "gamma":
        replacement = ["K_POINTS gamma"]
    elif specification["mode"] == "automatic":
        replacement = ["K_POINTS AUTOMATIC", " ".join(str(value) for value in specification["grid_shift"])]
    else:
        raise ValueError(f"Unsupported k-point mode: {specification}")
    return "\n".join(lines[:start] + replacement + lines[end:]) + "\n"


def apply_spin_model(text, model, starting_magnetization):
    for key in ("nspin", "noncolin", "lspinorb", "starting_magnetization"):
        text = remove_namelist_key(text, key)
    if model == "nonspin":
        return text
    if model == "collinear":
        text = set_namelist_value(text, "SYSTEM", "nspin", "2")
        return set_namelist_value(text, "SYSTEM", "starting_magnetization(1)", str(starting_magnetization))
    if model == "soc":
        text = set_namelist_value(text, "SYSTEM", "noncolin", ".true.")
        text = set_namelist_value(text, "SYSTEM", "lspinorb", ".true.")
        return set_namelist_value(text, "SYSTEM", "starting_magnetization(1)", str(starting_magnetization))
    raise ValueError(f"Unknown spin model: {model}")


def build_input(template, symbols, positions, profile, prefix, max_seconds):
    text = replace_positions(template, symbols, positions)
    for key, value in (
        ("calculation", "'scf'"),
        ("restart_mode", "'from_scratch'"),
        ("prefix", repr(prefix)),
        ("outdir", "'./out/'"),
        ("max_seconds", str(max_seconds)),
        ("disk_io", "'low'"),
    ):
        text = set_namelist_value(text, "CONTROL", key, value)
    for key, value in (
        ("ecutwfc", str(profile["ecutwfc_Ry"])),
        ("ecutrho", str(profile["ecutrho_Ry"])),
        ("degauss", str(profile["degauss_Ry"])),
    ):
        text = set_namelist_value(text, "SYSTEM", key, value)
    text = apply_spin_model(text, profile["spin_model"], profile.get("starting_magnetization", 0.5))
    text = replace_kpoints(text, profile["kpoints"])
    for key, value in (
        ("conv_thr", str(profile.get("conv_thr_Ry", "1.0d-6"))),
        ("electron_maxstep", str(profile.get("electron_maxstep", 800))),
        ("mixing_beta", str(profile.get("mixing_beta", 0.2))),
        ("mixing_mode", repr(profile.get("mixing_mode", "local-TF"))),
        ("mixing_ndim", str(profile.get("mixing_ndim", 8))),
        ("diagonalization", "'david'"),
        ("startingpot", "'atomic'"),
        ("startingwfc", "'atomic+random'"),
    ):
        text = set_namelist_value(text, "ELECTRONS", key, value)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_run = Path(config["source_run_dir"]).resolve()
    template_path = source_run / config.get("template", "pw_1.in")
    template_image = read_qe_image(template_path)
    iteration = int(config["source_iteration"])
    path_file = source_run / "out" / f"sb2te3.path{iteration}"
    history = parse_path_file(path_file, len(template_image["symbols"]))
    images = {row["image_index"]: row for row in history["images"]}
    walltime = config["resources"]["walltime"]
    max_seconds = int(config["resources"]["max_seconds"])
    if max_seconds >= walltime_seconds(walltime):
        raise ValueError("max_seconds must be below walltime")
    out_dir = Path(config["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for profile in config["profiles"]:
        for image_index in config["image_indices"]:
            image = images[int(image_index)]
            positions = [atom["position_A"] for atom in image["atoms"]]
            branch_id = f"{config['path_id']}_iter{iteration}_img{image_index}_{profile['id']}_s{config['seed']}"
            job_name = f"mb_p{config['path_id'].replace('-', '')}i{image_index}_{profile['job_slug']}_s{config['seed']}"
            branch_dir = out_dir / branch_id
            branch_dir.mkdir(parents=True, exist_ok=True)
            scf_input = branch_dir / "scf.in"
            scf_input.write_text(
                build_input(
                    template_path.read_text(), template_image["symbols"], positions,
                    profile, "sb2te3", max_seconds,
                )
            )
            identity = calculator_identity(scf_input, "protocol_sensitivity")
            resources = {
                **config["resources"],
                "ntasks": int(profile["ntasks"]),
                "kpoint_pools": int(profile["kpoint_pools"]),
            }
            manifest = {
                "schema_version": "1.0",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scientific_role": "fixed_geometry_qe_protocol_sensitivity",
                "branch_id": branch_id,
                "job_name": job_name,
                "path_id": config["path_id"],
                "source_iteration": iteration,
                "image_index_qe": int(image_index),
                "geometry_role": config["geometry_roles"][str(image_index)],
                "seed": int(config["seed"]),
                "profile": profile,
                "calculator_identity": identity,
                "source": {
                    "path_file": str(path_file),
                    "path_file_sha256": sha256_file(path_file),
                    "template": str(template_path),
                    "template_sha256": sha256_file(template_path),
                    "source_energy_eV": image["energy_eV"],
                },
                "generated_input": str(scf_input),
                "generated_input_sha256": sha256_file(scf_input),
                "resources": resources,
                "acceptance": {
                    "job_done": True,
                    "scf_converged": True,
                    "finite_energy": True,
                    "pair_delta_E_tolerance_eV": config["pair_delta_E_tolerance_eV"],
                    "compare_only_with_same_profile": True,
                },
            }
            manifest_path = branch_dir / "scf_sensitivity_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            jobs.append({**manifest, "manifest_path": str(manifest_path)})
    batch = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "paired_fixed_geometry_qe_protocol_sensitivity",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "jobs": jobs,
    }
    output = out_dir / "protocol_sensitivity_batch_manifest.json"
    output.write_text(json.dumps(batch, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
