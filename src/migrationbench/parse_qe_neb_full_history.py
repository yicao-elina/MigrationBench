#!/usr/bin/env python3
"""Join QE path restart coordinates/gradients with per-image PW energies/forces."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from analyze_neb_path_topology import displacement, read_qe_image
from parse_qe_neb_output import parse_neb_out
from parse_qe_neb_path_history import HARTREE_TO_EV, parse_path_file


RY_TO_EV = HARTREE_TO_EV / 2.0
BOHR_TO_A = 0.529177210903
RY_PER_BOHR_TO_EV_PER_A = RY_TO_EV / BOHR_TO_A
ENERGY_RE = re.compile(r"^!\s+total energy\s+=\s+([-+0-9.Ee]+)\s+Ry")
FORCE_RE = re.compile(
    r"^\s*atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+"
    r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
)
SCF_RE = re.compile(r"convergence has been achieved in\s+(\d+) iterations")


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_pw_blocks(path, atom_count):
    lines = path.read_text(errors="replace").splitlines()
    starts = [index for index, line in enumerate(lines) if ENERGY_RE.match(line)]
    blocks = []
    for block_index, start in enumerate(starts):
        stop = starts[block_index + 1] if block_index + 1 < len(starts) else len(lines)
        energy_ry = float(ENERGY_RE.match(lines[start]).group(1))
        scf_iterations = None
        forces = []
        in_forces = False
        for line in lines[start + 1 : stop]:
            scf = SCF_RE.search(line)
            if scf and scf_iterations is None:
                scf_iterations = int(scf.group(1))
            if "Forces acting on atoms" in line:
                in_forces = True
                continue
            if in_forces:
                match = FORCE_RE.match(line)
                if match:
                    forces.append([float(value) * RY_PER_BOHR_TO_EV_PER_A for value in match.groups()[1:]])
                    if len(forces) == atom_count:
                        break
        if len(forces) != atom_count:
            raise ValueError(f"Force block {block_index} in {path} has {len(forces)} of {atom_count} atoms")
        blocks.append(
            {
                "block_index": block_index,
                "energy_ry": energy_ry,
                "energy_eV": energy_ry * RY_TO_EV,
                "scf_iterations": scf_iterations,
                "forces_eV_A": forces,
            }
        )
    return blocks


def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def stable_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def structure_identity(symbols, cell, positions):
    structure = {
        "symbols": symbols,
        "cell_A": [[round(value, 10) for value in vector] for vector in cell],
        "positions_A": [[round(value, 10) for value in position] for position in positions],
        "pbc": [True, True, True],
    }
    return stable_hash(structure)


def reaction_coordinates(history, cell):
    cumulative = [0.0]
    for left, right in zip(history["images"], history["images"][1:]):
        squared = 0.0
        for left_atom, right_atom in zip(left["atoms"], right["atoms"]):
            vector = displacement(left_atom["position_A"], right_atom["position_A"], cell)
            squared += sum(value * value for value in vector)
        rms_step = math.sqrt(squared / len(left["atoms"]))
        cumulative.append(cumulative[-1] + rms_step)
    total = cumulative[-1]
    normalized = [value / total if total > 0.0 else 0.0 for value in cumulative]
    return cumulative, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pathway-id", required=True)
    parser.add_argument("--neb-out", type=Path, action="append", help="Ordered restart-lineage neb.out; defaults to RUN_DIR/neb.out")
    parser.add_argument("--energy-tolerance-eV", type=float, default=2.0e-4)
    parser.add_argument("--gradient-force-tolerance-eV-A", type=float, default=2.0e-5)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    template = read_qe_image(run_dir / "pw_1.in")
    atom_count = len(template["symbols"])
    path_files = sorted(
        (run_dir / "out").glob("*.path[0-9]*"),
        key=lambda value: int(re.search(r"path(\d+)$", value.name).group(1)),
    )
    histories = {int(re.search(r"path(\d+)$", path.name).group(1)): parse_path_file(path, atom_count) for path in path_files}
    labeled_iterations = sorted(iteration for iteration in histories if iteration > 0)
    image_count = histories[labeled_iterations[0]]["n_images"]
    pw_blocks = {
        image_index: parse_pw_blocks(run_dir / "out" / f"sb2te3_{image_index}" / "PW.out", atom_count)
        for image_index in range(1, image_count + 1)
    }
    neb_outputs = [path.resolve() for path in (args.neb_out or [run_dir / "neb.out"])]
    neb_index = {}
    neb_source = {}
    for neb_output in neb_outputs:
        neb = parse_neb_out(neb_output)
        for iteration in neb["complete_iterations"]:
            for image in iteration["images"]:
                key = (iteration["iteration"], image["image_index"])
                if key in neb_index and neb_index[key] != image:
                    raise ValueError(f"Conflicting NEB lineage record for {key}")
                neb_index[key] = image
                neb_source[key] = str(neb_output)

    rows = []
    frames = []
    run_id = run_dir.name
    max_energy_residual = 0.0
    max_gradient_force_residual = 0.0
    for iteration in labeled_iterations:
        history = histories[iteration]
        arc_A, arc_normalized = reaction_coordinates(history, template["cell"])
        for image in history["images"]:
            image_index = image["image_index"]
            blocks = pw_blocks[image_index]
            frozen_reuse = len(blocks) == 1
            block_index = 0 if frozen_reuse else iteration - 1
            if block_index >= len(blocks):
                continue
            block = blocks[block_index]
            energy_residual = abs(block["energy_eV"] - image["energy_eV"])
            residuals = []
            max_force = 0.0
            for atom, force in zip(image["atoms"], block["forces_eV_A"]):
                residual = [gradient + force_component for gradient, force_component in zip(atom["qe_path_gradient_eV_A"], force)]
                residuals.extend(residual)
                max_force = max(max_force, vector_norm(force))
            gradient_force_rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
            gradient_force_max = max(abs(value) for value in residuals)
            max_energy_residual = max(max_energy_residual, energy_residual)
            max_gradient_force_residual = max(max_gradient_force_residual, gradient_force_max)
            neb_row = neb_index.get((iteration, image_index), {})
            positions = [atom["position_A"] for atom in image["atoms"]]
            structure_hash = structure_identity(template["symbols"], template["cell"], positions)
            configuration_id = stable_hash(
                {
                    "structure_hash": structure_hash,
                    "pathway_id": args.pathway_id,
                    "path_iteration": iteration,
                    "image_index": image_index,
                }
            )
            row = {
                "iteration_record_id": stable_hash(
                    {"run_id": run_id, "path_iteration": iteration, "image_index": image_index}
                ),
                "configuration_id": configuration_id,
                "structure_hash": structure_hash,
                "run_id": run_id,
                "pathway_id": args.pathway_id,
                "path_iteration": iteration,
                "image_index": image_index,
                "pw_block_index": block_index,
                "frozen_endpoint_force_reused": frozen_reuse,
                "energy_eV": block["energy_eV"],
                "path_energy_eV": image["energy_eV"],
                "energy_residual_eV": energy_residual,
                "max_atomic_force_eV_A": max_force,
                "neb_residual_eV_A": neb_row.get("error_eV_A"),
                "neb_iteration_complete": bool(neb_row),
                "neb_source_file": neb_source.get((iteration, image_index)),
                "reaction_coordinate_A": arc_A[image_index - 1],
                "reaction_coordinate_normalized": arc_normalized[image_index - 1],
                "scf_iterations": block["scf_iterations"],
                "gradient_force_rms_eV_A": gradient_force_rms,
                "gradient_force_max_eV_A": gradient_force_max,
            }
            rows.append(row)
            frames.append((history, image, block, row))

    if max_energy_residual > args.energy_tolerance_eV:
        raise ValueError(f"PW/path energy mismatch {max_energy_residual} eV")
    if max_gradient_force_residual > args.gradient_force_tolerance_eV_A:
        raise ValueError(f"PW force/path gradient mismatch {max_gradient_force_residual} eV/A")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "neb_full_history.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lattice = " ".join(f"{value:.12f}" for vector in template["cell"] for value in vector)
    extxyz = out_dir / "neb_full_history.extxyz"
    with extxyz.open("w") as handle:
        for history, image, block, row in frames:
            handle.write(f"{atom_count}\n")
            handle.write(
                'Lattice="{}" Properties=species:S:1:pos:R:3:forces:R:3:qe_path_gradient:R:3:move_mask:I:3 '
                'energy={:.12f} configuration_id="{}" iteration_record_id="{}" pathway_id="{}" '
                'path_iteration={} image_index={} reaction_coordinate_A={:.12f} reaction_coordinate_normalized={:.12f} '
                'scf_iterations={} max_atomic_force_eV_A={:.12f} neb_residual_eV_A={} '
                'gradient_semantics="potential_energy_gradient_equals_negative_PW_force" '
                'frozen_endpoint_force_reused={} pbc="T T T"\n'.format(
                    lattice,
                    block["energy_eV"],
                    row["configuration_id"],
                    row["iteration_record_id"],
                    args.pathway_id,
                    history["iteration"],
                    image["image_index"],
                    row["reaction_coordinate_A"],
                    row["reaction_coordinate_normalized"],
                    block["scf_iterations"] if block["scf_iterations"] is not None else -1,
                    row["max_atomic_force_eV_A"],
                    row["neb_residual_eV_A"] if row["neb_residual_eV_A"] is not None else "nan",
                    str(row["frozen_endpoint_force_reused"]).lower(),
                )
            )
            for symbol, atom, force in zip(template["symbols"], image["atoms"], block["forces_eV_A"]):
                values = atom["position_A"] + force + atom["qe_path_gradient_eV_A"] + atom["move_mask"]
                handle.write(symbol + " " + " ".join(str(value) for value in values) + "\n")

    runtime_source = run_dir / "runtime_provenance.json"
    runtime_output = out_dir / "runtime_provenance.json"
    runtime_record = None
    if runtime_source.is_file():
        shutil.copy2(runtime_source, runtime_output)
        runtime_record = {
            "uri": str(runtime_output),
            "sha256": sha256_file(runtime_output),
            "source_uri": str(runtime_source),
            "source_sha256": sha256_file(runtime_source),
        }
        if runtime_record["sha256"] != runtime_record["source_sha256"]:
            raise RuntimeError("runtime provenance copy hash mismatch")

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_id": run_id,
        "pathway_id": args.pathway_id,
        "mapping": "path iteration n>=1 maps to interior-image PW block n-1; one-block frozen endpoint forces are explicitly reused",
        "path0": "excluded from DFT-labeled history because its zero energy/gradient fields are initialization placeholders",
        "vector_semantics": {
            "forces": "independent PW atomic force from PW.out",
            "qe_path_gradient": "potential-energy gradient from .pathN, numerically verified as negative PW force",
            "neb_residual": "scalar QE image error from neb.out; projected spring/tangent vector is not present in these raw artifacts",
        },
        "reaction_coordinate_definition": "cumulative minimum-image all-atom Cartesian RMS arc length within each iteration; normalized by final arc length",
        "n_labeled_iterations": len(labeled_iterations),
        "n_frames": len(frames),
        "max_energy_residual_eV": max_energy_residual,
        "max_gradient_plus_force_residual_eV_A": max_gradient_force_residual,
        "source_files": {
            "path_restart_sha256": {path.name: sha256_file(path) for path in path_files},
            "pw_output_sha256": {
                str(image_index): sha256_file(run_dir / "out" / f"sb2te3_{image_index}" / "PW.out")
                for image_index in range(1, image_count + 1)
            },
            "neb_lineage_sha256": {str(path): sha256_file(path) for path in neb_outputs},
            "runtime_provenance": runtime_record,
        },
        "outputs": {"csv": str(csv_path), "extxyz": str(extxyz), "runtime_provenance": runtime_record},
    }
    (out_dir / "neb_full_history_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ["n_labeled_iterations", "n_frames", "max_energy_residual_eV", "max_gradient_plus_force_residual_eV_A"]}, indent=2))


if __name__ == "__main__":
    main()
