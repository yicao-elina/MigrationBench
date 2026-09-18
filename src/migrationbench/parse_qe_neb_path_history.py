#!/usr/bin/env python3
"""Parse QE neb.x .pathN restart histories with exact iteration provenance."""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from analyze_neb_path_topology import read_qe_image
from parse_qe_neb_output import parse_neb_out


BOHR_TO_A = 0.529177210903
HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_A = HARTREE_TO_EV / BOHR_TO_A
IMAGE_RE = re.compile(r"^Image:\s+(\d+)\s*$")


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_path_file(path, atom_count):
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 10 or lines[0].strip() != "RESTART INFORMATION":
        raise ValueError(f"Not a QE path restart file: {path}")
    iteration = int(lines[1])
    n_images = int(lines[5])
    images = []
    cursor = 9
    while cursor < len(lines):
        match = IMAGE_RE.match(lines[cursor].strip())
        if not match:
            cursor += 1
            continue
        image_index = int(match.group(1))
        energy_hartree = float(lines[cursor + 1])
        atoms = []
        for line in lines[cursor + 2 : cursor + 2 + atom_count]:
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"Incomplete atom row in {path}: {line}")
            values = [float(value) for value in fields[:6]]
            atoms.append(
                {
                    "position_A": [value * BOHR_TO_A for value in values[:3]],
                    "qe_path_gradient_eV_A": [
                        value * HARTREE_PER_BOHR_TO_EV_PER_A for value in values[3:6]
                    ],
                    "move_mask": [int(value) for value in fields[6:9]] if len(fields) >= 9 else [-1, -1, -1],
                }
            )
        images.append(
            {
                "image_index": image_index,
                "energy_hartree": energy_hartree,
                "energy_eV": energy_hartree * HARTREE_TO_EV,
                "atoms": atoms,
            }
        )
        cursor += atom_count + 2
    if len(images) != n_images:
        raise ValueError(f"Expected {n_images} images in {path}, parsed {len(images)}")
    placeholder = iteration == 0 and all(abs(image["energy_hartree"]) < 1.0e-15 for image in images)
    for image in images:
        image["energy_valid"] = not placeholder
    return {"iteration": iteration, "n_images": n_images, "images": images, "energy_valid": not placeholder}


def write_extxyz(path, histories, symbols, cell, source_hashes):
    lattice = " ".join(f"{value:.12f}" for vector in cell for value in vector)
    with path.open("w") as handle:
        for history in histories:
            for image in history["images"]:
                handle.write(f"{len(symbols)}\n")
                relative = image["energy_eV"] - history["images"][0]["energy_eV"]
                energy = f'{image["energy_eV"]:.12f}' if image["energy_valid"] else "nan"
                relative_text = f"{relative:.12f}" if image["energy_valid"] else "nan"
                handle.write(
                    'Lattice="{}" Properties=species:S:1:pos:R:3:qe_path_gradient:R:3:move_mask:I:3 '
                    'energy={} relative_energy_eV={} energy_valid={} path_iteration={} image_index={} '
                    'pbc="T T T" position_unit="angstrom" gradient_unit="eV/angstrom" '
                    'gradient_semantics="QE_path_restart_gradient_not_independent_PW_force" source_sha256="{}"\n'.format(
                        lattice,
                        energy,
                        relative_text,
                        str(image["energy_valid"]).lower(),
                        history["iteration"],
                        image["image_index"],
                        source_hashes[history["iteration"]],
                    )
                )
                for symbol, atom in zip(symbols, image["atoms"]):
                    values = atom["position_A"] + atom["qe_path_gradient_eV_A"] + atom["move_mask"]
                    handle.write(symbol + " " + " ".join(str(value) for value in values) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    template = read_qe_image(run_dir / "pw_1.in")
    path_files = sorted(
        (run_dir / "out").glob("*.path[0-9]*"),
        key=lambda value: int(re.search(r"path(\d+)$", value.name).group(1)),
    )
    histories = [parse_path_file(path, len(template["symbols"])) for path in path_files]
    hashes = {row["iteration"]: sha256_file(path) for row, path in zip(histories, path_files)}
    qe_summary = parse_neb_out(run_dir / "neb.out")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    extxyz = out_dir / "neb_path_history.extxyz"
    write_extxyz(extxyz, histories, template["symbols"], template["cell"], hashes)

    rows = []
    for history in histories:
        reference = history["images"][0]["energy_eV"]
        for image in history["images"]:
            rows.append(
                {
                    "iteration": history["iteration"],
                    "image_index": image["image_index"],
                    "energy_eV": image["energy_eV"],
                    "relative_energy_eV": image["energy_eV"] - reference,
                    "energy_valid": image["energy_valid"],
                    "source_sha256": hashes[history["iteration"]],
                }
            )
    with (out_dir / "neb_path_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    last = histories[-1]
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "template_pw_input": str((run_dir / "pw_1.in").resolve()),
        "template_pw_sha256": sha256_file(run_dir / "pw_1.in"),
        "energy_unit_in_path_file": "hartree",
        "position_unit_in_path_file": "bohr",
        "gradient_unit_in_path_file": "hartree/bohr",
        "gradient_semantics": "QE path restart gradient; do not relabel as independent PW atomic force",
        "path0_semantics": "unlabeled initialization coordinates; zero energies and gradients are QE placeholders",
        "n_iterations": len(histories),
        "latest_available_iteration": last["iteration"],
        "latest_complete_neb_out_iteration": qe_summary["last_complete_iteration"],
        "latest_images": [
            {
                "image_index": image["image_index"],
                "energy_eV": image["energy_eV"],
                "relative_energy_eV": image["energy_eV"] - last["images"][0]["energy_eV"],
            }
            for image in last["images"]
        ],
        "iteration_sources": [
            {"iteration": row["iteration"], "path": str(path), "sha256": hashes[row["iteration"]]}
            for row, path in zip(histories, path_files)
        ],
        "outputs": {"extxyz": str(extxyz), "csv": str(out_dir / "neb_path_history.csv")},
    }
    (out_dir / "neb_path_history_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"iterations": len(histories), "latest": last["iteration"], "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
