#!/usr/bin/env python3
"""Parse QE pw.x relax output and preserve ionic iterations as extxyz."""

import argparse
import json
import math
import re
from pathlib import Path


RY_TO_EV = 13.605693122994
BOHR_TO_A = 0.529177210903
RY_BOHR_TO_EV_A = RY_TO_EV / BOHR_TO_A
ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-+0-9.Ee]+)\s+Ry")
FORCE_RE = re.compile(
    r"atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
)
TOTAL_FORCE_RE = re.compile(r"Total force\s+=\s+([-+0-9.Ee]+)")
SCF_ITER_RE = re.compile(r"iteration #\s+(\d+)")


def parse_positions(lines, start):
    header = lines[start].lower()
    if "angstrom" not in header:
        return None, start + 1
    symbols, positions = [], []
    cursor = start + 1
    while cursor < len(lines):
        fields = lines[cursor].split()
        if len(fields) < 4:
            break
        try:
            xyz = [float(value) for value in fields[1:4]]
        except ValueError:
            break
        symbols.append(fields[0])
        positions.append(xyz)
        cursor += 1
    return {"symbols": symbols, "positions_A": positions}, cursor


def parse_cell(lines, start):
    if "angstrom" not in lines[start].lower() or start + 3 >= len(lines):
        return None
    try:
        return [[float(value) for value in lines[start + offset].split()[:3]] for offset in (1, 2, 3)]
    except (ValueError, IndexError):
        return None


def input_geometry(path):
    if path is None or not Path(path).exists():
        return None
    lines = Path(path).read_text(errors="replace").splitlines()
    cell = None
    geometry = None
    for index, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            cell = parse_cell(lines, index) or cell
        if line.strip().upper().startswith("ATOMIC_POSITIONS"):
            geometry, _ = parse_positions(lines, index)
    if geometry:
        geometry["cell_A"] = cell
        geometry["geometry_source"] = "relax_input"
    return geometry


def same_positions(first, second, tolerance=1.0e-9):
    if not first or not second or first.get("symbols") != second.get("symbols"):
        return False
    return all(
        abs(float(a) - float(b)) <= tolerance
        for row_a, row_b in zip(first["positions_A"], second["positions_A"])
        for a, b in zip(row_a, row_b)
    )


def parse_relax_out(path, relax_input=None):
    lines = path.read_text(errors="replace").splitlines()
    steps = []
    latest_energy = None
    latest_forces = {}
    latest_total_force = None
    latest_cumulative_scf_iterations = 0
    latest_cell = None
    max_scf_iteration_seen = None
    total_scf_iterations_seen = 0
    scf_not_converged_count = 0
    bfgs_converged = False
    geometry_optimization_ended = False
    max_ionic_steps_reached = False
    max_seconds_reached = False
    current_geometry = input_geometry(relax_input)
    latest_printed_geometry = current_geometry
    evaluation_index = -1
    recorded_evaluation = -1
    in_d3_force_breakdown = False

    def append_evaluation(geometry, alignment):
        if geometry is None or latest_energy is None:
            return
        nat = len(geometry["symbols"])
        force_complete = len(latest_forces) == nat
        forces = [latest_forces.get(atom_index) for atom_index in range(nat)]
        finite_forces = [math.sqrt(sum(value * value for value in force)) for force in forces if force]
        steps.append({
            "ionic_step": len(steps),
            "symbols": geometry["symbols"],
            "positions_A": geometry["positions_A"],
            "cell_A": geometry.get("cell_A") or latest_cell,
            "geometry_source": geometry.get("geometry_source"),
            "force_geometry_alignment": alignment,
            "energy_eV": latest_energy,
            "forces_eV_A": forces,
            "force_complete": force_complete,
            "max_atom_force_eV_A": max(finite_forces) if finite_forces else None,
            "total_force_eV_A": latest_total_force,
            "cumulative_scf_iterations": latest_cumulative_scf_iterations,
        })
    for index, line in enumerate(lines):
        match = SCF_ITER_RE.search(line)
        if match:
            max_scf_iteration_seen = int(match.group(1))
            total_scf_iterations_seen += 1
        match = ENERGY_RE.search(line)
        if match:
            latest_energy = float(match.group(1)) * RY_TO_EV
            latest_cumulative_scf_iterations = total_scf_iterations_seen
            latest_forces = {}
            latest_total_force = None
            evaluation_index += 1
            in_d3_force_breakdown = False
        if "forces acting on atoms" in line.lower():
            in_d3_force_breakdown = False
        if "dft-d3 dispersion contribution to forces" in line.lower():
            in_d3_force_breakdown = True
        match = FORCE_RE.search(line)
        if match and not in_d3_force_breakdown:
            latest_forces[int(match.group(1)) - 1] = [
                float(match.group(axis)) * RY_BOHR_TO_EV_A for axis in (2, 3, 4)
            ]
        match = TOTAL_FORCE_RE.search(line)
        if match:
            latest_total_force = float(match.group(1)) * RY_BOHR_TO_EV_A
        if "convergence NOT achieved" in line:
            scf_not_converged_count += 1
        if "bfgs converged" in line.lower():
            bfgs_converged = True
        if "end of bfgs geometry optimization" in line.lower():
            geometry_optimization_ended = True
        if "maximum number of steps has been reached" in line.lower():
            max_ionic_steps_reached = True
        if "maximum cpu time exceeded" in line.lower() or "max_seconds" in line.lower() and "exceeded" in line.lower():
            max_seconds_reached = True
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            latest_cell = parse_cell(lines, index) or latest_cell
        if line.strip().upper().startswith("ATOMIC_POSITIONS"):
            geometry, _ = parse_positions(lines, index)
            if geometry:
                geometry["cell_A"] = latest_cell or (current_geometry or {}).get("cell_A")
                geometry["geometry_source"] = "qe_printed_atomic_positions"
                if evaluation_index != recorded_evaluation:
                    if current_geometry is None:
                        append_evaluation(geometry, "assumed_same_block_without_relax_input")
                    else:
                        append_evaluation(current_geometry, "input_or_previous_qe_update")
                    recorded_evaluation = evaluation_index
                current_geometry = geometry
                latest_printed_geometry = geometry
    if evaluation_index != recorded_evaluation and current_geometry is not None:
        append_evaluation(current_geometry, "input_or_previous_qe_update_at_eof")
    job_done = any("JOB DONE" in line for line in lines)
    final = steps[-1] if steps else None
    complete_force_steps = [step for step in steps if step["force_complete"]]
    final_complete_force_step = complete_force_steps[-1] if complete_force_steps else None
    latest_geometry_has_evaluated_forces = same_positions(latest_printed_geometry, final_complete_force_step)
    return {
        "relax_out": str(path),
        "job_done": job_done,
        "bfgs_converged": bfgs_converged,
        "geometry_optimization_ended": geometry_optimization_ended,
        "max_ionic_steps_reached": max_ionic_steps_reached,
        "max_seconds_reached": max_seconds_reached,
        "scf_not_converged_count": scf_not_converged_count,
        "max_scf_iteration_seen": max_scf_iteration_seen,
        "total_scf_iterations_seen": total_scf_iterations_seen,
        "n_ionic_steps": len(steps),
        "final_energy_eV": final["energy_eV"] if final else None,
        "final_max_atom_force_eV_A": (
            final_complete_force_step["max_atom_force_eV_A"]
            if final_complete_force_step else None
        ),
        "final_force_ionic_step": (
            final_complete_force_step["ionic_step"] if final_complete_force_step else None
        ),
        "trailing_incomplete_force_step": bool(final and not final["force_complete"]),
        "latest_geometry": latest_printed_geometry,
        "latest_geometry_has_evaluated_forces": latest_geometry_has_evaluated_forces,
        "steps": steps,
    }


def last_complete_force_step(parsed):
    complete = [step for step in parsed.get("steps", []) if step.get("force_complete")]
    return complete[-1] if complete else None


def last_printed_geometry(parsed):
    return parsed.get("latest_geometry")


def extxyz_header(step):
    lattice = step.get("cell_A")
    fields = []
    if lattice:
        flat = " ".join(f"{value:.12g}" for row in lattice for value in row)
        fields.append(f'Lattice="{flat}"')
    fields.extend(
        [
            "Properties=species:S:1:pos:R:3:forces:R:3",
            f"energy={step['energy_eV']:.12g}",
            f"ionic_step={step['ionic_step']}",
            "pbc=\"T T T\"" if lattice else "pbc=\"F F F\"",
        ]
    )
    return " ".join(fields)


def write_extxyz(path, steps):
    with path.open("w") as handle:
        for step in steps:
            if not step["force_complete"]:
                continue
            handle.write(f"{len(step['symbols'])}\n")
            handle.write(extxyz_header(step) + "\n")
            for symbol, position, force in zip(step["symbols"], step["positions_A"], step["forces_eV_A"]):
                values = position + force
                handle.write(symbol + " " + " ".join(f"{value:.12g}" for value in values) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("relax_out", type=Path)
    parser.add_argument("--relax-input", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-extxyz", type=Path)
    args = parser.parse_args()
    result = parse_relax_out(args.relax_out, args.relax_input)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_extxyz:
        args.output_extxyz.parent.mkdir(parents=True, exist_ok=True)
        write_extxyz(args.output_extxyz, result["steps"])
    print(json.dumps({key: value for key, value in result.items() if key != "steps"}, indent=2))


if __name__ == "__main__":
    main()
