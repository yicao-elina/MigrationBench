#!/usr/bin/env python3
"""Build exact 5x5-tile local periodic models from a 2050-atom extxyz frame."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
STEP_RE = re.compile(r"(?:^|\s)step=(-?\d+)(?:\s|$)")


def read_frame(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    comment = lines[1]
    rows = [line.split() for line in lines[2:2 + n]]
    cell = np.array(list(map(float, LATTICE_RE.search(comment).group(1).split()))).reshape(3, 3)
    species = [row[0] for row in rows]
    positions = np.array([[float(x) for x in row[1:4]] for row in rows])
    has_momenta = "momenta:R:3" in comment.split("Properties=", 1)[1].split()[0]
    forces = []
    momenta = []
    for row in rows:
        offset = 4
        if has_momenta:
            momenta.append([float(x) for x in row[offset:offset + 3]])
            offset += 3
        forces.append([float(x) for x in row[offset:offset + 3]])
    return n, comment, cell, np.array(species), positions, np.array(momenta), np.array(forces), rows, has_momenta


def write_frame(path, comment, cell, species, positions, momenta, forces):
    properties = "species:S:1:pos:R:3:"
    if momenta is not None:
        properties += "momenta:R:3:"
    properties += "forces:R:3"
    comment = re.sub(r"Properties=[^\s]+", "Properties=" + properties, comment)
    comment = re.sub(r"Lattice=\"[^\"]+\"", "Lattice=\"" + " ".join("%.14f" % x for x in cell.reshape(-1)) + "\"", comment)
    # The source energy/stress belong to the 2050-atom frame, not this reduced
    # cell. Do not carry them forward as if they were local-model observables.
    comment = re.sub(r"\s(?:energy|free_energy|stress)=\"[^\"]*\"", "", comment)
    comment = re.sub(r"\s(?:energy|free_energy)=[^\s]+", "", comment)
    with path.open("w") as handle:
        handle.write(str(len(species)) + "\n")
        handle.write(comment + " local_model=T\n")
        for i, symbol in enumerate(species):
            values = list(positions[i])
            if momenta is not None:
                values += list(momenta[i])
            values += list(forces[i])
            handle.write("%-2s % .12f % .12f % .12f" % (symbol, *values[:3]))
            cursor = 3
            if momenta is not None:
                handle.write(" % .12f % .12f % .12f" % tuple(values[cursor:cursor + 3]))
                cursor += 3
            handle.write(" % .12f % .12f % .12f\n" % tuple(values[cursor:cursor + 3]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-x", type=int, default=2)
    parser.add_argument("--tile-y", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_n, _, ref_cell, ref_species, ref_pos, _, _, _, _ = read_frame(args.reference)
    ref_cr = ref_pos[np.where(ref_species == "Cr")[0][0]] @ np.linalg.inv(ref_cell)
    report = {
        "reference": str(args.reference),
        "reference_natoms": ref_n,
        "reference_site": "unique_115 from cand_0016",
        "tile_rule": "fractional full-cell x/y bins floor(frac*5), tile=(2,1), local cell=(a/5,b/5,c)",
        "tile_x": args.tile_x,
        "tile_y": args.tile_y,
        "snapshots": [],
    }

    for source in sorted(args.input_dir.glob("snapshot_step_*.extxyz")):
        n, comment, cell, species, positions, momenta, forces, rows, has_momenta = read_frame(source)
        local_cell = cell.copy()
        local_cell[0] /= 5.0
        local_cell[1] /= 5.0
        frac = positions @ np.linalg.inv(cell)
        frac = frac - np.floor(frac)
        ref_frac = ref_pos @ np.linalg.inv(ref_cell)
        target_ref_global = ref_frac.copy()
        target_ref_global[:, 0] = (args.tile_x + ref_frac[:, 0]) / 5.0
        target_ref_global[:, 1] = (args.tile_y + ref_frac[:, 1]) / 5.0
        selected = []
        selected_distances = []
        for symbol in sorted(set(ref_species)):
            ref_indices = np.where(ref_species == symbol)[0]
            full_indices = np.where(species == symbol)[0]
            cost = np.empty((len(ref_indices), len(full_indices)))
            for i, ref_index in enumerate(ref_indices):
                delta = frac[full_indices] - target_ref_global[ref_index]
                delta -= np.round(delta)
                cost[i] = np.linalg.norm(delta @ cell, axis=1)
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                selected.append(int(full_indices[col]))
                selected_distances.append(float(cost[row, col]))
        indices = np.array(selected, dtype=int)
        # Unwrap selected atoms around the reference geometry so a boundary-crossing
        # thermal displacement does not change atom identity or cell membership.
        local_frac = np.empty((len(indices), 3))
        ref_order = []
        for symbol in sorted(set(ref_species)):
            ref_order.extend(list(np.where(ref_species == symbol)[0]))
        # linear_sum_assignment returns rows in sorted order for each species;
        # reconstruct the same order used by the selected list.
        cursor = 0
        for symbol in sorted(set(ref_species)):
            ref_indices = list(np.where(ref_species == symbol)[0])
            for ref_index in ref_indices:
                full_index = indices[cursor]
                delta = frac[full_index] - target_ref_global[ref_index]
                delta -= np.round(delta)
                local_frac[cursor] = ref_frac[ref_index] + delta * np.array([5.0, 5.0, 1.0])
                cursor += 1
        local_positions = local_frac @ local_cell
        local_species = species[indices]
        local_forces = forces[indices]
        local_momenta = momenta[indices] if has_momenta else None
        target_local = int(np.where(local_species == "Cr")[0][0])
        # Add the nearest additional Cr in the full periodic cell to make the
        # exact 82-atom local cell; the 81-atom branch intentionally omits it.
        target_full = indices[target_local]
        other_candidates = np.where(species == "Cr")[0]
        other_candidates = [int(i) for i in other_candidates if int(i) != int(target_full)]
        target_cart = positions[target_full]
        other_distances = []
        for i in other_candidates:
            delta = frac[i] - frac[target_full]
            delta -= np.round(delta)
            other_distances.append((float(np.linalg.norm(delta @ cell)), i))
        other_full = min(other_distances)[1]
        other_delta = frac[other_full] - target_ref_global[target_local]
        other_delta -= np.round(other_delta)
        other_local_frac = ref_cr + other_delta * np.array([5.0, 5.0, 1.0])
        exact_species = np.concatenate([local_species, np.array(["Cr"])])
        exact_positions = np.vstack([local_positions, other_local_frac @ local_cell])
        exact_forces = np.vstack([local_forces, forces[other_full]])
        exact_momenta = np.vstack([local_momenta, momenta[other_full]]) if has_momenta else None
        stem = source.stem
        exact_path = args.output_dir / (stem + "_local82.extxyz")
        reduced_path = args.output_dir / (stem + "_same_site81_approx.extxyz")
        write_frame(exact_path, comment, local_cell, exact_species, exact_positions, exact_momenta, exact_forces)
        write_frame(reduced_path, comment, local_cell, local_species, local_positions, local_momenta, local_forces)
        report["snapshots"].append({
            "source": str(source),
            "step": int(STEP_RE.search(comment).group(1)),
            "source_natoms": n,
            "tile_atoms": int(len(exact_species)),
            "tile_species_counts": {symbol: int((exact_species == symbol).sum()) for symbol in sorted(set(exact_species))},
            "local_cell": local_cell.tolist(),
            "target_cr_local_index_1based": target_local + 1,
            "other_cr_full_index_1based": other_full + 1,
            "reference_cr_distance_A": min(selected_distances),
            "selected_reference_assignment_max_distance_A": max(selected_distances),
            "second_cr_periodic_distance_A": min(other_distances)[0],
            "exact_local82": str(exact_path),
            "same_site81_approx": str(reduced_path),
            "limitations": [
                "82-atom tile preserves both Cr atoms and is the exact periodic subcell of the 2050 snapshot.",
                "81-atom approximation removes the second Cr from the same tile; it is not compositionally identical to the full MD cell.",
                "All positions remain fixed for single-point evaluation; no relaxation is implied.",
            ],
        })
    (args.output_dir / "local_tile_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
