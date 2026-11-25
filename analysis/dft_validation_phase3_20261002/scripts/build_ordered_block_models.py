#!/usr/bin/env python3
"""Extract one immutable 82-atom source block and its 81-atom one-Cr view."""

import argparse
import json
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
    species = np.array([row[0] for row in rows])
    positions = np.array([[float(x) for x in row[1:4]] for row in rows])
    has_momenta = "momenta:R:3" in comment.split("Properties=", 1)[1].split()[0]
    momenta = []
    forces = []
    for row in rows:
        cursor = 4
        if has_momenta:
            momenta.append([float(x) for x in row[cursor:cursor + 3]])
            cursor += 3
        forces.append([float(x) for x in row[cursor:cursor + 3]])
    return n, comment, cell, species, positions, np.array(momenta), np.array(forces), has_momenta


def write_frame(path, comment, cell, species, positions, momenta, forces):
    props = "species:S:1:pos:R:3:"
    if momenta is not None:
        props += "momenta:R:3:"
    props += "forces:R:3"
    comment = re.sub(r"Properties=[^\s]+", "Properties=" + props, comment)
    comment = re.sub(r"Lattice=\"[^\"]+\"", "Lattice=\"" + " ".join("%.14f" % x for x in cell.reshape(-1)) + "\"", comment)
    comment = re.sub(r"\s(?:energy|free_energy|stress)=\"[^\"]*\"", "", comment)
    comment = re.sub(r"\s(?:energy|free_energy)=[^\s]+", "", comment)
    with path.open("w") as handle:
        handle.write(str(len(species)) + "\n" + comment + " source_block_model=T\n")
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
    parser.add_argument("--block-index", type=int, default=7)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, _, ref_cell, ref_species, ref_positions, _, _, _ = read_frame(args.reference)
    ref_frac = ref_positions @ np.linalg.inv(ref_cell)
    ref_cr = ref_frac[np.where(ref_species == "Cr")[0][0]]
    report = {
        "reference": str(args.reference),
        "reference_site": "unique_115 from cand_0016",
        "block_index_zero_based": args.block_index,
        "block_rule": "preserve original atom ordering; block_size=82; no spatial reassignment",
        "snapshots": [],
    }
    for source in sorted(args.input_dir.glob("snapshot_step_*.extxyz")):
        n, comment, cell, species, positions, momenta, forces, has_momenta = read_frame(source)
        start = args.block_index * 82
        stop = start + 82
        if n < stop:
            raise ValueError("source has fewer than the requested block")
        block_species = species[start:stop]
        block_positions = positions[start:stop]
        block_forces = forces[start:stop]
        block_momenta = momenta[start:stop] if has_momenta else None
        local_cell = cell.copy()
        local_cell[0] /= 5.0
        local_cell[1] /= 5.0
        full_frac = block_positions @ np.linalg.inv(cell)
        local_frac = full_frac.copy()
        local_frac[:, :2] = (local_frac[:, :2] * 5.0) % 1.0
        cr_indices = np.where(block_species == "Cr")[0]
        cr_distances = []
        for index in cr_indices:
            delta = local_frac[index] - ref_cr
            delta -= np.round(delta)
            cr_distances.append(float(np.linalg.norm(delta @ local_cell)))
        target_cr = int(cr_indices[int(np.argmin(cr_distances))])
        other_cr = int(cr_indices[0] if cr_indices[1] == target_cr else cr_indices[1])
        keep = np.ones(82, dtype=bool)
        keep[other_cr] = False
        exact_path = args.output_dir / (source.stem + "_ordered82.extxyz")
        reduced_path = args.output_dir / (source.stem + "_same_site81_ordered.extxyz")
        local_positions = local_frac @ local_cell
        write_frame(exact_path, comment, local_cell, block_species, local_positions, block_momenta, block_forces)
        write_frame(reduced_path, comment, local_cell, block_species[keep], local_positions[keep], block_momenta[keep] if block_momenta is not None else None, block_forces[keep])
        report["snapshots"].append({
            "source": str(source),
            "step": int(STEP_RE.search(comment).group(1)),
            "source_atom_block_1based": [start + 1, stop],
            "target_cr_block_index_1based": target_cr + 1,
            "other_cr_block_index_1based": other_cr + 1,
            "target_cr_distance_to_unique_115_A": min(cr_distances),
            "block_max_force_eV_A": float(np.linalg.norm(block_forces, axis=1).max()),
            "exact_ordered82": str(exact_path),
            "same_site81_ordered": str(reduced_path),
            "species_counts_82": {symbol: int((block_species == symbol).sum()) for symbol in sorted(set(block_species))},
            "limitations": [
                "82-atom branch preserves the exact atom identities and both Cr atoms in the source block.",
                "81-atom branch removes only the second Cr; it is a controlled one-Cr composition approximation.",
                "The force columns are retained as source MLIP diagnostics, not DFT results; all coordinates are fixed.",
            ],
        })
    (args.output_dir / "ordered_block_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
