#!/usr/bin/env python3
"""Check NEB image geometry before spending DFT time.

This catches path initializations with huge inter-image jumps, unphysical close
contacts, or runaway migrant motion. It is intended as a pre-submit gate before
QE neb.x jobs.
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read


def choose_migrant(images, element):
    symbols = images[0].get_chemical_symbols()
    candidates = [i for i, s in enumerate(symbols) if s == element]
    if not candidates:
        raise ValueError("No atoms with migrant element %r" % element)
    if len(candidates) == 1:
        return candidates[0]
    start = images[0]
    end = images[-1]
    disp = [end.get_distance(i, i, mic=True, vector=True) for i in candidates]
    # ASE get_distance same atom gives zero, so fall back to Cartesian endpoint displacement.
    spos = start.get_positions(); epos = end.get_positions()
    norms = [float(np.linalg.norm(epos[i] - spos[i])) for i in candidates]
    return candidates[int(np.argmax(norms))]


def atom_distance(atoms, i, j):
    try:
        return float(atoms.get_distance(i, j, mic=True))
    except Exception:
        return float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))


def min_pair_distance(atoms):
    best = (math.inf, None, None, None, None)
    symbols = atoms.get_chemical_symbols()
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = atom_distance(atoms, i, j)
            if d < best[0]:
                best = (d, i, j, symbols[i], symbols[j])
    return best


def mic_displacement(a, b, idx):
    vec = b.get_positions()[idx] - a.get_positions()[idx]
    if a.cell.rank == 3 and any(a.pbc):
        ds = b.get_scaled_positions()[idx] - a.get_scaled_positions()[idx]
        ds -= np.rint(ds)
        vec = np.dot(ds, a.cell.array)
    return vec


def migrant_step(a, b, idx):
    try:
        return float(np.linalg.norm(mic_displacement(a, b, idx)))
    except Exception:
        return float(np.linalg.norm(b.positions[idx] - a.positions[idx]))


def all_atom_steps(a, b):
    cart = np.linalg.norm(b.get_positions() - a.get_positions(), axis=1)
    mic = []
    for idx in range(len(a)):
        try:
            mic.append(float(np.linalg.norm(mic_displacement(a, b, idx))))
        except Exception:
            mic.append(float(cart[idx]))
    return cart, np.asarray(mic, dtype=float)


def check(images, args):
    migrant = choose_migrant(images, args.migrant_element)
    symbols = images[0].get_chemical_symbols()
    host = [i for i, s in enumerate(symbols) if i != migrant]
    rows = []
    min_host = math.inf
    min_pair = math.inf
    pair_rows = []
    for image_index, atoms in enumerate(images):
        dmin = min(atom_distance(atoms, migrant, j) for j in host)
        min_host = min(min_host, dmin)
        pd, pi, pj, si, sj = min_pair_distance(atoms)
        min_pair = min(min_pair, pd)
        pair_rows.append({
            "image_index": image_index,
            "min_pair_distance_A": pd,
            "atom_i_zero_based": pi,
            "atom_j_zero_based": pj,
            "symbol_i": si,
            "symbol_j": sj,
        })
        rows.append({"image_index": image_index, "migrant_min_host_distance_A": dmin, "min_pair_distance_A": pd})
    steps = [migrant_step(a, b, migrant) for a, b in zip(images[:-1], images[1:])]
    all_cart_max = []
    all_mic_max = []
    worst_cart_atoms = []
    for image_pair_index, (a, b) in enumerate(zip(images[:-1], images[1:])):
        cart, mic = all_atom_steps(a, b)
        worst = int(np.argmax(cart))
        all_cart_max.append(float(cart[worst]))
        all_mic_max.append(float(mic.max()))
        worst_cart_atoms.append({
            "image_pair_index": image_pair_index,
            "atom_index_zero_based": worst,
            "symbol": a.get_chemical_symbols()[worst],
            "cartesian_step_A": float(cart[worst]),
            "mic_step_A": float(mic[worst]),
        })
    max_step = max(steps) if steps else 0.0
    max_all_cart = max(all_cart_max) if all_cart_max else 0.0
    max_all_mic = max(all_mic_max) if all_mic_max else 0.0
    total = sum(steps)
    endpoint = migrant_step(images[0], images[-1], migrant)
    status = "pass"
    reasons = []
    if max_step > args.max_migrant_step_A:
        status = "fail"
        reasons.append("max migrant inter-image step %.3f A exceeds %.3f A" % (max_step, args.max_migrant_step_A))
    if min_host < args.min_host_distance_A:
        status = "fail"
        reasons.append("min migrant-host distance %.3f A below %.3f A" % (min_host, args.min_host_distance_A))
    if min_pair < args.min_pair_distance_A:
        status = "fail"
        reasons.append("min all-pair distance %.3f A below %.3f A" % (min_pair, args.min_pair_distance_A))
    if total > args.max_total_migrant_path_A:
        status = "warn" if status == "pass" else status
        reasons.append("total migrant path %.3f A exceeds %.3f A" % (total, args.max_total_migrant_path_A))
    if max_all_cart > args.max_all_atom_cart_step_A:
        status = "fail"
        reasons.append("max all-atom Cartesian step %.3f A exceeds %.3f A; likely wrapped atoms or broken image mapping" % (max_all_cart, args.max_all_atom_cart_step_A))
    if max_all_mic > args.max_all_atom_mic_step_A:
        status = "warn" if status == "pass" else status
        reasons.append("max all-atom MIC step %.3f A exceeds %.3f A" % (max_all_mic, args.max_all_atom_mic_step_A))
    return {
        "status": status,
        "reasons": reasons,
        "n_images": len(images),
        "n_atoms": len(images[0]),
        "migrant_element": args.migrant_element,
        "migrant_index_zero_based": migrant,
        "max_migrant_step_A": max_step,
        "total_migrant_path_A": total,
        "endpoint_migrant_distance_A": endpoint,
        "min_migrant_host_distance_A": min_host,
        "min_pair_distance_A": min_pair,
        "min_pair_records": pair_rows,
        "max_all_atom_cartesian_step_A": max_all_cart,
        "max_all_atom_mic_step_A": max_all_mic,
        "worst_cartesian_atom_steps": worst_cart_atoms,
        "per_image": rows,
        "inter_image_migrant_steps_A": steps,
        "inter_image_all_atom_cartesian_max_A": all_cart_max,
        "inter_image_all_atom_mic_max_A": all_mic_max,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--migrant-element", default="Cr")
    parser.add_argument("--max-migrant-step-A", type=float, default=3.0)
    parser.add_argument("--max-total-migrant-path-A", type=float, default=18.0)
    parser.add_argument("--min-host-distance-A", type=float, default=1.4)
    parser.add_argument("--min-pair-distance-A", type=float, default=1.7)
    parser.add_argument("--max-all-atom-cart-step-A", type=float, default=8.0)
    parser.add_argument("--max-all-atom-mic-step-A", type=float, default=4.0)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    images = read(args.images, index=":")
    summary = check(images, args)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["images"] = str(Path(args.images).resolve())
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    if args.output_csv:
        fields = ["image_index", "migrant_min_host_distance_A", "min_pair_distance_A"]
        with Path(args.output_csv).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary["per_image"])
    print(json.dumps({k: summary[k] for k in ["status", "reasons", "max_migrant_step_A", "total_migrant_path_A", "min_migrant_host_distance_A", "min_pair_distance_A", "max_all_atom_cartesian_step_A", "max_all_atom_mic_step_A"]}, indent=2))
    if summary["status"] == "fail" or (args.fail_on_warn and summary["status"] == "warn"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
