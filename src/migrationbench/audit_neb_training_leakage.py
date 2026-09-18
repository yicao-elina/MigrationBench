#!/usr/bin/env python3
"""Audit NEB evaluation structures against MLFF train/valid/test geometries."""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from scipy.optimize import linear_sum_assignment


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def composition_key(atoms):
    return tuple(sorted(Counter(atoms.get_chemical_symbols()).items()))


def structure_fingerprint(atoms, decimals=5):
    symbols = atoms.get_chemical_symbols()
    distances = atoms.get_all_distances(mic=True)
    pairs = defaultdict(list)
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            pair = tuple(sorted((symbols[left], symbols[right])))
            pairs[pair].append(round(float(distances[left, right]), decimals))
    payload = {
        "composition": composition_key(atoms),
        "cell_metric": np.round(np.asarray(atoms.cell) @ np.asarray(atoms.cell).T, decimals).tolist(),
        "pair_distances": {"-".join(pair): sorted(values) for pair, values in sorted(pairs.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def periodic_fractional_from_anchor(atoms, anchor_index):
    fractional = atoms.get_scaled_positions(wrap=False)
    delta = fractional - fractional[anchor_index]
    delta -= np.round(delta)
    return delta


def local_environment(atoms, anchor_index, cutoff):
    source, neighbor, vectors = neighbor_list("ijD", atoms, cutoff)
    keep = source == anchor_index
    symbols = np.asarray(atoms.get_chemical_symbols())
    local_symbols = np.concatenate(([symbols[anchor_index]], symbols[neighbor[keep]]))
    local_vectors = np.vstack((np.zeros(3), vectors[keep]))
    return local_symbols, local_vectors


def local_environment_fingerprint(atoms, anchor_index, cutoff, decimals=5):
    symbols, vectors = local_environment(atoms, anchor_index, cutoff)
    pairs = defaultdict(list)
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            pair = tuple(sorted((str(symbols[left]), str(symbols[right]))))
            pairs[pair].append(round(float(np.linalg.norm(vectors[left] - vectors[right])), decimals))
    payload = {
        "composition": tuple(sorted(Counter(symbols).items())),
        "pair_distances": {"-".join(pair): sorted(values) for pair, values in sorted(pairs.items())},
        "cutoff_A": cutoff,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cr_local_species_rmsd(left, left_cr, right, right_cr, cutoff):
    left_symbols, left_vectors = local_environment(left, left_cr, cutoff)
    right_symbols, right_vectors = local_environment(right, right_cr, cutoff)
    if Counter(left_symbols) != Counter(right_symbols):
        return None
    squared = []
    for symbol in sorted(set(left_symbols)):
        left_indices = np.flatnonzero(left_symbols == symbol)
        right_indices = np.flatnonzero(right_symbols == symbol)
        difference = left_vectors[left_indices, None, :] - right_vectors[None, right_indices, :]
        cost = np.sum(difference**2, axis=2)
        row, column = linear_sum_assignment(cost)
        squared.extend(cost[row, column].tolist())
    return math.sqrt(sum(squared) / len(squared))


def soap_vectors(centered_structures, soap):
    vectors = []
    for atoms, center in centered_structures:
        vector = np.asarray(soap.create(atoms, centers=[center])[0], dtype=float)
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
    return np.asarray(vectors)


def quantiles(values):
    if not values:
        return {"min": None, "p01": None, "p05": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def parse_eval(value):
    if "=" not in value:
        raise ValueError("--eval must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def read_frames(path, limit=None):
    return read(path, index=f":{limit}" if limit else ":")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--eval", action="append", required=True, help="LABEL=EXTXYZ")
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--soap-r-cut", type=float, default=6.0)
    parser.add_argument("--soap-n-max", type=int, default=6)
    parser.add_argument("--soap-l-max", type=int, default=4)
    parser.add_argument("--exact-rmsd-A", type=float, default=0.01)
    parser.add_argument("--near-rmsd-A", type=float, default=0.05)
    parser.add_argument("--near-soap-cosine", type=float, default=1.0e-4)
    parser.add_argument("--rmsd-candidate-count", type=int, default=10)
    parser.add_argument("--local-cutoff-A", type=float, default=6.0)
    parser.add_argument("--max-reference-frames", type=int)
    parser.add_argument("--max-eval-frames", type=int)
    args = parser.parse_args()

    splits = {
        name: read_frames(path, args.max_reference_frames)
        for name, path in [("train", args.train), ("valid", args.valid), ("test", args.test)]
    }
    evaluations = []
    for value in args.eval:
        label, path = parse_eval(value)
        frames = read_frames(path, args.max_eval_frames)
        for frame_index, atoms in enumerate(frames):
            evaluations.append((label, frame_index, atoms))
    if not evaluations:
        raise ValueError("No evaluation structures")
    eval_centers = []
    for label, frame_index, atoms in evaluations:
        cr = [index for index, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Cr"]
        if len(cr) != 1:
            raise ValueError(f"Expected one Cr in evaluation {label} frame {frame_index}; found {len(cr)}")
        eval_centers.append((label, frame_index, atoms, cr[0]))
    reference_centers = {
        name: [
            (frame_index, cr_index, atoms)
            for frame_index, atoms in enumerate(rows)
            for cr_index, symbol in enumerate(atoms.get_chemical_symbols())
            if symbol == "Cr" and len(atoms) > 1
        ]
        for name, rows in splits.items()
    }
    if not reference_centers["train"]:
        raise ValueError("No Cr-centered periodic training environments")

    soap = SOAP(
        species=sorted({symbol for _, _, atoms in evaluations for symbol in atoms.get_chemical_symbols()}),
        periodic=True,
        r_cut=args.soap_r_cut,
        n_max=args.soap_n_max,
        l_max=args.soap_l_max,
        sparse=False,
        average="off",
    )
    eval_soap = soap_vectors([(atoms, cr) for _, _, atoms, cr in eval_centers], soap)
    eval_fingerprints = [structure_fingerprint(atoms) for _, _, atoms, _ in eval_centers]
    eval_local_fingerprints = [
        local_environment_fingerprint(atoms, cr, args.local_cutoff_A)
        for _, _, atoms, cr in eval_centers
    ]

    rows = []
    summaries = {}
    for split_name, centered_references in reference_centers.items():
        reference_indices = [index for index, _, _ in centered_references]
        reference_cr_indices = [cr for _, cr, _ in centered_references]
        reference_atoms = [atoms for _, _, atoms in centered_references]
        reference_soap = soap_vectors(
            [(atoms, cr) for _, cr, atoms in centered_references], soap
        )
        similarity = eval_soap @ reference_soap.T
        candidate_count = min(args.rmsd_candidate_count, len(reference_atoms))
        candidate_neighbors = np.argpartition(
            -similarity, kth=candidate_count - 1, axis=1
        )[:, :candidate_count]
        reference_fingerprints = defaultdict(list)
        for local_index, atoms in enumerate(reference_atoms):
            reference_fingerprints[structure_fingerprint(atoms)].append(local_index)
        reference_local_fingerprints = defaultdict(list)
        for local_index, (_, cr, atoms) in enumerate(centered_references):
            reference_local_fingerprints[
                local_environment_fingerprint(atoms, cr, args.local_cutoff_A)
            ].append(local_index)
        for eval_index, (label, frame_index, atoms, eval_cr) in enumerate(eval_centers):
            candidates = candidate_neighbors[eval_index]
            soap_reference = int(candidates[np.argmax(similarity[eval_index, candidates])])
            soap_distance = max(0.0, 1.0 - float(similarity[eval_index, soap_reference]))
            rmsd_candidates = [
                (
                    cr_local_species_rmsd(
                        atoms,
                        eval_cr,
                        reference_atoms[int(index)],
                        reference_cr_indices[int(index)],
                        args.local_cutoff_A,
                    ),
                    int(index),
                )
                for index in candidates
            ]
            rmsd_candidates = [item for item in rmsd_candidates if item[0] is not None]
            rmsd, rmsd_reference = (
                min(rmsd_candidates, key=lambda item: item[0])
                if rmsd_candidates
                else (None, None)
            )
            exact_indices = reference_fingerprints.get(eval_fingerprints[eval_index], [])
            exact_local_indices = reference_local_fingerprints.get(
                eval_local_fingerprints[eval_index], []
            )
            rows.append(
                {
                    "evaluation_pathway": label,
                    "evaluation_frame_index": frame_index,
                    "path_iteration": atoms.info.get("path_iteration"),
                    "image_index": atoms.info.get("image_index"),
                    "reference_split": split_name,
                    "soap_reference_frame_index": reference_indices[soap_reference],
                    "soap_reference_cr_index": reference_cr_indices[soap_reference],
                    "soap_cosine_distance": soap_distance,
                    "rmsd_reference_frame_index": reference_indices[rmsd_reference] if rmsd_reference is not None else None,
                    "rmsd_reference_cr_index": reference_cr_indices[rmsd_reference] if rmsd_reference is not None else None,
                    "cr_local_species_rmsd_A": rmsd,
                    "exact_global_fingerprint_match": bool(exact_indices),
                    "exact_reference_frame_indices": [reference_indices[index] for index in exact_indices],
                    "exact_local_fingerprint_match": bool(exact_local_indices),
                    "exact_local_reference_centers": [
                        [reference_indices[index], reference_cr_indices[index]]
                        for index in exact_local_indices
                    ],
                }
            )
        split_rows = [row for row in rows if row["reference_split"] == split_name]
        summaries[split_name] = {
            "total_frames": len(splits[split_name]),
            "periodic_frames_with_cr": len({index for index, _, _ in centered_references}),
            "cr_centered_reference_environments": len(reference_atoms),
            "exact_global_fingerprint_matches": sum(row["exact_global_fingerprint_match"] for row in split_rows),
            "exact_local_fingerprint_matches": sum(row["exact_local_fingerprint_match"] for row in split_rows),
            "local_rmsd_comparable_frames": sum(row["cr_local_species_rmsd_A"] is not None for row in split_rows),
            "rmsd_lte_exact_threshold": sum(row["cr_local_species_rmsd_A"] is not None and row["cr_local_species_rmsd_A"] <= args.exact_rmsd_A for row in split_rows),
            "rmsd_lte_near_threshold": sum(row["cr_local_species_rmsd_A"] is not None and row["cr_local_species_rmsd_A"] <= args.near_rmsd_A for row in split_rows),
            "soap_lte_near_threshold": sum(row["soap_cosine_distance"] <= args.near_soap_cosine for row in split_rows),
            "soap_cosine_distance": quantiles([row["soap_cosine_distance"] for row in split_rows]),
            "cr_local_species_rmsd_A": quantiles([row["cr_local_species_rmsd_A"] for row in split_rows if row["cr_local_species_rmsd_A"] is not None]),
        }

    train = summaries["train"]
    leakage_fail = (
        train["exact_global_fingerprint_matches"] > 0
        or train["exact_local_fingerprint_matches"] > 0
        or train["rmsd_lte_exact_threshold"] > 0
    )
    status = "fail_detected_training_overlap" if leakage_fail else "pass_no_detected_training_overlap_under_declared_metrics"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "nearest_neighbors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["exact_reference_frame_indices"] = json.dumps(output["exact_reference_frame_indices"])
            writer.writerow(output)
    provenance = {
        "model_checkpoint": file_record(args.model_checkpoint),
        "splits": {name: file_record(path) for name, path in [("train", args.train), ("valid", args.valid), ("test", args.test)]},
        "evaluations": {label: file_record(path) for label, path in map(parse_eval, args.eval)},
    }
    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "geometry leakage from target NEB evaluation frames into the checkpoint's declared fine-tuning train/valid/test files",
        "evaluation_compositions": [list(value) for value in sorted({composition_key(atoms) for _, _, atoms in evaluations})],
        "evaluation_frames": len(evaluations),
        "metrics": {
            "exact_fingerprint": "composition + cell metric + all species-pair MIC distances rounded to 1e-5 A",
            "exact_local_fingerprint": f"Cr-centered environment within {args.local_cutoff_A} A + all local species-pair distances rounded to 1e-5 A",
            "soap": {"center": "Cr", "r_cut_A": args.soap_r_cut, "n_max": args.soap_n_max, "l_max": args.soap_l_max, "distance": "cosine"},
            "rmsd": f"Cr-centered environments within {args.local_cutoff_A} A with per-species Hungarian assignment; no rotational alignment because cells share orientation",
            "rmsd_candidate_selection": f"minimum RMSD among the {args.rmsd_candidate_count} nearest Cr-centered SOAP neighbors",
        },
        "thresholds": {
            "exact_rmsd_A": args.exact_rmsd_A,
            "near_rmsd_A": args.near_rmsd_A,
            "near_soap_cosine": args.near_soap_cosine,
        },
        "summaries": summaries,
        "provenance": provenance,
        "limitations": [
            "A pass is bounded to the declared checkpoint files and metrics, not a claim about unknown pretraining corpora.",
            "SOAP-near alone is reported as distribution proximity and is not treated as leakage without an exact/low-RMSD match.",
        ],
    }
    (out_dir / "leakage_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": status, "evaluation_frames": len(evaluations), "summaries": summaries}, indent=2))
    if leakage_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
