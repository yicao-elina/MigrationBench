#!/usr/bin/env python3
"""Build an invariant, leakage-aware Cr site screening model.

The current site-search labels are MACE screening energies. This tool therefore
produces a proposal ranking, never a DFT-stability claim. Similar distorted
structures are grouped by their originating void family during validation.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_neb_path_topology import read_qe_image  # noqa: E402
from parse_qe_relax_output import parse_relax_out  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_cell(path):
    image = read_qe_image(path)
    if image["cell"] is None:
        raise ValueError(f"Reference input has no CELL_PARAMETERS: {path}")
    return np.asarray(image["cell"], dtype=float)


def minimum_image_vectors(center, positions, cell):
    vectors = np.asarray(positions, dtype=float) - np.asarray(center, dtype=float)
    fractional = vectors @ np.linalg.inv(cell)
    fractional -= np.rint(fractional)
    return fractional @ cell


def cutoff(distance, radius):
    if distance >= radius:
        return 0.0
    return 0.5 * (math.cos(math.pi * distance / radius) + 1.0)


def legendre_values(cosine):
    x = max(-1.0, min(1.0, cosine))
    return [
        1.0,
        x,
        0.5 * (3.0 * x * x - 1.0),
        0.5 * (5.0 * x**3 - 3.0 * x),
        0.125 * (35.0 * x**4 - 30.0 * x * x + 3.0),
    ]


def local_descriptor(image, cell, radial_cutoff=6.0, angular_cutoff=4.2):
    symbols = image["symbols"]
    cr_indices = [index for index, symbol in enumerate(symbols) if symbol == "Cr"]
    if len(cr_indices) != 1:
        raise ValueError("Expected exactly one Cr atom")
    cr_index = cr_indices[0]
    host_indices = [index for index in range(len(symbols)) if index != cr_index]
    host_symbols = [symbols[index] for index in host_indices]
    host_positions = [image["positions"][index] for index in host_indices]
    vectors = minimum_image_vectors(image["positions"][cr_index], host_positions, cell)
    distances = np.linalg.norm(vectors, axis=1)
    names, values = [], []

    centers = np.linspace(1.5, radial_cutoff, 19)
    width = 0.25
    for species in ("Sb", "Te"):
        mask = np.asarray([symbol == species for symbol in host_symbols])
        species_distances = distances[mask]
        for center in centers:
            value = sum(
                math.exp(-0.5 * ((distance - center) / width) ** 2)
                * cutoff(float(distance), radial_cutoff)
                for distance in species_distances
            )
            names.append(f"radial_{species}_{center:.2f}A")
            values.append(value)
        ordered = sorted(float(value) for value in species_distances)
        for rank in range(4):
            names.append(f"nearest_{species}_{rank + 1}_A")
            values.append(ordered[rank] if rank < len(ordered) else radial_cutoff)
        for radius in (2.8, 3.2, 3.6, 4.0):
            names.append(f"coord_{species}_{radius:.1f}A")
            values.append(sum(cutoff(float(distance), radius) for distance in species_distances))

    angular_mask = distances < angular_cutoff
    angular_vectors = vectors[angular_mask]
    angular_distances = distances[angular_mask]
    angular_symbols = [symbol for symbol, keep in zip(host_symbols, angular_mask) if keep]
    moments = {(a, b): np.zeros(5) for a, b in (("Sb", "Sb"), ("Sb", "Te"), ("Te", "Te"))}
    for first in range(len(angular_vectors)):
        for second in range(first + 1, len(angular_vectors)):
            pair = tuple(sorted((angular_symbols[first], angular_symbols[second])))
            if pair not in moments:
                continue
            cosine = float(
                np.dot(angular_vectors[first], angular_vectors[second])
                / (angular_distances[first] * angular_distances[second])
            )
            weight = cutoff(float(angular_distances[first]), angular_cutoff) * cutoff(
                float(angular_distances[second]), angular_cutoff
            )
            moments[pair] += weight * np.asarray(legendre_values(cosine))
    for pair in (("Sb", "Sb"), ("Sb", "Te"), ("Te", "Te")):
        for order, value in enumerate(moments[pair]):
            names.append(f"angular_{pair[0]}_{pair[1]}_P{order}")
            values.append(float(value))

    minimum_distance = float(np.min(distances))
    contact_penalty = sum(max(0.0, 2.3 - float(distance)) ** 2 for distance in distances)
    names.extend(["minimum_cr_host_distance_A", "contact_penalty_A2"])
    values.extend([minimum_distance, contact_penalty])
    return names, np.asarray(values, dtype=float)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first, second):
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


def void_family_groups(dedup_rows):
    families = defaultdict(set)
    union = UnionFind()
    for row in dedup_rows:
        family = row["input_label"].split("__", 1)[0]
        families[row["unique_id"]].add(family)
    for values in families.values():
        ordered = sorted(values)
        for value in ordered[1:]:
            union.union(ordered[0], value)
    return {
        site: min(union.find(family) for family in values)
        for site, values in families.items()
    }


def dft_label_status(input_path, output):
    if not output.exists():
        return "missing", None
    parsed = parse_relax_out(output, input_path)
    if output.stat().st_mtime < input_path.stat().st_mtime:
        return "historical_output_unlinked_to_current_input", parsed
    accepted = (
        parsed["job_done"]
        and parsed["bfgs_converged"]
        and parsed["scf_not_converged_count"] == 0
        and parsed["final_max_atom_force_eV_A"] is not None
        and parsed["final_max_atom_force_eV_A"] <= 0.05
    )
    return ("accepted_dft_local_minimum" if accepted else "dft_present_gate_failed"), parsed


def pairwise_order_accuracy(y_true, y_pred, minimum_delta=0.2):
    correct, total = 0, 0
    for first in range(len(y_true)):
        for second in range(first + 1, len(y_true)):
            true_delta = y_true[first] - y_true[second]
            if abs(true_delta) < minimum_delta:
                continue
            predicted_delta = y_pred[first] - y_pred[second]
            correct += (true_delta > 0) == (predicted_delta > 0)
            total += 1
    return correct / total if total else None


def grouped_oof(features, targets, groups, seed):
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    unique_groups = sorted(set(groups))
    folds = min(5, len(unique_groups))
    if folds < 3:
        raise ValueError("Need at least three independent void-family groups")
    predictions = np.full(len(targets), np.nan)
    uncertainties = np.full(len(targets), np.nan)
    split_index = np.full(len(targets), -1, dtype=int)
    for fold, (train, test) in enumerate(GroupKFold(n_splits=folds).split(features, targets, groups)):
        components = min(20, len(train) - 1, features.shape[1])
        kernel = ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(noise_level=0.05)
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, whiten=True, random_state=seed),
            GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, random_state=seed),
        )
        model.fit(features[train], targets[train])
        mean, std = model.predict(features[test], return_std=True)
        predictions[test], uncertainties[test], split_index[test] = mean, std, fold
    residual = targets - predictions
    stable_cutoff = float(np.quantile(targets, 0.20))
    predicted_stable = set(np.argsort(predictions)[: max(1, math.ceil(0.20 * len(targets)))])
    true_stable = set(np.where(targets <= stable_cutoff)[0])
    metrics = {
        "n_samples": len(targets),
        "n_void_family_groups": len(unique_groups),
        "n_group_folds": folds,
        "mae_eV": float(np.mean(np.abs(residual))),
        "rmse_eV": float(np.sqrt(np.mean(residual**2))),
        "spearman_rank_correlation": float(spearmanr(targets, predictions).statistic),
        "lowest_20pct_recall": len(predicted_stable & true_stable) / len(true_stable),
        "pairwise_order_accuracy_delta_gte_0p2eV": pairwise_order_accuracy(targets, predictions),
        "uncertainty_95pct_empirical_coverage": float(np.mean(np.abs(residual) <= 1.96 * uncertainties)),
    }
    return predictions, uncertainties, split_index, metrics


def diverse_low_energy_selection(features, energies, uncertainty, contact, count):
    energy_z = (energies - np.mean(energies)) / max(np.std(energies), 1.0e-12)
    uncertainty_z = (uncertainty - np.mean(uncertainty)) / max(np.std(uncertainty), 1.0e-12)
    score = energy_z + 0.35 * uncertainty_z + 5.0 * contact
    pool = list(np.argsort(score)[: min(len(score), max(count * 4, count))])
    standardized = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1.0e-12)
    selected = [pool.pop(0)]
    while pool and len(selected) < count:
        candidate = max(
            pool,
            key=lambda index: min(
                np.linalg.norm(standardized[index] - standardized[chosen]) for chosen in selected
            ) - 0.20 * score[index],
        )
        selected.append(candidate)
        pool.remove(candidate)
    return selected, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-table", type=Path, required=True)
    parser.add_argument("--qe-input-root", type=Path, required=True)
    parser.add_argument("--reference-cell-input", type=Path, required=True)
    parser.add_argument("--dedup-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recommendations", type=int, default=12)
    args = parser.parse_args()

    sites = read_csv(args.site_table)
    cell = reference_cell(args.reference_cell_input)
    family_groups = void_family_groups(read_csv(args.dedup_table))
    feature_names = None
    feature_rows, records, targets, groups = [], [], [], []
    compositions = set()
    for site in sites:
        raw_id = site["raw_site_id"]
        input_path = args.qe_input_root / raw_id / "relax.in"
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        image = read_qe_image(input_path)
        compositions.add(tuple(sorted((symbol, image["symbols"].count(symbol)) for symbol in set(image["symbols"]))))
        names, values = local_descriptor(image, cell)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("Descriptor schema drift")
        output_path = input_path.with_name("relax.out")
        status, parsed = dft_label_status(input_path, output_path)
        records.append(
            {
                **site,
                "void_family_group": family_groups.get(raw_id, raw_id),
                "local_input_path": str(input_path.resolve()),
                "local_input_sha256": sha256_file(input_path),
                "dft_relax_status": status,
                "input_mtime_utc": datetime.fromtimestamp(input_path.stat().st_mtime, timezone.utc).isoformat(),
                "output_mtime_utc": datetime.fromtimestamp(output_path.stat().st_mtime, timezone.utc).isoformat() if output_path.exists() else None,
                "dft_final_energy_eV": parsed["final_energy_eV"] if parsed else None,
                "dft_final_max_force_eV_A": parsed["final_max_atom_force_eV_A"] if parsed else None,
            }
        )
        feature_rows.append(values)
        targets.append(float(site["screening_energy_eV"]))
        groups.append(family_groups.get(raw_id, raw_id))
    if len(compositions) != 1:
        raise ValueError(f"Composition mismatch across candidate structures: {compositions}")

    features = np.vstack(feature_rows)
    targets = np.asarray(targets)
    predictions, uncertainty, fold, metrics = grouped_oof(features, targets, np.asarray(groups), args.seed)
    contact_index = feature_names.index("contact_penalty_A2")
    selected, scores = diverse_low_energy_selection(
        features, targets, uncertainty, features[:, contact_index], args.recommendations
    )
    selected_rank = {index: rank for rank, index in enumerate(selected, start=1)}
    enriched = []
    for index, (record, values) in enumerate(zip(records, feature_rows)):
        enriched.append(
            {
                **record,
                "oof_fold": int(fold[index]),
                "oof_predicted_mace_energy_eV": float(predictions[index]),
                "oof_predictive_std_eV": float(uncertainty[index]),
                "oof_residual_eV": float(targets[index] - predictions[index]),
                "proposal_score_lower_is_better": float(scores[index]),
                "recommended_rank": selected_rank.get(index),
                **{name: float(value) for name, value in zip(feature_names, values)},
            }
        )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "site_descriptors_and_oof_predictions.csv", enriched)
    write_csv(out_dir / "recommended_qe_sites.csv", [enriched[index] for index in selected])
    quality_gate = {
        "rmse_eV_lte_1p0": metrics["rmse_eV"] <= 1.0,
        "spearman_gte_0p5": metrics["spearman_rank_correlation"] >= 0.5,
        "lowest_20pct_recall_gte_0p5": metrics["lowest_20pct_recall"] >= 0.5,
        "independent_accepted_dft_labels_gte_20": sum(
            row["dft_relax_status"] == "accepted_dft_local_minimum" for row in records
        ) >= 20,
    }
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "mace_label_site_screening_proposer_not_dft_stability_model",
        "inputs": {
            "site_table": str(args.site_table.resolve()),
            "site_table_sha256": sha256_file(args.site_table),
            "dedup_table": str(args.dedup_table.resolve()),
            "dedup_table_sha256": sha256_file(args.dedup_table),
            "reference_cell_input": str(args.reference_cell_input.resolve()),
            "reference_cell_sha256": sha256_file(args.reference_cell_input),
        },
        "descriptor": {
            "name": "Cr_centered_species_resolved_radial_angular_invariant_v1",
            "dimension": len(feature_names),
            "periodic_minimum_image": True,
            "radial_cutoff_A": 6.0,
            "angular_cutoff_A": 4.2,
            "translation_rotation_permutation_invariant": True,
        },
        "validation": {
            "split_unit": "connected_source_void_family",
            "metrics": metrics,
            "quality_gate": quality_gate,
        },
        "counts": {
            "candidate_sites": len(records),
            "recommendations": len(selected),
            "dft_outputs_present": sum(row["dft_relax_status"] != "missing" for row in records),
            "historical_dft_outputs_unlinked_to_current_input": sum(
                row["dft_relax_status"] == "historical_output_unlinked_to_current_input"
                for row in records
            ),
            "accepted_dft_local_minima": sum(
                row["dft_relax_status"] == "accepted_dft_local_minimum" for row in records
            ),
        },
        "decision": {
            "screening_ranking_available": True,
            "dft_stability_model_allowed": all(quality_gate.values()),
            "qe_submission_allowed": False,
            "reason": "Current target is MACE screening energy and independent accepted DFT labels are insufficient.",
        },
        "seed": args.seed,
    }
    (out_dir / "proposer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "quality_gate": quality_gate, "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
