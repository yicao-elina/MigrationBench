#!/usr/bin/env python3
"""Analyze model disagreement without treating any model as a DFT reference."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from ase.io import read


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metrics(delta):
    x = np.asarray(delta, dtype=float).ravel()
    return {"mae": float(np.mean(np.abs(x))), "rmse": float(np.sqrt(np.mean(x * x))), "max_abs": float(np.max(np.abs(x))), "n": int(x.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-manifest", required=True, type=Path)
    ap.add_argument("--evaluation-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--local-radius", type=float, default=3.0)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.snapshot_manifest.read_text())
    records = [json.loads(x) for x in (args.evaluation_dir / "evaluation_records.jsonl").read_text().splitlines() if x.strip()]
    complete = [r for r in records if r["status"] == "complete"]
    by_step = {}
    for r in complete:
        by_step.setdefault(int(r["step"]), {})[r["model_id"]] = r
    model_ids = sorted({r["model_id"] for r in complete})
    steps = sorted(by_step)
    summary_rows = []
    pair_rows = []
    key_rows = []
    local_rows = []
    geometry_rows = []
    all_model_rows = []

    for step in steps:
        step_models = by_step[step]
        arrays = {}
        for m in step_models:
            raw = Path(step_models[m]["raw_output_npz"])
            if not raw.exists():
                raw = args.evaluation_dir / "npz" / raw.name
            arrays[m] = np.load(raw, allow_pickle=False)
        common = sorted(step_models)
        forces = np.stack([arrays[m]["forces"] for m in common])
        energies = np.array([float(arrays[m]["energy_eV"]) for m in common])
        stresses = np.stack([arrays[m]["stress"] for m in common])
        force_norms = np.linalg.norm(forces, axis=2)
        mean_force = forces.mean(axis=0)
        force_spread = np.max(np.linalg.norm(forces[:, :, :] - mean_force[None, :, :], axis=2), axis=0)
        pairwise_spread = np.max(np.linalg.norm(forces[:, None, :, :] - forces[None, :, :, :], axis=3), axis=(0, 1))
        # Pairwise model metrics are the primary model-independent comparison.
        for a, b in combinations(common, 2):
            d = forces[common.index(a)] - forces[common.index(b)]
            pair_rows.append({"step": step, "model_a": a, "model_b": b, "energy_a_eV": float(energies[common.index(a)]), "energy_b_eV": float(energies[common.index(b)]), "energy_abs_diff_eV": float(abs(energies[common.index(a)] - energies[common.index(b)])), "force_component_mae_eV_A": metrics(d)["mae"], "force_component_rmse_eV_A": metrics(d)["rmse"], "force_component_max_abs_eV_A": metrics(d)["max_abs"], "force_vector_mean_norm_eV_A": float(np.linalg.norm(d, axis=1).mean()), "force_vector_max_norm_eV_A": float(np.linalg.norm(d, axis=1).max()), "stress_frobenius_diff_eV_A3": float(np.linalg.norm(stresses[common.index(a)] - stresses[common.index(b)])), "stress_frobenius_diff_GPa": float(np.linalg.norm(stresses[common.index(a)] - stresses[common.index(b)]) * 160.21766208)})
        cr = np.array([i - 1 for i in manifest["identity"]["key_atom_indices_one_based"]], dtype=int)
        top_atom = int(np.argmax(pairwise_spread))
        positions = arrays[common[0]]["positions"]
        cell = arrays[common[0]]["cell"]
        # Fractional MIC distances to define the local disagreement environment.
        frac = positions @ np.linalg.inv(cell)
        df = frac - frac[top_atom]
        df -= np.rint(df)
        local_dist = np.linalg.norm(df @ cell, axis=1)
        local = local_dist <= args.local_radius
        for i in range(len(positions)):
            species = str(arrays[common[0]]["species"][i])
            key_rows.append({"step": step, "atom_index_one_based": i + 1, "species": species, "is_key_Cr": bool(i in set(cr.tolist())), "force_spread_max_pair_eV_A": float(pairwise_spread[i]), "force_spread_ensemble_eV_A": float(force_spread[i]), "force_norm_mean_eV_A": float(force_norms[:, i].mean()), "distance_to_top_disagreement_A": float(local_dist[i]), "in_top_local_shell": bool(local[i])})
        for m in common:
            mi = common.index(m)
            other = np.delete(forces, mi, axis=0).mean(axis=0)
            d = forces[mi] - other
            summary_rows.append({"step": step, "model_id": m, "n_models": len(common), "energy_eV": float(energies[mi]), "energy_minus_ensemble_mean_eV": float(energies[mi] - energies.mean()), "force_norm_mean_eV_A": float(force_norms[mi].mean()), "force_norm_max_eV_A": float(force_norms[mi].max()), "force_vs_other_models_mae_eV_A": metrics(d)["mae"], "force_vs_other_models_rmse_eV_A": metrics(d)["rmse"], "force_vs_other_models_max_abs_eV_A": metrics(d)["max_abs"], "key_Cr_force_norm_mean_eV_A": float(force_norms[mi, cr].mean()), "key_Cr_force_norm_max_eV_A": float(force_norms[mi, cr].max()), "stress_xx_GPa": float(stresses[mi, 0, 0] * 160.21766208), "stress_yy_GPa": float(stresses[mi, 1, 1] * 160.21766208), "stress_zz_GPa": float(stresses[mi, 2, 2] * 160.21766208)})
            all_model_rows.append({"step": step, "model_id": m, "force_component_mae_vs_other_eV_A": metrics(d)["mae"], "force_component_rmse_vs_other_eV_A": metrics(d)["rmse"], "force_component_max_abs_vs_other_eV_A": metrics(d)["max_abs"]})
        summary_rows.append({"step": step, "model_id": "ensemble", "n_models": len(common), "energy_eV": float(energies.mean()), "energy_minus_ensemble_mean_eV": 0.0, "force_norm_mean_eV_A": float(np.linalg.norm(mean_force, axis=1).mean()), "force_norm_max_eV_A": float(np.linalg.norm(mean_force, axis=1).max()), "force_vs_other_models_mae_eV_A": float(np.mean(np.abs(forces - mean_force[None, ...]))), "force_vs_other_models_rmse_eV_A": float(np.sqrt(np.mean((forces - mean_force[None, ...]) ** 2))), "force_vs_other_models_max_abs_eV_A": float(np.max(np.abs(forces - mean_force[None, ...]))), "key_Cr_force_norm_mean_eV_A": float(np.linalg.norm(mean_force[cr], axis=1).mean()), "key_Cr_force_norm_max_eV_A": float(np.linalg.norm(mean_force[cr], axis=1).max()), "stress_xx_GPa": float(stresses[:, 0, 0].mean() * 160.21766208), "stress_yy_GPa": float(stresses[:, 1, 1].mean() * 160.21766208), "stress_zz_GPa": float(stresses[:, 2, 2].mean() * 160.21766208)})
        geometry_rows.append({"step": step, "natoms": int(len(positions)), "cell_volume_A3": float(abs(np.linalg.det(cell))), "min_pair_distance_A": float(np.min(np.linalg.norm(((frac[:, None, :] - frac[None, :, :] + 0.5) % 1.0 - 0.5) @ cell, axis=2) + np.eye(len(positions)) * 1e9)), "top_disagreement_atom_one_based": top_atom + 1, "top_disagreement_atom_species": str(arrays[common[0]]["species"][top_atom]), "top_force_pairwise_spread_eV_A": float(pairwise_spread[top_atom]), "local_shell_atom_count_radius_A": int(local.sum()), "key_Cr_count": int(len(cr))})

    write_csv(out / "model_step_summary.csv", summary_rows)
    write_csv(out / "pairwise_force_comparison.csv", pair_rows)
    write_csv(out / "atom_force_disagreement.csv", key_rows)
    write_csv(out / "geometry_diagnostics.csv", geometry_rows)
    write_csv(out / "model_loo_disagreement.csv", all_model_rows)
    top_rows = sorted(key_rows, key=lambda r: (r["step"], -float(r["force_spread_max_pair_eV_A"])))
    write_csv(out / "top_disagreement_atoms.csv", [r for r in top_rows if int(r["atom_index_one_based"]) in {int(x["top_disagreement_atom_one_based"]) for x in geometry_rows}][:120])
    result = {"schema_version": "phase1-static-force-analysis-1.0", "source": manifest["source_path"], "source_sha256": manifest["source_sha256"], "steps": steps, "models_evaluated": model_ids, "evaluations_complete": len(complete), "evaluations_requested": 16, "key_atom_definition": "all Cr atoms from verified source species order; 1-based indices in snapshot_manifest.json", "reference_status": "No DFT reference; metrics quantify pairwise/ensemble model disagreement only.", "stress_status": "Stress comparisons use finite model stress tensors when available; units converted from eV/A^3 to GPa with 160.21766208.", "local_disagreement_definition": f"Atoms within {args.local_radius} A MIC of the atom with the largest pairwise force-vector spread at each step.", "files": {"summary": str(out / "model_step_summary.csv"), "pairwise": str(out / "pairwise_force_comparison.csv"), "atom": str(out / "atom_force_disagreement.csv"), "geometry": str(out / "geometry_diagnostics.csv")}}
    (out / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
