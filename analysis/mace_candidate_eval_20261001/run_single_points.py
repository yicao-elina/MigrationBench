from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from ase import __version__ as ase_version
from ase.io import read


FORCE_LIMIT = 5.0
MIN_DISTANCE_LIMIT = 0.8


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def software() -> dict:
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = "unavailable"
    try:
        import mace
        mace_v = getattr(mace, "__version__", "unknown")
    except Exception:
        mace_v = "unavailable"
    return {"python": sys.version.split()[0], "ase": ase_version, "torch": torch_v,
            "mace_torch": mace_v, "hostname": socket.gethostname(),
            "platform": platform.platform(), "git_head": os.environ.get("GIT_HEAD", "unknown")}


def geom(atoms):
    pos = np.asarray(atoms.positions, float)
    finite = bool(np.isfinite(pos).all())
    d = atoms.get_all_distances(mic=bool(np.any(atoms.pbc))) if finite else np.full((len(atoms), len(atoms)), np.nan)
    d[d == 0] = np.nan
    min_d = float(np.nanmin(d)) if np.isfinite(d).any() else np.nan
    cr = np.where(np.array(atoms.get_chemical_symbols()) == "Cr")[0]
    cr_i = int(cr[0]) if len(cr) else -1
    cr_pos = pos[cr_i] if cr_i >= 0 else np.array([np.nan, np.nan, np.nan])
    neigh = int(np.sum(d[cr_i] < 3.2)) if cr_i >= 0 else -1
    syms = np.array(atoms.get_chemical_symbols())
    te_neigh = int(np.sum((d[cr_i] < 3.2) & (syms == "Te"))) if cr_i >= 0 else -1
    sb_neigh = int(np.sum((d[cr_i] < 3.2) & (syms == "Sb"))) if cr_i >= 0 else -1
    return {"geometry_finite": finite, "min_interatomic_distance_A": min_d,
            "cr_x_A": float(cr_pos[0]), "cr_y_A": float(cr_pos[1]), "cr_z_A": float(cr_pos[2]),
            "cr_coordination_3p2_A": neigh, "cr_te_coordination_3p2_A": te_neigh,
            "cr_sb_coordination_3p2_A": sb_neigh, "n_atoms": len(atoms),
            "cell_volume_A3": float(abs(atoms.get_volume())) if atoms.cell.rank == 3 else np.nan,
            "pbc": json.dumps(atoms.pbc.tolist())}


def classify(row: dict) -> str:
    if row.get("runtime_error"): return "runtime_failure"
    if not row.get("geometry_finite", False) or not np.isfinite(row.get("energy_eV", np.nan)) or not np.isfinite(row.get("max_force_eV_A", np.nan)):
        return "nonfinite_energy_or_force"
    if np.isfinite(row.get("min_interatomic_distance_A", np.nan)) and row["min_interatomic_distance_A"] < MIN_DISTANCE_LIMIT:
        return "geometry_flying_or_atomic_overlap"
    if row.get("max_force_eV_A", np.nan) > FORCE_LIMIT:
        return "force_above_5_eV_A"
    return "success"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", required=True, help="TSV with model_id,model_label,model_path,model_family,seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    models = pd.read_csv(args.models, sep="\t")
    sw = software()
    results, failures = [], []
    for mr in models.to_dict("records"):
        mp = Path(str(mr["model_path"]))
        model_sha = sha256(mp) if mp.is_file() else "missing"
        for cr in manifest.to_dict("records"):
            base = {"candidate_id": cr["candidate_id"], "category": cr["category"], "mechanism_group": cr["mechanism_group"],
                    "endpoint_id": cr.get("endpoint_id", ""), "cov_id": cr.get("cov_id", ""), "image_index": cr.get("image_index", ""),
                    "model_id": mr["model_id"], "model_label": mr["model_label"], "model_family": mr.get("model_family", ""), "seed": mr.get("seed", ""),
                    "model_path": str(mp), "model_sha256": model_sha, "input_path": cr["input_path"], "input_sha256": cr["input_sha256"],
                    "software_json": json.dumps(sw, sort_keys=True), "runtime_error": "", "traceback": ""}
            row = dict(base)
            try:
                if not mp.is_file(): raise FileNotFoundError(str(mp))
                atoms = read(cr["input_path"], index=0)
                row.update(geom(atoms))
                from mace.calculators import MACECalculator
                atoms.calc = MACECalculator(model_paths=str(mp), device=args.device, default_dtype="float64")
                e = float(atoms.get_potential_energy())
                f = np.asarray(atoms.get_forces(), float)
                row.update({"energy_eV": e, "energy_per_atom_eV": e / len(atoms), "max_force_eV_A": float(np.max(np.linalg.norm(f, axis=1))),
                            "rms_force_eV_A": float(np.sqrt(np.mean(f * f))), "forces_finite": bool(np.isfinite(f).all())})
            except Exception as exc:
                row["runtime_error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=8)
            row["status"] = classify(row)
            results.append(row)
            if row["status"] != "success":
                failures.append({**row, "failure_class": row["status"]})
    rdf = pd.DataFrame(results)
    fdf = pd.DataFrame(failures)
    rdf.to_csv(out / "mace_single_point_results.csv", index=False)
    fdf.to_csv(out / "failure_ledger.csv", index=False)
    # Comparison is within candidate/model ensemble only: no absolute-energy cross-model ranking is emitted.
    ok = rdf[rdf.status.eq("success")].copy()
    if not ok.empty:
        summary = ok.groupby(["candidate_id", "category"], as_index=False).agg(
            n_models=("model_id", "nunique"), n_success=("status", "size"),
            energy_spread_within_candidate_eV=("energy_eV", lambda x: float(x.max() - x.min())),
            max_force_median_eV_A=("max_force_eV_A", "median"), max_force_max_eV_A=("max_force_eV_A", "max"),
            coordination_min=("cr_coordination_3p2_A", "min"), coordination_max=("cr_coordination_3p2_A", "max"),
            cr_z_spread_A=("cr_z_A", lambda x: float(x.max() - x.min())),
        )
        summary["interpretation"] = "within-candidate model spread; absolute total energies are not comparable across model families"
    else:
        summary = pd.DataFrame()
    summary.to_csv(out / "model_comparison_summary.csv", index=False)
    (out / "runtime_manifest.json").write_text(json.dumps({"manifest": str(args.manifest), "models": str(args.models), "software": sw, "n_rows": len(rdf), "n_failures": len(fdf)}, indent=2))
    print(rdf.groupby(["model_label", "status"]).size().to_string())


if __name__ == "__main__":
    main()
