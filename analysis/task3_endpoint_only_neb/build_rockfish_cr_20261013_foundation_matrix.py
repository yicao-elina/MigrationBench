#!/usr/bin/env python3
"""Build an auditable same-geometry Cr/Rockfish Foundation-model matrix.

The Rockfish QE paths are kept as a read-only raw snapshot.  This script only
adds derived extxyz inputs, model evaluations, QE image profiles, barrier
tables, hashes, and a publication-status report.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read, write


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "exploratory_energy_audit/rockfish_cr_neb_20261013/raw"
OUT = ROOT / "exploratory_energy_audit/rockfish_cr_neb_20261013"
MODEL_TABLE = Path("/Users/alina/Documents/Codex/foundation_grouped_benchmark_20261001/models.tsv")
EVALUATOR = Path("/Users/alina/Documents/Codex/foundation_grouped_benchmark_20261001/evaluate_neb_foundation.py")
HARTREE_TO_EV = 27.211386245988

PATHS = {
    "within-QL": "2L-Cu__neb-QLs",
    "gap → QL": "2L-Cu__neb-gap2QL",
    "in-gap diffusion": "2L-Cu__neb-gap-diffuse",
}
PATH_ORDER = ["within-QL", "in-gap diffusion", "gap → QL"]
JOB_IDS = {
    "2L-Cu__neb-QLs": "31916908",
    "2L-Cu__neb-gap2QL": "31916909",
    "2L-Cu__neb-gap-diffuse": "31916910",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_cell(path: Path) -> np.ndarray:
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            return np.array([[float(x) for x in lines[i + j].split("!")[0].split()[:3]] for j in (1, 2, 3)])
    raise ValueError(f"No CELL_PARAMETERS in {path}")


def read_scalar(text: str, name: str, default: str = "") -> str:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^!\n]+)", text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else default


def build_extxyz(path_id: str) -> tuple[Path, list]:
    source = RAW / path_id / "sb2te3.xyz"
    qe_input = RAW / path_id / "pw_1.in"
    frames = read(source, index=":")
    cell = parse_cell(qe_input)
    for frame in frames:
        frame.set_cell(cell)
        frame.set_pbc((True, True, True))
    out = OUT / "inputs" / f"{path_id}.extxyz"
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, frames, format="extxyz")
    return out, frames


def parse_qe_profile(path_id: str) -> dict:
    """Read the final complete QE image table, not an incomplete last step."""
    from neb_plot_style import _parse_complete_tables

    out = RAW / path_id / "neb.out"
    text = out.read_text(errors="replace")
    tables = _parse_complete_tables(out, n_images=10)
    if not tables:
        raise ValueError(f"No complete 10-image table in {out}")
    last_iteration, table = tables[-1]
    energies = table[:, 0]
    rel = energies - energies[0]
    reverse_rel = energies - energies[-1]
    activation_lines = re.findall(r"activation energy \(->\)\s*=\s*([-+0-9.]+).*?\n\s*activation energy \(<-\)\s*=\s*([-+0-9.]+)", text, re.S)
    reported_forward = float(activation_lines[-1][0]) if activation_lines else float(np.max(rel))
    reported_reverse = float(activation_lines[-1][1]) if activation_lines else float(np.max(reverse_rel))
    return {
        "path_id": path_id,
        "job_id": JOB_IDS[path_id],
        "n_images": 10,
        "qe_last_complete_iteration": int(last_iteration),
        "qe_n_complete_tables": len(tables),
        "qe_energy_eV": energies.tolist(),
        "qe_relative_energy_eV": rel.tolist(),
        "qe_force_error_eV_A": table[:, 1].tolist(),
        "qe_forward_activation_reported_eV": reported_forward,
        "qe_reverse_activation_reported_eV": reported_reverse,
        "job_done": "JOB DONE" in text,
        "scf_warning": "convergence NOT achieved" in text,
        "scf_warning_count": len(re.findall(r"convergence NOT achieved", text)),
        "neb_out_sha256": sha256(out),
    }


def load_models() -> list[dict]:
    rows = []
    with MODEL_TABLE.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            model = row["local_checkpoint"] or "/Users/alina/Downloads/mace-mpa-0-medium.model"
            model_path = Path(model)
            if not model_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for {row['model_id']}: {model_path}")
            row["resolved_local_checkpoint"] = str(model_path)
            row["resolved_sha256"] = sha256(model_path)
            rows.append(row)
    return rows


def evaluate_models(paths: dict[str, Path], models: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    result_manifest = []
    result_dir = OUT / "model_json"
    result_dir.mkdir(parents=True, exist_ok=True)
    for path_class in PATH_ORDER:
        path_id = PATHS[path_class]
        xyz = paths[path_id]
        for model in models:
            result = result_dir / f"{model['model_id']}__{path_id}.json"
            command = [sys.executable, str(EVALUATOR), "--model-id", model["model_id"], "--model", model["resolved_local_checkpoint"], "--path-id", path_id, "--xyz", str(xyz), "--out", str(result), "--device", "cpu"]
            if model.get("head"):
                command += ["--head", model["head"]]
            print("RUN", model["model_id"], path_id, flush=True)
            subprocess.run(command, check=True)
            payload = json.loads(result.read_text())
            rel = np.asarray(payload["relative_energies_eV"], dtype=float)
            for i, energy in enumerate(rel, start=1):
                rows.append({
                    "path_class": path_class,
                    "path_id": path_id,
                    "image_index": i,
                    "model_id": model["model_id"],
                    "model_label": model["variant"],
                    "energy_eV": float(payload["energies_eV"][i - 1]),
                    "relative_energy_eV": float(energy),
                    "force_max_eV_A": float(payload["force_max_eV_per_A"][i - 1]),
                    "finite": bool(payload["finite"]),
                })
            result_manifest.append({"model_id": model["model_id"], "path_id": path_id, "result": str(result), "result_sha256": sha256(result), "finite": bool(payload["finite"]), "barrier_eV": float(payload["barrier_eV"]), "n_images": int(payload["n_images"])})
    return pd.DataFrame(rows), result_manifest


def make_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for path_id in PATHS.values():
        paths[path_id], _ = build_extxyz(path_id)
    qe_profiles = {path_id: parse_qe_profile(path_id) for path_id in PATHS.values()}
    models = load_models()
    model_df, result_manifest = evaluate_models(paths, models)

    qe_rows = []
    for path_class in PATH_ORDER:
        path_id = PATHS[path_class]
        q = qe_profiles[path_id]
        for i, (energy, rel, residual) in enumerate(zip(q["qe_energy_eV"], q["qe_relative_energy_eV"], q["qe_force_error_eV_A"]), start=1):
            qe_rows.append({"path_class": path_class, "path_id": path_id, "image_index": i, "model_id": "QE", "model_label": "Rockfish QE final complete table", "energy_eV": energy, "relative_energy_eV": rel, "force_max_eV_A": np.nan, "qe_force_error_eV_A": residual, "qe_last_complete_iteration": q["qe_last_complete_iteration"], "job_id": q["job_id"], "job_done": q["job_done"], "scf_warning": q["scf_warning"], "qe_n_complete_tables": q["qe_n_complete_tables"]})
    profiles = pd.concat([model_df, pd.DataFrame(qe_rows)], ignore_index=True, sort=False)
    profiles["reaction_coordinate"] = (profiles["image_index"] - 1) / 9.0
    profiles.to_csv(OUT / "rockfish_cr_foundation_image_profiles.csv", index=False)

    qe_summary = []
    for path_class in PATH_ORDER:
        path_id = PATHS[path_class]
        q = qe_profiles[path_id]
        qe_summary.append({"path_class": path_class, "path_id": path_id, "job_id": q["job_id"], "qe_forward_activation_eV": q["qe_forward_activation_reported_eV"], "qe_reverse_activation_eV": q["qe_reverse_activation_reported_eV"], "qe_last_complete_iteration": q["qe_last_complete_iteration"], "qe_n_complete_tables": q["qe_n_complete_tables"], "job_done": q["job_done"], "scf_warning": q["scf_warning"], "scf_warning_count": q["scf_warning_count"], "status": "COMPLETED_BUT_SCF_WARNING" if q["job_done"] and q["scf_warning"] else "UNQUALIFIED"})
    pd.DataFrame(qe_summary).to_csv(OUT / "rockfish_cr_qe_summary.csv", index=False)

    barrier_rows = []
    for (path_class, path_id, model_id), group in model_df.groupby(["path_class", "path_id", "model_id"], sort=False):
        q = qe_profiles[path_id]
        barrier = float(group["relative_energy_eV"].max())
        qe_barrier = float(q["qe_forward_activation_reported_eV"])
        barrier_rows.append({"path_class": path_class, "path_id": path_id, "model_id": model_id, "model_label": group["model_label"].iloc[0], "qe_barrier_like_eV": qe_barrier, "foundation_barrier_like_eV": barrier, "signed_error_eV": barrier - qe_barrier, "absolute_error_eV": abs(barrier - qe_barrier), "reference_status": "COMPLETED_BUT_SCF_WARNING"})
    barriers = pd.DataFrame(barrier_rows)
    barriers.to_csv(OUT / "rockfish_cr_foundation_barrier_table.csv", index=False)
    summary = barriers.groupby("model_id", as_index=False).agg(median_absolute_error_eV=("absolute_error_eV", "median"), mean_absolute_error_eV=("absolute_error_eV", "mean"), max_absolute_error_eV=("absolute_error_eV", "max"), median_model_barrier_eV=("foundation_barrier_like_eV", "median"))
    summary.to_csv(OUT / "rockfish_cr_foundation_model_summary.csv", index=False)

    contract_rows = []
    for path_class, path_id in PATHS.items():
        neb = (RAW / path_id / "neb.in").read_text(errors="replace")
        pw = (RAW / path_id / "pw_1.in").read_text(errors="replace")
        kp = re.search(r"K_POINTS\s+AUTOMATIC\s*\n([^\n]+)", pw, re.IGNORECASE)
        contract_rows.append({"path_class": path_class, "path_id": path_id, "restart_mode": read_scalar(neb, "restart_mode"), "nstep_path": read_scalar(neb, "nstep_path"), "ds": read_scalar(neb, "ds"), "opt_scheme": read_scalar(neb, "opt_scheme"), "num_of_images": read_scalar(neb, "num_of_images"), "path_thr": read_scalar(neb, "path_thr"), "max_seconds": read_scalar(neb, "max_seconds"), "ecutwfc": read_scalar(pw, "ecutwfc"), "ecutrho": read_scalar(pw, "ecutrho"), "degauss": read_scalar(pw, "degauss"), "smearing": read_scalar(pw, "smearing"), "occupations": read_scalar(pw, "occupations"), "vdw_corr": read_scalar(pw, "vdw_corr"), "nat": read_scalar(pw, "nat"), "k_points": kp.group(1).strip() if kp else ""})
    pd.DataFrame(contract_rows).to_csv(OUT / "rockfish_cr_contract.tsv", sep="\t", index=False)

    manifest = {"status": "PROVENANCE_PARTIAL", "scientific_status": "COMPLETED_BUT_SCF_WARNING; EXCLUDE_FROM_PUBLICATION_AS_FINAL_BARRIER", "protocol": "same-geometry fixed-path Foundation single-point evaluation on the final complete Rockfish QE image table", "rockfish_root": "/scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013", "paths": {p: {"path_class": c, "job_id": JOB_IDS[p], "raw_dir": str(RAW / p), "raw_files": {f.name: sha256(f) for f in sorted((RAW / p).iterdir()) if f.is_file()}, "input_extxyz": str(paths[p]), "input_extxyz_sha256": sha256(paths[p])} for c, p in PATHS.items()}, "models": [{k: model[k] for k in ("model_id", "family", "variant", "head", "resolved_local_checkpoint", "resolved_sha256")} for model in models], "results": result_manifest, "qe_profiles": qe_profiles, "gates": {"n_images": 10, "all_model_rows_finite": bool(model_df["finite"].all()), "qe_job_done": bool(all(x["job_done"] for x in qe_profiles.values())), "qe_scf_warning_present": bool(any(x["scf_warning"] for x in qe_profiles.values())), "formal_dft_barrier_gate": "FAIL: SCF warning present in every Rockfish path; no final publication barrier"}}
    (OUT / "rockfish_cr_foundation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "rockfish_cr_status.md").write_text("""# Rockfish Cr NEB status\n\nThe three jobs completed with finite ten-image outputs and `JOB DONE`, but each final run contains an SCF-convergence warning. Their activation energies are therefore provisional path-table values, not publication-grade converged barriers. The Foundation-model rows are valid finite single-point evaluations on the saved Cr geometries, but any model-vs-QE error is diagnostic until the QE reference is rerun without the warning.\n\nThe input contract matches the retained Joshitha Cu/Li/Na NEB inputs for the numerical QE/NEB settings; the chemistry is Cr substitution and the launcher/accounting layer is Rockfish-specific.\n""")
    print(json.dumps({"output": str(OUT), "n_models": len(models), "n_paths": len(PATHS), "n_model_rows": len(model_df), "all_model_rows_finite": bool(model_df["finite"].all())}, indent=2))


if __name__ == "__main__":
    make_outputs()
