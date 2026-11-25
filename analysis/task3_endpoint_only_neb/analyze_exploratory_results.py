#!/usr/bin/env python3
"""Merge MACE path single points with DFT path profiles using model-internal relative energies."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "exploratory_energy_audit"
RESULTS = AUDIT / "full_results"
OUT = AUDIT / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def parse_file_name(name: str) -> tuple[str, str]:
    stem = name.removesuffix(".extxyz")
    path_id, variant = stem.rsplit("__", 1)
    return path_id, variant


def main() -> None:
    source = pd.read_csv(AUDIT / "path_sampling" / "path_sampling_manifest.csv")
    dft = source[["sample_id", "path_id", "image_index", "variant", "dft_path_energy_eV", "dft_path_sha256", "scientific_success"]].copy()
    dft["dft_relative_energy_eV"] = dft["dft_path_energy_eV"] - dft.groupby(["path_id", "variant"], sort=False)["dft_path_energy_eV"].transform("first")
    all_rows = []
    model_status = []
    for result_path in sorted(RESULTS.glob("*/mace_single_point_results.csv")):
        model_id = result_path.parent.name
        raw = pd.read_csv(result_path)
        raw[["path_id", "variant"]] = raw["sample_file"].apply(lambda x: pd.Series(parse_file_name(x)))
        raw["sample_id"] = raw.apply(lambda r: f"{r.path_id}__image{int(r.frame_index)+1:02d}__{r.variant}", axis=1)
        raw["image_index"] = raw["frame_index"] + 1
        raw["model_id"] = model_id
        raw["mace_relative_energy_eV"] = raw["energy_eV"] - raw.groupby(["path_id", "variant"], sort=False)["energy_eV"].transform("first")
        merged = dft.merge(raw[["sample_id", "model_id", "status", "scientific_success", "energy_eV", "mace_relative_energy_eV", "max_force_eV_A", "input_sha256", "model_sha256", "failure_reason"]], on="sample_id", how="left", suffixes=("_dft", "_mace"))
        merged["relative_energy_error_eV"] = merged["mace_relative_energy_eV"] - merged["dft_relative_energy_eV"]
        merged["interpretation"] = "within-model relative single-point profile; not DFT accuracy proof"
        all_rows.append(merged)
        model_status.append({"model_id": model_id, "n_rows": len(raw), "n_success": int(raw["scientific_success"].sum()), "n_failure": int((raw["status"] != "success").sum()), "model_sha256": raw["model_sha256"].dropna().iloc[0] if len(raw["model_sha256"].dropna()) else "", "failure_reasons": ";".join(sorted(set(str(x).splitlines()[-1] for x in raw.loc[raw.status != "success", "failure_reason"].dropna())))})
    image = pd.concat(all_rows, ignore_index=True)
    image.to_csv(OUT / "mace_dft_relative_image_metrics.csv", index=False)
    summary_rows = []
    for (model_id, path_id, variant), g in image.groupby(["model_id", "path_id", "variant"], sort=True):
        ok = g["status"] == "success"
        if ok.all():
            mg = g.sort_values("image_index")
            summary_rows.append({"model_id": model_id, "path_id": path_id, "variant": variant, "status": "success", "scientific_success": True, "n_images": int(ok.sum()), "dft_relative_barrier_eV": float(mg.dft_relative_energy_eV.max()), "mace_relative_barrier_eV": float(mg.mace_relative_energy_eV.max()), "barrier_difference_eV": float(mg.mace_relative_energy_eV.max() - mg.dft_relative_energy_eV.max()), "profile_mae_eV": float(np.mean(np.abs(mg.relative_energy_error_eV))), "profile_max_abs_error_eV": float(np.max(np.abs(mg.relative_energy_error_eV))), "mean_max_force_eV_A": float(mg.max_force_eV_A.mean()), "max_force_eV_A": float(mg.max_force_eV_A.max()), "failure_reason": "", "model_sha256": str(mg.model_sha256.iloc[0])})
        else:
            reason = ";".join(sorted(set(str(x).splitlines()[-1] for x in g.loc[~ok, "failure_reason"].dropna())))
            summary_rows.append({"model_id": model_id, "path_id": path_id, "variant": variant, "status": "runtime_failure", "scientific_success": False, "n_images": int(ok.sum()), "dft_relative_barrier_eV": "", "mace_relative_barrier_eV": "", "barrier_difference_eV": "", "profile_mae_eV": "", "profile_max_abs_error_eV": "", "mean_max_force_eV_A": "", "max_force_eV_A": "", "failure_reason": reason, "model_sha256": str(g.model_sha256.iloc[0])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "mace_dft_relative_path_summary.csv", index=False)
    # Perturbation response is only compared within the same model/path/image.
    base = image[image.variant == "base"][["model_id", "path_id", "image_index", "mace_relative_energy_eV", "status"]].rename(columns={"mace_relative_energy_eV": "base_mace_relative_energy_eV", "status": "base_status"})
    pert = image[image.variant != "base"].merge(base, on=["model_id", "path_id", "image_index"], how="left")
    pert["mace_perturbation_delta_eV"] = pert["mace_relative_energy_eV"] - pert["base_mace_relative_energy_eV"]
    pert.to_csv(OUT / "mace_perturbation_response.csv", index=False)
    pd.DataFrame(model_status).to_csv(OUT / "model_status.csv", index=False)
    manifest = {"n_dft_frames": len(dft), "n_model_image_rows": len(image), "n_successful_model_image_rows": int(image.scientific_success_mace.sum()), "models": model_status, "relative_energy_rule": "E_rel(model,path,variant,image)=E(model,path,variant,image)-E(model,path,variant,image_1)", "absolute_cross_model_energy_comparison": False, "dft_status": "non-equilibrium path samples; energy-table stability gate only", "scientific_claim_boundary": "finite single-point prediction and within-model profile/perturbation response, not DFT barrier accuracy"}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__": main()
