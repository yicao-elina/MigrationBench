#!/usr/bin/env python3
"""Recompute fixed-geometry MACE NEB barriers with path-matched DFT references."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data_processed" / "cluster" / "neb_benchmark" / "data"
NEB_SUMMARY = ROOT / "data_processed" / "cluster" / "inventory" / "rockfish_neb_out_summary.csv"
OUT = ROOT / "data_processed" / "cluster" / "neb_benchmark" / "unified_fixed_geometry_barriers_audit.csv"


STATUS_OVERRIDES = {
    "1-4": "numerically_stationary_unconverged",
}


def load_dft_refs() -> dict[str, dict]:
    df = pd.read_csv(NEB_SUMMARY)
    out = {}
    for _, row in df.iterrows():
        path = row["neb_out"]
        marker = "/3.neb/1.vdw_corr_DFT_D3/"
        if marker not in path:
            continue
        tail = path.split(marker, 1)[1]
        path_label = tail.split("/", 1)[0]
        if path_label not in {"1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"}:
            continue
        status = "converged" if bool(row["job_done"]) else "unconverged"
        status = STATUS_OVERRIDES.get(path_label, status)
        out[path_label] = {
            "reference_barrier_eV": float(row["dat_peak_eV"]) if not math.isnan(row["dat_peak_eV"]) else None,
            "reference_status": status,
            "reference_neb_out": path,
            "reference_n_images": int(row["n_images"]) if not math.isnan(row["n_images"]) else None,
        }
    return out


def load_implied_refs(path_label: str) -> dict[str, float]:
    f = BENCH / path_label / "neb_barrier_results.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return {row["model_name"]: float(row["barrier"]) - float(row["error_eV"]) for _, row in df.iterrows()}


def main() -> None:
    refs = load_dft_refs()
    rows = []
    for pred_path in sorted(BENCH.glob("1-*/neb_predictions.csv")):
        path_label = pred_path.parent.name
        if path_label not in refs:
            continue
        ref = refs[path_label]
        preds = pd.read_csv(pred_path)
        implied = load_implied_refs(path_label)
        for model, group in preds.groupby("model_name"):
            group = group.sort_values("image_index")
            rel = group["predicted_energy_eV"].astype(float) - float(group["predicted_energy_eV"].iloc[0])
            barrier = float(rel.max())
            dft_barrier = ref["reference_barrier_eV"]
            signed_error = None if dft_barrier is None else barrier - dft_barrier
            implied_ref = implied.get(model)
            ref_mismatch = implied_ref is not None and dft_barrier is not None and abs(implied_ref - dft_barrier) > 0.005
            manuscript_allowed = ref["reference_status"] in {"converged", "numerically_stationary_unconverged"} and not ref_mismatch
            rows.append(
                {
                    "path": path_label,
                    "model_name": model,
                    "protocol": "fixed_geometry_single_point",
                    "mace_barrier_eV": barrier,
                    "dft_reference_barrier_eV": dft_barrier,
                    "signed_error_eV": signed_error,
                    "abs_error_eV": None if signed_error is None else abs(signed_error),
                    "dft_reference_status": ref["reference_status"],
                    "dft_reference_neb_out": ref["reference_neb_out"],
                    "n_images": ref["reference_n_images"],
                    "existing_barrier_csv_implied_ref_eV": implied_ref,
                    "existing_barrier_csv_ref_mismatch": ref_mismatch,
                    "manuscript_allowed": manuscript_allowed,
                    "exclusion_reason": ""
                    if manuscript_allowed
                    else (
                        "existing barrier_results.csv used a different reference"
                        if ref_mismatch
                        else f"DFT reference status is {ref['reference_status']}"
                    ),
                    "prediction_csv": str(pred_path),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
