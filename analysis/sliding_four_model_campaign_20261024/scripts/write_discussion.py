#!/usr/bin/env python3
"""Write a data-bound discussion and caption suggestion after Stage 1 collection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, root_path, system, write_json


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    metric_path = root / "results/tables/fixed_metrics_per_model.csv"
    summary_path = root / "results/tables/fixed_family_summary.csv"
    metrics = rows(metric_path)
    summaries = rows(summary_path)
    if not metrics:
        raise SystemExit(f"missing metrics: {metric_path}")
    by_system = {system_id: [row for row in metrics if row.get("system_id") == system_id and row.get("run_status") == "PASS"] for system_id in config["systems"]}
    discussion = [
        "# Fig. 4b/c fixed-path discussion (Stage 1)",
        "",
        "The Stage 1 campaign evaluates MACE energies and forces on the ten retained DFT/QE geometries for pristine and Cr-in-gap Sb2Te3. The observable is a fixed-path relative-energy diagnostic: each profile is referenced to its own image 0, and no MACE geometry optimization or self-relaxed NEB convergence is used. The retained DFT reference is the no-CI ten-image QE path and remains classified as `COMPLETED_BUT_SCF_WARNING`. FR pseudopotential filenames are present in the archived inputs, but explicit noncollinear/spin-orbit flags were not independently confirmed in the retained input/output; this is therefore not upgraded to a verified SOC statement.",
        "",
    ]
    for system_id, spec in config["systems"].items():
        items = by_system[system_id]
        dft_peak = float(items[0]["dft_peak_eV"]) if items else float("nan")
        dft_index = int(float(items[0]["dft_peak_image_index"])) if items else -1
        discussion.append(f"## {spec['label']}")
        discussion.append("")
        discussion.append(f"The DFT reference reaches {fmt(dft_peak)} eV at image {dft_index} and has an image-9 endpoint drift of {fmt(float(items[0]['dft_endpoint_drift_eV']))} eV relative to image 0.")
        if items:
            best = min(items, key=lambda row: float(row.get("rmse_dft_eV", "nan")))
            discussion.append(f"Among the current checkpoint evaluations, the smallest ten-point RMSE is obtained by {best['model_label']} ({fmt(float(best['rmse_dft_eV']))} eV); its fixed-path peak is {fmt(float(best['max_relative_energy_eV']))} eV at image {best['peak_image_index']}, with endpoint drift {fmt(float(best['endpoint_drift_eV']))} eV and Spearman profile correlation {fmt(float(best['spearman_dft']), 2)}.")
            for family in ("foundation", "ft_600k", "ft_multit", "scratch"):
                family_rows = [row for row in items if row.get("model_family") == family]
                if family == "foundation":
                    labels = "; ".join(f"{row['model_label']}: peak {fmt(float(row['max_relative_energy_eV']))} eV, RMSE {fmt(float(row['rmse_dft_eV']))} eV" for row in family_rows)
                    discussion.append(f"The three Foundation entries are checkpoint variants rather than random-seed replicates and are reported individually: {labels}.")
                else:
                    subgroups = sorted({row.get("scratch_subfamily", "none") for row in family_rows})
                    for subgroup in subgroups:
                        subitems = [row for row in family_rows if row.get("scratch_subfamily", "none") == subgroup]
                        if not subitems:
                            continue
                        peaks = np.asarray([float(row["max_relative_energy_eV"]) for row in subitems])
                        rmses = np.asarray([float(row["rmse_dft_eV"]) for row in subitems])
                        name = {
                            "ft_600k": "FT-600K",
                            "ft_multit": "FT-MultiT",
                            "scratch_600k": "Scratch-600K",
                            "scratch_multit": "Scratch-MultiT",
                        }.get(family if subgroup == "none" else subgroup, family)
                        discussion.append(f"{name} gives a training-seed peak mean +/- SD of {fmt(float(peaks.mean()))} +/- {fmt(float(peaks.std(ddof=1)) if len(peaks) > 1 else float('nan'))} eV and RMSE mean +/- SD of {fmt(float(rmses.mean()))} +/- {fmt(float(rmses.std(ddof=1)) if len(rmses) > 1 else float('nan'))} eV (n={len(subitems)}).")
        discussion.append("")
    discussion.extend([
        "## Interpretation boundary",
        "",
        "The updated b/c panels support comparison of relative-energy profile shape, peak location, endpoint periodicity, and training-seed spread on a common fixed geometry. They do not establish a validated migration barrier: the MACE paths are not optimized, the DFT reference retains its warning-bearing status, and the fixed-path operator is not interchangeable with a self-relaxed NEB observable. The two Scratch subfamilies remain separate because both use max_L=2 and differ in training-data lineage; they are not collapsed into one Scratch mean. Legacy `dc41a989...` Foundation results are retained only for the A0 pipeline self-check and are excluded from the current panel and current conclusions.",
        "",
    ])
    discussion_path = root / "results/FIXED_PATH_DISCUSSION.md"
    discussion_path.write_text("\n".join(discussion) + "\n")
    caption = """Fig. 4b,c | Fixed-path relative-energy profiles for interlayer sliding in Sb2Te3. Black circles show the retained DFT reference from the ten-image no-CI Quantum ESPRESSO path; the reference remains classified as COMPLETED_BUT_SCF_WARNING. The current MACE evaluations use the same ten DFT geometries without relaxation. Colors follow the established FIG-168 mapping: Foundation (orange-red), FT-600K (dark blue), FT-MultiT (light blue), and from-scratch models (orange-yellow). Foundation checkpoint variants are shown individually and are not treated as a seed ensemble. FT-600K, FT-MultiT, Scratch-600K, and Scratch-MultiT show individual training-seed profiles together with mean +/- 1 SD bands where n=3; the two Scratch subfamilies are kept separate. These panels are fixed-path relative-energy diagnostics rather than self-relaxed or validated migration barriers. FR pseudopotential filenames are present in the retained DFT inputs, but explicit SOC flags were not independently confirmed."""
    caption_path = root / "results/FIXED_PATH_CAPTION_SUGGESTION.md"
    caption_path.write_text(caption + "\n")
    write_json(root / "results/FIXED_PATH_TEXT_MANIFEST.json", {
        "discussion": str(discussion_path),
        "caption": str(caption_path),
        "metrics": str(metric_path),
        "metrics_sha256": __import__("hashlib").sha256(metric_path.read_bytes()).hexdigest(),
        "boundary": "fixed-path diagnostic only",
    })
    print(discussion_path)
    print(caption_path)


if __name__ == "__main__":
    main()

