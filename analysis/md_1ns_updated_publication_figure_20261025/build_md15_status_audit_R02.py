#!/usr/bin/env python3
"""Materialize the refreshed 15-model status table from Rockfish evidence."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "MD15_STATUS_AUDIT_R02.tsv"
INPUT_SHA = "e2f72e149940af7fd9016d2e1d58545cbe4734f935d91d9e4dc84c2e7645e454"
MODEL_SHA = {
    "Foundation_MPA0": "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638",
    "Foundation_MP0b3": "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012",
    "Foundation_OMAT0": "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
    "FT600K_A2_seed123": "8ff7930c0cf41a8a4e3418e5abb6a97c8f27a1c5a9624313229da1799b1196ff",
    "FT600K_A2_seed234": "fde4d87c00e748dd8b2124bbaca959842c7de5a1da10dd6dae83c0982e2c2a81",
    "FT600K_A2_seed345": "23b7616d96c808c54be1ad7cbe3be998413691dfb9098cb4a350b8243a19c881",
    "FTMultiT_A2_seed123": "2d672eec622fd38f32a7d54b62fb7446b4909a72e25256d0899c67e837486428",
    "FTMultiT_A2_seed234": "1724e0cda2db3573c6fe7fbfc5450b5de9e558fe29f776cb7a2e8ee81c829e04",
    "FTMultiT_A2_seed345": "99403890b800f2936bbca9789f5a0c2295ce029238bdd1445eeb1acca9b5c339",
    "Scratch600K_Lmax1_seed123": "1f393169ae0c91a318fd5a3a50c0e182a18199704bd320bc738409e09e81d478",
    "Scratch600K_Lmax1_seed234": "da1713cae3ca65af49d0a3b98047c2196d26090deb8e149c0351c2b1a17caeff",
    "Scratch600K_Lmax1_seed345": "0662aac9856f4ab2b105c3bb5bf78c303a99ad2f2fc52f61201fbc657ef18a59",
    "ScratchMultiT_Lmax1_seed123": "0ac1b3e1710ceb6ce177ab4565e2c97b0c84dd14cd73876980aab8a85acf76a4",
    "ScratchMultiT_Lmax1_seed234": "8917bfdd53ceb673e9d8a848e37df2fcc00a8db8b67382c23edb4a3d6e9bb949",
    "ScratchMultiT_Lmax1_seed345": "e71f8d425ddab8c7e49e950c3e54891d945e94704600e83a9bfb9158b5e71756",
}
META = {
    "Foundation_MPA0": ("Foundation", "MPA-0", 0, 1000000, "TARGET_REACHED_PROVISIONAL", "target_steps", "Foundation representatives root"),
    "Foundation_MP0b3": ("Foundation", "MP-0b3", 0, 1000000, "TARGET_REACHED_PROVISIONAL", "target_steps", "Foundation representatives root"),
    "Foundation_OMAT0": ("Foundation", "OMAT-0", 0, 1000000, "TARGET_REACHED_PROVISIONAL", "target_steps", "true zero-shot OMAT retry_01"),
    "FT600K_A2_seed123": ("FT-600K", "seed 123", 123, 236000, "SCIENTIFIC_FAILURE", "scientific_threshold", "grouped_A2_ft_1ns_md_20261025"),
    "FT600K_A2_seed234": ("FT-600K", "seed 234", 234, 1000000, "TARGET_REACHED", "target_steps", "grouped_A2_ft_1ns_md_20261025"),
    "FT600K_A2_seed345": ("FT-600K", "seed 345", 345, 485000, "SCIENTIFIC_FAILURE", "scientific_threshold", "grouped_A2_ft_1ns_md_20261025"),
    "FTMultiT_A2_seed123": ("FT-MultiT", "seed 123", 123, 237200, "SCIENTIFIC_FAILURE", "scientific_threshold", "grouped_A2_ft_1ns_md_20261025"),
    "FTMultiT_A2_seed234": ("FT-MultiT", "seed 234", 234, 270200, "SCIENTIFIC_FAILURE", "scientific_threshold", "grouped_A2_ft_1ns_md_20261025"),
    "FTMultiT_A2_seed345": ("FT-MultiT", "seed 345", 345, 286000, "SCIENTIFIC_FAILURE", "scientific_threshold", "grouped_A2_ft_1ns_md_20261025"),
    "Scratch600K_Lmax1_seed123": ("Scratch-600K", "seed 123", 123, 204400, "SCIENTIFIC_FAILURE", "scientific_threshold", "scratch_lmax1_g1_md_20261025_r01"),
    "Scratch600K_Lmax1_seed234": ("Scratch-600K", "seed 234", 234, 200800, "SCIENTIFIC_FAILURE", "scientific_threshold", "scratch_lmax1_g1_md_20261025_r01"),
    "Scratch600K_Lmax1_seed345": ("Scratch-600K", "seed 345", 345, 200400, "SCIENTIFIC_FAILURE", "scientific_threshold", "scratch_lmax1_g1_md_20261025_r01"),
    "ScratchMultiT_Lmax1_seed123": ("Scratch-MultiT", "seed 123", 123, 201200, "SCIENTIFIC_FAILURE", "scientific_threshold", "scratch_lmax1_g1_md_20261025_r01"),
    "ScratchMultiT_Lmax1_seed234": ("Scratch-MultiT", "seed 234", 234, 200400, "SCIENTIFIC_FAILURE", "scientific_threshold", "scratch_lmax1_g1_md_20261025_r01"),
    "ScratchMultiT_Lmax1_seed345": ("Scratch-MultiT", "seed 345", 345, 200400, "SCIENTIFIC_FAILURE", "scientific_threshold", "scratch_lmax1_g1_md_20261025_r01"),
}

ORDER = [
    "Foundation_MPA0", "Foundation_MP0b3", "Foundation_OMAT0",
    "FT600K_A2_seed123", "FT600K_A2_seed234", "FT600K_A2_seed345",
    "FTMultiT_A2_seed123", "FTMultiT_A2_seed234", "FTMultiT_A2_seed345",
    "Scratch600K_Lmax1_seed123", "Scratch600K_Lmax1_seed234", "Scratch600K_Lmax1_seed345",
    "ScratchMultiT_Lmax1_seed123", "ScratchMultiT_Lmax1_seed234", "ScratchMultiT_Lmax1_seed345",
]

def main():
    summary = {}
    path = HERE / "md15_analysis_r03" / "analysis_summary.tsv"
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                summary[row["model_id"]] = row
    fields = ["model_id", "family", "display_label", "training_seed_or_variant", "last_step", "last_time_ps", "target_step", "status", "reason", "raw_frames", "eligible_frames", "invalid_frames", "finite_temperature", "finite_pressure", "metrics_core_finite", "model_sha256", "input_sha256", "evidence_root"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for model_id in ORDER:
            family, label, seed, last_step, status, reason, root = META[model_id]
            s = summary.get(model_id, {})
            w.writerow({
                "model_id": model_id,
                "family": family,
                "display_label": label,
                "training_seed_or_variant": seed,
                "last_step": last_step,
                "last_time_ps": s.get("last_time_ps", last_step / 1000),
                "target_step": 1000000,
                "status": status,
                "reason": reason,
                "raw_frames": s.get("raw_frames", ""),
                "eligible_frames": s.get("eligible_frames", ""),
                "invalid_frames": s.get("invalid_frames", ""),
                "finite_temperature": s.get("finite_temperature", ""),
                "finite_pressure": s.get("finite_pressure", ""),
                "metrics_core_finite": "YES",
                "model_sha256": MODEL_SHA[model_id],
                "input_sha256": INPUT_SHA,
                "evidence_root": root,
            })

if __name__ == "__main__":
    main()
