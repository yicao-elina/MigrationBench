#!/usr/bin/env python3
"""Export the multiscale probe data (static, MD, sliding) for the pitch video.

Every source file is resolved through the asset registry (chart_id/file_id ->
local_path), read read-only, and fingerprinted. NEB profiles come from
export_neb_profiles.py. Writes public/data/story.json.

Usage:
    python3 scripts/export_story_data.py [--registry PATH] [--root PATH]
"""

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from export_neb_profiles import DEFAULT_REGISTRY, DEFAULT_ROOT

OUT_PATH = Path(__file__).resolve().parent.parent / "public/data/story.json"

SOURCES = {
    "parity": ("chart_fig2_v16_temperature_plus_rmse_overview", "parity_summary"),
    "md_timecourse": ("chart_md_1ns_fig004_a_to_g_20261016", "timecourse"),
    "md_diffusion": ("chart_md_1ns_fig004_a_to_g_20261016", "dxy_summary"),
    "sliding_profiles": ("chart_interlayer_four_cr_gap_fixed_20261024", "fixed_profiles"),
    "sliding_metrics": ("chart_interlayer_four_cr_gap_fixed_20261024", "fixed_metrics"),
    "sliding_qe": ("chart_interlayer_r13_2x2_release_20261024", "qe_reference"),
}

# Sliding model_id prefix -> display family (matches the NEB/static families).
SLIDING_FAMILIES = {
    "foundation_omat0_medium": "Foundation (OMAT-0-M)",
    "ft_600k_grouped_A2": "FT-600K",
    "ft_multit_grouped_A2": "FT-MultiT",
    "scratch_600k_grouped_v1": "Scratch-600K",
}


def resolve_sources(registry: Path, root: Path) -> dict:
    resolved = {}
    with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as conn:
        for key, (chart_id, file_id) in SOURCES.items():
            row = conn.execute(
                "SELECT d.local_path, c.chart_status FROM chart_data_files d "
                "JOIN charts c USING (chart_id) WHERE d.chart_id = ? AND d.file_id = ?",
                (chart_id, file_id),
            ).fetchone()
            if row is None:
                sys.exit(f"Registry has no {chart_id}/{file_id}")
            path = Path(row[0]) if Path(row[0]).is_absolute() else root / row[0]
            if not path.is_file():
                sys.exit(f"Linked data file missing: {path}")
            resolved[key] = {
                "chart_id": chart_id,
                "chart_status": row[1],
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return resolved


def read_table(path: str) -> list[dict]:
    delimiter = "\t" if path.endswith(".tsv") else ","
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def static_summary(rows: list[dict]) -> list[dict]:
    by_family = defaultdict(lambda: {"e": [], "f": []})
    for r in rows:
        if r["split"] == "test":
            by_family[r["model"]]["e"].append(float(r["energy_rmse_meV_per_atom"]))
            by_family[r["model"]]["f"].append(float(r["force_rmse_meV_A"]))
    return [
        {"family": fam, "energy_rmse": mean_sd(v["e"]), "force_rmse": mean_sd(v["f"])}
        for fam, v in by_family.items()
    ]


def md_summary(rows: list[dict], diffusion: list[dict]) -> dict:
    series = defaultdict(list)
    for r in rows:
        series[(r["family"], r["observable"])].append(
            {
                "t_ps": float(r["time_ps"]),
                "mean": float(r["mean"]),
                "sd": float(r["sd"]),
                "n_seeds": int(r["n_training_seeds"]),
            }
        )
    return {
        "timecourse": [
            {"family": fam, "observable": obs, "points": sorted(pts, key=lambda p: p["t_ps"])}
            for (fam, obs), pts in series.items()
        ],
        "diffusion": [
            {
                "family": r["family"],
                "window": r["window"],
                "D_cm2_s": float(r["mean_D_cm2_s"]),
                "sd": float(r["sd_D_cm2_s"]),
                "n_seeds": int(r["n_training_seeds"]),
            }
            for r in diffusion
        ],
    }


def sliding_summary(profiles: list[dict], qe: list[dict]) -> dict:
    qe_rows = sorted(
        (r for r in qe if r["system"] == "cr_in_gap"), key=lambda r: int(r["image_index"])
    )
    reference = [
        {"s": float(r["image_fraction"]), "e": float(r["relative_energy_eV"])} for r in qe_rows
    ]
    grouped = defaultdict(lambda: defaultdict(list))
    for r in profiles:
        if r["system_id"] != "cr_in_gap" or r["finite_energy"] != "True":
            continue
        family = next((v for k, v in SLIDING_FAMILIES.items() if r["model_id"].startswith(k)), None)
        if family:
            grouped[family][int(r["image_index"])].append(float(r["relative_energy_eV"]))
    n_images = len(reference)
    families = []
    for family, images in grouped.items():
        points = [
            {"s": reference[i]["s"], **mean_sd(images[i])} for i in sorted(images) if i < n_images
        ]
        families.append({"family": family, "points": points})
    return {
        "reference": reference,
        "reference_status": qe_rows[0]["dft_status"] if qe_rows else None,
        "families": families,
    }


def sliding_metrics(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: {"rmse": [], "peak_image": [], "drift": []})
    for r in rows:
        family = next((v for k, v in SLIDING_FAMILIES.items() if r["model_id"].startswith(k)), None)
        if r["system_id"] != "cr_in_gap" or family is None:
            continue
        grouped[family]["rmse"].append(float(r["rmse_dft_eV"]))
        grouped[family]["peak_image"].append(float(r["peak_image_index"]))
        grouped[family]["drift"].append(float(r["endpoint_drift_eV"]))
    return [
        {"family": fam, **{k: mean_sd(v) for k, v in vals.items()}} for fam, vals in grouped.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    src = resolve_sources(args.registry, args.root)
    tables = {k: read_table(v["path"]) for k, v in src.items()}
    payload = {
        "exported_on": date.today().isoformat(),
        "sources": src,
        "static": static_summary(tables["parity"]),
        "md": md_summary(tables["md_timecourse"], tables["md_diffusion"]),
        "sliding": {
            **sliding_summary(tables["sliding_profiles"], tables["sliding_qe"]),
            "metrics": sliding_metrics(tables["sliding_metrics"]),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")
    for s in payload["static"]:
        print("static", s["family"], round(s["energy_rmse"]["mean"], 1), round(s["force_rmse"]["mean"]))
    for d in payload["md"]["diffusion"]:
        print("D", d["family"], d["window"], f"{d['D_cm2_s']:.2e}", d["n_seeds"])
    ref_peak = max(p["e"] for p in payload["sliding"]["reference"])
    print("sliding QE peak", round(ref_peak, 3), payload["sliding"]["reference_status"])
    for m in payload["sliding"]["metrics"]:
        print("sliding metric", m["family"], round(m["rmse"]["mean"], 2), m["peak_image"]["mean"], m["rmse"]["n"])
    for f in payload["sliding"]["families"]:
        print("sliding", f["family"], round(max(p["mean"] for p in f["points"]), 3), len(f["points"]))


if __name__ == "__main__":
    main()
