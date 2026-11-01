#!/usr/bin/env python3
"""Build Figure 2 with compressed groups and a clean composite gutter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import build_figure02_seed_boxclusters_v9 as v9
import build_figure02_temperature_reference_layout_v14 as v14


WORKTREE = Path("/Users/alina/.codex/worktrees/canva-figure-provenance")
OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_temperature_reference_layout_v16"
)


def build(out_dir: Path) -> dict[str, object]:
    v9._house_style()
    v9.BRACKET_STEP = 0.60
    v9.base.save = v14._save_exact
    v13 = v14.v13
    v12 = v13.v12
    compact_centers = {category: index * 2.00 for index, category in enumerate(v9.TEMP_CATEGORIES)}
    v13.TEMP_CENTERS = compact_centers
    v12.TEMP_CENTERS = compact_centers
    v12.TEMP_FAMILY_OFFSETS = v13.TEMP_FAMILY_OFFSETS
    v12.GROUP_HALF_WIDTH = v13.GROUP_HALF_WIDTH
    v12.RIGHT_MARGIN = 0.45
    v12.TEMP_BOX_WIDTH = 0.135
    v12.SEED_OFFSETS = np.asarray([-0.16, 0.0, 0.16])
    out_dir.mkdir(parents=True, exist_ok=True)
    records = v12._physical_records(v12._records())
    temperature, temperature_stats = v13._temperature_figure(records, out_dir)
    overview, overview_stats = v13._overview_figure(records, out_dir)
    composite = v13._composite_figure(records, out_dir)
    manifest = {
        "status": "RECONSTRUCTION",
        "previous_version": "05c53d7",
        "purpose": "compressed temperature groups with a reserved gutter before the overview y-axis label",
        "layout": {
            "temperature_centers": compact_centers,
            "temperature_box_width": v12.TEMP_BOX_WIDTH,
            "seed_offsets": v12.SEED_OFFSETS.tolist(),
            "right_margin_after_last_temperature_group": v12.RIGHT_MARGIN,
            "y_axis_join": "first x-axis segment starts at left y-axis x coordinate",
            "composite_canvas_inches": [16.0, 16.0],
            "overview_canvas_inches": [10.5, 10.5],
        },
        "data": {
            "source_mode": "readable_packaged_per_frame_csv",
            "frame_root": str(v9.FRAME_ROOT),
            "splits_merged": v9.base.SPLITS,
            "isolated_atoms_excluded": True,
            "temperature_families": v9.MULTIT_FAMILIES,
            "overview_families": v9.FAMILIES,
            "seeds": v9.SEEDS,
            "overview_scope": "all physical trajectory classes pooled within family and seed; train + test merged",
            "significance": {
                "temperature": "Scratch-MultiT vs FT-MultiT",
                "overview": ["Scratch vs FT · 600K", "Scratch vs FT · MultiT", "600K vs MultiT · FT"],
                "test": "paired two-sided t-test on three seed-level aggregates; exploratory n=3; no multiplicity correction",
            },
        },
        "outputs": {"temperature": temperature, "overview": overview, "composite": composite},
        "statistics": {"temperature": temperature_stats, "overview": overview_stats},
    }
    manifest_path = out_dir / "figure02_temperature_reference_layout_v16_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_temperature_reference_layout_v16_caption.md"
    caption_path.write_text(
        "# Figure 2 v16 compact composite gutter\n\n"
        "The three temperature groups are brought closer together while a dedicated right-side gutter keeps the last temperature group clear of the overview y-axis label in the square composite. The first temperature x-axis segment remains joined to the left y-axis.\n"
    )
    return {"manifest": str(manifest_path), "caption": str(caption_path), "outputs": manifest["outputs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
