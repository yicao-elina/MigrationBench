#!/usr/bin/env python3
"""Build Figure 2 v14 with compact groups and exact square output pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _figure02_latest_seed as v9
import _figure02_latest_compact as v13


WORKTREE = Path("/Users/alina/.codex/worktrees/canva-figure-provenance")
OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_temperature_reference_layout_v14"
)


def _save_exact(fig, out_dir: Path, stem: str) -> dict[str, str]:
    """Save without tight cropping so square canvases remain exactly square."""
    paths = {
        "png": out_dir / f"{stem}.png",
        "pdf": out_dir / f"{stem}.pdf",
        "svg": out_dir / f"{stem}.svg",
    }
    for path in paths.values():
        fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return {key: str(value) for key, value in paths.items()}


def build(out_dir: Path) -> dict[str, object]:
    v9._house_style()
    v9.BRACKET_STEP = 0.60
    v9.base.save = _save_exact
    v13.v12.TEMP_CENTERS = v13.TEMP_CENTERS
    v13.v12.TEMP_FAMILY_OFFSETS = v13.TEMP_FAMILY_OFFSETS
    v13.v12.GROUP_HALF_WIDTH = v13.GROUP_HALF_WIDTH
    out_dir.mkdir(parents=True, exist_ok=True)
    records = v13.v12._physical_records(v13.v12._records())
    temperature, temperature_stats = v13._temperature_figure(records, out_dir)
    overview, overview_stats = v13._overview_figure(records, out_dir)
    composite = v13._composite_figure(records, out_dir)
    manifest = {
        "status": "RECONSTRUCTION",
        "previous_version": "e82b464",
        "purpose": "compact temperature groups with exact square standalone and composite overview pages",
        "save_mode": "exact_canvas_no_tight_crop",
        "layout": {
            "temperature_centers": v13.TEMP_CENTERS,
            "group_half_width": v13.GROUP_HALF_WIDTH,
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
    manifest_path = out_dir / "figure02_temperature_reference_layout_v14_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_temperature_reference_layout_v14_caption.md"
    caption_path.write_text(
        "# Figure 2 v14 compact reference-style layout\n\n"
        "The temperature groups use compact inter-group spacing. The standalone Scratch-vs-FT overview and the combined board preserve exact square page canvases by disabling tight cropping during export. All frame-level points and three seed-specific boxplots are retained, with train and held-out test records merged.\n"
    )
    return {"manifest": str(manifest_path), "caption": str(caption_path), "outputs": manifest["outputs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
