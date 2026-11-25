#!/usr/bin/env python3
"""Reproduce the Figure 2 v16 seed-boxcluster gallery case.

The canonical Figure 2 builder is imported directly so this gallery case does
not fork the scientific plotting implementation. The wrapper adds only a
provenance manifest and a stable, self-contained output location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in CASE_DIR.parents if (parent / ".git").exists())
SCRIPT_ROOT = REPO_ROOT / "docs/reproducibility/canva_figure_provenance/scripts"
DATA_ROOT = REPO_ROOT / "docs/reproducibility/canva_figure_provenance/data/figure02"
CANONICAL_BUILDER = SCRIPT_ROOT / "build_figure02_latest.py"
CANONICAL_SCRIPTS = [
    "_figure02_latest_base.py",
    "_figure02_latest_aggregate.py",
    "_figure02_latest_compact.py",
    "_figure02_latest_layout.py",
    "_figure02_latest_panels.py",
    "_figure02_latest_seed.py",
    "_figure02_latest_square.py",
    "build_figure02_latest.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def relative_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in sorted(paths)}


def input_files() -> list[Path]:
    return sorted(path for path in DATA_ROOT.rglob("*") if path.is_file())


def build(out_dir: Path) -> dict[str, object]:
    sys.path.insert(0, str(SCRIPT_ROOT))
    import build_figure02_latest as canonical  # noqa: PLC0415

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = canonical.build(out_dir)
    output_paths = sorted(path for path in out_dir.iterdir() if path.is_file())
    manifest = {
        "gallery_case": "boxplot_cluster_reference_v04",
        "description": "Figure 2 v16 seed-resolved clustered boxplot and square composite",
        "source_revision": "941baf8",
        "repository_revision_at_build": git_revision(),
        "entrypoint": str(Path(__file__).relative_to(REPO_ROOT)),
        "canonical_builder": str(CANONICAL_BUILDER.relative_to(REPO_ROOT)),
        "canonical_script_sha256": relative_hashes(
            [SCRIPT_ROOT / name for name in CANONICAL_SCRIPTS]
        ),
        "input_sha256": relative_hashes(input_files()),
        "outputs": {
            str(path.relative_to(REPO_ROOT)): sha256(path) for path in output_paths
        },
        "canonical_build_result": result,
        "visual_grammar": {
            "family_palette": {
                "Scratch-600K": "#FF9F1B",
                "FT-600K": "#002E71",
                "Scratch-MultiT": "#F4C95D",
                "FT-MultiT": "#67ACE5",
            },
            "seed_hatches": {"123": "", "234": "///", "345": "\\\\\\"},
            "typography": {
                "font_family": "Arial first, sans-serif fallback",
                "tick_pt": 20,
                "axis_label_pt": 22,
                "panel_title_pt": 22,
                "composite_heading_pt": 23,
            },
            "strokes_pt": {
                "axis": 1.8,
                "box_whisker_cap": 1.45,
                "median": 2.0,
                "hatch": 1.35,
            },
            "temperature_box_width": 0.135,
            "temperature_seed_offsets": [-0.16, 0.0, 0.16],
            "composite_canvas_inches": [16.0, 16.0],
            "composite_grid": "2x5; temperature left 3/5, overview right 2/5",
        },
    }
    manifest_path = CASE_DIR / "gallery_case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=CASE_DIR / "outputs")
    args = parser.parse_args()
    manifest = build(args.out_dir)
    print(json.dumps({"manifest": str(CASE_DIR / "gallery_case_manifest.json"), "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
