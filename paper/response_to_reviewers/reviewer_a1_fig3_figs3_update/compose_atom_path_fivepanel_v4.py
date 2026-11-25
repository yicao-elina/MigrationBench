"""Compose the standalone DFT and MLIP five-panel atom-path figures.

The panels are intentionally label-free and arrow-free.  They are designed
to be placed above the clean PES operator figure by the author in the final
layout, while the branch names and image ordering remain explicit in the
manifest and filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ATLAS = Path(
    __import__("os").environ.get(
        "MIGRATIONBENCH_PLOT_ATLAS", "/Users/alina/Documents/Codex/plot_library"
    )
)
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas import apply_publication_style, save_figure  # noqa: E402


ATOM_DIR = HERE / "atom_renders_v4"
OUTPUT_STEM = HERE / "fig3d_figS3_atom_path"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_bbox(image: np.ndarray, pad: int = 20) -> tuple[int, int, int, int]:
    """Return a padded non-transparent crop box for an RGBA image."""
    if image.ndim != 3 or image.shape[-1] != 4:
        return (0, 0, image.shape[1], image.shape[0])
    alpha = image[..., 3]
    ys, xs = np.where(alpha > 0.02)
    if len(xs) == 0:
        return (0, 0, image.shape[1], image.shape[0])
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(image.shape[0], int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def load_image(path: Path) -> np.ndarray:
    image = plt.imread(path)
    x0, y0, x1, y1 = alpha_bbox(image)
    return image[y0:y1, x0:x1]


def compose(branch: str, output: Path) -> None:
    image_paths = [ATOM_DIR / f"{branch}_{index:02d}.png" for index in range(5)]
    missing = [path for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing atom render(s): " + ", ".join(map(str, missing)))

    apply_publication_style(10.8)
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "dejavusans"})

    images = [load_image(path) for path in image_paths]
    aspect_ratios = [image.shape[0] / image.shape[1] for image in images]
    # The cropped QL renders are portrait-oriented.  A wide, shallow canvas
    # keeps all five images readable while avoiding an artificial text band.
    fig = plt.figure(figsize=(12.0, 3.25), facecolor="white")
    grid = fig.add_gridspec(
        1,
        5,
        left=0.012,
        right=0.988,
        bottom=0.035,
        top=0.965,
        wspace=0.028,
        width_ratios=[1.0 / ratio for ratio in aspect_ratios],
    )
    for index, image in enumerate(images):
        ax = fig.add_subplot(grid[0, index])
        ax.imshow(image, interpolation="lanczos", aspect="equal")
        ax.set_axis_off()
        ax.set_facecolor("white")
        # A very restrained border separates adjacent frames without adding
        # another semantic colour or an annotation to the art.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
            spine.set_edgecolor("#D8D8D8")

    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        save_figure(fig, output.with_suffix(f".{ext}"), dpi=360)
    plt.close(fig)

    manifest = {
        "figure_id": f"A1-ATOM-PATH-{branch.upper()}-FIVEPANEL-V4",
        "status": "STANDALONE_COORDINATE_DERIVED_STRUCTURAL_VISUAL_ANCHOR",
        "branch": branch,
        "frame_count": 5,
        "panel_order": [0, 1, 2, 3, 4],
        "input_render_manifest": str((ATOM_DIR / "atom_render_manifest.json").relative_to(REPO)),
        "input_frames": [
            {
                "image_index": index,
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
            }
            for index, path in enumerate(image_paths)
        ],
        "visual_policy": {
            "labels": False,
            "arrows": False,
            "arrowheads": False,
            "transparent_atom_source": True,
            "background": "white",
            "crop": "alpha-bounds plus 20-pixel padding",
        },
        "notes": [
            "This is an independent five-panel atom figure; atom images are not embedded in the clean PES operator main figure.",
            "DFT and MLIP branch identity is carried by the filename and manifest so the author can add final labels during layout.",
            "The renders are structural visual anchors, not evidence of a converged self-consistent DFT NEB.",
        ],
    }
    manifest["script_sha256"] = sha256(Path(__file__))
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    for branch in ("dft", "mlip"):
        compose(branch, args.output_dir / f"fig3d_figS3_atom_path_{branch}_fivepanel_v4")


if __name__ == "__main__":
    main()
