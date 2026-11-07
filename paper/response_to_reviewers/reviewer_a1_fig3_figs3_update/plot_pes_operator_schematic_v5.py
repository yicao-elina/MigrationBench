"""v5 PES operator schematic with explicit five-image anchor points.

v5 keeps the clean v4 composition, but restores the five image anchors needed
to map the independent A/1/TS/3/B atom panels onto the curves.  Fixed-path
anchors share the same reaction-coordinate positions, while the self-relaxed
q1-q2 anchors are shown at their distinct coordinates.  Pair connectors are
plain, non-directional lines; no explanatory arrows or callout text are used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V4_PATH = HERE / "plot_pes_operator_schematic_v4.py"
DFT_XYZ = REPO / "tests/fixtures/historical_neb/legacy_61/1-4/sb2te3.xyz"
MLIP_XYZ = HERE / "data/self_relaxed_foundation_path_61_1-4.extxyz"

ATLAS = Path(
    __import__("os").environ.get(
        "MIGRATIONBENCH_PLOT_ATLAS", "/Users/alina/Documents/Codex/plot_library"
    )
)
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas import apply_publication_style, save_figure  # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402


def load_v4_module():
    spec = importlib.util.spec_from_file_location("pes_operator_v4", V4_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {V4_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V4 = load_v4_module()
V2 = V4.V2

INK = V4.INK
DFT_PATH_COLOUR = V4.DFT_PATH_COLOUR
MLIP_PATH_COLOUR = V4.MLIP_PATH_COLOUR
LIGHT_GREY = V4.LIGHT_GREY

IMAGE_LABELS = ("A", "1", "TS", "3", "B")
IMAGE_MARKER_SIZE = 23
IMAGE_LABEL_FONTSIZE = 7.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scatter_anchor_3d(ax, q_dft: np.ndarray, q_mlip: np.ndarray, mode: str) -> None:
    """Add the five coordinate-derived image anchors without annotations."""
    z_floor = 0.018
    if mode == "fixed":
        z_dft, z_mlip = V2.surfaces(q_dft[:, 0], q_dft[:, 1])
        ax.scatter(
            q_dft[:, 0], q_dft[:, 1], z_dft,
            s=IMAGE_MARKER_SIZE, color=INK, edgecolor="white", linewidth=0.55,
            depthshade=False, zorder=35,
        )
        ax.scatter(
            q_dft[:, 0], q_dft[:, 1], z_mlip,
            s=IMAGE_MARKER_SIZE, color=MLIP_PATH_COLOUR, edgecolor="white", linewidth=0.55,
            depthshade=False, zorder=36,
        )
        ax.scatter(
            q_dft[:, 0], q_dft[:, 1], np.full(len(q_dft), z_floor),
            s=10, color=DFT_PATH_COLOUR, edgecolor="white", linewidth=0.35,
            depthshade=False, zorder=37,
        )
    else:
        z_dft, _ = V2.surfaces(q_dft[:, 0], q_dft[:, 1])
        _, z_mlip = V2.surfaces(q_mlip[:, 0], q_mlip[:, 1])
        ax.scatter(
            q_dft[:, 0], q_dft[:, 1], z_dft,
            s=IMAGE_MARKER_SIZE, color=INK, edgecolor="white", linewidth=0.55,
            depthshade=False, zorder=35,
        )
        ax.scatter(
            q_mlip[:, 0], q_mlip[:, 1], z_mlip,
            s=IMAGE_MARKER_SIZE, color=MLIP_PATH_COLOUR, edgecolor="white", linewidth=0.55,
            depthshade=False, zorder=36,
        )
        ax.scatter(
            q_dft[:, 0], q_dft[:, 1], np.full(len(q_dft), z_floor),
            s=10, color=DFT_PATH_COLOUR, edgecolor="white", linewidth=0.35,
            depthshade=False, zorder=37,
        )
        ax.scatter(
            q_mlip[:, 0], q_mlip[:, 1], np.full(len(q_mlip), z_floor),
            s=10, color=MLIP_PATH_COLOUR, edgecolor="white", linewidth=0.35,
            depthshade=False, zorder=38,
        )


def draw_surfaces_marked(ax, q_dft: np.ndarray, q_mlip: np.ndarray, mode: str) -> None:
    V4.draw_surfaces_clean(ax, q_dft, q_mlip, mode)
    scatter_anchor_3d(ax, q_dft, q_mlip, mode)


def synthetic_fixed_profile_marked(ax, q_dft: np.ndarray) -> None:
    V4.synthetic_fixed_profile(ax, q_dft)
    s_anchor = np.linspace(0.0, 1.0, len(q_dft))
    dft_anchor, mlip_anchor = V2.surfaces(q_dft[:, 0], q_dft[:, 1])
    dft_anchor -= dft_anchor[0]
    mlip_anchor -= mlip_anchor[0]
    y_bottom, _ = ax.get_ylim()
    # The faint vertical guides make the fixed-path image correspondence
    # explicit: each DFT/MLIP pair has the same horizontal s coordinate.
    for s, y_dft, y_mlip in zip(s_anchor, dft_anchor, mlip_anchor):
        ax.plot(
            [s, s], [y_bottom, max(float(y_dft), float(y_mlip))],
            color=LIGHT_GREY, lw=0.55, alpha=0.50, zorder=1,
        )
    ax.scatter(
        s_anchor, dft_anchor, s=IMAGE_MARKER_SIZE,
        color=INK, edgecolor="white", linewidth=0.55, zorder=5,
    )
    ax.scatter(
        s_anchor, mlip_anchor, s=IMAGE_MARKER_SIZE,
        color=MLIP_PATH_COLOUR, edgecolor="white", linewidth=0.55, zorder=5,
    )
    for s, y_dft, y_mlip, label in zip(
        s_anchor, dft_anchor, mlip_anchor, IMAGE_LABELS
    ):
        ax.text(
            float(s),
            max(float(y_dft), float(y_mlip)) + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=IMAGE_LABEL_FONTSIZE,
            color=INK,
            zorder=6,
        )


def operator_panel_marked(spec, q_dft: np.ndarray, q_mlip: np.ndarray) -> None:
    inner = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=spec, wspace=0.38, width_ratios=(1.03, 1.0)
    )
    ax_geo = plt.subplot(inner[0, 0])
    ax_energy = plt.subplot(inner[0, 1])
    _, qd_x, qd_y = V2.smooth_path(q_dft)
    _, qm_x, qm_y = V2.smooth_path(q_mlip)

    ax_geo.plot(qd_x, qd_y, color=INK, lw=1.7, ls=(0, (4, 2)), label="DFT path")
    ax_geo.plot(qm_x, qm_y, color=MLIP_PATH_COLOUR, lw=2.0, label="MLIP path")
    # Non-directional paired-image connectors: these show the geometry shift
    # without reintroducing the distracting arrows removed in v4.
    for dft_point, mlip_point in zip(q_dft, q_mlip):
        ax_geo.plot(
            [dft_point[0], mlip_point[0]],
            [dft_point[1], mlip_point[1]],
            color=LIGHT_GREY, lw=0.65, alpha=0.85, zorder=2,
        )
    ax_geo.scatter(
        q_dft[:, 0], q_dft[:, 1], s=IMAGE_MARKER_SIZE,
        color=INK, edgecolor="white", linewidth=0.55, zorder=5,
    )
    ax_geo.scatter(
        q_mlip[:, 0], q_mlip[:, 1], s=IMAGE_MARKER_SIZE,
        color=MLIP_PATH_COLOUR, edgecolor="white", linewidth=0.55, zorder=5,
    )
    label_offsets = ((3, 4), (3, 4), (3, 4), (3, 4), (-12, 4))
    for point, label, offset in zip(q_mlip, IMAGE_LABELS, label_offsets):
        ax_geo.annotate(
            label,
            xy=(float(point[0]), float(point[1])),
            xytext=offset,
            textcoords="offset points",
            fontsize=IMAGE_LABEL_FONTSIZE,
            color=MLIP_PATH_COLOUR,
            ha="left",
            va="bottom",
            zorder=6,
        )
    ax_geo.set_xlabel("$q_1$", fontsize=11.0)
    ax_geo.set_ylabel("$q_2$", fontsize=11.0)
    ax_geo.set_xlim(-0.06, 1.06)
    ax_geo.set_ylim(-0.06, 0.48)
    ax_geo.set_xticks([0.0, 0.5, 1.0])
    ax_geo.set_yticks([0.0, 0.20, 0.40])
    ax_geo.legend(
        loc="upper left", frameon=False, fontsize=8.1, handlelength=1.5,
        borderpad=0.1, labelspacing=0.2,
    )
    V4.clean_axes(ax_geo)

    s_anchor = np.linspace(0.0, 1.0, 5)
    e_dft = np.array([0.04, 0.34, 0.53, 0.30, 0.05])
    e_mlip = np.array([0.035, 0.24, 0.39, 0.22, 0.04])
    s_dense = np.linspace(0.0, 1.0, 240)
    e_dft_dense = V2.monotone_cubic(s_anchor, e_dft, s_dense)
    e_mlip_dense = V2.monotone_cubic(s_anchor, e_mlip, s_dense)
    ax_energy.plot(
        s_dense, e_dft_dense, color=DFT_PATH_COLOUR, lw=1.75,
        ls=(0, (4, 2)), label="DFT",
    )
    ax_energy.plot(
        s_dense, e_mlip_dense, color=MLIP_PATH_COLOUR, lw=2.0,
        label="MLIP",
    )
    ax_energy.scatter(
        s_anchor, e_dft, s=IMAGE_MARKER_SIZE,
        color=DFT_PATH_COLOUR, edgecolor="white", linewidth=0.55, zorder=5,
    )
    ax_energy.scatter(
        s_anchor, e_mlip, s=IMAGE_MARKER_SIZE,
        color=MLIP_PATH_COLOUR, edgecolor="white", linewidth=0.55, zorder=5,
    )
    ax_energy.set_xlabel("path progress, $s$", fontsize=11.0)
    ax_energy.set_ylabel("relative energy", fontsize=11.0)
    ax_energy.set_xlim(-0.03, 1.03)
    ax_energy.set_ylim(-0.02, 0.62)
    ax_energy.set_xticks([0.0, 0.5, 1.0])
    ax_energy.set_yticks([0.0, 0.3, 0.6])
    ax_energy.legend(
        loc="upper left", frameon=False, fontsize=8.0, handlelength=1.5,
        borderpad=0.1, labelspacing=0.2,
    )
    V4.clean_axes(ax_energy)


def render(output: Path) -> None:
    output = output.resolve()
    for path in (DFT_XYZ, MLIP_XYZ):
        if not path.exists():
            raise FileNotFoundError(path)

    apply_publication_style(10.8)
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "dejavusans"})
    dft_frames = V2.read_xyz_frames(DFT_XYZ)
    mlip_frames = V2.read_xyz_frames(MLIP_XYZ)
    q_dft, q_mlip, q_metadata = V2.path_coordinates(dft_frames, mlip_frames)

    fig = plt.figure(figsize=(9.35, 5.85), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=(3.08, 1.62),
        left=0.060,
        right=0.985,
        top=0.875,
        bottom=0.145,
        wspace=0.14,
        hspace=0.43,
    )
    ax_fixed = fig.add_subplot(gs[0, 0], projection="3d")
    ax_self = fig.add_subplot(gs[0, 1], projection="3d")
    draw_surfaces_marked(ax_fixed, q_dft, q_mlip, "fixed")
    draw_surfaces_marked(ax_self, q_dft, q_mlip, "self")

    surface_handles = [
        V4.Patch(facecolor=V4.DFT_SURFACE_HIGH, edgecolor="none", alpha=0.58, label="DFT PES"),
        V4.Patch(facecolor=V4.MLIP_SURFACE_HIGH, edgecolor="none", alpha=0.55, label="MLIP PES"),
    ]
    fig.legend(
        handles=surface_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.945),
        ncol=2,
        frameon=False,
        fontsize=10.2,
        handlelength=1.25,
        columnspacing=1.25,
        borderpad=0.1,
        labelspacing=0.25,
    )

    ax_profile = fig.add_subplot(gs[1, 0])
    synthetic_fixed_profile_marked(ax_profile, q_dft)
    operator_panel_marked(gs[1, 1], q_dft, q_mlip)

    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        save_figure(fig, output.with_suffix(f".{ext}"), dpi=360)
    plt.close(fig)

    path_csv = output.with_name("pes_operator_v5_path_coordinates.csv")
    with path_csv.open("w") as handle:
        handle.write("branch,image_index,image_label,q1,q2,geometry_status\n")
        for branch, q, status in (
            ("DFT_fixed", q_dft, "COORDINATE_DERIVED"),
            ("MLIP_self_relaxed", q_mlip, "COORDINATE_DERIVED_PROVISIONAL"),
        ):
            for index, (q1, q2) in enumerate(q):
                handle.write(
                    f"{branch},{index},{IMAGE_LABELS[index]},{q1:.10f},{q2:.10f},{status}\n"
                )

    manifest = {
        "figure_id": "A1-PES-OPERATOR-SCHEMATIC-V5",
        "status": "RECONSTRUCTION_OPERATOR_SCHEMATIC_WITH_EXPLICIT_FIVE_IMAGE_ANCHORS",
        "base_release": "fig3d_figS3_pes_operator_schematic_v4",
        "surface_status": "synthetic reduced-coordinate PES; z-dependent colour is a visual distinction, not a quantitative PES",
        "fixed_energy_profile_status": "synthetic section evaluated from the same synthetic DFT/MLIP surfaces used above",
        "geometry_status": "five-image coordinate-derived 61-atom DFT fixture and current Foundation MLIP self-relaxed path",
        "self_rescore_status": "synthetic operator definition only; same-system DFT re-score remains unavailable",
        "atom_figure_mapping": {
            "image_labels": list(IMAGE_LABELS),
            "image_indices": [0, 1, 2, 3, 4],
            "standalone_figures": [
                "fig3d_figS3_atom_path_dft_fivepanel_v4",
                "fig3d_figS3_atom_path_mlip_fivepanel_v4",
            ],
        },
        "anchor_policy": {
            "fixed_path": "five DFT/MLIP pairs share the same s positions; faint vertical guides make the alignment explicit",
            "self_relaxed_geometry": "five DFT and five MLIP markers are shown at their q1-q2 coordinates; paired geometry shifts are plain non-directional segments",
            "self_relaxed_energy": "five markers are shown on both synthetic re-score curves at the five path-progress anchors",
            "arrows": False,
            "arrowheads": False,
        },
        "q_coordinate_definition": q_metadata,
        "palette": {
            "name": V4.ANALYTICAL_PALETTE_NAME,
            "roles": {
                "DFT": V4.DFT_PATH_COLOUR,
                "MLIP": V4.MLIP_PATH_COLOUR,
                "DFT_surface_high": V4.DFT_SURFACE_HIGH,
                "MLIP_surface_high": V4.MLIP_SURFACE_HIGH,
                "DFT_surface_low": V4.DFT_SURFACE_LOW,
                "MLIP_surface_low": V4.MLIP_SURFACE_LOW,
            },
            "high_saturation_red_used": False,
        },
        "inputs": {str(path.relative_to(REPO)): sha256(path) for path in (DFT_XYZ, MLIP_XYZ)},
        "path_coordinate_csv": str(path_csv.relative_to(REPO)),
        "source_scripts": {
            "v4_base": str(V4_PATH.relative_to(REPO)),
            "v4_sha256": sha256(V4_PATH),
        },
        "notes": [
            "The five image anchors correspond left-to-right to A, 1, TS, 3, B in the independent atom strips.",
            "The markers are restored as the minimal visual correspondence cue requested in review; no raster atom image is embedded in the main figure.",
            "The lower panels remain synthetic operator schematics and do not imply an unavailable self-consistent DFT re-score.",
            "No historical 0.410 eV record is used.",
        ],
    }
    manifest["script_sha256"] = sha256(Path(__file__))
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
