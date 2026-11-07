"""Clean main PES-operator schematic for the reviewer-response figure.

v4 separates the structural atom renders from the operator figure.  The main
figure therefore contains only the two synthetic reduced-coordinate PES
surfaces, their smooth paths and projections, axes, and concise legends.  All
callout arrows, step annotations, energy-mismatch arrows, decomposition text,
and atom thumbnails are intentionally omitted.  The independent atom figures
are composed by ``compose_atom_path_fivepanel_v4.py``.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V3_PATH = HERE / "plot_pes_operator_schematic_v3.py"
DFT_XYZ = REPO / "tests/fixtures/historical_neb/legacy_61/1-4/sb2te3.xyz"
MLIP_XYZ = HERE / "data/self_relaxed_foundation_path_61_1-4.extxyz"

ATLAS = Path(
    __import__("os").environ.get(
        "MIGRATIONBENCH_PLOT_ATLAS", "/Users/alina/Documents/Codex/plot_library"
    )
)
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas import apply_publication_style, save_figure  # noqa: E402
from plot_atlas.palettes import get_palette  # noqa: E402


def load_v3_module():
    spec = importlib.util.spec_from_file_location("pes_operator_v3", V3_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {V3_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V3 = load_v3_module()
V2 = V3.V2

ANALYTICAL_PALETTE_NAME = "nature_steered_dynamics"
ANALYTICAL_PALETTE = get_palette(ANALYTICAL_PALETTE_NAME)


def blend_with_white(hex_colour: str, white_fraction: float) -> str:
    """Make a restrained surface tint from a registered Atlas colour."""
    value = hex_colour.lstrip("#")
    rgb = np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    mixed = (1.0 - white_fraction) * rgb + white_fraction * np.ones(3)
    return "#" + "".join(f"{int(round(component * 255)):02X}" for component in mixed)


INK = ANALYTICAL_PALETTE["ink"]
DFT_PATH_COLOUR = ANALYTICAL_PALETTE["roles"]["pbe"]
MLIP_PATH_COLOUR = ANALYTICAL_PALETTE["roles"]["ml_global"]
DFT_SURFACE_HIGH = DFT_PATH_COLOUR
MLIP_SURFACE_HIGH = MLIP_PATH_COLOUR
DFT_SURFACE_LOW = blend_with_white(DFT_SURFACE_HIGH, 0.82)
MLIP_SURFACE_LOW = blend_with_white(MLIP_SURFACE_HIGH, 0.82)
FLOOR_GREY = blend_with_white(INK, 0.55)
LIGHT_GREY = blend_with_white(INK, 0.72)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def surface_cmap(low: str, high: str, name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [low, high])


def set_3d_style_clean(ax, title: str, q2_limits: tuple[float, float]) -> None:
    ax.text2D(
        0.01,
        1.015,
        title,
        transform=ax.transAxes,
        fontsize=13.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(*q2_limits)
    ax.set_zlim(0.0, 0.95)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.20, 0.40])
    ax.set_zticks([0.0, 0.4, 0.8])
    ax.set_xlabel(r"$q_1$", labelpad=1, fontsize=12.0)
    ax.set_ylabel(r"$q_2$", labelpad=1, fontsize=12.0)
    ax.set_zlabel("energy", labelpad=2, fontsize=11.5)
    ax.tick_params(labelsize=9.2, width=0.75, pad=1, length=3)
    ax.view_init(elev=29, azim=-62)
    ax.set_box_aspect((1.40, 0.72, 0.88))
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.50, 0.55, 0.60, 0.13)
        axis._axinfo["grid"]["linewidth"] = 0.35
        axis.line.set_color(INK)
        axis.line.set_linewidth(0.7)


def draw_surfaces_clean(ax, q_dft: np.ndarray, q_mlip: np.ndarray, mode: str) -> None:
    """Draw the two distinguishable PES manifolds without callouts/arrows."""
    q2_limits = (-0.12, 0.48)
    q1_grid = np.linspace(-0.06, 1.06, 65)
    q2_grid = np.linspace(q2_limits[0], q2_limits[1], 65)
    xx, yy = np.meshgrid(q1_grid, q2_grid)
    zz_dft, zz_mlip = V2.surfaces(xx, yy)
    dft_norm = Normalize(vmin=float(zz_dft.min()), vmax=float(zz_dft.max()))
    mlip_norm = Normalize(vmin=float(zz_mlip.min()), vmax=float(zz_mlip.max()))

    ax.plot_surface(
        xx,
        yy,
        zz_dft,
        facecolors=surface_cmap(DFT_SURFACE_LOW, DFT_SURFACE_HIGH, "dft_surface")(
            dft_norm(zz_dft)
        ),
        alpha=0.58,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
    )
    ax.plot_surface(
        xx,
        yy,
        zz_mlip,
        facecolors=surface_cmap(MLIP_SURFACE_LOW, MLIP_SURFACE_HIGH, "mlip_surface")(
            mlip_norm(zz_mlip)
        ),
        alpha=0.48,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
    )
    ax.plot_wireframe(
        xx[::8, ::8],
        yy[::8, ::8],
        zz_dft[::8, ::8],
        color=DFT_SURFACE_HIGH,
        alpha=0.25,
        linewidth=0.32,
    )
    ax.plot_wireframe(
        xx[::8, ::8],
        yy[::8, ::8],
        zz_mlip[::8, ::8],
        color=MLIP_SURFACE_HIGH,
        alpha=0.22,
        linewidth=0.32,
    )

    _, qd_x, qd_y = V2.smooth_path(q_dft)
    _, qm_x, qm_y = V2.smooth_path(q_mlip)
    z_floor = 0.018
    z_dft_dft, z_mlip_dft = V2.surfaces(qd_x, qd_y)
    z_dft_mlip, z_mlip_mlip = V2.surfaces(qm_x, qm_y)

    # The floor curves are the explicit 2-D projections of the paths.  They
    # are the only geometry connectors retained in this clean figure.
    ax.plot(
        qd_x,
        qd_y,
        np.full_like(qd_x, z_floor),
        color=DFT_PATH_COLOUR,
        lw=1.35,
        ls=(0, (4, 2)),
        alpha=0.95,
        zorder=18,
    )
    if mode == "fixed":
        ax.plot(qd_x, qd_y, z_dft_dft, color=INK, lw=2.10, zorder=20)
        ax.plot(qd_x, qd_y, z_mlip_dft, color=MLIP_PATH_COLOUR, lw=1.95, zorder=21)
    else:
        ax.plot(
            qm_x,
            qm_y,
            np.full_like(qm_x, z_floor),
            color=MLIP_PATH_COLOUR,
            lw=1.35,
            ls=(0, (4, 2)),
            alpha=0.95,
            zorder=18,
        )
        ax.plot(
            qd_x,
            qd_y,
            z_dft_dft,
            color=INK,
            lw=1.75,
            ls=(0, (4, 2)),
            zorder=20,
        )
        ax.plot(qm_x, qm_y, z_mlip_mlip, color=MLIP_PATH_COLOUR, lw=2.20, zorder=23)
        ax.plot(
            qm_x,
            qm_y,
            z_dft_mlip,
            color=DFT_PATH_COLOUR,
            lw=1.15,
            ls=(0, (2, 2)),
            alpha=0.88,
            zorder=22,
        )

    set_3d_style_clean(
        ax,
        "(a) Fixed-path scoring" if mode == "fixed" else "(b) Self-relaxed NEB",
        q2_limits,
    )


def clean_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.3, width=0.8, length=3.2)


def synthetic_fixed_profile(ax, q_dft: np.ndarray) -> None:
    s_dense, qx, qy = V2.smooth_path(q_dft)
    dft, mlip = V2.surfaces(qx, qy)
    dft = dft - dft[0]
    mlip = mlip - mlip[0]
    ax.plot(s_dense, dft, color=INK, lw=2.0, label="DFT")
    ax.plot(s_dense, mlip, color=MLIP_PATH_COLOUR, lw=2.0, label="MLIP")
    ax.set_xlabel("path progress, $s$", fontsize=11.0)
    ax.set_ylabel("relative energy", fontsize=11.0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(min(float(dft.min()), float(mlip.min())) - 0.08, max(float(dft.max()), float(mlip.max())) + 0.10)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.legend(loc="upper right", frameon=False, fontsize=8.9, handlelength=1.8, borderpad=0.1, labelspacing=0.25)
    clean_axes(ax)


def operator_panel_clean(spec, q_dft: np.ndarray, q_mlip: np.ndarray) -> None:
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.38, width_ratios=(1.03, 1.0))
    ax_geo = plt.subplot(inner[0, 0])
    ax_energy = plt.subplot(inner[0, 1])
    _, qd_x, qd_y = V2.smooth_path(q_dft)
    _, qm_x, qm_y = V2.smooth_path(q_mlip)

    ax_geo.plot(qd_x, qd_y, color=INK, lw=1.7, ls=(0, (4, 2)), label="DFT path")
    ax_geo.plot(qm_x, qm_y, color=MLIP_PATH_COLOUR, lw=2.0, label="MLIP path")
    ax_geo.set_xlabel("$q_1$", fontsize=11.0)
    ax_geo.set_ylabel("$q_2$", fontsize=11.0)
    ax_geo.set_xlim(-0.06, 1.06)
    ax_geo.set_ylim(-0.06, 0.48)
    ax_geo.set_xticks([0.0, 0.5, 1.0])
    ax_geo.set_yticks([0.0, 0.20, 0.40])
    ax_geo.legend(loc="upper left", frameon=False, fontsize=8.1, handlelength=1.5, borderpad=0.1, labelspacing=0.2)
    clean_axes(ax_geo)

    # Synthetic lower-right curves are a visual section of the same operator
    # story; no unavailable self-consistent DFT re-score is implied.
    s_anchor = np.linspace(0.0, 1.0, 5)
    e_dft = np.array([0.04, 0.34, 0.53, 0.30, 0.05])
    e_mlip = np.array([0.035, 0.24, 0.39, 0.22, 0.04])
    s_dense = np.linspace(0.0, 1.0, 240)
    e_dft_dense = V2.monotone_cubic(s_anchor, e_dft, s_dense)
    e_mlip_dense = V2.monotone_cubic(s_anchor, e_mlip, s_dense)
    ax_energy.plot(s_dense, e_dft_dense, color=DFT_PATH_COLOUR, lw=1.75, ls=(0, (4, 2)), label="DFT")
    ax_energy.plot(s_dense, e_mlip_dense, color=MLIP_PATH_COLOUR, lw=2.0, label="MLIP")
    ax_energy.set_xlabel("path progress, $s$", fontsize=11.0)
    ax_energy.set_ylabel("relative energy", fontsize=11.0)
    ax_energy.set_xlim(-0.03, 1.03)
    ax_energy.set_ylim(-0.02, 0.62)
    ax_energy.set_xticks([0.0, 0.5, 1.0])
    ax_energy.set_yticks([0.0, 0.3, 0.6])
    ax_energy.legend(loc="upper left", frameon=False, fontsize=8.0, handlelength=1.5, borderpad=0.1, labelspacing=0.2)
    clean_axes(ax_energy)


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

    # The upper axes retain the open white band formerly occupied by the atom
    # thumbnails.  No raster atom image is embedded in this figure.
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
    draw_surfaces_clean(ax_fixed, q_dft, q_mlip, "fixed")
    draw_surfaces_clean(ax_self, q_dft, q_mlip, "self")

    surface_handles = [
        Patch(facecolor=DFT_SURFACE_HIGH, edgecolor="none", alpha=0.58, label="DFT PES"),
        Patch(facecolor=MLIP_SURFACE_HIGH, edgecolor="none", alpha=0.55, label="MLIP PES"),
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
    synthetic_fixed_profile(ax_profile, q_dft)
    operator_panel_clean(gs[1, 1], q_dft, q_mlip)

    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        save_figure(fig, output.with_suffix(f".{ext}"), dpi=360)
    plt.close(fig)

    path_csv = output.with_name("pes_operator_v4_path_coordinates.csv")
    with path_csv.open("w") as handle:
        handle.write("branch,image_index,q1,q2,geometry_status\n")
        for branch, q, status in (
            ("DFT_fixed", q_dft, "COORDINATE_DERIVED"),
            ("MLIP_self_relaxed", q_mlip, "COORDINATE_DERIVED_PROVISIONAL"),
        ):
            for index, (q1, q2) in enumerate(q):
                handle.write(f"{branch},{index},{q1:.10f},{q2:.10f},{status}\n")

    manifest = {
        "figure_id": "A1-PES-OPERATOR-SCHEMATIC-V4",
        "status": "RECONSTRUCTION_OPERATOR_SCHEMATIC_CLEAN_MAIN_WITH_SEPARATE_ATOM_FIGURES",
        "surface_status": "synthetic reduced-coordinate PES; z-dependent colour is a visual distinction, not a quantitative PES",
        "fixed_energy_profile_status": "synthetic section evaluated from the same synthetic DFT/MLIP surfaces used above",
        "geometry_status": "five-image coordinate-derived 61-atom DFT fixture and current Foundation MLIP self-relaxed path",
        "self_rescore_status": "synthetic operator definition only; same-system DFT re-score remains unavailable",
        "atom_figure_policy": "atom thumbnails removed from main figure and composed as two independent five-panel figures",
        "upper_band_policy": "open white space retained above the PES surfaces for later author-side atom placement",
        "omitted_from_main": [
            "atom thumbnails",
            "directional arrows and arrowheads",
            "vertical energy-mismatch arrows",
            "geometry-shift connectors",
            "delta-E fixed callout",
            "step 1 / step 2 callout text",
            "model-mismatch callout",
            "bottom decomposition equation",
        ],
        "q_coordinate_definition": q_metadata,
        "palette": {
            "name": ANALYTICAL_PALETTE_NAME,
            "roles": {
                "DFT": DFT_PATH_COLOUR,
                "MLIP": MLIP_PATH_COLOUR,
                "DFT_surface_high": DFT_SURFACE_HIGH,
                "MLIP_surface_high": MLIP_SURFACE_HIGH,
                "DFT_surface_low": DFT_SURFACE_LOW,
                "MLIP_surface_low": MLIP_SURFACE_LOW,
            },
            "high_saturation_red_used": False,
        },
        "inputs": {str(path.relative_to(REPO)): sha256(path) for path in (DFT_XYZ, MLIP_XYZ)},
        "path_coordinate_csv": str(path_csv.relative_to(REPO)),
        "separate_atom_figures": {
            "renderer": "render_atom_frames_v4.py",
            "composition": "compose_atom_path_fivepanel_v4.py",
            "render_manifest": "atom_renders_v4/atom_render_manifest.json",
        },
        "source_scripts": {
            "v3_base": str(V3_PATH.relative_to(REPO)),
            "v3_sha256": sha256(V3_PATH),
        },
        "notes": [
            "Both lower panels are synthetic schematic sections so the two operator branches remain visually consistent.",
            "The upper paths include their 2-D projections on the q1-q2 floor; no arrows are used to explain the operators.",
            "The five atom snapshots are coordinate-derived structural visual anchors only; they do not establish DFT convergence or a publication barrier.",
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
