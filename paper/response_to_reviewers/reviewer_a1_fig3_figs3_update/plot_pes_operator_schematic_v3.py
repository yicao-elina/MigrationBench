"""v3 operator schematic with a unified synthetic energy story.

The v3 composition keeps the coordinate-derived QL geometry and atom renders,
but removes the real fixed-path energy profile from the lower-left panel.  Both
lower energy panels now use the same smooth synthetic PES operator curves as
the upper surfaces.  The shared surface legend is placed in a dedicated
figure-level band, and all axes/ticks are enlarged for final-size inspection.
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
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V2_PATH = HERE / "plot_pes_operator_schematic_v2.py"
ATOM_DIR = HERE / "atom_renders_v3"
DFT_XYZ = REPO / "tests/fixtures/historical_neb/legacy_61/1-4/sb2te3.xyz"
MLIP_XYZ = HERE / "data/self_relaxed_foundation_path_61_1-4.extxyz"

ATLAS = Path(
    __import__("os").environ.get(
        "MIGRATIONBENCH_PLOT_ATLAS", "/Users/alina/Documents/Codex/plot_library"
    )
)
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas import apply_publication_style, save_figure  # noqa: E402


def load_v2_module():
    spec = importlib.util.spec_from_file_location("pes_operator_v2", V2_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {V2_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_v2_module()

INK = V2.INK
DFT_GREY = V2.DFT_GREY
DFT_SURFACE_LOW = V2.DFT_SURFACE_LOW
DFT_SURFACE_HIGH = V2.DFT_SURFACE_HIGH
MLIP_BLUE = V2.MLIP_BLUE
MLIP_SURFACE_LOW = V2.MLIP_SURFACE_LOW
MLIP_SURFACE_HIGH = V2.MLIP_SURFACE_HIGH
PATH_RED = V2.PATH_RED
ERROR_RED = V2.ERROR_RED
FLOOR_GREY = V2.FLOOR_GREY


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def surface_cmap(low: str, high: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("surface", [low, high])


def set_3d_style_v3(ax, title: str, q2_limits: tuple[float, float]) -> None:
    ax.text2D(
        0.01,
        1.015,
        title,
        transform=ax.transAxes,
        fontsize=13.2,
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


def draw_surfaces_v3(ax, q_dft: np.ndarray, q_mlip: np.ndarray, mode: str) -> None:
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
        facecolors=surface_cmap(DFT_SURFACE_LOW, DFT_SURFACE_HIGH)(dft_norm(zz_dft)),
        alpha=0.56,
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
        facecolors=surface_cmap(MLIP_SURFACE_LOW, MLIP_SURFACE_HIGH)(mlip_norm(zz_mlip)),
        alpha=0.48,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
    )
    ax.plot_wireframe(xx[::8, ::8], yy[::8, ::8], zz_dft[::8, ::8], color="#74818C", alpha=0.22, linewidth=0.32)
    ax.plot_wireframe(xx[::8, ::8], yy[::8, ::8], zz_mlip[::8, ::8], color="#3E89C9", alpha=0.20, linewidth=0.32)

    _, qd_x, qd_y = V2.smooth_path(q_dft)
    _, qm_x, qm_y = V2.smooth_path(q_mlip)
    z_floor = 0.018
    z_dft_dft, z_mlip_dft = V2.surfaces(qd_x, qd_y)
    z_dft_mlip, z_mlip_mlip = V2.surfaces(qm_x, qm_y)

    ax.plot(qd_x, qd_y, np.full_like(qd_x, z_floor), color=INK, lw=1.45, ls=(0, (4, 2)), alpha=0.95, zorder=18)
    if mode == "fixed":
        ax.plot(qd_x, qd_y, z_dft_dft, color=INK, lw=2.15, zorder=20)
        ax.plot(qd_x, qd_y, z_mlip_dft, color=MLIP_BLUE, lw=1.95, zorder=21)
        for index in range(1, 4):
            qx, qy = q_dft[index]
            zd, zm = V2.surfaces(np.array([qx]), np.array([qy]))
            lo, hi = sorted((float(zd[0]), float(zm[0])))
            ax.plot([qx, qx], [qy, qy], [lo, hi], color=ERROR_RED, lw=1.55, alpha=0.95)
            ax.scatter([qx, qx], [qy, qy], [lo, hi], s=18, color=[INK, MLIP_BLUE], edgecolor="white", linewidth=0.45, depthshade=False)
            ax.plot([qx, qx], [qy, qy], [z_floor, lo], color=FLOOR_GREY, lw=0.55, ls=(0, (2, 2)), alpha=0.65)
        ax.text2D(
            0.07,
            0.64,
            r"same geometry $R_i^{\rm DFT}$",
            transform=ax.transAxes,
            fontsize=10.7,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.98, "pad": 1.5},
        )
        ax.text2D(
            0.09,
            0.58,
            r"$\delta E_{\rm fixed}$",
            transform=ax.transAxes,
            fontsize=12.0,
            fontweight="bold",
            color=ERROR_RED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.98, "pad": 1.5},
        )
    else:
        ax.plot(qm_x, qm_y, np.full_like(qm_x, z_floor), color=PATH_RED, lw=1.45, ls=(0, (4, 2)), alpha=0.95, zorder=18)
        ax.plot(qd_x, qd_y, z_dft_dft, color=INK, lw=1.75, ls=(0, (4, 2)), zorder=20)
        ax.plot(qm_x, qm_y, z_mlip_mlip, color=PATH_RED, lw=2.25, zorder=23)
        ax.plot(qm_x, qm_y, z_dft_mlip, color=DFT_GREY, lw=1.20, ls=(0, (2, 2)), alpha=0.82, zorder=22)
        for index in range(1, 4):
            qx_d, qy_d = q_dft[index]
            qx_m, qy_m = q_mlip[index]
            ax.plot([qx_d, qx_m], [qy_d, qy_m], [z_floor, z_floor], color=PATH_RED, lw=1.20, ls=(0, (2, 2)), alpha=0.94)
            ax.scatter([qx_d, qx_m], [qy_d, qy_m], [z_floor, z_floor], s=15, color=[INK, PATH_RED], edgecolor="white", linewidth=0.4, depthshade=False)
            ax.plot([qx_d, qx_d], [qy_d, qy_d], [z_floor, float(z_dft_dft[np.argmin(np.abs(qd_x - qx_d))])], color=FLOOR_GREY, lw=0.50, ls=(0, (2, 2)), alpha=0.55)
            ax.plot([qx_m, qx_m], [qy_m, qy_m], [z_floor, float(z_mlip_mlip[np.argmin(np.abs(qm_x - qx_m))])], color=FLOOR_GREY, lw=0.50, ls=(0, (2, 2)), alpha=0.55)
        qx, qy = q_mlip[2]
        dft_at_mlip, mlip_at_mlip = V2.surfaces(np.array([qx]), np.array([qy]))
        lo, hi = sorted((float(dft_at_mlip[0]), float(mlip_at_mlip[0])))
        ax.plot([qx, qx], [qy, qy], [lo, hi], color=ERROR_RED, lw=1.60, zorder=25)
        ax.scatter([qx, qx], [qy, qy], [lo, hi], s=18, color=[DFT_GREY, PATH_RED], edgecolor="white", linewidth=0.45, depthshade=False)
        ax.text2D(
            0.06,
            0.64,
            "1  geometry shift",
            transform=ax.transAxes,
            fontsize=10.7,
            fontweight="bold",
            color=PATH_RED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.98, "pad": 1.5},
        )
        ax.text2D(
            0.07,
            0.58,
            "2  energy mismatch",
            transform=ax.transAxes,
            fontsize=10.7,
            fontweight="bold",
            color=ERROR_RED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.98, "pad": 1.5},
        )
    ax.scatter(q_dft[:, 0], q_dft[:, 1], V2.surfaces(q_dft[:, 0], q_dft[:, 1])[0], s=21, color=INK, edgecolor="white", linewidth=0.5, depthshade=False, zorder=30)
    if mode == "self":
        ax.scatter(q_mlip[:, 0], q_mlip[:, 1], V2.surfaces(q_mlip[:, 0], q_mlip[:, 1])[1], s=21, color=PATH_RED, edgecolor="white", linewidth=0.5, depthshade=False, zorder=31)
    set_3d_style_v3(ax, "(a) Fixed-path scoring" if mode == "fixed" else "(b) Self-relaxed NEB", q2_limits)


def add_snapshot_strip_v3(ax, branch: str, labels: tuple[str, ...], edgecolor: str) -> None:
    for index, label in enumerate(labels):
        left = 0.006 + index * 0.198
        inset = ax.inset_axes([left, 0.68, 0.185, 0.235])
        image = plt.imread(ATOM_DIR / f"{branch}_{index:02d}.png")
        inset.imshow(image, aspect="equal")
        inset.set_axis_off()
        inset.patch.set_facecolor("white")
        inset.patch.set_edgecolor(edgecolor)
        inset.patch.set_linewidth(0.80)
        inset.text(0.05, 0.95, label, transform=inset.transAxes, fontsize=8.6, fontweight="bold", color=INK, ha="left", va="top")


def synthetic_fixed_profile(ax, q_dft: np.ndarray) -> None:
    s_dense, qx, qy = V2.smooth_path(q_dft)
    dft, mlip = V2.surfaces(qx, qy)
    dft = dft - dft[0]
    mlip = mlip - mlip[0]
    ax.plot(s_dense, dft, color=INK, lw=2.0, label="DFT / fixed path", zorder=3)
    ax.plot(s_dense, mlip, color=MLIP_BLUE, lw=2.0, label="MLIP / same path", zorder=3)
    s_anchor = np.linspace(0.0, 1.0, len(q_dft))
    dft_anchor, mlip_anchor = V2.surfaces(q_dft[:, 0], q_dft[:, 1])
    dft_anchor -= dft_anchor[0]
    mlip_anchor -= mlip_anchor[0]
    ax.scatter(s_anchor, dft_anchor, s=24, color=INK, edgecolor="white", linewidth=0.45, zorder=4)
    ax.scatter(s_anchor, mlip_anchor, s=24, color=MLIP_BLUE, edgecolor="white", linewidth=0.45, zorder=4)
    for index in (1, 2, 3):
        ax.annotate("", xy=(s_anchor[index], mlip_anchor[index]), xytext=(s_anchor[index], dft_anchor[index]), arrowprops={"arrowstyle": "<->", "color": ERROR_RED, "lw": 1.0})
    ax.text(0.03, 0.94, r"$\delta E_{\rm fixed}$", transform=ax.transAxes, fontsize=11.0, color=ERROR_RED, va="top", fontweight="bold")
    ax.set_title("fixed-path section (schematic)", loc="left", fontsize=10.8, fontweight="bold", pad=5)
    ax.set_xlabel("path progress, $s$", fontsize=11.0)
    ax.set_ylabel("relative energy", fontsize=11.0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(min(float(dft.min()), float(mlip.min())) - 0.08, max(float(dft.max()), float(mlip.max())) + 0.10)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.legend(loc="upper right", frameon=False, fontsize=8.9, handlelength=1.8, borderpad=0.1, labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.3, width=0.8, length=3.2)


def operator_panel_v3(spec, q_dft: np.ndarray, q_mlip: np.ndarray) -> None:
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.38, width_ratios=(1.03, 1.0))
    ax_geo = plt.subplot(inner[0, 0])
    ax_energy = plt.subplot(inner[0, 1])
    _, qd_x, qd_y = V2.smooth_path(q_dft)
    _, qm_x, qm_y = V2.smooth_path(q_mlip)
    ax_geo.plot(qd_x, qd_y, color=INK, lw=1.7, ls=(0, (4, 2)), label="DFT path")
    ax_geo.plot(qm_x, qm_y, color=PATH_RED, lw=2.0, label="MLIP relaxed")
    ax_geo.scatter(q_dft[:, 0], q_dft[:, 1], s=22, color=INK, zorder=4)
    ax_geo.scatter(q_mlip[:, 0], q_mlip[:, 1], s=22, color=PATH_RED, zorder=4)
    for index in (1, 2, 3):
        ax_geo.annotate("", xy=(q_mlip[index, 0], q_mlip[index, 1]), xytext=(q_dft[index, 0], q_dft[index, 1]), arrowprops={"arrowstyle": "<->", "color": PATH_RED, "lw": 0.95})
    ax_geo.annotate("geometry shift", xy=(0.53, 0.25), xycoords="data", xytext=(0.86, 0.80), textcoords="axes fraction", fontsize=8.7, color=PATH_RED, ha="center", va="center", fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.3}, arrowprops={"arrowstyle": "-", "color": PATH_RED, "lw": 0.70, "connectionstyle": "arc3,rad=-0.18"})
    ax_geo.set_title("1  relax the path", loc="left", fontsize=10.8, fontweight="bold", pad=5)
    ax_geo.set_xlabel("$q_1$", fontsize=11.0)
    ax_geo.set_ylabel("$q_2$", fontsize=11.0)
    ax_geo.set_xlim(-0.06, 1.06)
    ax_geo.set_ylim(-0.06, 0.48)
    ax_geo.set_xticks([0.0, 0.5, 1.0])
    ax_geo.set_yticks([0.0, 0.20, 0.40])
    ax_geo.legend(loc="upper left", frameon=False, fontsize=8.1, handlelength=1.5, borderpad=0.1, labelspacing=0.2)
    ax_geo.spines[["top", "right"]].set_visible(False)
    ax_geo.tick_params(labelsize=9.0, width=0.75, length=3.0)

    s_anchor = np.linspace(0.0, 1.0, 5)
    e_dft = np.array([0.04, 0.34, 0.53, 0.30, 0.05])
    e_mlip = np.array([0.035, 0.24, 0.39, 0.22, 0.04])
    s_dense = np.linspace(0.0, 1.0, 240)
    e_dft_dense = V2.monotone_cubic(s_anchor, e_dft, s_dense)
    e_mlip_dense = V2.monotone_cubic(s_anchor, e_mlip, s_dense)
    ax_energy.plot(s_dense, e_dft_dense, color=DFT_GREY, lw=1.75, ls=(0, (4, 2)), label="DFT at relaxed")
    ax_energy.plot(s_dense, e_mlip_dense, color=PATH_RED, lw=2.0, label="MLIP at relaxed")
    ax_energy.scatter(s_anchor, e_dft, s=22, facecolor="white", edgecolor=DFT_GREY, linewidth=0.85, zorder=4)
    ax_energy.scatter(s_anchor, e_mlip, s=22, facecolor="#F5A3AF", edgecolor=PATH_RED, linewidth=0.85, zorder=4)
    ax_energy.annotate("", xy=(0.5, e_dft[2]), xytext=(0.5, e_mlip[2]), arrowprops={"arrowstyle": "<->", "color": ERROR_RED, "lw": 1.05})
    ax_energy.annotate("model mismatch", xy=(0.50, 0.46), xycoords="data", xytext=(0.78, 0.70), textcoords="axes fraction", fontsize=8.4, color=ERROR_RED, ha="center", va="center", fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.3}, arrowprops={"arrowstyle": "-", "color": ERROR_RED, "lw": 0.70, "connectionstyle": "arc3,rad=0.18"})
    ax_energy.set_title("2  re-score on relaxed path (schematic)", loc="left", fontsize=10.8, fontweight="bold", pad=5)
    ax_energy.set_xlabel("path progress, $s$", fontsize=11.0)
    ax_energy.set_ylabel("relative energy", fontsize=11.0)
    ax_energy.set_xlim(-0.03, 1.03)
    ax_energy.set_ylim(-0.02, 0.62)
    ax_energy.set_xticks([0.0, 0.5, 1.0])
    ax_energy.set_yticks([0.0, 0.3, 0.6])
    ax_energy.legend(loc="upper left", frameon=False, fontsize=8.0, handlelength=1.5, borderpad=0.1, labelspacing=0.2)
    ax_energy.spines[["top", "right"]].set_visible(False)
    ax_energy.tick_params(labelsize=9.0, width=0.75, length=3.0)


def render(output: Path) -> None:
    output = output.resolve()
    for path in (DFT_XYZ, MLIP_XYZ):
        if not path.exists():
            raise FileNotFoundError(path)
    for branch in ("dft", "mlip"):
        for index in range(5):
            if not (ATOM_DIR / f"{branch}_{index:02d}.png").exists():
                raise FileNotFoundError(ATOM_DIR / f"{branch}_{index:02d}.png")

    apply_publication_style(10.8)
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "dejavusans"})
    dft_frames = V2.read_xyz_frames(DFT_XYZ)
    mlip_frames = V2.read_xyz_frames(MLIP_XYZ)
    q_dft, q_mlip, q_metadata = V2.path_coordinates(dft_frames, mlip_frames)

    fig = plt.figure(figsize=(9.35, 6.35), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=(3.30, 1.62),
        left=0.060,
        right=0.985,
        top=0.915,
        bottom=0.145,
        wspace=0.14,
        hspace=0.39,
    )
    ax_fixed = fig.add_subplot(gs[0, 0], projection="3d")
    ax_self = fig.add_subplot(gs[0, 1], projection="3d")
    draw_surfaces_v3(ax_fixed, q_dft, q_mlip, "fixed")
    draw_surfaces_v3(ax_self, q_dft, q_mlip, "self")
    add_snapshot_strip_v3(ax_fixed, "dft", ("A", "1", "TS", "3", "B"), INK)
    add_snapshot_strip_v3(ax_self, "mlip", ("A", "1", "TS", "3", "B"), PATH_RED)
    surface_handles = [
        Patch(facecolor=DFT_SURFACE_HIGH, edgecolor="none", alpha=0.58, label="DFT PES"),
        Patch(facecolor=MLIP_SURFACE_HIGH, edgecolor="none", alpha=0.55, label="MLIP PES"),
    ]
    fig.legend(handles=surface_handles, loc="upper center", bbox_to_anchor=(0.50, 0.945), ncol=2, frameon=False, fontsize=10.2, handlelength=1.25, columnspacing=1.25, borderpad=0.1, labelspacing=0.25)

    ax_profile = fig.add_subplot(gs[1, 0])
    synthetic_fixed_profile(ax_profile, q_dft)
    operator_panel_v3(gs[1, 1], q_dft, q_mlip)
    # The lower-left curves are literal sections of the same synthetic paths
    # drawn above.  These two restrained connectors make that relationship
    # explicit without adding another prose legend inside the panels.
    for color, upper_xy, lower_xy in (
        (INK, (0.53, 0.31), (0.50, 0.99)),
        (MLIP_BLUE, (0.56, 0.28), (0.56, 0.99)),
    ):
        fig.add_artist(
            ConnectionPatch(
                xyA=upper_xy,
                coordsA=ax_fixed.transAxes,
                xyB=lower_xy,
                coordsB=ax_profile.transAxes,
                arrowstyle="->",
                mutation_scale=8,
                lw=0.80,
                ls=(0, (2, 2)),
                color=color,
                alpha=0.62,
                clip_on=False,
            )
        )
    fig.text(
        0.545,
        0.045,
        r"$\Delta E_i=[E_{\rm DFT}(R_i^{\rm MLIP})-E_{\rm DFT}(R_i^{\rm DFT})]_{\rm geometry/path}+[E_{\rm MLIP}(R_i^{\rm MLIP})-E_{\rm DFT}(R_i^{\rm MLIP})]_{\rm energy\ model}$",
        ha="center",
        va="center",
        fontsize=8.7,
        color=INK,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        save_figure(fig, output.with_suffix(f".{ext}"), dpi=360)
    plt.close(fig)

    path_csv = output.with_name("pes_operator_v3_path_coordinates.csv")
    with path_csv.open("w") as handle:
        handle.write("branch,image_index,q1,q2,geometry_status\n")
        for branch, q, status in (("DFT_fixed", q_dft, "COORDINATE_DERIVED"), ("MLIP_self_relaxed", q_mlip, "COORDINATE_DERIVED_PROVISIONAL")):
            for index, (q1, q2) in enumerate(q):
                handle.write(f"{branch},{index},{q1:.10f},{q2:.10f},{status}\n")
    manifest = {
        "figure_id": "A1-PES-OPERATOR-SCHEMATIC-V3",
        "status": "RECONSTRUCTION_OPERATOR_SCHEMATIC_WITH_COORDINATE_DERIVED_QL_VISUALS",
        "surface_status": "synthetic reduced-coordinate PES; z-dependent color is a visual distinction, not a quantitative PES",
        "fixed_energy_profile_status": "synthetic section evaluated from the same synthetic DFT/MLIP surfaces used above; real fixed-path profile intentionally excluded for visual consistency",
        "geometry_status": "five-image coordinate-derived 61-atom DFT fixture and current Foundation MLIP self-relaxed path",
        "self_rescore_status": "synthetic operator definition only; same-system DFT re-score remains unavailable",
        "atom_render_status": "compact QL cross-section with explicit five-image Cr trajectory; structural visual anchor only",
        "q_coordinate_definition": q_metadata,
        "inputs": {str(path.relative_to(REPO)): sha256(path) for path in (DFT_XYZ, MLIP_XYZ)},
        "atom_render_manifest": str((ATOM_DIR / "atom_render_manifest.json").relative_to(REPO)),
        "path_coordinate_csv": str(path_csv.relative_to(REPO)),
        "source_scripts": {"v2_base": str(V2_PATH.relative_to(REPO)), "v2_sha256": sha256(V2_PATH)},
        "notes": [
            "All energy/profile panels are synthetic operator schematics for visual consistency.",
            "The upper and lower path curves reuse the same smooth synthetic surface evaluation, so the lower panels read as sections of the upper operators.",
            "The five atom snapshots use coordinate-derived QL cross-sections and an explicit Cr trajectory; they do not establish DFT convergence or a publication barrier.",
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
