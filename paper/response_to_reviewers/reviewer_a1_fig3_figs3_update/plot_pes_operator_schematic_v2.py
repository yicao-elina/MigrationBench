"""Publication-oriented v2 of the fixed-path/self-relaxed NEB operator figure.

This is a hybrid evidence figure:

* the fixed-path profile is the current 1--4 DFT candidate profile and the
  FT-600K evaluation on the same five geometries;
* the q1/q2 path coordinates are derived from the actual Cr displacement in
  the 61-atom DFT path and the current Foundation MLIP self-relaxed path;
* the atom snapshots are POV-Ray renders of those same five frames;
* the reduced PES surfaces and the second-stage re-score curves are a smooth,
  deterministic operator schematic, not a quantitative PES or a completed
  same-system DFT re-score.

All text, axes, paths, equations, and annotations remain vector/editable in
the SVG export.  Only the atomistic thumbnails are embedded raster images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from matplotlib.patches import Patch

ATLAS = Path(
    __import__("os").environ.get(
        "MIGRATIONBENCH_PLOT_ATLAS", "plot_library"
    )
)
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas import apply_publication_style, save_figure  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DFT_PROFILE = REPO / "data/processed/dft_neb_profiles.csv"
MODEL_PROFILE = REPO / "data/raw_compact/revision1/neb_benchmark/data/1-4/neb_predictions.csv"
DFT_XYZ = REPO / "tests/fixtures/historical_neb/legacy_61/1-4/sb2te3.xyz"
MLIP_XYZ = HERE / "data/self_relaxed_foundation_path_61_1-4.extxyz"
ATOM_DIR = HERE / "atom_renders_v2"

INK = "#111111"
DFT_GREY = "#394653"
DFT_SURFACE_LOW = "#E6EBF0"
DFT_SURFACE_HIGH = "#445462"
MLIP_BLUE = "#2F72B8"
MLIP_SURFACE_LOW = "#D8EEFF"
MLIP_SURFACE_HIGH = "#1D5E9F"
PATH_RED = "#C83E4D"
ERROR_RED = "#D1374B"
FLOOR_GREY = "#9EA8B2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_xyz_frames(path: Path) -> list[list[tuple[str, float, float, float]]]:
    """Read ordinary XYZ and extXYZ without requiring ASE in the plot venv."""
    lines = path.read_text().splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        n_atoms = int(lines[cursor].strip())
        start = cursor + 2
        atoms = []
        for line in lines[start : start + n_atoms]:
            fields = line.split()
            atoms.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
        frames.append(atoms)
        cursor = start + n_atoms
    return frames


def load_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dft_rows = []
    with DFT_PROFILE.open() as handle:
        for row in csv.DictReader(handle):
            if row["path"] == "1-4":
                dft_rows.append(row)
    dft_rows.sort(key=lambda row: int(row["image_index"]))
    dft = np.array([float(row["rel_energy_eV"]) for row in dft_rows])

    model_rows = []
    with MODEL_PROFILE.open() as handle:
        for row in csv.DictReader(handle):
            if row["model_name"] == "FT-600K":
                model_rows.append(row)
    model_rows.sort(key=lambda row: int(row["image_index"]))
    mlip = np.array([float(row["predicted_energy_eV"]) for row in model_rows])
    mlip = mlip - mlip[0]
    s = np.linspace(0.0, 1.0, len(dft))
    return s, dft, mlip


def path_coordinates(
    dft_frames: list[list[tuple[str, float, float, float]]],
    mlip_frames: list[list[tuple[str, float, float, float]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Project the migrant's real Cartesian displacement into two q axes.

    q1 is the Cr displacement along the common A-to-B endpoint vector,
    normalized by the endpoint distance. q2 is the orthogonal Cr displacement
    component, normalized by the same distance. This is a transparent reduced
    coordinate for displaying the actual two paths; it is not a claim that the
    3N-dimensional PES has only two physical degrees of freedom.
    """
    dft_symbols = [row[0] for row in dft_frames[0]]
    mlip_symbols = [row[0] for row in mlip_frames[0]]
    if dft_symbols != mlip_symbols:
        raise ValueError("DFT and MLIP structures do not have the same atom order")
    if len(dft_frames) != 5 or len(mlip_frames) != 5:
        raise ValueError("Expected five images in each path")
    cr_index = dft_symbols.index("Cr")

    # Read only the Cr coordinate so the reduced axes remain interpretable.
    dft_cr = np.array(
        [[dft_frames[i][cr_index][1], dft_frames[i][cr_index][2], dft_frames[i][cr_index][3]] for i in range(5)]
    )
    mlip_cr = np.array(
        [[mlip_frames[i][cr_index][1], mlip_frames[i][cr_index][2], mlip_frames[i][cr_index][3]] for i in range(5)]
    )
    origin = dft_cr[0]
    endpoint = dft_cr[-1] - origin
    endpoint_distance = float(np.linalg.norm(endpoint))
    if endpoint_distance <= 1e-8:
        raise ValueError("Endpoint displacement is too small for q1/q2 projection")
    e1 = endpoint / endpoint_distance
    residual = dft_cr[2] - origin - np.dot(dft_cr[2] - origin, e1) * e1
    e2_norm = float(np.linalg.norm(residual))
    if e2_norm <= 1e-8:
        raise ValueError("DFT central image does not define a transverse q2 axis")
    e2 = residual / e2_norm

    def project(coords: np.ndarray) -> np.ndarray:
        displacement = coords - origin
        return np.column_stack(
            [displacement @ e1 / endpoint_distance, displacement @ e2 / endpoint_distance]
        )

    q_dft = project(dft_cr)
    q_mlip = project(mlip_cr)
    metadata = {
        "cr_index_zero_based": float(cr_index),
        "endpoint_distance_A": endpoint_distance,
        "q1_axis_x": float(e1[0]),
        "q1_axis_y": float(e1[1]),
        "q1_axis_z": float(e1[2]),
        "q2_axis_x": float(e2[0]),
        "q2_axis_y": float(e2[1]),
        "q2_axis_z": float(e2[2]),
    }
    return q_dft, q_mlip, metadata


def smooth_path(q: np.ndarray, n: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, len(q))
    dense = np.linspace(0.0, 1.0, n)
    x = monotone_cubic(s, q[:, 0], dense)
    y = monotone_cubic(s, q[:, 1], dense)
    return dense, x, y


def monotone_cubic(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Small NumPy-only PCHIP implementation for smooth, non-overshooting paths."""
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    if len(y) == 2:
        slopes[:] = delta[0]
    else:
        for i in range(1, len(y) - 1):
            if delta[i - 1] * delta[i] <= 0.0:
                slopes[i] = 0.0
            else:
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        slopes[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
        slopes[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
        if slopes[0] * delta[0] <= 0.0:
            slopes[0] = 0.0
        elif abs(slopes[0]) > 3.0 * abs(delta[0]):
            slopes[0] = 3.0 * delta[0]
        if slopes[-1] * delta[-1] <= 0.0:
            slopes[-1] = 0.0
        elif abs(slopes[-1]) > 3.0 * abs(delta[-1]):
            slopes[-1] = 3.0 * delta[-1]
    values = np.empty_like(x_new, dtype=float)
    indices = np.searchsorted(x, x_new, side="right") - 1
    indices = np.clip(indices, 0, len(x) - 2)
    t = (x_new - x[indices]) / h[indices]
    h00 = (2.0 * t**3 - 3.0 * t**2 + 1.0)
    h10 = (t**3 - 2.0 * t**2 + t)
    h01 = (-2.0 * t**3 + 3.0 * t**2)
    h11 = (t**3 - t**2)
    values[:] = h00 * y[indices] + h10 * h[indices] * slopes[indices] + h01 * y[indices + 1] + h11 * h[indices] * slopes[indices + 1]
    return values


def surfaces(q1: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smooth reduced-coordinate surfaces used only for the operator schematic."""
    background = (
        0.10
        + 0.028 * (q2 + 0.05) ** 2
        + 0.035 * np.sin(2.0 * np.pi * q1) * np.cos(4.0 * np.pi * q2)
    )
    dft = background + 0.58 * np.exp(
        -((q1 - 0.52) / 0.24) ** 2 - ((q2 - 0.15) / 0.13) ** 2
    )
    dft += 0.11 * np.exp(-((q1 - 0.16) / 0.15) ** 2 - ((q2 + 0.01) / 0.08) ** 2)
    mlip = (
        0.075
        + 0.025 * (q2 + 0.015) ** 2
        + 0.030 * np.sin(2.0 * np.pi * q1 + 0.28) * np.cos(3.0 * np.pi * q2)
        + 0.46 * np.exp(-((q1 - 0.62) / 0.29) ** 2 - ((q2 - 0.20) / 0.17) ** 2)
    )
    return dft, mlip


def surface_cmap(low: str, high: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("surface", [low, high])


def set_3d_style(ax, title: str, q2_limits: tuple[float, float]) -> None:
    ax.text2D(
        0.01,
        1.015,
        title,
        transform=ax.transAxes,
        fontsize=11.2,
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
    ax.set_xlabel(r"$q_1$", labelpad=-1, fontsize=9.5)
    ax.set_ylabel(r"$q_2$", labelpad=-2, fontsize=9.5)
    ax.set_zlabel("energy", labelpad=-5, fontsize=9.0)
    ax.tick_params(labelsize=7.0, width=0.6, pad=0)
    ax.view_init(elev=29, azim=-62)
    ax.set_box_aspect((1.40, 0.72, 0.88))
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.50, 0.55, 0.60, 0.13)
        axis._axinfo["grid"]["linewidth"] = 0.35
        axis.line.set_color(INK)
        axis.line.set_linewidth(0.6)


def add_surface_legend(ax) -> None:
    handles = [
        Patch(facecolor=DFT_SURFACE_HIGH, edgecolor="none", alpha=0.58, label="DFT PES"),
        Patch(facecolor=MLIP_SURFACE_HIGH, edgecolor="none", alpha=0.55, label="MLIP PES"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.02, 0.66),
        frameon=False,
        fontsize=7.3,
        handlelength=1.0,
        borderpad=0.1,
        labelspacing=0.25,
    )


def draw_surfaces(ax, q_dft: np.ndarray, q_mlip: np.ndarray, mode: str) -> None:
    q2_limits = (-0.12, 0.48)
    q1_grid = np.linspace(-0.06, 1.06, 65)
    q2_grid = np.linspace(q2_limits[0], q2_limits[1], 65)
    xx, yy = np.meshgrid(q1_grid, q2_grid)
    zz_dft, zz_mlip = surfaces(xx, yy)
    dft_norm = Normalize(vmin=float(zz_dft.min()), vmax=float(zz_dft.max()))
    mlip_norm = Normalize(vmin=float(zz_mlip.min()), vmax=float(zz_mlip.max()))
    dft_cmap = surface_cmap(DFT_SURFACE_LOW, DFT_SURFACE_HIGH)
    mlip_cmap = surface_cmap(MLIP_SURFACE_LOW, MLIP_SURFACE_HIGH)

    ax.plot_surface(
        xx,
        yy,
        zz_dft,
        facecolors=dft_cmap(dft_norm(zz_dft)),
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
        facecolors=mlip_cmap(mlip_norm(zz_mlip)),
        alpha=0.48,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
    )
    ax.plot_wireframe(xx[::8, ::8], yy[::8, ::8], zz_dft[::8, ::8], color="#74818C", alpha=0.22, linewidth=0.32)
    ax.plot_wireframe(xx[::8, ::8], yy[::8, ::8], zz_mlip[::8, ::8], color="#3E89C9", alpha=0.20, linewidth=0.32)

    qd_dense_s, qd_x, qd_y = smooth_path(q_dft)
    qm_dense_s, qm_x, qm_y = smooth_path(q_mlip)
    del qd_dense_s, qm_dense_s
    z_floor = 0.018
    z_dft_dft, z_mlip_dft = surfaces(qd_x, qd_y)
    z_dft_mlip, z_mlip_mlip = surfaces(qm_x, qm_y)

    # Base-plane projections are intentionally darker than the translucent
    # surfaces so that the one-path versus two-path distinction survives print.
    ax.plot(qd_x, qd_y, np.full_like(qd_x, z_floor), color=INK, lw=1.35, ls=(0, (4, 2)), alpha=0.90)
    if mode == "fixed":
        ax.plot(qd_x, qd_y, z_dft_dft, color=INK, lw=2.05, zorder=20)
        ax.plot(qd_x, qd_y, z_mlip_dft, color=MLIP_BLUE, lw=1.85, zorder=21)
        for index in range(1, 4):
            qx, qy = q_dft[index]
            zd, zm = surfaces(np.array([qx]), np.array([qy]))
            lo, hi = sorted((float(zd[0]), float(zm[0])))
            ax.plot([qx, qx], [qy, qy], [lo, hi], color=ERROR_RED, lw=1.45, alpha=0.95)
            ax.scatter([qx, qx], [qy, qy], [lo, hi], s=16, color=[INK, MLIP_BLUE], edgecolor="white", linewidth=0.4, depthshade=False)
            ax.plot([qx, qx], [qy, qy], [z_floor, lo], color=FLOOR_GREY, lw=0.50, ls=(0, (2, 2)), alpha=0.65)
        ax.text2D(
            0.08,
            0.60,
            r"same geometry $R_i^{\rm DFT}$",
            transform=ax.transAxes,
            fontsize=8.0,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.2},
        )
        ax.text2D(
            0.11,
            0.54,
            r"$\delta E_{\rm fixed}$",
            transform=ax.transAxes,
            fontsize=9.2,
            fontweight="bold",
            color=ERROR_RED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.2},
        )
    else:
        ax.plot(qm_x, qm_y, np.full_like(qm_x, z_floor), color=PATH_RED, lw=1.35, ls=(0, (4, 2)), alpha=0.92, zorder=18)
        ax.plot(qd_x, qd_y, z_dft_dft, color=INK, lw=1.65, ls=(0, (4, 2)), zorder=20)
        ax.plot(qm_x, qm_y, z_mlip_mlip, color=PATH_RED, lw=2.15, zorder=23)
        # This grey trace is the DFT surface evaluated at the MLIP path
        # coordinates, shown as the second-stage comparison target.
        ax.plot(qm_x, qm_y, z_dft_mlip, color=DFT_GREY, lw=1.15, ls=(0, (2, 2)), alpha=0.82, zorder=22)
        for index in range(1, 4):
            qx_d, qy_d = q_dft[index]
            qx_m, qy_m = q_mlip[index]
            ax.plot([qx_d, qx_m], [qy_d, qy_m], [z_floor, z_floor], color=PATH_RED, lw=1.15, ls=(0, (2, 2)), alpha=0.92)
            ax.scatter([qx_d, qx_m], [qy_d, qy_m], [z_floor, z_floor], s=13, color=[INK, PATH_RED], edgecolor="white", linewidth=0.35, depthshade=False)
            ax.plot([qx_d, qx_d], [qy_d, qy_d], [z_floor, float(z_dft_dft[np.argmin(np.abs(qd_x - qx_d))])], color=FLOOR_GREY, lw=0.45, ls=(0, (2, 2)), alpha=0.55)
            ax.plot([qx_m, qx_m], [qy_m, qy_m], [z_floor, float(z_mlip_mlip[np.argmin(np.abs(qm_x - qx_m))])], color=FLOOR_GREY, lw=0.45, ls=(0, (2, 2)), alpha=0.55)
        qx, qy = q_mlip[2]
        dft_at_mlip, mlip_at_mlip = surfaces(np.array([qx]), np.array([qy]))
        lo, hi = sorted((float(dft_at_mlip[0]), float(mlip_at_mlip[0])))
        ax.plot([qx, qx], [qy, qy], [lo, hi], color=ERROR_RED, lw=1.50, zorder=25)
        ax.scatter([qx, qx], [qy, qy], [lo, hi], s=16, color=[DFT_GREY, PATH_RED], edgecolor="white", linewidth=0.4, depthshade=False)
        ax.text2D(
            0.06,
            0.59,
            r"1  geometry shift",
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=PATH_RED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
        )
        ax.text2D(
            0.34,
            0.49,
            "2  energy mismatch",
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=ERROR_RED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
        )
    ax.scatter(q_dft[:, 0], q_dft[:, 1], surfaces(q_dft[:, 0], q_dft[:, 1])[0], s=18, color=INK, edgecolor="white", linewidth=0.45, depthshade=False, zorder=30)
    if mode == "self":
        ax.scatter(q_mlip[:, 0], q_mlip[:, 1], surfaces(q_mlip[:, 0], q_mlip[:, 1])[1], s=18, color=PATH_RED, edgecolor="white", linewidth=0.45, depthshade=False, zorder=31)
    add_surface_legend(ax)
    set_3d_style(ax, "(a) Fixed-path scoring" if mode == "fixed" else "(b) Self-relaxed NEB", q2_limits)


def add_snapshot_strip(ax, branch: str, labels: tuple[str, ...], edgecolor: str) -> None:
    for index, label in enumerate(labels):
        left = 0.012 + index * 0.195
        inset = ax.inset_axes([left, 0.73, 0.17, 0.20])
        image = plt.imread(ATOM_DIR / f"{branch}_{index:02d}.png")
        inset.imshow(image, aspect="auto")
        inset.set_axis_off()
        inset.patch.set_facecolor("white")
        inset.patch.set_edgecolor(edgecolor)
        inset.patch.set_linewidth(0.65)
        inset.text(0.05, 0.92, label, transform=inset.transAxes, fontsize=6.8, fontweight="bold", color=INK, ha="left", va="top")


def fixed_profile_panel(ax, s: np.ndarray, dft: np.ndarray, mlip: np.ndarray) -> None:
    ax.plot(s, dft, color=INK, lw=1.75, marker="o", ms=4.0, label="DFT on fixed path", zorder=3)
    ax.plot(s, mlip, color=MLIP_BLUE, lw=1.75, marker="o", ms=3.8, label="FT-600K on same path", zorder=3)
    for index in (1, 2, 3):
        ax.annotate("", xy=(s[index], mlip[index]), xytext=(s[index], dft[index]), arrowprops={"arrowstyle": "<->", "color": ERROR_RED, "lw": 0.95})
    ax.text(
        0.03,
        0.96,
        r"$\delta E_i=E_{\rm MLIP}(R_i^{\rm DFT})-E_{\rm DFT}(R_i^{\rm DFT})$",
        transform=ax.transAxes,
        fontsize=7.9,
        color=ERROR_RED,
        va="top",
    )
    ax.set_xlabel("reaction coordinate, $s$", fontsize=9.0)
    ax.set_ylabel("relative energy (eV)", fontsize=9.0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.25, 0.65)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.legend(loc="lower left", frameon=False, fontsize=7.1, handlelength=1.7, borderpad=0.1, labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.4, width=0.7, length=3)
    ax.set_title("fixed-path energy profile", loc="left", fontsize=8.8, fontweight="bold", pad=4)


def operator_panel(spec, q_dft: np.ndarray, q_mlip: np.ndarray) -> None:
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.34, width_ratios=(1.03, 1.0))
    ax_geo = plt.subplot(inner[0, 0])
    ax_energy = plt.subplot(inner[0, 1])

    dense, qd_x, qd_y = smooth_path(q_dft)
    _, qm_x, qm_y = smooth_path(q_mlip)
    del dense
    ax_geo.plot(qd_x, qd_y, color=INK, lw=1.45, ls=(0, (4, 2)), label="DFT reference")
    ax_geo.plot(qm_x, qm_y, color=PATH_RED, lw=1.75, label="MLIP self-relaxed")
    ax_geo.scatter(q_dft[:, 0], q_dft[:, 1], s=16, color=INK, zorder=4)
    ax_geo.scatter(q_mlip[:, 0], q_mlip[:, 1], s=16, color=PATH_RED, zorder=4)
    for index in (1, 2, 3):
        ax_geo.annotate("", xy=(q_mlip[index, 0], q_mlip[index, 1]), xytext=(q_dft[index, 0], q_dft[index, 1]), arrowprops={"arrowstyle": "<->", "color": PATH_RED, "lw": 0.85})
    ax_geo.annotate(
        "geometry shift",
        xy=(0.53, 0.25),
        xycoords="data",
        xytext=(0.86, 0.82),
        textcoords="axes fraction",
        fontsize=6.8,
        color=PATH_RED,
        ha="center",
        va="center",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.0},
        arrowprops={"arrowstyle": "-", "color": PATH_RED, "lw": 0.65, "connectionstyle": "arc3,rad=-0.18"},
    )
    ax_geo.set_title("1  relax the path", loc="left", fontsize=8.8, fontweight="bold", pad=4)
    ax_geo.set_xlabel("$q_1$", fontsize=8.5)
    ax_geo.set_ylabel("$q_2$", fontsize=8.5)
    ax_geo.set_xlim(-0.06, 1.06)
    ax_geo.set_ylim(-0.06, 0.48)
    ax_geo.set_xticks([0.0, 0.5, 1.0])
    ax_geo.set_yticks([0.0, 0.20, 0.40])
    ax_geo.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=1.4, borderpad=0.1, labelspacing=0.2)
    ax_geo.spines[["top", "right"]].set_visible(False)
    ax_geo.tick_params(labelsize=6.9, width=0.65, length=2.5)

    s_anchor = np.linspace(0.0, 1.0, 5)
    e_dft = np.array([0.04, 0.34, 0.53, 0.30, 0.05])
    e_mlip = np.array([0.035, 0.24, 0.39, 0.22, 0.04])
    s_dense = np.linspace(0.0, 1.0, 220)
    e_dft_dense = monotone_cubic(s_anchor, e_dft, s_dense)
    e_mlip_dense = monotone_cubic(s_anchor, e_mlip, s_dense)
    ax_energy.plot(s_dense, e_dft_dense, color=DFT_GREY, lw=1.45, ls=(0, (4, 2)), label=r"DFT at $R_i^{\rm MLIP}$")
    ax_energy.plot(s_dense, e_mlip_dense, color=PATH_RED, lw=1.75, label=r"MLIP at $R_i^{\rm MLIP}$")
    ax_energy.scatter(s_anchor, e_dft, s=14, facecolor="white", edgecolor=DFT_GREY, linewidth=0.75, zorder=4)
    ax_energy.scatter(s_anchor, e_mlip, s=14, facecolor="#F5A3AF", edgecolor=PATH_RED, linewidth=0.75, zorder=4)
    ax_energy.annotate("", xy=(0.5, e_dft[2]), xytext=(0.5, e_mlip[2]), arrowprops={"arrowstyle": "<->", "color": ERROR_RED, "lw": 0.95})
    ax_energy.annotate(
        "model mismatch",
        xy=(0.50, 0.46),
        xycoords="data",
        xytext=(0.77, 0.70),
        textcoords="axes fraction",
        fontsize=6.7,
        color=ERROR_RED,
        ha="center",
        va="center",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.0},
        arrowprops={"arrowstyle": "-", "color": ERROR_RED, "lw": 0.65, "connectionstyle": "arc3,rad=0.18"},
    )
    ax_energy.set_title("2  re-score on relaxed path", loc="left", fontsize=8.8, fontweight="bold", pad=4)
    ax_energy.set_xlabel("path progress, $s$", fontsize=8.5)
    ax_energy.set_ylabel("relative energy", fontsize=8.5)
    ax_energy.set_xlim(-0.03, 1.03)
    ax_energy.set_ylim(-0.02, 0.62)
    ax_energy.set_xticks([0.0, 0.5, 1.0])
    ax_energy.set_yticks([0.0, 0.3, 0.6])
    ax_energy.legend(loc="upper left", frameon=False, fontsize=6.4, handlelength=1.4, borderpad=0.1, labelspacing=0.2)
    ax_energy.spines[["top", "right"]].set_visible(False)
    ax_energy.tick_params(labelsize=6.9, width=0.65, length=2.5)


def write_path_csv(path: Path, q_dft: np.ndarray, q_mlip: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["branch", "image_index", "q1", "q2", "geometry_status"])
        for branch, q, status in (
            ("DFT_fixed", q_dft, "VERIFIED_STRUCTURE_SOURCE"),
            ("MLIP_self_relaxed", q_mlip, "PROVISIONAL_MLIP_PRECONDITIONER"),
        ):
            for index, (q1, q2) in enumerate(q):
                writer.writerow([branch, index, f"{q1:.10f}", f"{q2:.10f}", status])


def render(output: Path) -> None:
    output = output.resolve()
    required = [DFT_PROFILE, MODEL_PROFILE, DFT_XYZ, MLIP_XYZ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input(s): " + ", ".join(missing))
    for branch in ("dft", "mlip"):
        for index in range(5):
            if not (ATOM_DIR / f"{branch}_{index:02d}.png").exists():
                raise FileNotFoundError(f"Missing atom render {ATOM_DIR / f'{branch}_{index:02d}.png'}")

    apply_publication_style(9.2)
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "dejavusans"})
    s, dft, mlip = load_profiles()
    dft_frames = read_xyz_frames(DFT_XYZ)
    mlip_frames = read_xyz_frames(MLIP_XYZ)
    q_dft, q_mlip, q_metadata = path_coordinates(dft_frames, mlip_frames)

    fig = plt.figure(figsize=(8.45, 5.55), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=(3.40, 1.56),
        left=0.050,
        right=0.985,
        top=0.945,
        bottom=0.145,
        wspace=0.13,
        hspace=0.34,
    )
    ax_fixed = fig.add_subplot(gs[0, 0], projection="3d")
    ax_self = fig.add_subplot(gs[0, 1], projection="3d")
    draw_surfaces(ax_fixed, q_dft, q_mlip, "fixed")
    draw_surfaces(ax_self, q_dft, q_mlip, "self")
    add_snapshot_strip(ax_fixed, "dft", ("A", "1", "TS", "3", "B"), INK)
    add_snapshot_strip(ax_self, "mlip", ("A", "1", "TS", "3", "B"), PATH_RED)
    operator_panel(gs[1, 1], q_dft, q_mlip)
    fixed_profile_panel(fig.add_subplot(gs[1, 0]), s, dft, mlip)

    fig.text(
        0.545,
        0.047,
        r"$\Delta E_i=[E_{\rm DFT}(R_i^{\rm MLIP})-E_{\rm DFT}(R_i^{\rm DFT})]_{\rm geometry/path}+[E_{\rm MLIP}(R_i^{\rm MLIP})-E_{\rm DFT}(R_i^{\rm MLIP})]_{\rm energy\ model}$",
        ha="center",
        va="center",
        fontsize=8.1,
        color=INK,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        save_figure(fig, output.with_suffix(f".{ext}"), dpi=360)
    plt.close(fig)

    path_csv = output.with_name("pes_operator_v2_path_coordinates.csv")
    write_path_csv(path_csv, q_dft, q_mlip)
    manifest = {
        "figure_id": "A1-PES-OPERATOR-SCHEMATIC-V2",
        "status": "RECONSTRUCTION_SCHEMATIC_WITH_REAL_PATH_GEOMETRY",
        "surface_status": "synthetic reduced-coordinate PES; color is z-dependent and not a quantitative PES",
        "fixed_profile_status": "derived from current 1-4 processed DFT candidate profile and FT-600K fixed-path predictions",
        "fixed_geometry_status": "five-image 61-atom DFT fixture",
        "self_geometry_status": "five-image 61-atom Foundation MLIP self-relaxed path; converged under the MLIP optimization potential but provisional for publication",
        "self_rescore_status": "operator definition only; same-system DFT re-score is not available and is not plotted as data",
        "q_coordinate_definition": q_metadata,
        "equations": {
            "fixed": "deltaE_i = E_MLIP(R_i^DFT) - E_DFT(R_i^DFT)",
            "self": "Gamma_MLIP = NEB_MLIP(R_A,R_B)",
            "decomposition": "DeltaE_i = [E_DFT(R_i^MLIP)-E_DFT(R_i^DFT)] + [E_MLIP(R_i^MLIP)-E_DFT(R_i^MLIP)]",
        },
        "inputs": {
            str(path.relative_to(REPO)): sha256(path)
            for path in required
        },
        "atom_render_manifest": str((ATOM_DIR / "atom_render_manifest.json").relative_to(REPO)),
        "path_coordinate_csv": str(path_csv.relative_to(REPO)),
        "notes": [
            "The five atomistic snapshots are structural anchors and contain no labels.",
            "The right-hand path uses actual Foundation MLIP self-relaxed geometries with shared endpoints, not the excluded historical 0.410 eV record.",
            "The lower-right re-score curves are synthetic smooth operator definitions because a matched DFT re-score on the self-relaxed path is still missing.",
            "No publication barrier or model ranking is inferred from the schematic surfaces or synthetic re-score curves.",
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
