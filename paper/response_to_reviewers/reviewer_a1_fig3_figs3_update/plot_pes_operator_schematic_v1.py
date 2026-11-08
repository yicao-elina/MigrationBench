"""Nature-style schematic of fixed-path and self-relaxed NEB operators.

The figure is deliberately a hybrid of evidence and definition:

* the lower-left fixed-path profile uses the surviving 1--4 path data;
* the atomic thumbnails are rendered from the 61-atom path files;
* the two reduced-coordinate energy landscapes are a transparent schematic
  projection, because the real PES lives in 3N dimensions;
* the self-relaxed panel defines the two-stage operator and does not claim a
  completed same-system DFT re-score.

Outputs are written next to this script as PDF, SVG, PNG, and a JSON manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import sys

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
MLIP_XYZ = REPO / "data/raw_compact/revision1/mb_mlff_1_4_s42_30763473/mlff_1-4_s42/mlff_neb_images.extxyz"


INK = "#111111"
DFT_GREY = "#424A53"
MLIP_BLUE = "#2F72B8"
MLIP_BLUE_PALE = "#9CC7F2"
ERROR_RED = "#C83E4D"
PATH_RED = "#C8333E"
ATOM_TE = "#A89AAE"
ATOM_SB = "#6D8FB9"
ATOM_CR = "#D79B1E"
ATOM_EDGE = "#29313A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_xyz_frames(path: Path) -> list[list[tuple[str, float, float, float]]]:
    """Read ordinary XYZ and extXYZ files without depending on ASE."""
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


def reduced_surfaces(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A smooth 2D projection used only to illustrate the operator geometry."""
    dft = (
        0.08
        + 0.15 * (y - 0.50) ** 2
        + 0.66 * np.exp(-((x - 0.50) / 0.20) ** 2 - ((y - 0.50) / 0.25) ** 2)
        + 0.08 * np.sin(2.0 * np.pi * x) * np.cos(np.pi * y)
    )
    mlip = (
        0.04
        + 0.11 * (y - 0.47) ** 2
        + 0.53 * np.exp(-((x - 0.49) / 0.27) ** 2 - ((y - 0.47) / 0.35) ** 2)
        + 0.035 * np.sin(2.0 * np.pi * x + 0.22) * np.cos(np.pi * y)
    )
    return dft, mlip


def dft_path(x: np.ndarray) -> np.ndarray:
    return 0.34 + 0.14 * np.sin(np.pi * x)


def mlip_path(x: np.ndarray) -> np.ndarray:
    # Same endpoints, visibly different interior geometry.
    return dft_path(x) + 0.20 * np.sin(np.pi * x) ** 2 * (0.68 - 1.35 * x)


def style_3d_axis(ax, title: str) -> None:
    ax.text2D(0.02, 1.025, title, transform=ax.transAxes, fontsize=11.5,
              fontweight="bold", color=INK, ha="left", va="bottom")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(r"$q_1$", labelpad=-7, fontsize=9, color=INK)
    ax.set_ylabel(r"$q_2$", labelpad=-8, fontsize=9, color=INK)
    ax.set_zlabel("energy", labelpad=-8, fontsize=8.5, color=INK)
    ax.view_init(elev=29, azim=-60)
    ax.set_box_aspect((1.15, 0.95, 0.75))
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
        axis.line.set_color(INK)
        axis.line.set_linewidth(0.65)


def add_surface_legend(ax) -> None:
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 1.3}
    ax.text2D(0.73, 0.34, "DFT PES", transform=ax.transAxes, fontsize=8.7,
              fontweight="bold", color=DFT_GREY,
              bbox=box)
    ax.text2D(0.76, 0.27, "MLIP PES", transform=ax.transAxes, fontsize=8.7,
              fontweight="bold", color=MLIP_BLUE,
              bbox=box)


def plot_surface_panel(ax, mode: str) -> None:
    grid = np.linspace(0.0, 1.0, 31)
    xx, yy = np.meshgrid(grid, grid)
    zz_dft, zz_mlip = reduced_surfaces(xx, yy)
    ax.plot_surface(xx, yy, zz_dft, color=DFT_GREY, alpha=0.28,
                    linewidth=0.0, antialiased=True, shade=True, rstride=1, cstride=1)
    ax.plot_surface(xx, yy, zz_mlip, color=MLIP_BLUE_PALE, alpha=0.40,
                    linewidth=0.0, antialiased=True, shade=True, rstride=1, cstride=1)
    # A sparse wireframe makes the two surfaces legible at manuscript scale.
    ax.plot_wireframe(xx[::5, ::5], yy[::5, ::5], zz_dft[::5, ::5],
                      color="#7B858E", alpha=0.15, linewidth=0.35)
    ax.plot_wireframe(xx[::5, ::5], yy[::5, ::5], zz_mlip[::5, ::5],
                      color="#4D91D0", alpha=0.18, linewidth=0.35)

    q = np.linspace(0.0, 1.0, 100)
    yd = dft_path(q)
    ym = mlip_path(q)
    zd_dft, zd_mlip = reduced_surfaces(q, yd)
    zm_dft, zm_mlip = reduced_surfaces(q, ym)
    del zm_dft

    if mode == "fixed":
        ax.plot(q, yd, zd_dft, color=INK, linewidth=2.2, zorder=20)
        ax.plot(q, yd, zd_mlip, color=MLIP_BLUE, linewidth=2.0, zorder=21)
        ax.scatter(q[[0, 50, 99]], yd[[0, 50, 99]], zd_dft[[0, 50, 99]],
                   s=18, color=INK, edgecolor="white", linewidth=0.5, depthshade=False)
        ax.scatter(q[[0, 50, 99]], yd[[0, 50, 99]], zd_mlip[[0, 50, 99]],
                   s=18, color=MLIP_BLUE, edgecolor="white", linewidth=0.5, depthshade=False)
        for idx in (20, 50, 80):
            ax.plot([q[idx], q[idx]], [yd[idx], yd[idx]],
                    [zd_dft[idx], zd_mlip[idx]], color=ERROR_RED, linewidth=1.45)
            ax.scatter([q[idx]], [yd[idx]], [zd_dft[idx]], s=16, color=INK,
                       edgecolor="white", linewidth=0.45, depthshade=False)
            ax.scatter([q[idx]], [yd[idx]], [zd_mlip[idx]], s=16, color=MLIP_BLUE,
                       edgecolor="white", linewidth=0.45, depthshade=False)
    else:
        ax.plot(q, yd, zd_dft, color=INK, linewidth=1.9,
                linestyle=(0, (4, 2)), zorder=20)
        ax.plot(q, ym, zm_mlip, color=PATH_RED, linewidth=2.2, zorder=22)
        ax.scatter(q[[0, 50, 99]], yd[[0, 50, 99]], zd_dft[[0, 50, 99]],
                   s=18, color=INK, edgecolor="white", linewidth=0.5, depthshade=False)
        ax.scatter(q[[0, 50, 99]], ym[[0, 50, 99]], zm_mlip[[0, 50, 99]],
                   s=18, color=PATH_RED, edgecolor="white", linewidth=0.5, depthshade=False)
        idx = 52
        # Stage 1: lateral shift in reduced geometry coordinates, shown on the base plane.
        ax.plot([q[idx], q[idx]], [yd[idx], ym[idx]], [0.025, 0.025],
                color=PATH_RED, linewidth=1.25, linestyle=(0, (2, 2)))
        ax.scatter([q[idx], q[idx]], [yd[idx], ym[idx]], [0.025, 0.025],
                   s=12, color=PATH_RED, edgecolor="white", linewidth=0.4,
                   depthshade=False)
        # Stage 2: compare the two energies at the new MLIP geometry.
        dft_at_mlip = reduced_surfaces(np.array([q[idx]]), np.array([ym[idx]]))[0][0]
        mlip_at_mlip = reduced_surfaces(np.array([q[idx]]), np.array([ym[idx]]))[1][0]
        ax.plot([q[idx], q[idx]], [ym[idx], ym[idx]], [dft_at_mlip, mlip_at_mlip],
                color=ERROR_RED, linewidth=1.45)
        ax.scatter([q[idx]], [ym[idx]], [dft_at_mlip], s=16, color=DFT_GREY,
                   edgecolor="white", linewidth=0.45, depthshade=False)
        ax.scatter([q[idx]], [ym[idx]], [mlip_at_mlip], s=16, color=PATH_RED,
                   edgecolor="white", linewidth=0.45, depthshade=False)
    add_surface_legend(ax)
    style_3d_axis(ax, "(a) Fixed-path scoring" if mode == "fixed" else "(b) Self-relaxed NEB")
    if mode == "fixed":
        ax.text2D(0.13, 0.59, r"$\Delta E_{\rm fixed}$", transform=ax.transAxes,
                  fontsize=9.2, fontweight="bold", color=ERROR_RED, zorder=1000,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1.5})
    else:
        ax.text2D(0.14, 0.34, "geometry shift", transform=ax.transAxes,
                  fontsize=8.2, fontweight="bold", color=PATH_RED, zorder=1000,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1.2})
        ax.text2D(0.35, 0.59, "energy mismatch", transform=ax.transAxes,
                  fontsize=8.6, fontweight="bold", color=ERROR_RED, zorder=1000,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1.2})


def draw_atom_snapshot(ax, atoms, title: str) -> None:
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    symbols = [symbol for symbol, *_ in atoms]
    cr_positions = coords[np.array([symbol == "Cr" for symbol in symbols])]
    center = cr_positions[0]
    mask = (
        (coords[:, 0] > center[0] - 4.9)
        & (coords[:, 0] < center[0] + 4.9)
        & (coords[:, 2] > center[2] - 5.6)
        & (coords[:, 2] < center[2] + 5.6)
    )
    if mask.sum() < 12:
        mask[:] = True
    coords = coords[mask]
    symbols = [symbol for symbol, keep in zip(symbols, mask) if keep]
    proj_x = coords[:, 0] + 0.23 * coords[:, 1]
    proj_y = coords[:, 2] + 0.045 * coords[:, 1]
    depth = coords[:, 1]
    ux = (proj_x - proj_x.min()) / max(np.ptp(proj_x), 1e-8)
    uy = (proj_y - proj_y.min()) / max(np.ptp(proj_y), 1e-8)
    depth_norm = (depth - depth.min()) / max(np.ptp(depth), 1e-8)
    colors = {"Te": ATOM_TE, "Sb": ATOM_SB, "Cr": ATOM_CR}
    order = np.argsort(depth)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#343B43",
                           linewidth=0.65, zorder=0))
    for i in order:
        radius = 0.018 + 0.014 * depth_norm[i]
        symbol = symbols[i]
        edge = "#20252A" if symbol == "Cr" else ATOM_EDGE
        lw = 0.72 if symbol == "Cr" else 0.35
        ax.add_patch(Circle((ux[i], uy[i]), radius, facecolor=colors.get(symbol, "#999999"),
                            edgecolor=edge, linewidth=lw, zorder=2 + depth_norm[i]))
        if symbol == "Cr":
            ax.add_patch(Circle((ux[i] - radius * 0.25, uy[i] + radius * 0.28),
                                radius * 0.22, facecolor="white", edgecolor="none",
                                alpha=0.75, zorder=5))
            ax.add_patch(Circle((ux[i], uy[i]), radius * 1.25, facecolor="none",
                                edgecolor=ERROR_RED, linewidth=0.7, zorder=6))
    ax.text(0.05, 0.93, title, fontsize=6.4, fontweight="bold", color=INK,
            ha="left", va="top", zorder=10)


def add_snapshot_strip(ax3d, frames, frame_indices) -> None:
    # Inset atom renders use the actual 61-atom path coordinates. They are
    # aligned with the left-to-right reaction sequence, not used as a new
    # quantitative path definition.
    for pos, idx, title in zip((0.05, 0.37, 0.69), frame_indices, ("A", "TS", "B")):
        inset = ax3d.inset_axes([pos, 0.72, 0.25, 0.22])
        draw_atom_snapshot(inset, frames[idx], title)


def fixed_profile_panel(ax, s, dft, mlip) -> None:
    ax.plot(s, dft, color=INK, linewidth=1.8, marker="o", markersize=3.8,
            label="DFT candidate", zorder=3)
    ax.plot(s, mlip, color=MLIP_BLUE, linewidth=1.8, marker="o", markersize=3.6,
            label="MLIP on same path", zorder=3)
    for idx in (1, 2, 3):
        ax.annotate("", xy=(s[idx], mlip[idx]), xytext=(s[idx], dft[idx]),
                    arrowprops={"arrowstyle": "<->", "color": ERROR_RED, "lw": 0.95})
    ax.text(0.03, 0.94, r"$\delta E_i = E_{\rm MLIP}(R_i^{\rm DFT}) - E_{\rm DFT}(R_i^{\rm DFT})$",
            transform=ax.transAxes, fontsize=8.0, color=ERROR_RED, va="top")
    ax.set_xlabel("reaction coordinate, $s$", fontsize=8.8)
    ax.set_ylabel("relative energy (eV)", fontsize=8.8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(min(-0.25, dft.min() - 0.05), max(0.65, mlip.max() + 0.10))
    ax.legend(loc="lower left", frameon=False, fontsize=7.2, handlelength=1.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5, width=0.7, length=3)


def self_operator_panel(spec) -> None:
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.42, width_ratios=(1.05, 1.0))
    ax_geo = plt.subplot(inner[0, 0])
    ax_energy = plt.subplot(inner[0, 1])
    q = np.linspace(0.0, 1.0, 100)
    yd = dft_path(q)
    ym = mlip_path(q)
    ax_geo.plot(q, yd, color=INK, linewidth=1.5, linestyle=(0, (4, 2)), label="DFT reference")
    ax_geo.plot(q, ym, color=PATH_RED, linewidth=1.8, label="MLIP-relaxed")
    ax_geo.scatter(q[[0, 50, 99]], yd[[0, 50, 99]], s=14, color=INK, zorder=3)
    ax_geo.scatter(q[[0, 50, 99]], ym[[0, 50, 99]], s=14, color=PATH_RED, zorder=3)
    idx = 52
    ax_geo.annotate("", xy=(q[idx], ym[idx]), xytext=(q[idx], yd[idx]),
                    arrowprops={"arrowstyle": "<->", "color": PATH_RED, "lw": 1.0})
    ax_geo.text(0.52, 0.05, "geometry shift", transform=ax_geo.transAxes,
                fontsize=7.4, color=PATH_RED, ha="center")
    ax_geo.set_xlabel("$q_1$", fontsize=8.2)
    ax_geo.set_ylabel("$q_2$", fontsize=8.2)
    ax_geo.set_title("1  path relaxation", loc="left", fontsize=8.8, fontweight="bold")
    ax_geo.spines[["top", "right"]].set_visible(False)
    ax_geo.tick_params(labelsize=7.0, width=0.65, length=2.5)

    s = np.linspace(0.0, 1.0, 5)
    e_dft = 0.05 + 0.46 * np.sin(np.pi * s) ** 1.45
    ax_energy.plot(s, e_dft, color=INK, linewidth=1.45, linestyle=(0, (4, 2)))
    # The red trace is a deliberately schematic MLIP energy trace on its own path.
    e_self = 0.04 + 0.58 * np.sin(np.pi * s) ** 1.6 + 0.05 * np.sin(3 * np.pi * s)
    ax_energy.plot(s, e_self, color=PATH_RED, linewidth=1.75)
    ax_energy.annotate("", xy=(s[2], e_self[2]), xytext=(s[2], e_dft[2]),
                       arrowprops={"arrowstyle": "<->", "color": ERROR_RED, "lw": 1.0})
    ax_energy.text(0.50, 0.94, "energy mismatch", transform=ax_energy.transAxes,
                   fontsize=7.6, color=ERROR_RED, ha="center", va="top")
    ax_energy.set_xlabel("path progress, $s$", fontsize=8.2)
    ax_energy.set_ylabel("energy", fontsize=8.2)
    ax_energy.set_title("2  re-score on relaxed path (schematic)", loc="left",
                        fontsize=8.8, fontweight="bold")
    ax_energy.spines[["top", "right"]].set_visible(False)
    ax_energy.tick_params(labelsize=7.0, width=0.65, length=2.5)


def render(output: Path) -> None:
    apply_publication_style(8.6)
    s, dft, mlip = load_profiles()
    dft_frames = read_xyz_frames(DFT_XYZ)
    mlip_frames = read_xyz_frames(MLIP_XYZ)
    if len(dft_frames) < 5 or len(mlip_frames) < 5:
        raise RuntimeError("Expected five path images for both atomistic strips")

    # Work at approximately Nature's 180-mm double-column width so that the
    # typography is judged at the intended placement rather than after a
    # large down-scaling step in LaTeX.
    fig = plt.figure(figsize=(7.4, 4.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=(3.35, 1.55),
                          left=0.055, right=0.985, top=0.94, bottom=0.14,
                          wspace=0.16, hspace=0.36)
    ax_fixed = fig.add_subplot(gs[0, 0], projection="3d")
    ax_self = fig.add_subplot(gs[0, 1], projection="3d")
    plot_surface_panel(ax_fixed, "fixed")
    plot_surface_panel(ax_self, "self")
    add_snapshot_strip(ax_fixed, dft_frames, (0, 2, 4))
    add_snapshot_strip(ax_self, mlip_frames, (0, 2, 4))

    ax_profile = fig.add_subplot(gs[1, 0])
    fixed_profile_panel(ax_profile, s, dft, mlip)
    self_operator_panel(gs[1, 1])

    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        save_figure(fig, output.with_suffix(f".{ext}"), dpi=360)
    plt.close(fig)

    manifest = {
        "figure_id": "A1-PES-OPERATOR-SCHEMATIC-V1",
        "status": "RECONSTRUCTION_SCHEMATIC",
        "surface_status": "synthetic reduced-coordinate projection; not a quantitative PES",
        "fixed_profile_status": "derived from current 1-4 processed path profile and FT-600K predictions",
        "self_relaxed_status": "operator definition; same-system DFT re-score not asserted",
        "equations": {
            "fixed": "deltaE_i = E_MLIP(R_i^DFT) - E_DFT(R_i^DFT)",
            "self": "Gamma_MLIP = NEB_MLIP(R_A,R_B), then compare E_MLIP(R_i^MLIP) with E_DFT(R_i^MLIP)",
            "decomposition": "DeltaE_total = [E_DFT(R_i^MLIP)-E_DFT(R_i^DFT)] + [E_MLIP(R_i^MLIP)-E_DFT(R_i^MLIP)]",
        },
        "inputs": {
            str(path.relative_to(REPO)): sha256(path)
            for path in (DFT_PROFILE, MODEL_PROFILE, DFT_XYZ, MLIP_XYZ)
        },
        "notes": [
            "The 61-atom DFT fixture is used for structural visual anchoring only.",
            "The right-hand atom strip comes from a finite FT-600K MLIP pre-NEB path and is not presented as a converged self-NEB result.",
            "Do not use this schematic to rank models or to validate the historical 0.410 eV record.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
