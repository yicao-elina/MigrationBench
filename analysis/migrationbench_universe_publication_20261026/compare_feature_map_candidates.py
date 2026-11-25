#!/usr/bin/env python3
"""Compare feature-preserving configuration-universe layouts.

Unlike the presentation map, these candidates do not translate task islands to
manual anchors. Candidate coordinates come from UMAP/densMAP applied to the
common Cr-local SOAP/PCA representation, with an optional weak geometric
augmentation from measured Cr-shell observables.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.preprocessing import RobustScaler

from compose_comprehensive_task_map import (
    DERIVED,
    FIGURES,
    MD_SAMPLE,
    MD_MANIFEST,
    TASK_COLORS,
    TASK_ORDER,
    SUBLEVEL_COLORS,
    SUBLEVEL_ORDER,
    add_path_lines,
    load_md_rows,
    load_reference_shell,
    local_soap,
    sublevel,
    task_group,
    protocol_style,
    size_from_distortion,
)


CANDIDATE_DIR = FIGURES / "feature_map_candidates"


def load_common_data() -> tuple[pd.DataFrame, np.ndarray]:
    base = pd.read_csv(DERIVED / "coordination_distortion_points.csv")
    base_vectors = np.load(DERIVED / "structure_universe_local_soap_vectors.npz")["vectors"]
    if len(base) != len(base_vectors):
        raise RuntimeError(f"base metadata/vector mismatch: {len(base)} != {len(base_vectors)}")
    md, md_vectors = load_md_rows(load_reference_shell())
    df = pd.concat([base, md], ignore_index=True, sort=False)
    df["task_group"] = df.apply(task_group, axis=1)
    df["sublevel"] = df.apply(sublevel, axis=1)
    df["protocol_style"] = df.apply(protocol_style, axis=1)
    return df, np.vstack([base_vectors, md_vectors])


def project_common_pca(X: np.ndarray) -> np.ndarray:
    fit = np.load(DERIVED / "cr_local_soap_fit.npz")
    mean = fit["mean"]
    scale = fit["scale"]
    pca_mean = fit["pca_mean"]
    components = fit["components"]
    Z = (X - mean) / np.where(scale == 0, 1.0, scale)
    return (Z - pca_mean) @ components.T


def geometry_augmentation(df: pd.DataFrame, weight: float) -> np.ndarray:
    columns = [
        "rms_delta_A",
        "coordination_number_delta",
        "species_count_delta",
        "bond_length_std_A",
    ]
    geom = df[columns].apply(pd.to_numeric, errors="coerce")
    geom = geom.fillna(geom.median()).fillna(0.0)
    scaled = RobustScaler(quantile_range=(10, 90)).fit_transform(geom.to_numpy(float))
    scaled = np.clip(scaled, -4.0, 4.0)
    return weight * scaled


def run_umap(features: np.ndarray, *, densmap: bool, n_neighbors: int, min_dist: float) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        init="spectral",
        densmap=densmap,
        dens_lambda=2.0 if densmap else 0.0,
        dens_frac=0.3 if densmap else 0.0,
        random_state=42,
        n_components=2,
    )
    return reducer.fit_transform(features)


def normalize_xy(xy: np.ndarray) -> np.ndarray:
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    center = (lo + hi) / 2.0
    scale = max(float(np.max(hi - lo)), 1e-12)
    return 18.0 * (xy - center) / scale


def rank_equalize_xy(xy: np.ndarray) -> np.ndarray:
    """Equalize only the two display marginals, preserving feature-space order."""
    out = np.empty_like(xy, dtype=float)
    n = len(xy)
    for axis in range(2):
        order = np.argsort(xy[:, axis], kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        out[:, axis] = ranks / max(n - 1, 1)
    return out


def render_candidate(
    df: pd.DataFrame,
    xy: np.ndarray,
    name: str,
    subtitle: str,
    equalize: bool = False,
) -> None:
    out = CANDIDATE_DIR / f"{name}.png"
    work = df.copy()
    if equalize:
        xy = rank_equalize_xy(xy)
    xy = normalize_xy(xy)
    work["x"] = xy[:, 0]
    work["y"] = xy[:, 1]
    work["size"] = size_from_distortion(work["rms_delta_A"], foreground=True)
    work["size_bg"] = size_from_distortion(work["rms_delta_A"], foreground=False)

    fig, ax = plt.subplots(figsize=(15.2, 10.8), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.795)
    reference = work[work["task_group"].eq("L0 DFT Reference")]
    ax.scatter(
        reference["x"], reference["y"], s=reference["size_bg"],
        color=SUBLEVEL_COLORS["Static DFT"], alpha=0.16,
        edgecolors="none", rasterized=True, zorder=1,
    )
    for (task, level), group in work[~work["task_group"].eq("L0 DFT Reference")].groupby(
        ["task_group", "sublevel"], sort=False
    ):
        edge = "#222222" if level.startswith("DFT") else "white"
        ax.scatter(
            group["x"], group["y"], s=group["size"],
            color=SUBLEVEL_COLORS[level], alpha=0.78,
            edgecolors=edge, linewidths=0.55, rasterized=True, zorder=3,
        )

    # Path lines use the actual feature-space coordinates; no task anchors.
    for _, group in work[work["path_key"].notna()].groupby("path_key", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values(["frame", "sample_frame"], na_position="last")
        linestyle = "--" if group["protocol_style"].iloc[0] == "self-relaxed" else "-"
        ax.plot(group["x"], group["y"], color=TASK_COLORS[group["task_group"].iloc[0]],
                linewidth=0.55, alpha=0.22, linestyle=linestyle, zorder=2)

    for task in TASK_ORDER:
        group = work[work["task_group"].eq(task)]
        if group.empty:
            continue
        ax.text(
            group["x"].median(), group["y"].median(), task,
            color=TASK_COLORS[task], fontsize=11.5, fontweight="bold",
            ha="center", va="center", zorder=6,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
        )

    ax.set_xlim(-10.2, 10.2)
    ax.set_ylim(-10.2, 10.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(2.2)
    ax.set_title("MigrationBench configuration universe", fontsize=23, fontweight="bold", pad=42)
    ax.set_xlabel("Feature-map axis 1", fontsize=15, fontweight="bold", labelpad=11)
    ax.set_ylabel("Feature-map axis 2", fontsize=15, fontweight="bold", labelpad=11)
    ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#555555")

    task_handles = [Patch(facecolor=TASK_COLORS[name], edgecolor="white", label=name) for name in TASK_ORDER]
    task_legend = ax.legend(handles=task_handles, loc="upper center", bbox_to_anchor=(0.5, 1.145),
                            ncol=4, prop={"weight": "bold", "size": 10.2}, frameon=False,
                            handlelength=1.0, columnspacing=1.1, handletextpad=0.35)
    ax.add_artist(task_legend)
    observed = [name for name in SUBLEVEL_ORDER if (work["sublevel"] == name).any()]
    sub_handles = [Patch(facecolor=SUBLEVEL_COLORS[name], edgecolor="#222222" if name.startswith("DFT") else "white",
                          linewidth=0.7, label=name) for name in observed]
    sub_legend = ax.legend(handles=sub_handles, title="Sublayers", loc="upper left", bbox_to_anchor=(0.01, 0.99),
                           ncol=2, prop={"weight": "bold", "size": 9.0}, title_fontsize=9.5,
                           handletextpad=0.35, columnspacing=0.9, frameon=False)
    ax.add_artist(sub_legend)
    protocol_handles = [
        Line2D([0], [0], color="#555555", linewidth=1.1, linestyle="-", label="fixed / sliding"),
        Line2D([0], [0], color="#555555", linewidth=1.1, linestyle="--", label="self-relaxed"),
    ]
    protocol_legend = ax.legend(handles=protocol_handles, title="Path protocol", loc="upper right",
                                bbox_to_anchor=(0.99, 0.99), prop={"weight": "bold", "size": 9.0},
                                title_fontsize=9.5, frameon=False, handletextpad=0.45)
    ax.add_artist(protocol_legend)
    size_handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#777777",
                            markeredgecolor="white", markersize=float(np.sqrt(18.0 + 330.0 * (v / 0.8) ** 0.68)),
                            label=f"{v:.1f} Å") for v in [0.1, 0.3, 0.6]]
    ax.legend(handles=size_handles, title="Cr-shell RMS Δr", loc="lower right", bbox_to_anchor=(1.0, 0.0),
              ncol=3, prop={"weight": "bold", "size": 9.5}, title_fontsize=10.0,
              handletextpad=0.35, columnspacing=0.8, frameon=False)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=420, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df, X = load_common_data()
    pc = project_common_pca(X)
    soap = pc[:, :30]
    candidates = {
        "candidate_01_soap_umap": (soap, False, 30, 0.18, "Common SOAP/PCA → UMAP; no post-hoc island translation"),
        "candidate_02_soap_densmap": (soap, True, 45, 0.22, "Common SOAP/PCA → density-preserving densMAP"),
        "candidate_03_soap_geometry_densmap": (
            np.hstack([soap, geometry_augmentation(df, 0.28)]),
            True,
            45,
            0.22,
            "SOAP/PCA + weak measured Cr-shell geometry → densMAP",
        ),
        "candidate_07_soap_geometry_umap": (
            np.hstack([soap, geometry_augmentation(df, 0.28)]),
            False,
            30,
            0.18,
            "SOAP/PCA + weak measured Cr-shell geometry → UMAP",
        ),
    }
    summary = {}
    for name, (features, densmap, neighbors, min_dist, subtitle) in candidates.items():
        xy = run_umap(features, densmap=densmap, n_neighbors=neighbors, min_dist=min_dist)
        render_candidate(df, xy, name, subtitle)
        summary[name] = {
            "densmap": densmap,
            "n_neighbors": neighbors,
            "min_dist": min_dist,
            "feature_dimensions": int(features.shape[1]),
        }
    local_xy = run_umap(pc[:, :30], densmap=False, n_neighbors=15, min_dist=0.10)
    render_candidate(
        df,
        local_xy,
        "candidate_05_soap_umap_local",
        "Common SOAP/PCA → local UMAP; n_neighbors=15, no post-hoc translation",
    )
    render_candidate(
        df,
        local_xy,
        "candidate_06_soap_umap_equalized",
        "SOAP/PCA → local UMAP + global rank equalization for display density",
        equalize=True,
    )
    summary["candidate_05_soap_umap_local"] = {
        "densmap": False,
        "n_neighbors": 15,
        "min_dist": 0.10,
        "feature_dimensions": 30,
    }
    summary["candidate_06_soap_umap_equalized"] = {
        "base": "candidate_05_soap_umap_local",
        "display_transform": "global rank equalization of UMAP x/y only",
        "feature_dimensions": 30,
    }
    # Direct linear baseline: first two common SOAP/PCA axes, with no
    # nonlinear embedding and no post-hoc movement of task groups.
    render_candidate(
        df,
        pc[:, :2],
        "candidate_04_soap_pca",
        "Common SOAP/PCA axes; direct linear projection, no post-hoc translation",
    )
    summary["candidate_04_soap_pca"] = {
        "method": "common SOAP/PCA",
        "feature_dimensions": 2,
    }
    (CANDIDATE_DIR / "candidate_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
