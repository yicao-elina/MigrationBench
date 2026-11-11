#!/usr/bin/env python3
"""Render a reproducible first-pass MigrationBench configuration universe.

PCA coordinates are the audited common SOAP/PCA coordinates from the existing
task-conditioned atlas. UMAP and t-SNE are deliberately applied to the same
retained 2-D coordinate table, so their scope is explicit and reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
DERIVED = HERE / "derived"
REF = Path("/Users/alina/Project/migrationbench-paper-repro/analysis/task_conditioned_distribution_atlas_20261018/derived/grouped_reference_embedding.csv")
MD = Path("/Users/alina/Project/migrationbench-paper-repro/analysis/task_conditioned_distribution_atlas_20261018/derived/md_distribution_points.csv")
TASK_LONG = Path("/Users/alina/Project/migrationbench-paper-repro/analysis/task_conditioned_distribution_atlas_20261018/derived/task_distribution_long.csv")

COLORS = {
    "train": "#253494",
    "valid": "#2CA25F",
    "test": "#F28E2B",
    "neb": "#B2182B",
    "Foundation": "#D55E00",
    "FT-600K": "#0072B2",
    "FT-MultiT": "#009E73",
    "Scratch": "#CC79A7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 11.5,
        "axes.labelsize": 12.5,
        "axes.titlesize": 14.0,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.15,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": False,
    })


def load_points() -> pd.DataFrame:
    ref = pd.read_csv(REF)
    ref = ref.loc[ref["split"].isin(["train", "valid", "test", "neb"])].copy()
    ref = ref.rename(columns={"split": "group", "pca1": "x_pca", "pca2": "y_pca"})
    ref["point_id"] = [f"reference_{i:04d}" for i in range(len(ref))]
    ref["task_layer"] = np.where(ref["group"].eq("neb"), "fixed_path_NEB", "static")
    ref["lane"] = "DFT_reference"
    ref["model_family"] = ""
    ref["protocol"] = "grouped_SOAP_PCA"
    ref["scientific_status"] = "DESCRIPTIVE_HISTORICAL_COMMON_SPACE"
    ref["source_file"] = ref["source_file"].fillna("")
    ref["frame"] = ref["frame"].astype(str)

    md = pd.read_csv(MD)
    md = md.loc[md["soap_status"].eq("ok")].copy()
    # Keep a deterministic, balanced occupancy sample. The full MD table and
    # short-contact exclusions remain available through the source manifest.
    chunks = []
    for family, part in md.groupby("family", sort=True):
        part = part.sort_values(["frame", "source_file"]).reset_index(drop=True)
        take = min(420, len(part))
        idx = np.linspace(0, len(part) - 1, take, dtype=int)
        chunks.append(part.iloc[np.unique(idx)])
    md = pd.concat(chunks, ignore_index=True)
    md = md.rename(columns={"pca1": "x_pca", "pca2": "y_pca", "frame": "frame"})
    md["point_id"] = [f"md_{i:04d}" for i in range(len(md))]
    md["task_layer"] = "equilibrium_MD"
    md["group"] = md["family"]
    md["lane"] = "MLIP_generated"
    md["model_family"] = md["family"]
    md["protocol"] = "600K_1fs_2050atom_SOAP_PCA"
    md["scientific_status"] = "DESCRIPTIVE_MLIP_ONLY"
    md["natoms"] = md["natoms"].astype(str)
    md["frame"] = md["frame"].astype(str)

    cols = ["point_id", "task_layer", "lane", "group", "model_family", "protocol",
            "scientific_status", "source_file", "frame", "natoms", "x_pca", "y_pca"]
    out = pd.concat([ref[cols], md[cols]], ignore_index=True)
    out["x_pca"] = pd.to_numeric(out["x_pca"], errors="coerce")
    out["y_pca"] = pd.to_numeric(out["y_pca"], errors="coerce")
    out = out.dropna(subset=["x_pca", "y_pca"]).reset_index(drop=True)
    return out


def add_nonlinear_views(points: pd.DataFrame) -> pd.DataFrame:
    xy = points[["x_pca", "y_pca"]].to_numpy(float)
    z = StandardScaler().fit_transform(xy)
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.18,
        metric="euclidean",
        init="spectral",
        random_state=42,
        n_components=2,
    )
    points[["x_umap", "y_umap"]] = reducer.fit_transform(z)
    perplexity = min(35.0, max(5.0, (len(points) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=42,
        method="barnes_hut",
    )
    points[["x_tsne", "y_tsne"]] = tsne.fit_transform(z)
    return points


def style_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, pad=10)
    ax.tick_params(direction="out")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.15)


def plot_view(points: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str,
              title: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 6.6), dpi=180)
    # The reference cloud is the orientation layer; task-specific model clouds
    # are foreground points. Plot order keeps all scientific classes visible.
    ref_order = ["train", "valid", "test", "neb"]
    for group in ref_order:
        part = points.loc[(points["lane"] == "DFT_reference") & points["group"].eq(group)]
        if part.empty:
            continue
        marker = "D" if group == "neb" else "o"
        ax.scatter(part[x], part[y], s=13 if group != "neb" else 42,
                   marker=marker, c=COLORS[group], alpha=0.48 if group != "neb" else 0.9,
                   linewidths=0.35 if group == "neb" else 0,
                   edgecolors="#222222" if group == "neb" else "none",
                   rasterized=True, label=f"DFT {group}")
    for family in ["Foundation", "FT-600K", "FT-MultiT", "Scratch"]:
        part = points.loc[(points["lane"] == "MLIP_generated") & points["model_family"].eq(family)]
        if part.empty:
            continue
        ax.scatter(part[x], part[y], s=8.5, c=COLORS[family], alpha=0.28,
                   linewidths=0, rasterized=True, label=f"MLIP MD · {family}")
    style_axis(ax, xlabel, ylabel, title)
    ax.legend(loc="upper right", fontsize=9.2, ncol=2, columnspacing=1.0,
              handletextpad=0.45, borderaxespad=0.2)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=320 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_style()
    points = add_nonlinear_views(load_points())
    DERIVED.mkdir(parents=True, exist_ok=True)
    points.to_csv(DERIVED / "data_universe_points.csv", index=False)
    manifest = {
        "schema_version": "1.0",
        "status": "RECONSTRUCTION",
        "created": "2026-10-26",
        "representation": {
            "pca": "audited grouped Cr/Sb/Te SOAP/PCA coordinates",
            "umap": "UMAP applied to standardized retained 2-D SOAP/PCA coordinates",
            "tsne": "t-SNE applied to standardized retained 2-D SOAP/PCA coordinates",
            "seed": 42,
            "umap": {"n_neighbors": 30, "min_dist": 0.18, "metric": "euclidean"},
            "tsne": {"perplexity": 35, "max_iter": 1500, "init": "pca"},
        },
        "inputs": [
            {"path": str(REF), "sha256": sha256(REF), "role": "grouped_reference_SOAP_PCA"},
            {"path": str(MD), "sha256": sha256(MD), "role": "MLIP_MD_SOAP_PCA"},
            {"path": str(TASK_LONG), "sha256": sha256(TASK_LONG), "role": "task_status_crosswalk"},
        ],
        "counts": {
            "n_points": int(len(points)),
            "reference_points": int((points["lane"] == "DFT_reference").sum()),
            "mlip_md_points": int((points["lane"] == "MLIP_generated").sum()),
            "md_sample_per_family": 420,
        },
        "scientific_boundary": "MLIP MD occupancy is descriptive and not a DFT error; short-contact-skipped frames are retained in the source table but excluded from the manifold coordinates.",
    }
    (DERIVED / "data_universe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_view(points, "x_pca", "y_pca", "SOAP/PCA-1", "SOAP/PCA-2",
              "MigrationBench configuration universe · PCA", "migrationbench_data_universe_pca")
    plot_view(points, "x_umap", "y_umap", "UMAP-1", "UMAP-2",
              "MigrationBench configuration universe · UMAP", "migrationbench_data_universe_umap")
    plot_view(points, "x_tsne", "y_tsne", "t-SNE-1", "t-SNE-2",
              "MigrationBench configuration universe · t-SNE", "migrationbench_data_universe_tsne")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()

