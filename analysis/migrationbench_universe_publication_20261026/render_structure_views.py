#!/usr/bin/env python3
"""Re-render structure-universe views from the saved derived table."""

from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_structure_universe as atlas


def main() -> None:
    atlas.configure_style()
    atlas.FIGURES.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(atlas.DERIVED / "structure_universe_points.csv")
    reps = pd.read_csv(atlas.DERIVED / "cluster_representatives.csv")
    atlas.plot_local_view(points, "x_pca", "y_pca", "Cr-local SOAP/PCA-1", "Cr-local SOAP/PCA-2", "MigrationBench structure universe · PCA", "structure_universe_pca", reps)
    atlas.plot_local_view(points, "x_umap", "y_umap", "UMAP-1", "UMAP-2", "MigrationBench structure universe · UMAP", "structure_universe_umap", reps)
    atlas.plot_local_view(points, "x_tsne", "y_tsne", "t-SNE-1", "t-SNE-2", "MigrationBench structure universe · t-SNE", "structure_universe_tsne", reps)
    atlas.plot_distance_view(points, reps)
    atlas.plot_path_profiles(points)
    print({"points": len(points), "representatives": len(reps), "figures": str(atlas.FIGURES)})


if __name__ == "__main__":
    main()
