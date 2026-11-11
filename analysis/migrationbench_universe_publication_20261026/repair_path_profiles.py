#!/usr/bin/env python3
"""Repair path metadata in an already-computed structure table.

The first descriptor pass completed before the energy-key fallback was added.
This small additive repair reads only the saved source file/frame references;
it does not recompute SOAP or change coordinates.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_structure_universe as atlas


def main() -> None:
    path = atlas.DERIVED / "structure_universe_points.csv"
    df = pd.read_csv(path)
    df["system"] = df["system"].astype(str).str.replace("interlayer_interlayer_", "interlayer_", regex=False)
    # Read each source EXTXYZ once.  Re-opening the same multi-frame file for
    # every row is both slow and, for incomplete/empty exploratory files, can
    # leave the repair process waiting in the parser.  Missing frames remain
    # NaN and are retained as an explicit data-status signal.
    path_rows = df[df["source_kind"].isin(["mlip_self_neb", "mlip_interlayer_self"])]
    for source_file, group in path_rows.groupby("source_file", sort=False):
        try:
            source = Path(str(source_file))
            if not source.exists() or source.stat().st_size == 0:
                continue
            # Some archived macOS files are present as compressed/dataless
            # placeholders.  Opening one can block in the filesystem provider;
            # retain its NaNs and let the manifest mark that source as pending.
            flags = subprocess.run(
                ["ls", "-lO", str(source)], capture_output=True, text=True, check=False
            ).stdout
            if "dataless" in flags:
                continue
            frames = read(str(source), index=":")
            if not isinstance(frames, list):
                frames = [frames]
            for idx, row in group.iterrows():
                frame = int(row["frame"])
                if frame < 0 or frame >= len(frames):
                    continue
                atoms = frames[frame]
                value = atoms.info.get("energy", atoms.info.get("mlff_rel_energy_eV", np.nan))
                df.loc[idx, "energy_eV"] = atlas.finite_float(value)
        except Exception:
            continue

    # The compact DFT 1--4 EXTXYZ carries coordinates/path labels but not the
    # archived QE energies. Rejoin the audited per-image profile by frame and
    # reference it to image 0, keeping it separate from MLIP self-NEB values.
    dft_energy_file = Path(
        "/Users/alina/Project/migrationbench-paper-repro/data/processed/foundation_grouped/dft_1-4_neb_energies.csv"
    )
    dft_rows = df[df["source_kind"] == "dft_cr_fixed_neb"]
    if dft_energy_file.exists() and not dft_rows.empty:
        dft_energy = pd.read_csv(dft_energy_file)["energy"].to_numpy(dtype=float)
        dft_energy = dft_energy - dft_energy[0]
        for idx, row in dft_rows.iterrows():
            frame = int(row["frame"])
            if 0 <= frame < len(dft_energy):
                df.loc[idx, "energy_eV"] = float(dft_energy[frame])
    df = atlas.finish_path_profiles(df)
    df.to_csv(path, index=False)
    reps = pd.read_csv(atlas.DERIVED / "cluster_representatives.csv")
    atlas.configure_style()
    atlas.plot_local_view(df, "x_pca", "y_pca", "Cr-local SOAP/PCA-1", "Cr-local SOAP/PCA-2", "MigrationBench structure universe · PCA", "structure_universe_pca", reps)
    atlas.plot_local_view(df, "x_umap", "y_umap", "UMAP-1", "UMAP-2", "MigrationBench structure universe · UMAP", "structure_universe_umap", reps)
    atlas.plot_local_view(df, "x_tsne", "y_tsne", "t-SNE-1", "t-SNE-2", "MigrationBench structure universe · t-SNE", "structure_universe_tsne", reps)
    atlas.plot_distance_view(df, reps)
    atlas.plot_path_profiles(df)
    print(df.groupby("source_kind")["energy_eV"].apply(lambda x: int(x.notna().sum())).to_dict())


if __name__ == "__main__":
    main()
