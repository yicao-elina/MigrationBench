#!/usr/bin/env python3
"""Build the structure-resolved MigrationBench configuration universe.

This release is deliberately additive.  It does not overwrite the earlier
whole-cell MD atlas.  Instead it creates a common Cr-centred local SOAP space
for every locally available structure, including grouped reference frames,
DFT reference paths, MLIP self-relaxed paths, and interlayer-sliding paths.

The raw MD SOAP vectors are not present in this checkout.  The retained MD
atlas is therefore kept as a separate evidence layer; this script never mixes
its retained whole-cell PCA coordinates with the new Cr-local coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import iread, read, write
from dscribe.descriptors import SOAP
from scipy.spatial import ConvexHull
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import umap


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
BLENDER_DIR = HERE / "blender"

PAPER_REPRO = Path("/Users/alina/Project/migrationbench-paper-repro")
ATLAS = PAPER_REPRO / "analysis/task_conditioned_distribution_atlas_20261018"
NEB_MATRIX = PAPER_REPRO / "analysis/neb_protocol_matrix_20261018"
RAW_SLIDING = PAPER_REPRO / "data/raw_compact/sliding_recovered_20261024/dft_2.2-layer-dislocation"

GROUPED = ATLAS / "raw_cache/grouped_a2"
NEB_STATUS = NEB_MATRIX / "derived/neb_matrix_status.tsv"

SOAP_KW = {
    "species": ["Cr", "Sb", "Te"],
    "periodic": False,
    "r_cut": 5.0,
    "n_max": 4,
    "l_max": 4,
    "sigma": 0.5,
    "average": "off",
    "sparse": False,
}

# Model hues are held fixed across every local-space view.  Task difficulty
# is represented by marker shape and the distance-to-training gradient, not by
# silently changing the model hue.
MODEL_COLORS = {
    "Foundation": "#E69F00",
    "Scratch-600K": "#0072B2",
    "Scratch-MultiT": "#7B2CBF",
    "FT-600K": "#009E73",
    "FT-MultiT": "#CC79A7",
    "DFT": "#202020",
    "Reference": "#8DA9BF",
}
DISTANCE_COLORS = {
    "near_training": "#B7E3C0",
    "intermediate": "#FEC44F",
    "far_from_training": "#D95F0E",
}
MARKERS = {
    "L0_reference": "o",
    "L2_DFT_path": "D",
    "L3_self_relaxed_NEB": "^",
    "L3_interlayer_self": "s",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def local_soap(soap: SOAP, atoms: Atoms) -> np.ndarray | None:
    """Return the mean Cr-centred local SOAP vector for one structure.

    A single descriptor contract is used for all structures: non-periodic
    Cr-centred SOAP with the same species, cutoff, n/l limits, and Gaussian
    width.  The choice is intentional because a number of legacy self-NEB
    files omit a cell even though their coordinates are valid.  The method is
    common across all rows, and the cell/pbc omission is preserved in the
    source manifest rather than repaired silently.
    """

    centers = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Cr"]
    if not centers:
        return None
    work = atoms.copy()
    work.set_pbc(False)
    # SOAP(periodic=False) ignores the cell, but clearing pbc makes the
    # descriptor contract explicit for legacy files with incomplete headers.
    desc = np.asarray(soap.create(work, centers=centers), dtype=np.float64)
    if desc.ndim != 2 or not np.isfinite(desc).all():
        return None
    return np.mean(desc, axis=0)


def parse_axsf(path: Path) -> list[Atoms]:
    """Parse the Quantum ESPRESSO/AIMS-style AXSF path used by the DFT NEB."""

    lines = path.read_text(errors="replace").splitlines()
    primvec = None
    for i, line in enumerate(lines):
        if line.strip().upper() == "PRIMVEC" and i + 3 < len(lines):
            primvec = np.array([[float(x) for x in lines[i + j + 1].split()[:3]] for j in range(3)])
            break
    if primvec is None:
        raise ValueError(f"No PRIMVEC found in {path}")
    frames: list[Atoms] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().upper().startswith("PRIMCOORD"):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            natoms = int(lines[i].split()[0])
            i += 1
            symbols = []
            positions = []
            for _ in range(natoms):
                parts = lines[i].split()
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
                i += 1
            frames.append(Atoms(symbols=symbols, positions=positions, cell=primvec, pbc=True))
            continue
        i += 1
    if not frames:
        raise ValueError(f"No PRIMCOORD frames found in {path}")
    return frames


def parse_path_energies(path: Path) -> list[float]:
    text = path.read_text(errors="replace")
    values = []
    matches = list(re.finditer(r"^Image:\s*\d+\s*$", text, flags=re.MULTILINE))
    for match in matches:
        tail = text[match.end() :].splitlines()
        value = float("nan")
        for line in tail:
            if line.strip():
                try:
                    value = float(line.strip().split()[0])
                except (ValueError, IndexError):
                    pass
                break
        values.append(value)
    return values


def model_family_from_id(model_id: str) -> str:
    mapping = {
        "foundation_omat_ft": "Foundation",
        "ft600k_seed123": "FT-600K",
        "ft600k_seed234": "FT-600K",
        "ft600k_seed345": "FT-600K",
        "scratch_multit_seed123": "Scratch-MultiT",
        "scratch_multit_seed234": "Scratch-MultiT",
        "scratch_multit_seed345": "Scratch-MultiT",
        "ft_multit_seed123": "FT-MultiT",
        "ft_multit_seed234": "FT-MultiT",
        "ft_multit_seed345": "FT-MultiT",
    }
    return mapping.get(model_id, "Unknown")


def self_model_tag(model_id: str) -> str:
    return {
        "foundation_omat_ft": "foundation",
        "ft600k_seed123": "ft600k",
        "scratch_multit_seed123": "scratch_multit",
        "ft_multit_seed123": "ft_multit",
    }.get(model_id, model_id)


def available_self_extxyz(row: pd.Series) -> tuple[Path | None, str]:
    source = Path(str(row["source"]))
    direct = source.parent / "mlff_neb_images.extxyz"
    if direct.exists():
        return direct, "provisional_current"

    model_id = str(row["model_id"])
    path_id = str(row["path_id"])
    cell_atoms = int(row["cell_atoms"])

    # The legacy Foundation 81-atom paths are retained in the compact raw
    # cache; they are useful geometry but carry the legacy provenance flag.
    if model_id == "foundation_omat_ft" and cell_atoms == 81:
        candidates = sorted(
            (PAPER_REPRO / "data/raw_compact/revision1").glob(f"mb_mlff_cpu_81_{path_id}__s42_*/mlff_81_{path_id}_s42/mlff_neb_images.extxyz")
        )
        if candidates:
            return candidates[0], "legacy_geometry_provenance_partial"

    # MultiT retry geometry is locally present under the pending archive even
    # though the canonical status manifest points to a separate run directory.
    pending_root = NEB_MATRIX / "raw/self_gpu_retry_pending"
    for candidate in pending_root.rglob("mlff_neb_images.extxyz"):
        text = str(candidate)
        if path_id in text and model_id in text:
            return candidate, "exploratory_pending_geometry"
    return None, "missing_local_extxyz"


def make_record(**kwargs) -> dict:
    base = {
        "source_kind": "",
        "source_file": "",
        "source_sha256": "",
        "frame": -1,
        "natoms": -1,
        "split": "",
        "task_layer": "",
        "protocol": "",
        "system": "",
        "path_id": "",
        "path_key": "",
        "model_family": "",
        "model_label": "",
        "model_id": "",
        "model_status": "",
        "scientific_status": "",
        "geometry_status": "",
        "energy_eV": float("nan"),
        "source_has_cell": False,
        "renderable": False,
    }
    base.update(kwargs)
    return base


def add_grouped_reference(soap: SOAP, records: list[dict], vectors: list[np.ndarray]) -> None:
    for split in ["train", "valid", "test"]:
        path = GROUPED / f"{split}.xyz"
        for frame, atoms in enumerate(iread(str(path), index=":")):
            vector = local_soap(soap, atoms)
            if vector is None:
                continue
            records.append(
                make_record(
                    source_kind="grouped_reference",
                    source_file=str(path),
                    source_sha256=sha256(path),
                    frame=frame,
                    natoms=len(atoms),
                    split=split,
                    task_layer="L0_reference",
                    protocol="grouped_reference_Cr_local_SOAP",
                    system="grouped_600K_reference",
                    model_family="Reference",
                    model_label=f"Grouped {split}",
                    model_status="REFERENCE_DATASET",
                    scientific_status="REFERENCE_CONFIGURATION",
                    geometry_status="available",
                    source_has_cell=bool(atoms.pbc.any()),
                    renderable=True,
                )
            )
            vectors.append(vector)


def add_fixed_dft_neb(soap: SOAP, records: list[dict], vectors: list[np.ndarray]) -> None:
    path = GROUPED / "neb_1-4.extxyz"
    for frame, atoms in enumerate(iread(str(path), index=":")):
        vector = local_soap(soap, atoms)
        if vector is None:
            continue
        records.append(
            make_record(
                source_kind="dft_cr_fixed_neb",
                source_file=str(path),
                source_sha256=sha256(path),
                frame=frame,
                natoms=len(atoms),
                split="dft_neb",
                task_layer="L2_DFT_path",
                protocol="DFT_fixed_path_NEB_61",
                system="Cr_migration_61",
                path_id="1-4",
                path_key="dft_cr_61_1-4",
                model_family="DFT",
                model_label="DFT reference",
                model_id="dft_cr61",
                model_status="COMPLETED_BUT_SCF_WARNING",
                scientific_status="PROVISIONAL_DFT_REFERENCE",
                geometry_status="available",
                source_has_cell=bool(atoms.pbc.any()),
                renderable=True,
            )
        )
        vectors.append(vector)


def add_dft_interlayer(soap: SOAP, records: list[dict], vectors: list[np.ndarray]) -> None:
    for system, folder, atoms_count in [
        ("interlayer_bulk_2x2", "1.bulk", 80),
        ("interlayer_cr_gap_2x2", "2.cr-in-gap", 81),
    ]:
        axsf = RAW_SLIDING / folder / "sb2te3.axsf"
        path_file = RAW_SLIDING / folder / "sb2te3.path"
        energies = parse_path_energies(path_file)
        frames = parse_axsf(axsf)
        for frame, atoms in enumerate(frames):
            vector = local_soap(soap, atoms)
            if vector is None:
                continue
            records.append(
                make_record(
                    source_kind="dft_interlayer_neb",
                    source_file=str(axsf),
                    source_sha256=sha256(axsf),
                    frame=frame,
                    natoms=len(atoms),
                    split="dft_interlayer",
                    task_layer="L2_DFT_path",
                    protocol="DFT_interlayer_NEB",
                    system=system,
                    path_id="dft",
                    path_key=f"dft_interlayer_{system}",
                    model_family="DFT",
                    model_label="DFT reference",
                    model_id="dft_interlayer",
                    model_status="COMPLETED_BUT_SCF_WARNING",
                    scientific_status="PROVISIONAL_DFT_REFERENCE",
                    geometry_status="available",
                    energy_eV=energies[frame] if frame < len(energies) else float("nan"),
                    source_has_cell=True,
                    renderable=True,
                )
            )
            vectors.append(vector)


def add_self_neb(soap: SOAP, records: list[dict], vectors: list[np.ndarray], coverage: list[dict]) -> None:
    status = pd.read_csv(NEB_STATUS, sep="\t")
    status = status.loc[status["protocol"].eq("self_relaxed")].copy()
    seen = set()
    for _, row in status.iterrows():
        family = model_family_from_id(str(row["model_id"]))
        extxyz, geometry_status = available_self_extxyz(row)
        key = (int(row["cell_atoms"]), str(row["path_id"]), str(row["model_id"]))
        coverage.append(
            {
                "task": "Cr_self_relaxed_NEB",
                "system": f"Cr_migration_{int(row['cell_atoms'])}",
                "path_id": str(row["path_id"]),
                "model_id": str(row["model_id"]),
                "model_family": family,
                "status_registry": str(row["publication_status"]),
                "run_status": str(row["run_status"]),
                "geometry_status": geometry_status,
                "source_file": str(extxyz) if extxyz else "",
            }
        )
        if extxyz is None or key in seen:
            continue
        seen.add(key)
        source_status = str(row["publication_status"])
        for frame, atoms in enumerate(iread(str(extxyz), index=":")):
            vector = local_soap(soap, atoms)
            if vector is None:
                continue
            energy = finite_float(atoms.info.get("energy", atoms.info.get("mlff_rel_energy_eV", np.nan)))
            records.append(
                make_record(
                    source_kind="mlip_self_neb",
                    source_file=str(extxyz),
                    source_sha256=sha256(extxyz),
                    frame=frame,
                    natoms=len(atoms),
                    split="mlip_self_neb",
                    task_layer="L3_self_relaxed_NEB",
                    protocol="MLIP_self_relaxed_NEB_endpoint_proxy",
                    system=f"Cr_migration_{int(row['cell_atoms'])}",
                    path_id=str(row["path_id"]),
                    path_key=f"mlip_self_{row['model_id']}_{row['cell_atoms']}_{row['path_id']}",
                    model_family=family,
                    model_label=str(row["model_label"]),
                    model_id=str(row["model_id"]),
                    model_status=source_status,
                    scientific_status=(
                        "EXPLORATORY_PENDING_GEOMETRY" if geometry_status == "exploratory_pending_geometry" else source_status
                    ),
                    geometry_status=geometry_status,
                    energy_eV=energy,
                    source_has_cell=bool(atoms.pbc.any()),
                    renderable=True,
                )
            )
            vectors.append(vector)

    # Make the missing target family explicit.  The current self-NEB matrix
    # has no Scratch-600K rows, so its absence must not disappear in a plot.
    for path_id in ["1-2", "1-3", "1-4", "1-5", "1-7", "neb_1", "neb_2", "neb_3", "neb_4", "neb_5"]:
        atoms_count = 61 if path_id.startswith("1-") else 81
        coverage.append(
            {
                "task": "Cr_self_relaxed_NEB",
                "system": f"Cr_migration_{atoms_count}",
                "path_id": path_id,
                "model_id": "scratch_600k",
                "model_family": "Scratch-600K",
                "status_registry": "NOT_PRESENT_IN_SELF_MATRIX",
                "run_status": "MISSING",
                "geometry_status": "missing_local_extxyz",
                "source_file": "",
            }
        )


def add_interlayer_self(soap: SOAP, records: list[dict], vectors: list[np.ndarray], coverage: list[dict]) -> None:
    root = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_2x2_20261024/runs/self")
    files = sorted(root.rglob("mlff_neb_images.extxyz"))
    for path in files:
        parts = path.parts
        try:
            system = parts[parts.index("self") + 1]
            variant = parts[parts.index("self") + 2]
        except (ValueError, IndexError):
            system, variant = "unknown", path.parent.name
        label = f"Foundation-{variant.replace('_', ' ')}"
        for frame, atoms in enumerate(iread(str(path), index=":")):
            vector = local_soap(soap, atoms)
            if vector is None:
                continue
            energy = finite_float(atoms.info.get("energy", atoms.info.get("mlff_rel_energy_eV", np.nan)))
            records.append(
                make_record(
                    source_kind="mlip_interlayer_self",
                    source_file=str(path),
                    source_sha256=sha256(path),
                    frame=frame,
                    natoms=len(atoms),
                    split="mlip_interlayer_self",
                    task_layer="L3_interlayer_self",
                    protocol="MLIP_self_relaxed_interlayer_endpoint_proxy",
                    system=system,
                    path_id=variant,
                    path_key=f"mlip_interlayer_{system}_{variant}",
                    model_family="Foundation",
                    model_label=label,
                    model_id=f"foundation_{variant}",
                    model_status="CONVERGED_PROXY",
                    scientific_status="CONVERGED_PROXY_ENDPOINT_ONLY",
                    geometry_status="available",
                    energy_eV=energy,
                    source_has_cell=bool(atoms.pbc.any()),
                    renderable=True,
                )
            )
            vectors.append(vector)

    # The current local interlayer self geometry campaign is Foundation-only.
    # Keep the five requested families visible as coverage rows.
    for family in ["Foundation", "Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT"]:
        coverage.append(
            {
                "task": "interlayer_self_relaxed",
                "system": "bulk_2x2 + Cr-in-gap_2x2",
                "path_id": "all_local_paths",
                "model_id": family,
                "model_family": family,
                "status_registry": "AVAILABLE" if family == "Foundation" else "MISSING_LOCAL_EXTXYZ",
                "run_status": "CONVERGED_PROXY" if family == "Foundation" else "MISSING",
                "geometry_status": "available" if family == "Foundation" else "missing_local_extxyz",
                "source_file": str(root) if family == "Foundation" else "",
            }
        )


def assign_distances_and_clusters(df: pd.DataFrame, X: np.ndarray) -> tuple[pd.DataFrame, dict, np.ndarray]:
    train_mask = df["source_kind"].eq("grouped_reference") & df["split"].eq("train")
    mean = X[train_mask.to_numpy()].mean(axis=0)
    scale = X[train_mask.to_numpy()].std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale

    n_pc = min(50, Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=n_pc, whiten=True, random_state=0)
    pca.fit(Z[train_mask.to_numpy()])
    pc = pca.transform(Z)
    df["x_pca"] = pc[:, 0]
    df["y_pca"] = pc[:, 1]

    train_pc = pc[train_mask.to_numpy()]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(train_pc[:, : min(30, n_pc)])
    distances = nn.kneighbors(pc[:, : min(30, n_pc)], return_distance=True)[0][:, 0]
    df["distance_to_training"] = distances
    non_train = distances[~train_mask.to_numpy()]
    q50, q90 = np.quantile(non_train, [0.5, 0.9]) if len(non_train) else (1.0, 2.0)
    df["distance_bin"] = np.where(
        distances <= q50,
        "near_training",
        np.where(distances <= q90, "intermediate", "far_from_training"),
    )
    df.loc[train_mask, "distance_bin"] = "near_training"

    embed = pc[:, : min(30, n_pc)]
    clusterer = HDBSCAN(min_cluster_size=max(20, min(45, len(df) // 30)), min_samples=10, cluster_selection_method="eom")
    labels = clusterer.fit_predict(embed)
    df["cluster_raw"] = labels
    counts = Counter(int(x) for x in labels if int(x) >= 0)
    ordered = [x for x, _ in counts.most_common()]
    remap = {old: f"C{i + 1:02d}" for i, old in enumerate(ordered)}
    df["cluster"] = [remap.get(int(x), "noise") for x in labels]
    cluster_color_map = {}
    for i, label in enumerate([f"C{i + 1:02d}" for i in range(len(ordered))]):
        cluster_color_map[label] = mpl.colors.to_hex(plt.get_cmap("tab20")(i % 20))

    fit = {
        "descriptor": SOAP_KW,
        "fit_split": "grouped_reference/train",
        "n_train": int(train_mask.sum()),
        "n_features": int(X.shape[1]),
        "n_pca_components_for_nonlinear": int(n_pc),
        "pca_explained_variance_ratio_first2": pca.explained_variance_ratio_[:2].tolist(),
        "distance_metric": "nearest-neighbor Euclidean distance in first 30 whitened PCA coordinates",
        "distance_thresholds": {"near_to_intermediate": float(q50), "intermediate_to_far": float(q90)},
        "clusterer": {"method": "sklearn.HDBSCAN", "min_cluster_size": int(max(20, min(45, len(df) // 30))), "min_samples": 10},
        "cluster_sizes": {remap.get(k, "noise"): int(v) for k, v in counts.items()},
        "cluster_colors": cluster_color_map,
        "pca": pca,
        "mean": mean,
        "scale": scale,
    }
    df["cluster_color"] = df["cluster"].map(cluster_color_map).fillna("#B8B8B8")
    df["model_color"] = df["model_family"].map(MODEL_COLORS).fillna("#777777")
    return df, fit, pc


def save_representatives(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    render_input_dir = DERIVED / "render_inputs"
    render_input_dir.mkdir(parents=True, exist_ok=True)
    reps = []
    clusters = [c for c in df["cluster"].unique() if c != "noise"]
    # Prefer clusters that contain path geometries; otherwise the local map can
    # be dominated by repetitive grouped-training frames.
    stats = []
    for cluster in clusters:
        part = df[df["cluster"].eq(cluster)].copy()
        special = (~part["source_kind"].eq("grouped_reference")).sum()
        stats.append((cluster, len(part), int(special)))
    stats.sort(key=lambda x: (x[2], x[1]), reverse=True)
    selected_clusters = [x[0] for x in stats[:8]]

    for cluster in selected_clusters:
        part = df[df["cluster"].eq(cluster)].copy()
        # Find the descriptor-medoid among available, renderable structures.
        part = part[part["renderable"].astype(bool)]
        if part.empty:
            continue
        # HDBSCAN labels already represent local density; use the point nearest
        # the cluster's PCA centroid as a deterministic medoid proxy.
        center = part[["x_pca", "y_pca"]].to_numpy(float).mean(axis=0)
        part["_d"] = np.linalg.norm(part[["x_pca", "y_pca"]].to_numpy(float) - center, axis=1)
        row = part.sort_values(["_d", "distance_to_training"]).iloc[0]
        atoms = load_atoms_for_row(row)
        if atoms is None:
            continue
        out = render_input_dir / f"representative_{cluster}.extxyz"
        atoms.info["universe_cluster"] = cluster
        atoms.info["source_file"] = row["source_file"]
        atoms.info["source_frame"] = int(row["frame"])
        write(str(out), atoms, format="extxyz")
        reps.append(
            {
                "cluster": cluster,
                "cluster_color": row["cluster_color"],
                "source_kind": row["source_kind"],
                "source_file": row["source_file"],
                "source_frame": int(row["frame"]),
                "render_input": str(out),
                "task_layer": row["task_layer"],
                "model_family": row["model_family"],
                "model_label": row["model_label"],
                "system": row["system"],
                "path_id": row["path_id"],
                "distance_to_training": float(row["distance_to_training"]),
                "scientific_status": row["scientific_status"],
            }
        )
    reps_df = pd.DataFrame(reps)
    reps_df.to_csv(DERIVED / "cluster_representatives.csv", index=False)
    return reps_df


def load_atoms_for_row(row: pd.Series) -> Atoms | None:
    try:
        if row["source_kind"] == "dft_interlayer_neb":
            frames = parse_axsf(Path(row["source_file"]))
            return frames[int(row["frame"])]
        return read(str(row["source_file"]), index=int(row["frame"]))
    except Exception as exc:
        print(f"[warn] cannot load representative {row['source_file']} frame {row['frame']}: {exc}")
        return None


def finish_path_profiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["path_image_fraction"] = np.nan
    out["relative_energy_eV"] = np.nan
    for key, idx in out[out["path_key"].ne("")].groupby("path_key").groups.items():
        part = out.loc[idx].sort_values("frame")
        out.loc[part.index, "path_image_fraction"] = np.linspace(0.0, 1.0, len(part))
        energies = part["energy_eV"].to_numpy(float)
        if np.isfinite(energies).sum() >= 2:
            finite = energies[np.isfinite(energies)]
            out.loc[part.index, "relative_energy_eV"] = energies - finite[0]
    return out


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.05,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_cluster_hulls(ax, df: pd.DataFrame, x: str, y: str) -> None:
    for cluster, part in df[df["cluster"].ne("noise")].groupby("cluster"):
        if len(part) < 12:
            continue
        xy = part[[x, y]].to_numpy(float)
        if not np.isfinite(xy).all() or len(xy) < 3:
            continue
        # Convex hulls are only a visual density guide; they are not class
        # boundaries and are intentionally very faint.
        try:
            hull = ConvexHull(xy)
        except Exception:
            continue
        color = part["cluster_color"].iloc[0]
        poly = Polygon(xy[hull.vertices], closed=True, facecolor=color, edgecolor=color, alpha=0.045, linewidth=0.7, zorder=0)
        ax.add_patch(poly)


def plot_local_view(df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str, stem: str, reps: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=180)
    add_cluster_hulls(ax, df, x, y)

    ref = df[df["source_kind"].eq("grouped_reference")]
    ax.scatter(ref[x], ref[y], s=4.5, c="#B8CBD8", alpha=0.22, linewidths=0, rasterized=True, zorder=1)

    for task_layer, marker in MARKERS.items():
        part = df[df["task_layer"].eq(task_layer)]
        if part.empty:
            continue
        for source_group, group in part.groupby("model_family"):
            color = MODEL_COLORS.get(source_group, "#666666")
            pending = group["geometry_status"].eq("exploratory_pending_geometry")
            stable = group[~pending]
            if not stable.empty:
                ax.scatter(stable[x], stable[y], s=24 if task_layer != "L2_DFT_path" else 32,
                           marker=marker, c=color, alpha=0.82, edgecolors="white", linewidths=0.35,
                           rasterized=True, zorder=3)
            if not pending.empty:
                pending_group = group[pending]
                ax.scatter(pending_group[x], pending_group[y], s=30, marker=marker, facecolors="none",
                           edgecolors=color, linewidths=1.0, alpha=0.9, rasterized=True, zorder=4)

        # Connect path images, preserving route structure in the embedding.
        for _, group in part.groupby("path_key"):
            if len(group) < 2 or not group["path_key"].iloc[0]:
                continue
            group = group.sort_values("frame")
            color = MODEL_COLORS.get(group["model_family"].iloc[0], "#666666")
            ax.plot(group[x], group[y], color=color, alpha=0.32, linewidth=0.85, zorder=2)

    if not reps.empty:
        for _, row in reps.iterrows():
            point = df[(df["source_file"].eq(row["source_file"])) & (df["frame"].eq(row["source_frame"]))]
            if point.empty:
                continue
            x0, y0 = float(point.iloc[0][x]), float(point.iloc[0][y])
            ax.scatter([x0], [y0], s=122, marker="*", facecolors="white", edgecolors=row["cluster_color"], linewidths=1.5, zorder=7)
            ax.text(x0, y0, row["cluster"], ha="center", va="center", fontsize=7.2, fontweight="bold", color="#222222", zorder=8)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, pad=10)
    ax.tick_params(direction="out")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[family], markeredgecolor="white", markersize=7, label=family)
        for family in ["Foundation", "Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT"]
    ]
    handles += [
        Line2D([0], [0], marker=MARKERS[task], color="#444444", linestyle="none", markersize=6, label=task.replace("L0_", "").replace("L2_", "").replace("L3_", ""))
        for task in ["L2_DFT_path", "L3_self_relaxed_NEB", "L3_interlayer_self"]
    ]
    ax.legend(handles=handles, loc="best", fontsize=7.5, ncol=2, handletextpad=0.35, columnspacing=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=360 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_distance_view(df: pd.DataFrame, reps: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=180)
    sc = ax.scatter(df["x_umap"], df["y_umap"], c=np.log1p(df["distance_to_training"]), cmap="YlOrRd", s=8, alpha=0.62, linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("log(1 + nearest training distance)\nCr-local SOAP/PCA space", fontsize=10)
    for _, row in reps.iterrows():
        point = df[(df["source_file"].eq(row["source_file"])) & (df["frame"].eq(row["source_frame"]))]
        if point.empty:
            continue
        p = point.iloc[0]
        ax.scatter([p["x_umap"]], [p["y_umap"]], s=112, marker="*", facecolors="white", edgecolors=row["cluster_color"], linewidths=1.4, zorder=5)
        ax.text(p["x_umap"], p["y_umap"], row["cluster"], ha="center", va="center", fontsize=7, fontweight="bold", zorder=6)
    ax.set_xlabel("UMAP-1", fontweight="bold")
    ax.set_ylabel("UMAP-2", fontweight="bold")
    ax.set_title("Training-distance gradient · structure-resolved universe", pad=10)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"structure_universe_distance.{ext}", dpi=360 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_path_profiles(df: pd.DataFrame) -> None:
    # A compact diagnostic of route shape and energy where energy metadata are
    # actually present.  It keeps fixed-path and self-relaxed paths in separate
    # subplots and never converts the curves into a common barrier ranking.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), dpi=180)
    panels = [
        ("Cr migration · 61 atom · 1-4", df[(df["system"].eq("Cr_migration_61")) & (df["path_id"].eq("1-4"))]),
        ("Interlayer bulk · 2×2", df[df["system"].eq("interlayer_bulk_2x2")]),
        ("Interlayer Cr-in-gap · 2×2", df[df["system"].eq("interlayer_cr_gap_2x2")]),
    ]
    for ax, (title, part) in zip(axes, panels):
        for key, group in part.groupby("path_key"):
            group = group.sort_values("frame")
            if group["relative_energy_eV"].notna().sum() < 2:
                continue
            family = group["model_family"].iloc[0]
            color = MODEL_COLORS.get(family, "#666666")
            style = "--" if group["geometry_status"].iloc[0] == "exploratory_pending_geometry" else "-"
            label = group["model_label"].iloc[0]
            ax.plot(group["path_image_fraction"], group["relative_energy_eV"], marker="o", markersize=2.5, linewidth=1.2, color=color, linestyle=style, label=label)
        ax.set_title(title, fontsize=10.2, pad=8)
        ax.set_xlabel("path image fraction")
        ax.set_ylabel("relative energy (eV)")
        ax.tick_params(direction="out")
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(0.95)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"path_geometry_energy_profiles.{ext}", dpi=360 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def save_manifest(df: pd.DataFrame, fit: dict, coverage: list[dict], reps: pd.DataFrame, inputs: list[Path]) -> None:
    out = {
        "schema_version": "1.0",
        "created": "2026-10-26",
        "status": "PROVENANCE_AWARE_STRUCTURE_RELEASE",
        "representation": {
            "name": "Cr-centred local SOAP",
            "descriptor": SOAP_KW,
            "pca_fit": "grouped_reference/train only",
            "nonlinear_views": "UMAP and t-SNE fit to the first 30 whitened local-SOAP PCA coordinates",
            "umap": {"n_neighbors": 30, "min_dist": 0.18, "metric": "euclidean", "random_state": 42},
            "tsne": {"perplexity": 35, "max_iter": 1500, "init": "pca", "random_state": 42},
        },
        "fit_summary": {k: v for k, v in fit.items() if k not in {"pca", "mean", "scale"}},
        "counts": {
            "all_structure_points": int(len(df)),
            "grouped_reference": int((df["source_kind"] == "grouped_reference").sum()),
            "dft_paths": int(df["source_kind"].str.startswith("dft_").sum()),
            "mlip_self_neb": int((df["source_kind"] == "mlip_self_neb").sum()),
            "mlip_interlayer_self": int((df["source_kind"] == "mlip_interlayer_self").sum()),
            "cluster_representatives": int(len(reps)),
        },
        "coverage": coverage,
        "source_inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs if p.exists()],
        "scientific_boundaries": [
            "The local structure map is a Cr-centred representation; it is not the same descriptor as the retained whole-cell MD atlas.",
            "MLIP self-relaxed paths are model-PES endpoint proxies, not DFT minimum-energy paths or validated barriers.",
            "DFT interlayer references retain COMPLETED_BUT_SCF_WARNING and are diagnostic reference geometries.",
            "Pending MultiT geometries are shown as exploratory open markers and are not promoted to converged publication evidence.",
            "Distance to grouped training is a descriptor-space diagnostic, not a universal task-difficulty score.",
        ],
    }
    (DERIVED / "structure_universe_manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    pd.DataFrame(coverage).to_csv(DERIVED / "coverage_matrix.csv", index=False)


def main() -> None:
    configure_style()
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    BLENDER_DIR.mkdir(parents=True, exist_ok=True)

    soap = SOAP(**SOAP_KW)
    records: list[dict] = []
    vectors: list[np.ndarray] = []
    coverage: list[dict] = []
    add_grouped_reference(soap, records, vectors)
    add_fixed_dft_neb(soap, records, vectors)
    add_dft_interlayer(soap, records, vectors)
    add_self_neb(soap, records, vectors, coverage)
    add_interlayer_self(soap, records, vectors, coverage)

    if not records:
        raise RuntimeError("No structure records were generated")
    df = pd.DataFrame(records)
    X = np.vstack(vectors)
    if len(df) != X.shape[0]:
        raise RuntimeError(f"metadata/vector mismatch: {len(df)} != {X.shape[0]}")
    df, fit, pc = assign_distances_and_clusters(df, X)

    nonlinear_input = pc[:, : min(30, pc.shape[1])]
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.18, metric="euclidean", init="spectral", random_state=42, n_components=2)
    df[["x_umap", "y_umap"]] = reducer.fit_transform(nonlinear_input)
    perplexity = min(35.0, max(5.0, (len(df) - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", max_iter=1500, random_state=42, method="barnes_hut")
    df[["x_tsne", "y_tsne"]] = tsne.fit_transform(nonlinear_input)
    df = finish_path_profiles(df)

    # Save coordinates and a compact descriptor-fit artifact.
    df.to_csv(DERIVED / "structure_universe_points.csv", index=False)
    np.savez_compressed(
        DERIVED / "cr_local_soap_fit.npz",
        mean=fit["mean"], scale=fit["scale"],
        pca_mean=fit["pca"].mean_, components=fit["pca"].components_,
        explained_variance=fit["pca"].explained_variance_, explained_variance_ratio=fit["pca"].explained_variance_ratio_,
    )
    np.savez_compressed(DERIVED / "structure_universe_local_soap_vectors.npz", vectors=X)

    reps = save_representatives(df, fit)
    inputs = [GROUPED / f"{s}.xyz" for s in ["train", "valid", "test"]] + [GROUPED / "neb_1-4.extxyz", NEB_STATUS]
    inputs += [RAW_SLIDING / folder / "sb2te3.axsf" for folder in ["1.bulk", "2.cr-in-gap"]]
    save_manifest(df, fit, coverage, reps, inputs)

    plot_local_view(df, "x_pca", "y_pca", "Cr-local SOAP/PCA-1", "Cr-local SOAP/PCA-2", "MigrationBench structure universe · PCA", "structure_universe_pca", reps)
    plot_local_view(df, "x_umap", "y_umap", "UMAP-1", "UMAP-2", "MigrationBench structure universe · UMAP", "structure_universe_umap", reps)
    plot_local_view(df, "x_tsne", "y_tsne", "t-SNE-1", "t-SNE-2", "MigrationBench structure universe · t-SNE", "structure_universe_tsne", reps)
    plot_distance_view(df, reps)
    plot_path_profiles(df)

    # A small job table is the handoff between the analysis layer and the
    # bright/transparent Blender renderer.
    if not reps.empty:
        jobs = reps.copy()
        jobs["output_dir"] = jobs["cluster"].map(lambda c: str(BLENDER_DIR / "cluster_renders" / c))
        jobs.to_csv(BLENDER_DIR / "render_jobs.tsv", sep="\t", index=False)

    print(json.dumps({
        "points": int(len(df)),
        "source_counts": df["source_kind"].value_counts().to_dict(),
        "model_counts": df["model_family"].value_counts().to_dict(),
        "clusters": df["cluster"].value_counts().head(12).to_dict(),
        "representatives": int(len(reps)),
        "coverage_rows": int(len(coverage)),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
