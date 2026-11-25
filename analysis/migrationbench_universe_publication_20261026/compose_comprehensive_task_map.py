#!/usr/bin/env python3
"""Build the comprehensive, task-conditioned configuration universe map.

The map uses one common Cr-centred SOAP/PCA/UMAP/t-SNE contract for the
audited static/path structures and a uniformly sampled set of 15 MD branches.
The reference manifold retains its common-space footprint; sparse task layers
keep their within-layer UMAP geometry but are placed in separated display
regions so four physical regimes and their protocol sublayers can be read on
one page.
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
from ase import Atoms
from ase.io import iread
from dscribe.descriptors import SOAP
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"

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
CUTOFF_A = 3.75

# Parent colours encode physical configuration regimes. Sublayer colours use
# the same hue with controlled saturation/lightness changes; method/protocol
# remain separate metadata and edge/path cues.
TASK_ORDER = ["L0 DFT Reference", "L1 Thermal Equilibrium", "L2 NEB Migration", "L3 Layer Sliding"]
TASK_COLORS = {
    "L0 DFT Reference": "#6E8793",
    "L1 Thermal Equilibrium": "#7658C7",
    "L2 NEB Migration": "#2D7FB5",
    "L3 Layer Sliding": "#E8792F",
}

SUBLEVEL_ORDER = [
    "Static DFT",
    "DFT-AIMD",
    "MLIP-MD",
    "DFT Fixed",
    "MLIP Relaxed",
    "DFT Sliding",
    "MLIP Sliding",
]
SUBLEVEL_COLORS = {
    "Static DFT": "#587482",
    "DFT-AIMD": "#4B3199",
    "MLIP-MD": "#A28DE2",
    "DFT Fixed": "#1F5D88",
    "MLIP Relaxed": "#69A9D3",
    "DFT Sliding": "#C45C20",
    "MLIP Sliding": "#F3B16F",
}

TASK_ANCHORS = {
    "L1 Thermal Equilibrium": (-5.7, 4.7),
    "L2 NEB Migration": (-1.0, 5.8),
    "L3 Layer Sliding": (4.8, -4.3),
}
TASK_SCALES = {
    "L1 Thermal Equilibrium": 2.30,
    "L2 NEB Migration": 1.75,
    "L3 Layer Sliding": 2.25,
}

MD_SAMPLE = DERIVED / "md_map_sample.extxyz"
MD_MANIFEST = DERIVED / "md_map_sample_manifest.json"
MD_AUDIT = Path(
    "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/"
    "md_1ns_updated_publication_figure_20261025/MD15_STATUS_AUDIT_R02.tsv"
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "axes.linewidth": 2.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def local_soap(soap: SOAP, atoms: Atoms) -> np.ndarray | None:
    centers = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Cr"]
    if not centers:
        return None
    work = atoms.copy()
    work.set_pbc(False)
    desc = np.asarray(soap.create(work, centers=centers), dtype=np.float64)
    if desc.ndim != 2 or not np.isfinite(desc).all():
        return None
    return np.mean(desc, axis=0)


def valid_mic(atoms: Atoms) -> bool:
    try:
        return bool(np.any(atoms.pbc) and atoms.cell.rank >= 2)
    except Exception:
        return False


def shell_data_for_focus(atoms: Atoms, focus: int, cutoff: float = CUTOFF_A) -> dict:
    others = np.asarray([i for i in range(len(atoms)) if i != focus], dtype=int)
    mic = valid_mic(atoms)
    try:
        vectors = np.asarray(atoms.get_distances(focus, others, mic=mic, vector=True), dtype=float)
    except Exception:
        vectors = np.asarray(atoms.get_positions()[others] - atoms.positions[focus], dtype=float)
    distances = np.linalg.norm(vectors, axis=1)
    mask = np.isfinite(distances) & (distances <= cutoff)
    shell_indices = others[mask]
    shell_vectors = vectors[mask]
    shell_distances = distances[mask]
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)[shell_indices]
    order = np.argsort(shell_distances)
    shell_indices = shell_indices[order]
    shell_vectors = shell_vectors[order]
    shell_distances = shell_distances[order]
    symbols = symbols[order]
    angles = []
    if len(shell_vectors) >= 2:
        unit = shell_vectors / np.maximum(shell_distances[:, None], 1e-12)
        cosine = np.clip(unit @ unit.T, -1.0, 1.0)
        tri = np.triu_indices(len(unit), k=1)
        angles = np.degrees(np.arccos(cosine[tri])).tolist()
    by_species = {}
    for species in sorted(set(symbols.tolist())):
        by_species[species] = [float(x) for x in shell_distances[symbols == species]]
    return {
        "focus_index": int(focus),
        "mic_used": mic,
        "indices": [int(x) for x in shell_indices],
        "symbols": [str(x) for x in symbols],
        "distances_A": [float(x) for x in shell_distances],
        "by_species": by_species,
        "angles_deg": [float(x) for x in angles],
    }


def paired_species_delta(shell: dict, reference: dict) -> tuple[float, float, int, int]:
    deltas = []
    count_delta = 0
    for species in sorted(set(shell["by_species"]) | set(reference["by_species"])):
        current = np.asarray(shell["by_species"].get(species, []), dtype=float)
        ref = np.asarray(reference["by_species"].get(species, []), dtype=float)
        count_delta += abs(len(current) - len(ref))
        n = min(len(current), len(ref))
        if n:
            deltas.extend((current[:n] - ref[:n]).tolist())
    if not deltas:
        return float("nan"), float("nan"), count_delta, 0
    values = np.asarray(deltas, dtype=float)
    return float(np.mean(np.abs(values))), float(np.sqrt(np.mean(values**2))), count_delta, len(values)


def metrics_from_shell(shell: dict, reference: dict) -> dict:
    distances = np.asarray(shell["distances_A"], dtype=float)
    mean_abs, rms, count_delta, paired = paired_species_delta(shell, reference)
    angles = np.asarray(shell["angles_deg"], dtype=float)
    return {
        "focus_index": shell["focus_index"],
        "coordination_number": int(len(distances)),
        "n_Sb": int(sum(symbol == "Sb" for symbol in shell["symbols"])),
        "n_Te": int(sum(symbol == "Te" for symbol in shell["symbols"])),
        "min_bond_A": float(np.min(distances)) if len(distances) else float("nan"),
        "max_bond_A": float(np.max(distances)) if len(distances) else float("nan"),
        "mean_bond_A": float(np.mean(distances)) if len(distances) else float("nan"),
        "bond_length_std_A": float(np.std(distances)) if len(distances) else float("nan"),
        "bond_length_range_A": float(np.ptp(distances)) if len(distances) else float("nan"),
        "mean_abs_delta_A": mean_abs,
        "rms_delta_A": rms,
        "coordination_number_delta": int(len(distances) - len(reference["distances_A"])),
        "species_count_delta": int(count_delta),
        "paired_bonds": int(paired),
        "mean_neighbor_angle_deg": float(np.mean(angles)) if len(angles) else float("nan"),
        "neighbor_angle_std_deg": float(np.std(angles)) if len(angles) else float("nan"),
    }


def aggregate_md_metrics(atoms: Atoms, reference: dict) -> dict:
    centers = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Cr"]
    rows = []
    for focus in centers:
        try:
            rows.append(metrics_from_shell(shell_data_for_focus(atoms, focus), reference))
        except Exception:
            continue
    if not rows:
        return {"focus_index": -1, "n_cr_centers": 0, "rms_delta_A": float("nan")}
    result = {"focus_index": int(rows[0]["focus_index"]), "n_cr_centers": len(rows)}
    integer_keys = {"coordination_number", "n_Sb", "n_Te", "paired_bonds"}
    for key in rows[0]:
        if key == "focus_index":
            continue
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        if key in integer_keys and np.isfinite(values).any():
            result[key] = int(round(result[key]))
    return result


def load_reference_shell() -> dict:
    with (DERIVED / "coordination_shells.json").open() as handle:
        return json.load(handle)["reference"]["shell"]


def md_status_map() -> dict[str, dict]:
    if not MD_AUDIT.exists():
        return {}
    audit = pd.read_csv(MD_AUDIT, sep="\t")
    return {str(row["model_id"]): row.to_dict() for _, row in audit.iterrows()}


def load_md_rows(reference_shell: dict) -> tuple[pd.DataFrame, np.ndarray]:
    if not MD_SAMPLE.exists() or not MD_MANIFEST.exists():
        raise FileNotFoundError("MD sample and manifest are required; run the remote extractor first")
    manifest = json.loads(MD_MANIFEST.read_text())
    frame_meta = {int(row["sample_frame"]): row for row in manifest["frames"]}
    status = md_status_map()
    soap = SOAP(**SOAP_KW)
    records = []
    vectors = []
    for sample_frame, atoms in enumerate(iread(str(MD_SAMPLE), index=":")):
        meta = frame_meta.get(sample_frame)
        if meta is None:
            continue
        vector = local_soap(soap, atoms)
        if vector is None:
            continue
        model_id = str(meta["model_id"])
        audit = status.get(model_id, {})
        family = str(meta.get("family", audit.get("family", "MD")))
        frame = int(meta["source_frame"])
        nframes = next(
            (int(row.get("nframes", 0)) for row in manifest["sources"] if row.get("model_id") == model_id),
            0,
        )
        info = atoms.info
        metrics = aggregate_md_metrics(atoms, reference_shell)
        record = {
            "source_kind": "md_trajectory",
            "source_file": str(meta["source_path"]),
            "source_sha256": str(meta["source_sha256"]),
            "frame": frame,
            "sample_frame": sample_frame,
            "natoms": int(len(atoms)),
            "split": "md_sample",
            "task_layer": "L1_thermal",
            "protocol": "MD_600K_Cr_local_SOAP",
            "system": "CrSb2Te3_2050",
            "path_id": model_id,
            "path_key": "md_" + model_id,
            "model_family": family,
            "model_label": str(audit.get("display_label", model_id)),
            "model_id": model_id,
            "model_status": str(audit.get("status", "REMOTE_SAMPLED")),
            "scientific_status": "MD_TARGET_REACHED_PROVISIONAL"
            if "TARGET_REACHED" in str(audit.get("status", ""))
            else "MD_SCIENTIFIC_THRESHOLD_FAILURE",
            "geometry_status": "remote_uniform_sample",
            "energy_eV": float(info.get("energy", np.nan)),
            "source_has_cell": bool(np.any(atoms.pbc)),
            "renderable": True,
            "source_path_exists": True,
            "path_image_fraction": frame / max(nframes - 1, 1),
            "relative_energy_eV": np.nan,
            "process": "MD thermal",
            "temperature_K": float(info.get("temperature", np.nan)),
            "time_ps": float(info.get("time_fs", np.nan)) / 1000.0,
            "step": int(info.get("step", frame)),
            "nframes_source": nframes,
            "md_aggregation": "mean_over_Cr_centers",
        }
        record.update(metrics)
        records.append(record)
        vectors.append(vector)
    return pd.DataFrame(records), np.vstack(vectors)


def task_group(row: pd.Series) -> str:
    source = str(row.get("source_kind", ""))
    if source == "grouped_reference":
        return "L0 DFT Reference"
    if source == "md_trajectory":
        return "L1 Thermal Equilibrium"
    if source == "dft_cr_fixed_neb":
        return "L2 NEB Migration"
    if source == "mlip_self_neb":
        return "L2 NEB Migration"
    if source in {"dft_interlayer_neb", "mlip_interlayer_self"}:
        return "L3 Layer Sliding"
    return "Other"


def sublevel(row: pd.Series) -> str:
    source = str(row.get("source_kind", ""))
    if source == "grouped_reference":
        return "Static DFT"
    if source == "md_trajectory":
        protocol = str(row.get("protocol", "")).upper()
        family = str(row.get("model_family", "")).upper()
        return "DFT-AIMD" if "AIMD" in protocol or family == "DFT" else "MLIP-MD"
    if source == "dft_cr_fixed_neb":
        return "DFT Fixed"
    if source == "mlip_self_neb":
        return "MLIP Relaxed"
    if source == "dft_interlayer_neb":
        return "DFT Sliding"
    if source == "mlip_interlayer_self":
        return "MLIP Sliding"
    return "Other"


def protocol_style(row: pd.Series) -> str:
    source = str(row.get("source_kind", ""))
    if source == "dft_cr_fixed_neb":
        return "fixed"
    if source == "mlip_self_neb":
        return "self-relaxed"
    if source in {"dft_interlayer_neb", "mlip_interlayer_self"}:
        return "sliding"
    if source == "md_trajectory":
        return "thermal"
    return "reference"


def project_common_pca(X: np.ndarray) -> np.ndarray:
    fit = np.load(DERIVED / "cr_local_soap_fit.npz")
    mean = fit["mean"]
    scale = fit["scale"]
    pca_mean = fit["pca_mean"]
    components = fit["components"]
    Z = (X - mean) / np.where(scale == 0, 1.0, scale)
    return (Z - pca_mean) @ components.T


def fit_common_views(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pc = project_common_pca(X)
    embed = pc[:, : min(30, pc.shape[1])]
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.18,
        metric="euclidean",
        init="spectral",
        random_state=42,
        n_components=2,
    )
    umap_xy = reducer.fit_transform(embed)
    perplexity = min(35.0, max(5.0, (len(embed) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=42,
        method="barnes_hut",
    )
    tsne_xy = tsne.fit_transform(embed)
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(embed[train_mask, : min(30, embed.shape[1])])
    distances = nearest.kneighbors(embed[:, : min(30, embed.shape[1])], return_distance=True)[0][:, 0]
    return pc, umap_xy, tsne_xy, distances


def raw_canvas_coordinates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw_x = df["x_umap"].to_numpy(float)
    raw_y = df["y_umap"].to_numpy(float)
    xlo, xhi = np.percentile(raw_x, [2, 98])
    ylo, yhi = np.percentile(raw_y, [2, 98])
    x = 8.8 * (raw_x - xlo) / max(xhi - xlo, 1e-12) - 8.8
    y = 8.8 * (raw_y - ylo) / max(yhi - ylo, 1e-12) - 8.8
    return x, y


def balanced_task_coordinates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x, y = raw_canvas_coordinates(df)
    result_x = x.copy()
    result_y = y.copy()
    reference_mask = df["task_group"].eq("L0 DFT Reference").to_numpy()
    for name, anchor in TASK_ANCHORS.items():
        mask = df["task_group"].eq(name).to_numpy()
        if mask.sum() == 0:
            continue
        local_x = x[mask] - np.median(x[mask])
        local_y = y[mask] - np.median(y[mask])
        scale = max(np.percentile(np.abs(local_x), 90), np.percentile(np.abs(local_y), 90), 1e-6)
        result_x[mask] = anchor[0] + TASK_SCALES[name] * local_x / scale
        result_y[mask] = anchor[1] + TASK_SCALES[name] * local_y / scale
    result_x[reference_mask] = x[reference_mask]
    result_y[reference_mask] = y[reference_mask]
    return result_x, result_y


def size_from_distortion(values: pd.Series, foreground: bool) -> np.ndarray:
    distortion = values.to_numpy(float)
    finite = np.isfinite(distortion)
    reference = np.nanpercentile(distortion[finite], 98) if finite.any() else 1.0
    reference = max(float(reference), 1e-6)
    normalized = np.clip(np.nan_to_num(distortion, nan=0.0) / reference, 0.0, 1.4)
    base = 5.0 if not foreground else 18.0
    span = 58.0 if not foreground else 330.0
    return base + span * np.power(normalized, 0.68)


def add_hulls(ax, df: pd.DataFrame) -> None:
    for name, group in df.groupby("task_group", sort=False):
        if name not in TASK_COLORS or len(group) < 4:
            continue
        xy = group[["display_x", "display_y"]].to_numpy(float)
        try:
            hull = ConvexHull(xy)
        except Exception:
            continue
        alpha = 0.04 if name == "L0 DFT Reference" else 0.08
        ax.add_patch(
            Polygon(
                xy[hull.vertices],
                closed=True,
                facecolor=TASK_COLORS[name],
                edgecolor=TASK_COLORS[name],
                linewidth=1.15,
                alpha=alpha,
                zorder=0,
            )
        )


def add_path_lines(ax, df: pd.DataFrame) -> None:
    path_rows = df[df["path_key"].notna() & df["path_key"].astype(str).ne("nan")]
    for _, group in path_rows.groupby("path_key", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values(["frame", "sample_frame"], na_position="last")
        name = str(group["task_group"].iloc[0])
        protocol = str(group["protocol_style"].iloc[0])
        linestyle = "--" if protocol == "self-relaxed" else "-"
        ax.plot(
            group["display_x"],
            group["display_y"],
            color=TASK_COLORS.get(name, "#777777"),
            linewidth=0.65,
            alpha=0.25,
            linestyle=linestyle,
            zorder=1,
        )


def make_figure() -> None:
    configure_style()
    base = pd.read_csv(DERIVED / "coordination_distortion_points.csv")
    base_vectors = np.load(DERIVED / "structure_universe_local_soap_vectors.npz")["vectors"]
    if len(base) != len(base_vectors):
        raise RuntimeError(f"base metadata/vector mismatch: {len(base)} != {len(base_vectors)}")

    reference_shell = load_reference_shell()
    md, md_vectors = load_md_rows(reference_shell)
    df = pd.concat([base, md], ignore_index=True, sort=False)
    X = np.vstack([base_vectors, md_vectors])
    df["task_group"] = df.apply(task_group, axis=1)
    df["sublevel"] = df.apply(sublevel, axis=1)
    df["protocol_style"] = df.apply(protocol_style, axis=1)
    train_mask = df["source_kind"].eq("grouped_reference") & df["split"].eq("train")
    pc, umap_xy, tsne_xy, distances = fit_common_views(X, train_mask.to_numpy())
    df["x_pca"] = pc[:, 0]
    df["y_pca"] = pc[:, 1]
    df["x_umap"] = umap_xy[:, 0]
    df["y_umap"] = umap_xy[:, 1]
    df["x_tsne"] = tsne_xy[:, 0]
    df["y_tsne"] = tsne_xy[:, 1]
    df["distance_to_training"] = distances
    non_train = distances[~train_mask.to_numpy()]
    q50, q90 = np.quantile(non_train, [0.5, 0.9]) if len(non_train) else (1.0, 2.0)
    df["distance_bin"] = np.where(
        distances <= q50,
        "near_training",
        np.where(distances <= q90, "intermediate", "far_from_training"),
    )
    df.loc[train_mask, "distance_bin"] = "near_training"
    df["task_label"] = df["task_group"].where(df["task_group"].isin(TASK_COLORS), "Other")
    df["display_x"], df["display_y"] = balanced_task_coordinates(df)
    df["display_size"] = size_from_distortion(df["rms_delta_A"], foreground=True)
    df["display_size_background"] = size_from_distortion(df["rms_delta_A"], foreground=False)
    df.to_csv(DERIVED / "comprehensive_task_map_points.csv", index=False)
    np.savez_compressed(
        DERIVED / "comprehensive_task_map_vectors.npz",
        vectors=X,
        pca=pc,
        umap=umap_xy,
        tsne=tsne_xy,
    )

    fit_manifest = {
        "descriptor": SOAP_KW,
        "coordination_cutoff_A": CUTOFF_A,
        "pca_fit": "existing grouped_reference/train Cr-local SOAP fit",
        "nonlinear_input": "first 30 whitened PCA coordinates",
        "umap": {"n_neighbors": 30, "min_dist": 0.18, "metric": "euclidean", "random_state": 42},
        "tsne": {"perplexity": min(35.0, max(5.0, (len(df) - 1) / 3.0)), "random_state": 42},
        "counts": {name: int((df["task_group"] == name).sum()) for name in TASK_ORDER},
        "sublevel_counts": {name: int((df["sublevel"] == name).sum()) for name in SUBLEVEL_ORDER},
        "task_labels": TASK_ORDER,
        "sublevel_labels": SUBLEVEL_ORDER,
        "visual_encoding": {
            "parent_hue": "physical configuration regime",
            "saturation_lightness": "DFT/MLIP or AIMD/MD sublevel",
            "path_line": "solid=fixed/sliding, dashed=self-relaxed",
            "circle_area": "Cr-shell RMS bond-length distortion",
        },
        "md_sample_manifest": str(MD_MANIFEST),
        "md_shell_aggregation": "mean over Cr centres in each 2050-atom MD frame",
        "selected_map": {
            "descriptor": "common Cr-local SOAP, projected through the shared PCA fit",
            "embedding": "UMAP on the first 30 common PCA coordinates",
            "geometry_overlay": "marker area encodes measured Cr-shell RMS bond-length distortion",
            "display_calibration": "task-conditioned spacing with within-task UMAP geometry preserved",
        },
        "display_layout": "feature-guided spaced display; four physical-regime medians separated while within-regime UMAP geometry is preserved",
        "display_distance_warning": "the spaced display is not a new metric distance; use the raw feature-map candidates for quantitative proximity claims",
        "axis_labels": {"x": "Feature-map axis 1", "y": "Feature-map axis 2"},
    }
    (DERIVED / "comprehensive_task_map_fit.json").write_text(json.dumps(fit_manifest, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(15.2, 10.8), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.80)
    add_hulls(ax, df)
    add_path_lines(ax, df)

    reference = df[df["task_group"].eq("L0 DFT Reference")]
    ax.scatter(
        reference["display_x"],
        reference["display_y"],
        s=reference["display_size_background"],
        color=SUBLEVEL_COLORS["Static DFT"],
        alpha=0.16,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    foreground = df[~df["task_group"].eq("L0 DFT Reference")]
    for (name, level), group in foreground.groupby(["task_group", "sublevel"], sort=False):
        edge = "#222222" if level.startswith("DFT") else "white"
        ax.scatter(
            group["display_x"],
            group["display_y"],
            s=group["display_size"],
            color=SUBLEVEL_COLORS.get(level, TASK_COLORS.get(name, "#777777")),
            alpha=0.80,
            edgecolors=edge,
            linewidths=0.50,
            rasterized=True,
            zorder=3,
        )

    # Place every task label at its own displayed cluster median, including
    # L0. This avoids the previous fixed offset that left L0 outside its hull.
    for name in TASK_ORDER:
        group = df[df["task_group"].eq(name)]
        if group.empty:
            continue
        x0 = float(group["display_x"].median())
        y0 = float(group["display_y"].median())
        ax.text(
            x0,
            y0,
            name,
            color=TASK_COLORS[name],
            fontsize=12.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
        )

    ax.set_xlim(-10.2, 10.2)
    ax.set_ylim(-10.2, 10.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(2.2)
    ax.set_title("MigrationBench configuration universe", fontsize=23, fontweight="bold", pad=12)
    ax.set_xlabel("Feature-map axis 1", labelpad=11, fontsize=15, fontweight="bold")
    ax.set_ylabel("Feature-map axis 2", labelpad=11, fontsize=15, fontweight="bold")

    task_handles = [
        Patch(facecolor=TASK_COLORS[name], edgecolor="white", linewidth=0.5, alpha=0.90, label=name)
        for name in TASK_ORDER
    ]
    observed_sublevels = [name for name in SUBLEVEL_ORDER if (df["sublevel"] == name).any()]
    sublevel_handles = [
        Patch(
            facecolor=SUBLEVEL_COLORS[name],
            edgecolor="#222222" if name.startswith("DFT") else "white",
            linewidth=0.7,
            alpha=0.92,
            label=name,
        )
        for name in observed_sublevels
    ]
    protocol_handles = [
        Line2D([0], [0], color="#555555", linewidth=1.1, linestyle="-", label="fixed / sliding"),
        Line2D([0], [0], color="#555555", linewidth=1.1, linestyle="--", label="self-relaxed"),
    ]
    distortion_values = [0.1, 0.3, 0.6]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#7A7A7A",
            markeredgecolor="white",
            markersize=float(np.sqrt(18.0 + 330.0 * (v / 0.8) ** 0.68)),
            label=f"{v:.1f} Å",
        )
        for v in distortion_values
    ]
    unified_handles = task_handles + sublevel_handles + protocol_handles + size_handles
    fig.legend(
        handles=unified_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        title="Task / data layer / path protocol / Cr-shell RMS Δr",
        fontsize=9.0,
        prop={"weight": "bold", "size": 9.0},
        title_fontsize=10.5,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="#222222",
        facecolor="white",
        borderpad=0.65,
        handlelength=1.15,
        handletextpad=0.35,
        columnspacing=1.0,
        labelspacing=0.42,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            FIGURES / f"migrationbench_configuration_universe_comprehensive.{ext}",
            dpi=420 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(json.dumps(fit_manifest["counts"], indent=2))


if __name__ == "__main__":
    make_figure()
