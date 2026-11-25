#!/usr/bin/env python3
"""Quantify Cr coordination-shell distortion for the universe-map structures.

Every structure is evaluated with the same Cr-centred distance-qualified shell:
neighbours within 3.75 Å, minimum-image distances when a valid periodic cell is
present. The reference is a Cr-doped 81-atom host endpoint, because the
undoped pristine Sb2Te3 structure has no focal Cr and therefore cannot define a
Cr bonding environment.

The script deliberately keeps coordination geometry separate from energy,
NEB convergence, and model ranking. The paired shell deltas are structural
descriptors for figure annotation and renderer selection, not a chemical bond
order inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from matplotlib.colors import Normalize


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
POINTS_PATH = DERIVED / "structure_universe_points.csv"
REPS_PATH = DERIVED / "cluster_representatives.csv"
REFERENCE_PATH = Path("/Users/alina/Project/migrationbench-paper-repro/experiments/foundation_model_md/data/raw/sb2te3_cr_81atom.xyz")
CUTOFF_A = 3.75

MODEL_COLORS = {
    "Foundation": "#E69F00",
    "Scratch-600K": "#0072B2",
    "Scratch-MultiT": "#7B2CBF",
    "FT-600K": "#009E73",
    "FT-MultiT": "#CC79A7",
    "DFT": "#202020",
    "Reference": "#8DA9BF",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_one(path: Path, frame: int = 0) -> Atoms:
    if path.suffix.lower() == ".axsf":
        return read_axsf(path)[frame]
    obj = read(str(path), index=int(frame))
    if isinstance(obj, list):
        return obj[0]
    return obj


def read_axsf(path: Path) -> list[Atoms]:
    """Read the small AXSF trajectory used by the recovered DFT path."""
    lines = path.read_text(encoding="utf-8").splitlines()
    cell = None
    for i, line in enumerate(lines):
        if line.strip().upper() == "PRIMVEC" and i + 3 < len(lines):
            cell = np.asarray([[float(x) for x in lines[i + j].split()[:3]] for j in range(1, 4)], dtype=float)
            break
    frames: list[Atoms] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().upper().startswith("PRIMCOORD"):
            i += 1
            continue
        if i + 1 >= len(lines):
            break
        n = int(lines[i + 1].split()[0])
        symbols = []
        positions = []
        for row in lines[i + 2 : i + 2 + n]:
            parts = row.split()
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])
        frames.append(Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True if cell is not None else False))
        i += n + 2
    if not frames:
        raise ValueError(f"No PRIMCOORD frames found in {path}")
    return frames


def valid_mic(atoms: Atoms) -> bool:
    try:
        return bool(np.any(atoms.pbc) and atoms.cell.rank >= 2)
    except Exception:
        return False


def focal_cr_index(atoms: Atoms) -> int:
    indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Cr"]
    if not indices:
        raise ValueError("No Cr atom found")
    return indices[0]


def shell_data(atoms: Atoms, cutoff: float = CUTOFF_A) -> dict:
    focus = focal_cr_index(atoms)
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

    # Cr-centred angular disorder is reported only for the selected shell.
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
        "vectors_A": [[float(v) for v in row] for row in shell_vectors],
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


def process_label(row: pd.Series) -> str:
    kind = str(row.get("source_kind", ""))
    family = str(row.get("model_family", ""))
    if kind == "grouped_reference":
        return "training/reference"
    if kind == "dft_cr_fixed_neb":
        return "DFT fixed migration path"
    if kind == "dft_interlayer_neb":
        return "DFT interlayer path"
    if kind == "mlip_interlayer_self":
        return f"{family} interlayer self"
    if kind == "mlip_self_neb":
        return f"{family} self-NEB"
    return kind


def load_point_metrics(points: pd.DataFrame, reference_shell: dict) -> pd.DataFrame:
    records = []
    for source_file, group in points.groupby("source_file", sort=False):
        path = Path(str(source_file))
        frames = sorted(set(int(x) for x in group["frame"].tolist()))
        try:
            if path.suffix.lower() == ".axsf":
                loaded = read_axsf(path)
            else:
                loaded = read(str(path), index=":")
                if not isinstance(loaded, list):
                    loaded = [loaded]
            structures = {frame: loaded[frame] for frame in frames}
        except Exception as exc:
            print(f"[warn] cannot read {path}: {exc}")
            continue
        for _, row in group.iterrows():
            frame = int(row["frame"])
            atoms = structures.get(frame)
            if atoms is None:
                continue
            try:
                shell = shell_data(atoms)
                metrics = metrics_from_shell(shell, reference_shell)
            except Exception as exc:
                print(f"[warn] metrics failed {path} frame {frame}: {exc}")
                continue
            record = row.to_dict()
            record.update(metrics)
            record["process"] = process_label(row)
            record["source_path_exists"] = path.exists()
            records.append(record)
    return pd.DataFrame(records)


def load_representative_metrics(reps: pd.DataFrame, reference_shell: dict) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    shells = {}
    for _, row in reps.iterrows():
        path = Path(str(row["render_input"]))
        atoms = read_one(path, 0)
        shell = shell_data(atoms)
        metric = metrics_from_shell(shell, reference_shell)
        record = row.to_dict()
        record.update(metric)
        record["process"] = process_label(row)
        record["source_path_exists"] = path.exists()
        rows.append(record)
        shells[str(row["cluster"])] = shell
    return pd.DataFrame(rows), shells


def save_tables(all_metrics: pd.DataFrame, reps_metrics: pd.DataFrame, reference_shell: dict, reference_metrics: dict) -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(DERIVED / "coordination_distortion_points.csv", index=False)
    reps_metrics.to_csv(DERIVED / "coordination_distortion.csv", index=False)
    payload = {
        "reference": {"name": "PERFECT_REFERENCE", "path": str(REFERENCE_PATH), "cutoff_A": CUTOFF_A, "metrics": reference_metrics, "shell": reference_shell},
    }
    for _, row in reps_metrics.iterrows():
        payload[str(row["cluster"])] = {"metrics": {key: (None if pd.isna(row.get(key)) else row.get(key)) for key in metrics_columns()}, "source": str(row["render_input"])}
    (DERIVED / "coordination_shells.json").write_text(json.dumps(payload, indent=2) + "\n")


def metrics_columns() -> list[str]:
    return [
        "focus_index", "coordination_number", "n_Sb", "n_Te", "min_bond_A", "max_bond_A",
        "mean_bond_A", "bond_length_std_A", "bond_length_range_A", "mean_abs_delta_A", "rms_delta_A",
        "coordination_number_delta", "species_count_delta", "paired_bonds", "mean_neighbor_angle_deg", "neighbor_angle_std_deg",
    ]


def cluster_color(row: pd.Series) -> str:
    value = str(row.get("cluster_color", ""))
    if value and value.lower() != "nan":
        return value
    return MODEL_COLORS.get(str(row.get("model_family", "")), "#666666")


def add_spines(ax) -> None:
    ax.tick_params(direction="out", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def make_figure(points: pd.DataFrame, reps: pd.DataFrame, reference_shell: dict, reference_metrics: dict) -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    ref_distances = np.asarray(reference_shell["distances_A"], dtype=float)

    fig = plt.figure(figsize=(15.0, 10.2), dpi=180, facecolor="white")
    grid = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.30)
    ax_shell = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])
    ax_umap = fig.add_subplot(grid[1, 0])
    ax_process = fig.add_subplot(grid[1, 1])

    # (a) Ranked shell distances make individual bond compression/elongation visible.
    ranks = np.arange(1, len(ref_distances) + 1)
    ax_shell.plot(ranks, ref_distances, "o-", color="#202020", lw=2.2, ms=5, label="perfect-Cr-host reference", zorder=5)
    for _, row in reps.iterrows():
        distances = np.asarray(json.loads(row["_distances_json"]), dtype=float)
        x = np.arange(1, len(distances) + 1)
        ax_shell.plot(x, distances, "o-", color=cluster_color(row), lw=1.0, ms=3.2, alpha=0.92, label=str(row["cluster"]))
    ax_shell.axhline(CUTOFF_A, color="#666666", ls="--", lw=0.8, label="coordination cutoff")
    ax_shell.set_xlabel("ranked Cr–neighbour index")
    ax_shell.set_ylabel("Cr–neighbour distance (Å)")
    ax_shell.set_title("(a) Coordination-shell fingerprint", loc="left", fontweight="bold")
    ax_shell.legend(loc="upper left", fontsize=6.4, ncol=2, handlelength=1.6, columnspacing=0.7)
    add_spines(ax_shell)

    # (b) Representative distortion relative to the same reference.
    ordered = reps.sort_values(["rms_delta_A", "cluster"], na_position="last").reset_index(drop=True)
    y = np.arange(len(ordered))
    values = ordered["mean_abs_delta_A"].to_numpy(float)
    ax_delta.barh(y, values, color=[cluster_color(row) for _, row in ordered.iterrows()], alpha=0.86, edgecolor="#333333", linewidth=0.35)
    for yi, (_, row) in enumerate(ordered.iterrows()):
        ax_delta.text(max(values[yi], 0) + 0.012, yi, f"CN={int(row['coordination_number'])}, ΔCN={int(row['coordination_number_delta']):+d}", va="center", fontsize=6.5)
    ax_delta.set_yticks(y, ordered["cluster"].tolist())
    ax_delta.set_xlabel("mean paired |Δr| from reference (Å)")
    ax_delta.set_title("(b) Representative bonding distortion", loc="left", fontweight="bold")
    ax_delta.invert_yaxis()
    add_spines(ax_delta)

    # (c) Put the shell distortion back onto the common UMAP universe.
    finite = points[np.isfinite(points["rms_delta_A"])]
    sc = ax_umap.scatter(
        finite["x_umap"], finite["y_umap"], c=finite["rms_delta_A"], cmap="magma", norm=Normalize(vmin=0.0, vmax=max(0.35, float(finite["rms_delta_A"].quantile(0.98)))),
        s=6, alpha=0.52, linewidths=0, rasterized=True,
    )
    for _, row in reps.iterrows():
        point = points[(points["source_file"].eq(row["source_file"])) & (points["frame"].eq(int(row["source_frame"])))].head(1)
        if point.empty:
            continue
        ax_umap.scatter(point["x_umap"], point["y_umap"], marker="*", s=80, facecolors="white", edgecolors=cluster_color(row), linewidths=1.0, zorder=6)
        ax_umap.text(float(point["x_umap"].iloc[0]), float(point["y_umap"].iloc[0]), str(row["cluster"]), fontsize=6, ha="center", va="center", zorder=7)
    ax_umap.set_xlabel("UMAP-1")
    ax_umap.set_ylabel("UMAP-2")
    ax_umap.set_title("(c) UMAP coloured by Cr-shell RMS distortion", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax_umap, fraction=0.046, pad=0.04)
    cb.set_label("paired shell RMS Δr (Å)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    add_spines(ax_umap)

    # (d) Process/model separation is shown as distributions, not as a rank.
    order = [
        "training/reference", "DFT fixed migration path", "DFT interlayer path",
        "Foundation self-NEB", "FT-600K self-NEB", "Scratch-MultiT self-NEB", "FT-MultiT self-NEB",
        "Foundation interlayer self",
    ]
    groups = [(name, points[points["process"].eq(name)]["rms_delta_A"].dropna().to_numpy(float)) for name in order]
    groups = [(name, values) for name, values in groups if len(values)]
    xpos = np.arange(len(groups))
    for i, (name, values) in enumerate(groups):
        color = MODEL_COLORS.get(name.split()[0], "#7A7A7A")
        if name == "training/reference":
            color = "#8DA9BF"
        jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.array([0.0])
        ax_process.scatter(np.full(len(values), i) + jitter, values, s=8, color=color, alpha=0.28, edgecolors="none", rasterized=True)
        ax_process.hlines(np.median(values), i - 0.24, i + 0.24, color="#222222", lw=1.3)
        ax_process.text(i, float(np.nanmax(values)) + 0.02, f"n={len(values)}", ha="center", va="bottom", fontsize=6)
    ax_process.set_xticks(xpos, [name.replace(" ", "\n") for name, _ in groups], rotation=0, fontsize=6.4)
    ax_process.set_ylabel("paired shell RMS Δr (Å)")
    ax_process.set_title("(d) Distortion by process/model", loc="left", fontweight="bold")
    add_spines(ax_process)

    fig.suptitle("Cr coordination-sphere distortion relative to a perfect-Cr-host reference", y=0.982, fontsize=16, fontweight="bold")
    fig.text(
        0.5, 0.012,
        "Same 3.75 Å distance-qualified shell for every structure; cylinders in Blender are coordination guides, not inferred covalent bond orders. "
        "Structural distortion is not an energy-barrier or model-ranking metric.",
        ha="center", va="bottom", fontsize=7.5, color="#444444",
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"coordination_distortion.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = pd.read_csv(POINTS_PATH)
    reps = pd.read_csv(REPS_PATH)
    reference_atoms = read_one(REFERENCE_PATH, 0)
    reference_shell = shell_data(reference_atoms)
    reference_metrics = metrics_from_shell(reference_shell, reference_shell)
    all_metrics = load_point_metrics(points, reference_shell)
    reps_metrics, rep_shells = load_representative_metrics(reps, reference_shell)
    reps_metrics["_distances_json"] = [json.dumps(rep_shells[str(cluster)]["distances_A"]) for cluster in reps_metrics["cluster"]]
    save_tables(all_metrics, reps_metrics, reference_shell, reference_metrics)
    make_figure(all_metrics, reps_metrics, reference_shell, reference_metrics)
    print(json.dumps({"reference": reference_metrics, "point_rows": len(all_metrics), "representatives": len(reps_metrics), "cutoff_A": CUTOFF_A}, indent=2))


if __name__ == "__main__":
    main()
