#!/usr/bin/env python3
"""Audit original plus exact from-scratch replicas without changing MD outputs.

This is the all-replicate companion to ``diagnose_from_scratch.py``.  It keeps
the original two-root CLI compatible while accepting multiple reproduction
roots.  The output is deliberately additive: full metrics, selected-frame
features, a replica ledger, snapshot provenance, optional force disagreement,
and an auditable diagnostic manifest.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import iread, read
from ase.neighborlist import neighbor_list
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

THRESHOLDS = {
    "temperature_K": 5000.0,
    "max_force_eV_A": 100.0,
    "min_distance_A": 0.5,
}
SEEDS = (123, 234, 345)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path, default: Optional[Dict] = None) -> Dict:
    if not path.exists():
        return {} if default is None else default
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except Exception:
        return {} if default is None else default


def first_bad(metrics: pd.DataFrame) -> Tuple[Optional[int], Optional[str]]:
    if metrics.empty:
        return None, None
    finite = np.isfinite(metrics[["temperature_K", "max_force_eV_A", "min_distance_A"]]).all(axis=1)
    bad = (
        (metrics["temperature_K"] > THRESHOLDS["temperature_K"])
        | (metrics["max_force_eV_A"] > THRESHOLDS["max_force_eV_A"])
        | (metrics["min_distance_A"] < THRESHOLDS["min_distance_A"])
        | ~finite
    )
    indices = np.flatnonzero(bad.to_numpy())
    if len(indices) == 0:
        return None, None
    index = int(indices[0])
    row = metrics.iloc[index]
    tests = {
        "temperature": float(row["temperature_K"]) > THRESHOLDS["temperature_K"],
        "force": float(row["max_force_eV_A"]) > THRESHOLDS["max_force_eV_A"],
        "minimum_distance": float(row["min_distance_A"]) < THRESHOLDS["min_distance_A"],
    }
    trigger = "+".join(name for name, value in tests.items() if value)
    return index, trigger or "nonfinite"


def evenly_spaced(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    if len(values) <= n:
        return values
    indices = np.linspace(0, len(values) - 1, n).round().astype(int)
    return values[np.unique(indices)]


def selected_steps(metrics: pd.DataFrame, onset_ps: float) -> Dict[str, np.ndarray]:
    time = metrics["time_ps"].to_numpy(float)
    steps = metrics["step"].to_numpy(int)
    windows = {
        "stable_early": (5.0, 15.0),
        "stable_pre": (max(0.0, onset_ps - 30.0), max(0.0, onset_ps - 15.0)),
        "precursor": (max(0.0, onset_ps - 5.0), onset_ps),
        "threshold": (onset_ps, onset_ps),
    }
    selected = {}
    for name, (lo, hi) in windows.items():
        selected[name] = evenly_spaced(steps[(time >= lo) & (time <= hi)], 24 if name != "threshold" else 1)
    if not len(selected["stable_pre"]):
        selected["stable_pre"] = selected["stable_early"]
    return selected


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if len(values) else float("nan")


def _coordination(atoms: Atoms, central: str, neighbor: str, cutoff: float) -> Tuple[float, float]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j = neighbor_list("ij", atoms, cutoff=cutoff)
    mask = (symbols[i] == central) & (symbols[j] == neighbor)
    if not np.any(mask):
        return float("nan"), float("nan")
    counts = np.bincount(i[mask], minlength=len(atoms))
    selected = counts[symbols == central]
    return float(np.mean(selected)), float(np.std(selected))


def _pair_distance(atoms: Atoms, a: str, b: str, cutoff: float = 6.0) -> Tuple[float, str]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, distances = neighbor_list("ijd", atoms, cutoff=cutoff)
    mask = ((symbols[i] == a) & (symbols[j] == b)) | ((symbols[i] == b) & (symbols[j] == a))
    if not np.any(mask):
        return float("nan"), ""
    index = int(np.flatnonzero(mask)[np.argmin(distances[mask])])
    return float(distances[index]), f"{symbols[i[index]]}-{symbols[j[index]]}"


def _minimum_pair(atoms: Atoms, cutoff: float = 6.0) -> Tuple[float, str]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, distances = neighbor_list("ijd", atoms, cutoff=cutoff)
    if not len(distances):
        return float("nan"), ""
    index = int(np.argmin(distances))
    return float(distances[index]), f"{symbols[i[index]]}-{symbols[j[index]]}"


def _local_angle_summary(atoms: Atoms, central: str = "Cr", neighbor: str = "Te", cutoff: float = 3.5) -> Tuple[float, float]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, shifts = neighbor_list("ijS", atoms, cutoff=cutoff)
    vectors = atoms.positions[j] + shifts @ atoms.cell.array - atoms.positions[i]
    values = []
    for center in np.flatnonzero(symbols == central):
        mask = (i == center) & (symbols[j] == neighbor)
        vec = vectors[mask]
        if len(vec) < 2:
            continue
        norms = np.linalg.norm(vec, axis=1)
        vec = vec / norms[:, None]
        cosine = np.clip(vec @ vec.T, -1.0, 1.0)
        values.extend(np.degrees(np.arccos(cosine[np.triu_indices(len(vec), 1)])).tolist())
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def _force_arrays(atoms: Atoms) -> np.ndarray:
    values = atoms.arrays.get("forces")
    if values is None:
        return np.full((len(atoms), 3), np.nan)
    return np.asarray(values, dtype=float)


def _minimum_image_displacement(atoms: Atoms, reference: Atoms) -> np.ndarray:
    try:
        frac = atoms.get_scaled_positions(wrap=False)
        ref_frac = reference.get_scaled_positions(wrap=False)
        delta = frac - ref_frac
        delta -= np.rint(delta)
        return delta @ atoms.cell.array
    except Exception:
        return np.full((len(atoms), 3), np.nan)


def _local_force_environment(atoms: Atoms, atom_index: int, cutoff: float = 3.2) -> Dict[str, float]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    if atom_index < 0 or atom_index >= len(atoms):
        return {"top_force_n_neighbors_3p2": float("nan")}
    i, j, distances = neighbor_list("ijd", atoms, cutoff=cutoff)
    mask = i == atom_index
    out = {"top_force_n_neighbors_3p2": float(np.sum(mask))}
    for symbol in ("Cr", "Sb", "Te"):
        out[f"top_force_{symbol}_neighbors_3p2"] = float(np.sum(mask & (symbols[j] == symbol)))
    out["top_force_min_neighbor_A"] = _safe_min(distances[mask])
    return out


def _bond_statistics(atoms: Atoms, left: str, right: str, cutoff: float = 4.0) -> Dict[str, float]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, distances = neighbor_list("ijd", atoms, cutoff=cutoff)
    mask = ((symbols[i] == left) & (symbols[j] == right)) | ((symbols[i] == right) & (symbols[j] == left))
    values = np.asarray(distances[mask], dtype=float)
    if not len(values):
        return {f"{left}{right}_bond_q10_A": float("nan"), f"{left}{right}_bond_median_A": float("nan"), f"{left}{right}_bond_q90_A": float("nan")}
    return {
        f"{left}{right}_bond_q10_A": float(np.quantile(values, 0.10)),
        f"{left}{right}_bond_median_A": float(np.quantile(values, 0.50)),
        f"{left}{right}_bond_q90_A": float(np.quantile(values, 0.90)),
    }


def structural_features(atoms: Atoms, reference: Optional[Atoms] = None) -> Dict[str, Union[float, str, bool]]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    forces = _force_arrays(atoms)
    force_norm = np.linalg.norm(forces, axis=1) if np.isfinite(forces).all() else np.full(len(atoms), np.nan)
    worst = int(np.nanargmax(force_norm)) if np.isfinite(force_norm).any() else -1
    min_dist, min_pair = _minimum_pair(atoms)
    momenta = np.asarray(atoms.get_momenta(), dtype=float) if atoms.has("momenta") else np.full((len(atoms), 3), np.nan)
    masses = atoms.get_masses()[:, None]
    kinetic_atom = 0.5 * np.sum(momenta * momenta / masses, axis=1) if np.isfinite(momenta).all() else np.full(len(atoms), np.nan)
    positive_kinetic = kinetic_atom[np.isfinite(kinetic_atom)]
    top_k = max(1, int(math.ceil(0.01 * len(atoms))))
    tail_fraction = float(np.sum(np.sort(positive_kinetic)[-top_k:]) / np.sum(positive_kinetic)) if len(positive_kinetic) and np.sum(positive_kinetic) > 0 else float("nan")
    velocity = momenta / masses if np.isfinite(momenta).all() else np.full((len(atoms), 3), np.nan)
    velocity_norm_A_ps = np.linalg.norm(velocity, axis=1) * 1000.0
    cr_mask = symbols == "Cr"
    out: Dict[str, Union[float, str, bool]] = {
        "n_Cr": float(np.sum(symbols == "Cr")),
        "n_Sb": float(np.sum(symbols == "Sb")),
        "n_Te": float(np.sum(symbols == "Te")),
        "volume_A3": float(atoms.get_volume()),
        "natoms_observed": len(atoms),
        "pbc_observed": "".join("1" if bool(x) else "0" for x in atoms.pbc),
        "natoms_ok": bool(len(atoms) == 2050),
        "pbc_ok": bool(tuple(bool(x) for x in atoms.pbc) == (True, True, True)),
        "finite_positions": bool(np.isfinite(atoms.positions).all()),
        "finite_momenta": bool(np.isfinite(momenta).all()),
        "finite_forces": bool(np.isfinite(forces).all()),
        "force_max_norm_eV_A": float(np.nanmax(force_norm)) if np.isfinite(force_norm).any() else float("nan"),
        "force_rms_norm_eV_A": float(np.sqrt(np.nanmean(force_norm * force_norm))) if np.isfinite(force_norm).any() else float("nan"),
        "force_q95_norm_eV_A": float(np.nanquantile(force_norm, 0.95)) if np.isfinite(force_norm).any() else float("nan"),
        "force_q99_norm_eV_A": float(np.nanquantile(force_norm, 0.99)) if np.isfinite(force_norm).any() else float("nan"),
        "worst_force_norm_eV_A": float(force_norm[worst]) if worst >= 0 else float("nan"),
        "worst_force_atom_index": worst,
        "worst_force_atom_species": symbols[worst] if worst >= 0 else "",
        "min_pair_distance_computed_A": min_dist,
        "min_pair_species": min_pair,
        "kinetic_energy_eV": float(np.nansum(kinetic_atom)) if np.isfinite(kinetic_atom).any() else float("nan"),
        "kinetic_top_1pct_fraction": tail_fraction,
        "max_atomic_kinetic_energy_eV": float(np.nanmax(kinetic_atom)) if np.isfinite(kinetic_atom).any() else float("nan"),
        "Cr_velocity_median_A_ps": float(np.nanmedian(velocity_norm_A_ps[cr_mask])) if np.isfinite(velocity_norm_A_ps[cr_mask]).any() else float("nan"),
        "Cr_velocity_max_A_ps": float(np.nanmax(velocity_norm_A_ps[cr_mask])) if np.isfinite(velocity_norm_A_ps[cr_mask]).any() else float("nan"),
        "Cr_kinetic_median_eV": float(np.nanmedian(kinetic_atom[cr_mask])) if np.isfinite(kinetic_atom[cr_mask]).any() else float("nan"),
        "Cr_kinetic_max_eV": float(np.nanmax(kinetic_atom[cr_mask])) if np.isfinite(kinetic_atom[cr_mask]).any() else float("nan"),
    }
    if reference is not None and len(reference) == len(atoms):
        displacement = _minimum_image_displacement(atoms, reference)
        cr = symbols == "Cr"
        out["Cr_displacement_median_A"] = float(np.nanmedian(np.linalg.norm(displacement[cr], axis=1)))
        out["Cr_displacement_max_A"] = float(np.nanmax(np.linalg.norm(displacement[cr], axis=1)))
    for cutoff in (2.8, 3.2, 3.6, 4.0):
        tag = str(cutoff).replace(".", "p")
        for neighbor in ("Te", "Sb", "Cr"):
            mean, std = _coordination(atoms, "Cr", neighbor, cutoff)
            out[f"Cr_{neighbor}_coord_{tag}_mean"] = mean
            out[f"Cr_{neighbor}_coord_{tag}_std"] = std
    for left, right in (("Cr", "Te"), ("Cr", "Sb"), ("Cr", "Cr"), ("Sb", "Te")):
        distance, pair = _pair_distance(atoms, left, right)
        out[f"min_{left}{right}_A"] = distance
        out[f"min_{left}{right}_pair"] = pair
        out.update(_bond_statistics(atoms, left, right))
    angle_mean, angle_std = _local_angle_summary(atoms)
    out["CrTe_angle_mean_deg"] = angle_mean
    out["CrTe_angle_std_deg"] = angle_std
    out.update(_local_force_environment(atoms, worst))
    return out


def scalar_features(atoms: Atoms, row: Union[Dict, pd.Series], reference: Optional[Atoms] = None) -> Dict:
    out = {
        "step": int(row["step"]),
        "time_ps": float(row["time_ps"]),
        "potential_energy_eV": float(row["potential_energy_eV"]),
        "temperature_K": float(row["temperature_K"]),
        "max_force_eV_A": float(row["max_force_eV_A"]),
        "rms_force_eV_A": float(row["rms_force_eV_A"]),
        "min_distance_A": float(row["min_distance_A"]),
        "total_energy_eV": float(row["potential_energy_eV"]) + float(structural_features(atoms).get("kinetic_energy_eV", float("nan"))),
    }
    return {**out, **structural_features(atoms, reference)}


def load_soap():
    from dscribe.descriptors import SOAP

    return SOAP(species=["Cr", "Sb", "Te"], periodic=True, r_cut=5.0, n_max=4, l_max=4, average="inner", sigma=0.5)


def sample_extxyz(path: Path, limit: int) -> List[Atoms]:
    frames = list(iread(path, format="extxyz"))
    if len(frames) <= limit:
        return frames
    indices = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[i] for i in np.unique(indices)]


def reference_soap(data_root: Path, neb_root: Path, soap, limit: int) -> Tuple[np.ndarray, List[Dict]]:
    vectors: List[np.ndarray] = []
    meta: List[Dict] = []
    for label in ("train", "valid", "validation", "test"):
        candidates = [data_root / f"{label}.xyz", data_root / f"{label}.extxyz"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None or label == "validation" and (data_root / "valid.xyz").exists():
            continue
        canonical = "valid" if label == "validation" else label
        for index, atoms in enumerate(sample_extxyz(path, limit)):
            vectors.append(np.asarray(soap.create(atoms), dtype=float))
            meta.append({"reference": canonical, "source": str(path), "index": index})
    for path in sorted(neb_root.glob("**/neb_soap_features.csv")):
        try:
            matrix = np.loadtxt(path, delimiter=",")
            matrix = np.atleast_2d(matrix)
        except Exception:
            continue
        if matrix.shape[1] != 390:
            continue
        for index, vector in enumerate(matrix):
            vectors.append(np.asarray(vector, dtype=float))
            meta.append({"reference": "neb", "source": str(path), "index": index})
    if not vectors:
        raise RuntimeError(f"no descriptor references found below {data_root} and {neb_root}")
    return np.asarray(vectors), meta


def fit_distribution(reference: np.ndarray, meta: List[Dict]) -> Tuple[np.ndarray, np.ndarray, PCA, Dict]:
    labels = np.asarray([row["reference"] for row in meta])
    train = reference[labels == "train"]
    if len(train) < 3:
        raise RuntimeError("not enough training reference descriptors for distribution fit")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    train_z = (train - mean) / scale
    pca = PCA(n_components=min(20, train_z.shape[0] - 1, train_z.shape[1]), whiten=True, random_state=0)
    train_pca = pca.fit_transform(train_z)
    k = min(6, len(train_pca))
    nn_train = NearestNeighbors(n_neighbors=k).fit(train_pca)
    distances, _ = nn_train.kneighbors(train_pca)
    baseline = distances[:, 1:].mean(axis=1) if k > 1 else distances[:, 0]
    return mean, scale, pca, {
        "train_knn_q95": float(np.quantile(baseline, 0.95)),
        "train_knn_q99": float(np.quantile(baseline, 0.99)),
        "pca_components": int(pca.n_components_),
        "n_train": len(train),
        "n_reference": len(reference),
    }


def add_distances(rows: List[Dict], vectors: np.ndarray, reference: np.ndarray, meta: List[Dict], mean: np.ndarray, scale: np.ndarray, pca: PCA, gates: Dict) -> None:
    labels = np.asarray([row["reference"] for row in meta])
    ref_pca = pca.transform((reference - mean) / scale)
    md_pca = pca.transform((vectors - mean) / scale)
    by_label = {label: ref_pca[labels == label] for label in sorted(set(labels))}
    for row, point in zip(rows, md_pca):
        for label, cloud in by_label.items():
            row[f"distance_to_{label}"] = float(np.min(np.linalg.norm(cloud - point[None, :], axis=1))) if len(cloud) else float("nan")
        row["pca1"] = float(point[0])
        row["pca2"] = float(point[1]) if len(point) > 1 else 0.0
        row["outside_train_q95"] = bool(row.get("distance_to_train", np.nan) > gates["train_knn_q95"])
        row["outside_train_q99"] = bool(row.get("distance_to_train", np.nan) > gates["train_knn_q99"])


def iter_run_frames(run: Path, chosen_steps: Set[int]) -> Iterable[Tuple[int, Atoms]]:
    trajectory = run / "trajectory.extxyz"
    if not trajectory.exists():
        return
    for atoms in iread(trajectory, format="extxyz"):
        step = int(atoms.info.get("step", -1))
        if step in chosen_steps:
            yield step, atoms


def _run_candidates(root: Path, family: str, seed: int, md_seed: int) -> List[Path]:
    candidates = []
    for run in sorted((root / "runs").glob("*")):
        manifest = json_read(run / "run_manifest.json")
        if manifest.get("family") == family and int(manifest.get("training_seed", -1)) == seed and int(manifest.get("md_seed", -1)) == md_seed:
            candidates.append(run)
    return sorted(candidates, key=lambda path: (0 if path.name.endswith("__r00") else 1, path.name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--reproduce-root", type=Path, action="append", default=[])
    parser.add_argument("--reproduce-label", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--neb-root", type=Path, required=True)
    parser.add_argument("--frames-per-reference", type=int, default=250)
    parser.add_argument("--force-disagreement", action="store_true")
    parser.add_argument("--prepare-dft-dir", type=Path)
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    labels = args.reproduce_label or [f"reproduce-{index + 1}" for index in range(len(args.reproduce_root))]
    if len(labels) != len(args.reproduce_root):
        raise SystemExit("--reproduce-label must be supplied once per --reproduce-root")
    roots = [("original", args.original_root)] + list(zip(labels, args.reproduce_root))
    soap = load_soap()
    reference, ref_meta = reference_soap(args.data_root, args.neb_root, soap, args.frames_per_reference)
    mean, scale, pca, gates = fit_distribution(reference, ref_meta)

    run_specs = []
    ledger_rows = []
    initial_state_rows = []
    input_cache: Dict[str, Atoms] = {}
    for source_name, root in roots:
        for family, seed in (("ablation_lmax_2", 123), ("ablation_lmax_2", 234), ("baseline", 345)):
            candidates = _run_candidates(root, family, seed, 20261005)
            run = candidates[0] if candidates else None
            manifest = json_read(run / "run_manifest.json") if run else {}
            status = json_read(run / "status.json") if run else {}
            metrics_path = run / "metrics.csv" if run else Path()
            trajectory_path = run / "trajectory.extxyz" if run else Path()
            metrics = pd.read_csv(metrics_path) if run and metrics_path.exists() else pd.DataFrame()
            onset_index, trigger = first_bad(metrics)
            onset_step = int(metrics.iloc[onset_index]["step"]) if onset_index is not None else (int(metrics.iloc[-1]["step"]) if not metrics.empty else -1)
            onset_ps = onset_step * float(manifest.get("timestep_fs", 1.0)) * 1.0e-3 if onset_step >= 0 else float("nan")
            terminal = status.get("status") in {"complete", "failed", "interrupted"} and bool(metrics_path.exists())
            if status.get("status") == "failed" and status.get("stop_reason") == "instability_threshold":
                acceptance = "unstable_threshold"
            elif status.get("status") == "complete" and int(status.get("last_step", -1)) >= int(status.get("target_step", 0)):
                acceptance = "complete_to_target"
            elif run:
                acceptance = "incomplete_or_running"
            else:
                acceptance = "missing"
            ledger_rows.append({
                "source": source_name,
                "root": str(root),
                "family": family,
                "training_seed": seed,
                "run": run.name if run else "",
                "run_path": str(run) if run else "",
                "status": status.get("status", "missing"),
                "stop_reason": status.get("stop_reason", ""),
                "last_step": status.get("last_step", ""),
                "target_step": status.get("target_step", ""),
                "terminal": terminal,
                "acceptance": acceptance,
                "model_sha256": manifest.get("model_sha256", ""),
                "input_sha256": manifest.get("input_sha256", ""),
                "md_seed": manifest.get("md_seed", 20261005),
                "slurm_job_id": manifest.get("slurm_job_id", ""),
                "slurm_array_task_id": manifest.get("slurm_array_task_id", ""),
                "host": manifest.get("host", ""),
                "onset_step": onset_step,
                "onset_ps": onset_ps,
                "trigger": trigger or ("not_observed" if metrics.empty or onset_index is None else ""),
            })
            if run is None or metrics.empty or not trajectory_path.exists():
                continue
            input_path = Path(str(manifest.get("input", "")))
            input_key = str(input_path)
            if input_key not in input_cache and input_path.exists():
                input_cache[input_key] = read(input_path, index=-1)
            reference_atoms = input_cache.get(input_key)
            try:
                first_atoms = read(trajectory_path, index=0, format="extxyz")
                if reference_atoms is not None and len(first_atoms) == len(reference_atoms):
                    initial_state_rows.append({
                        "source": source_name,
                        "run": run.name,
                        "family": family,
                        "training_seed": seed,
                        "host": manifest.get("host", ""),
                        "slurm_job_id": manifest.get("slurm_job_id", ""),
                        "slurm_array_task_id": manifest.get("slurm_array_task_id", ""),
                        "input_sha256": manifest.get("input_sha256", ""),
                        "first_step": first_atoms.info.get("step", ""),
                        "position_max_abs_A": float(np.max(np.abs(first_atoms.positions - reference_atoms.positions))),
                        "momentum_max_abs": float(np.max(np.abs(first_atoms.get_momenta() - reference_atoms.get_momenta()))) if first_atoms.has("momenta") and reference_atoms.has("momenta") else float("nan"),
                        "cell_max_abs_A": float(np.max(np.abs(first_atoms.cell.array - reference_atoms.cell.array))),
                        "species_equal": first_atoms.get_chemical_symbols() == reference_atoms.get_chemical_symbols(),
                        "pbc_equal": tuple(bool(value) for value in first_atoms.pbc) == tuple(bool(value) for value in reference_atoms.pbc),
                        "natoms_equal": len(first_atoms) == len(reference_atoms),
                        "finite_positions": bool(np.isfinite(first_atoms.positions).all()),
                        "finite_momenta": bool(np.isfinite(first_atoms.get_momenta()).all()) if first_atoms.has("momenta") else False,
                    })
            except Exception:
                initial_state_rows.append({"source": source_name, "run": run.name, "family": family, "training_seed": seed, "initial_state_audit_error": True})
            metrics["temperature_derivative_K_ps"] = np.gradient(metrics["temperature_K"].to_numpy(float), metrics["time_ps"].to_numpy(float), edge_order=1)
            metrics["relative_time_ps"] = metrics["time_ps"] - onset_ps
            metrics["source"] = source_name
            metrics["run"] = run.name
            metrics["family"] = family
            metrics["training_seed"] = seed
            metrics["model_sha256"] = manifest.get("model_sha256", "")
            run_specs.append({
                "source": source_name,
                "root": root,
                "run": run,
                "manifest": manifest,
                "status": status,
                "metrics": metrics,
                "onset_step": onset_step,
                "onset_ps": onset_ps,
                "trigger": trigger or "not_observed",
                "reference_atoms": reference_atoms,
            })

    metrics_all = pd.concat([spec["metrics"] for spec in run_specs], ignore_index=True) if run_specs else pd.DataFrame()
    if not metrics_all.empty:
        metrics_all.to_csv(out / "metrics_all.csv", index=False)

    frame_rows: List[Dict] = []
    frame_vectors: List[np.ndarray] = []
    force_frames: List[Tuple[Dict, Atoms]] = []
    run_summary: List[Dict] = []
    snapshot_rows: List[Dict] = []
    for spec in run_specs:
        manifest = spec["manifest"]
        metrics = spec["metrics"]
        selected = selected_steps(metrics, spec["onset_ps"])
        force_targets = {int(step) for window in ("stable_pre", "precursor", "threshold") for step in evenly_spaced(selected[window], 3)}
        current_calculator = None
        if args.force_disagreement:
            from mace.calculators import MACECalculator

            current_calculator = MACECalculator(model_paths=str(manifest["model"]), device="cpu", default_dtype="float64")
        row_by_step = {int(row["step"]): row for row in metrics.to_dict("records")}
        wanted = {int(step) for values in selected.values() for step in values}
        window_by_step = {int(step): window for window, values in selected.items() for step in values}
        matched = 0
        for step, atoms in iter_run_frames(spec["run"], wanted):
            if current_calculator is not None and step in force_targets:
                atoms.calc = current_calculator
                atoms.arrays["forces"] = np.asarray(atoms.get_forces(), dtype=float)
                # Keep the exact-model force array in the snapshot, but detach
                # the calculator before ASE writes extxyz; otherwise ASE tries
                # to export the same ``forces`` field twice.
                atoms.calc = None
            metric = row_by_step[step]
            row = {
                "source": spec["source"],
                "root": str(spec["root"]),
                "run": spec["run"].name,
                "family": manifest.get("family"),
                "training_seed": manifest.get("training_seed"),
                "md_seed": manifest.get("md_seed"),
                "model_sha256": manifest.get("model_sha256"),
                "window": window_by_step[step],
                "onset_step": spec["onset_step"],
                "onset_ps": spec["onset_ps"],
                "trigger": spec["trigger"],
                **scalar_features(atoms, metric, spec["reference_atoms"]),
            }
            snapshot_dir = out / "snapshots" / f"{spec['source']}__{spec['run'].name}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"{window_by_step[step]}__step_{step:07d}.extxyz"
            atoms.write(snapshot_path, format="extxyz")
            row["snapshot_path"] = str(snapshot_path)
            frame_rows.append(row)
            frame_vectors.append(np.asarray(soap.create(atoms), dtype=float))
            snapshot_rows.append({
                "source": spec["source"],
                "run": spec["run"].name,
                "training_seed": manifest.get("training_seed"),
                "family": manifest.get("family"),
                "window": window_by_step[step],
                "step": step,
                "time_ps": float(metric["time_ps"]),
                "snapshot_path": str(snapshot_path),
                "model_sha256": manifest.get("model_sha256"),
                "input_sha256": manifest.get("input_sha256"),
            })
            if args.force_disagreement and step in force_targets:
                force_frames.append((row.copy(), atoms.copy()))
            matched += 1
        run_summary.append({
            "source": spec["source"],
            "root": str(spec["root"]),
            "run": spec["run"].name,
            "family": manifest.get("family"),
            "training_seed": manifest.get("training_seed"),
            "md_seed": manifest.get("md_seed"),
            "model_sha256": manifest.get("model_sha256"),
            "input_sha256": manifest.get("input_sha256"),
            "host": manifest.get("host", ""),
            "slurm_job_id": manifest.get("slurm_job_id", ""),
            "slurm_array_task_id": manifest.get("slurm_array_task_id", ""),
            "status": spec["status"].get("status", ""),
            "stop_reason": spec["status"].get("stop_reason", ""),
            "terminal": spec["status"].get("status") in {"complete", "failed", "interrupted"},
            "onset_step": spec["onset_step"],
            "onset_ps": spec["onset_ps"],
            "trigger": spec["trigger"],
            "last_step": int(metrics.iloc[-1]["step"]),
            "last_time_ps": float(metrics.iloc[-1]["time_ps"]),
            "max_temperature_K": float(metrics["temperature_K"].max()),
            "max_force_eV_A": float(metrics["max_force_eV_A"].max()),
            "min_distance_A": float(metrics["min_distance_A"].min()),
            "n_selected_frames": matched,
        })

    if frame_rows:
        add_distances(frame_rows, np.asarray(frame_vectors), reference, ref_meta, mean, scale, pca, gates)
        fields = sorted({key for row in frame_rows for key in row})
        with (out / "frame_features.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(frame_rows)
    for path, rows in ((out / "run_summary.csv", run_summary), (out / "replica_ledger.csv", ledger_rows), (out / "snapshot_manifest.csv", snapshot_rows), (out / "initial_state_audit.csv", initial_state_rows)):
        fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    ref_pca = pca.transform((reference - mean) / scale)
    with (out / "reference_embedding.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reference", "source", "index", "pca1", "pca2"])
        writer.writeheader()
        for item, point in zip(ref_meta, ref_pca):
            writer.writerow({"reference": item["reference"], "source": item["source"], "index": item["index"], "pca1": float(point[0]), "pca2": float(point[1]) if len(point) > 1 else 0.0})

    if frame_rows and snapshot_rows:
        snapshot_by_key = {(row["source"], row["run"], row["step"]): row for row in frame_rows}
        for snapshot in snapshot_rows:
            row = snapshot_by_key.get((snapshot["source"], snapshot["run"], snapshot["step"]), {})
            for key in ("pca1", "pca2", "distance_to_train", "distance_to_valid", "distance_to_test", "distance_to_neb", "outside_train_q95", "outside_train_q99"):
                snapshot[key] = row.get(key, "")
        with (out / "snapshot_manifest.csv").open("w", newline="") as handle:
            fields = sorted({key for row in snapshot_rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(snapshot_rows)

    if args.force_disagreement and force_frames:
        from mace.calculators import MACECalculator

        model_paths = {spec["manifest"].get("model_sha256"): Path(spec["manifest"]["model"]) for spec in run_specs if spec["manifest"].get("model_sha256")}
        calculators = {model_sha: MACECalculator(model_paths=str(path), device="cpu", default_dtype="float64") for model_sha, path in sorted(model_paths.items())}
        disagreement_rows = []
        for row, atoms in force_frames:
            predictions = {}
            for model_sha, calculator in calculators.items():
                atoms.calc = calculator
                predictions[model_sha] = np.asarray(atoms.get_forces(), dtype=float)
            model_keys = sorted(predictions)
            stack = np.stack([predictions[key] for key in model_keys])
            spread = np.std(stack, axis=0)
            result = {
                "source": row["source"], "run": row["run"], "training_seed": row["training_seed"],
                "family": row["family"], "window": row["window"], "step": row["step"],
                "time_ps": row["time_ps"], "onset_ps": row["onset_ps"], "current_model_sha256": row["model_sha256"],
                "ensemble_force_rms_std_eV_A": float(np.sqrt(np.mean(spread * spread))),
                "ensemble_force_max_std_eV_A": float(np.max(np.linalg.norm(spread, axis=1))),
            }
            for left_index, left in enumerate(model_keys):
                for right in model_keys[left_index + 1:]:
                    delta = predictions[left] - predictions[right]
                    result[f"pair_rms_{left[:8]}_{right[:8]}_eV_A"] = float(np.sqrt(np.mean(delta * delta)))
                    result[f"pair_max_{left[:8]}_{right[:8]}_eV_A"] = float(np.max(np.linalg.norm(delta, axis=1)))
            disagreement_rows.append(result)
        fields = sorted({key for row in disagreement_rows for key in row})
        with (out / "force_disagreement.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(disagreement_rows)

    if args.prepare_dft_dir:
        args.prepare_dft_dir.mkdir(parents=True, exist_ok=True)
        candidates = [row for row in frame_rows if row.get("window") == "threshold" or row.get("outside_train_q99") is True]
        dft_rows = []
        for row in candidates:
            source = Path(row["snapshot_path"])
            target = args.prepare_dft_dir / f"{row['source']}__seed{int(row['training_seed'])}__{row['window']}__step{int(row['step']):07d}.extxyz"
            if not target.exists():
                target.write_bytes(source.read_bytes())
            dft_rows.append({"submitted": False, "reason": "prepared_only", "source_snapshot": row["snapshot_path"], "prepared_extxyz": str(target), "source": row["source"], "training_seed": row["training_seed"], "window": row["window"], "step": row["step"], "time_ps": row["time_ps"], "model_sha256": row["model_sha256"], "distance_to_train": row.get("distance_to_train", ""), "distance_to_neb": row.get("distance_to_neb", "")})
        with (args.prepare_dft_dir / "manifest.csv").open("w", newline="") as handle:
            fields = sorted({key for row in dft_rows for key in row}) if dft_rows else ["submitted", "reason"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dft_rows)

    manifest = {
        "schema": "migrationbench_from_scratch_failure_attribution_v2",
        "roots": [{"source": source, "root": str(root)} for source, root in roots],
        "data_root": str(args.data_root),
        "neb_root": str(args.neb_root),
        "soap": {"species": ["Cr", "Sb", "Te"], "r_cut_A": 5.0, "n_max": 4, "l_max": 4, "average": "inner", "sigma_A": 0.5},
        "thresholds": THRESHOLDS,
        "distribution_fit": gates,
        "script_sha256": sha256_file(Path(__file__)),
        "script_hashes": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("diagnose_from_scratch_all.py", "plot_failure_attribution_all.py", "write_failure_report_all.py", "run_attribution_all.slurm")
            if Path(__file__).with_name(name).exists()
        },
        "provenance": {
            "model_sha256": sorted({str(row["model_sha256"]) for row in ledger_rows if row.get("model_sha256")}),
            "input_sha256": sorted({str(row["input_sha256"]) for row in ledger_rows if row.get("input_sha256")}),
            "families": sorted({str(row["family"]) for row in ledger_rows}),
            "training_seeds": sorted({int(row["training_seed"]) for row in ledger_rows}),
            "md_seeds": sorted({int(row["md_seed"]) for row in ledger_rows}) if ledger_rows and "md_seed" in ledger_rows[0] else [20261005],
        },
        "n_runs": len(run_summary),
        "n_frame_rows": len(frame_rows),
        "n_reference_rows": len(ref_meta),
        "n_initial_state_audits": len(initial_state_rows),
        "force_disagreement": bool(args.force_disagreement),
        "dft_prepared_only": str(args.prepare_dft_dir) if args.prepare_dft_dir else None,
        "note": "Descriptor distances are distribution-novelty diagnostics, not calibrated uncertainty or proof of physical causality. Failed branches are unstable and have no Dxy reported.",
    }
    (out / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runs": len(run_summary), "frame_rows": len(frame_rows), "references": len(ref_meta), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
