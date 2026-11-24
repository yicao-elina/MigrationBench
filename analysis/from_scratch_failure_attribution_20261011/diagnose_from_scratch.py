#!/usr/bin/env python3
"""Provenance-first failure attribution for the three from-scratch MD branches.

The script deliberately separates:
  * the original runs from their exact reproductions;
  * thermodynamic/force threshold onset from a possible physical transition;
  * stable, precursor, and threshold-frame environments; and
  * local structural novelty from model-force disagreement.

It is designed to run in the Rockfish MACE environment on the shared CPU
partition.  It writes only to a new analysis directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import iread
from ase.neighborlist import neighbor_list
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


THRESHOLDS = {
    "temperature_K": 5000.0,
    "max_force_eV_A": 100.0,
    "min_distance_A": 0.5,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_bad(metrics: pd.DataFrame) -> tuple[int | None, str | None]:
    bad = (
        (metrics["temperature_K"] > THRESHOLDS["temperature_K"])
        | (metrics["max_force_eV_A"] > THRESHOLDS["max_force_eV_A"])
        | (metrics["min_distance_A"] < THRESHOLDS["min_distance_A"])
        | ~np.isfinite(metrics[["temperature_K", "max_force_eV_A", "min_distance_A"]]).all(axis=1)
    )
    indices = np.flatnonzero(bad.to_numpy())
    if len(indices) == 0:
        return None, None
    index = int(indices[0])
    row = metrics.iloc[index]
    candidates = {
        "temperature": float(row["temperature_K"]) > THRESHOLDS["temperature_K"],
        "force": float(row["max_force_eV_A"]) > THRESHOLDS["max_force_eV_A"],
        "minimum_distance": float(row["min_distance_A"]) < THRESHOLDS["min_distance_A"],
    }
    trigger = "+".join(name for name, value in candidates.items() if value)
    return index, trigger or "nonfinite"


def evenly_spaced(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) <= n:
        return values
    indices = np.linspace(0, len(values) - 1, n).round().astype(int)
    return values[np.unique(indices)]


def selected_steps(metrics: pd.DataFrame, onset_ps: float) -> dict[str, np.ndarray]:
    time = metrics["time_ps"].to_numpy(float)
    steps = metrics["step"].to_numpy(int)
    windows = {
        "stable_early": (5.0, 15.0),
        "stable_pre": (max(0.0, onset_ps - 30.0), max(0.0, onset_ps - 15.0)),
        "precursor": (max(0.0, onset_ps - 5.0), onset_ps),
        "threshold": (onset_ps, onset_ps),
    }
    selected: dict[str, np.ndarray] = {}
    for name, (lo, hi) in windows.items():
        mask = (time >= lo) & (time <= hi)
        candidates = steps[mask]
        selected[name] = evenly_spaced(candidates, 24 if name != "threshold" else 1)
    if not len(selected["stable_pre"]):
        selected["stable_pre"] = selected["stable_early"]
    return selected


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if len(values) else float("nan")


def _coordination(atoms: Atoms, central: str, neighbor: str, cutoff: float) -> tuple[float, float]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, distances = neighbor_list("ijd", atoms, cutoff=cutoff)
    mask = (symbols[i] == central) & (symbols[j] == neighbor)
    if not np.any(mask):
        return float("nan"), float("nan")
    counts = np.bincount(i[mask], minlength=len(atoms))
    selected = counts[symbols == central]
    return float(np.mean(selected)), float(np.std(selected))


def _pair_distance(atoms: Atoms, a: str, b: str, cutoff: float = 5.0) -> float:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, distances = neighbor_list("ijd", atoms, cutoff=cutoff)
    mask = ((symbols[i] == a) & (symbols[j] == b)) | ((symbols[i] == b) & (symbols[j] == a))
    return _safe_min(distances[mask])


def _local_angle_summary(atoms: Atoms, central: str = "Cr", neighbor: str = "Te", cutoff: float = 3.5) -> tuple[float, float]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, shifts, distances = neighbor_list("ijSd", atoms, cutoff=cutoff)
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
        upper = cosine[np.triu_indices(len(vec), 1)]
        values.extend(np.degrees(np.arccos(upper)).tolist())
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def structural_features(atoms: Atoms) -> dict[str, float]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    features: dict[str, float] = {
        "n_Cr": float(np.sum(symbols == "Cr")),
        "n_Sb": float(np.sum(symbols == "Sb")),
        "n_Te": float(np.sum(symbols == "Te")),
        "volume_A3": float(atoms.get_volume()),
    }
    for cutoff in (2.8, 3.2, 3.6, 4.0):
        for neighbor in ("Te", "Sb", "Cr"):
            mean, std = _coordination(atoms, "Cr", neighbor, cutoff)
            tag = str(cutoff).replace(".", "p")
            features[f"Cr_{neighbor}_coord_{tag}_mean"] = mean
            features[f"Cr_{neighbor}_coord_{tag}_std"] = std
    features["min_CrTe_A"] = _pair_distance(atoms, "Cr", "Te")
    features["min_CrSb_A"] = _pair_distance(atoms, "Cr", "Sb")
    features["min_CrCr_A"] = _pair_distance(atoms, "Cr", "Cr")
    features["min_SbTe_A"] = _pair_distance(atoms, "Sb", "Te")
    angle_mean, angle_std = _local_angle_summary(atoms)
    features["CrTe_angle_mean_deg"] = angle_mean
    features["CrTe_angle_std_deg"] = angle_std
    return features


def scalar_features(atoms: Atoms, row: dict | pd.Series) -> dict[str, float]:
    forces = np.asarray(atoms.arrays.get("forces", np.full((len(atoms), 3), np.nan)), dtype=float)
    force_norm = np.linalg.norm(forces, axis=1) if np.isfinite(forces).all() else np.full(len(atoms), np.nan)
    worst = int(np.nanargmax(force_norm)) if np.isfinite(force_norm).any() else -1
    out = {
        "step": int(row["step"]),
        "time_ps": float(row["time_ps"]),
        "potential_energy_eV": float(row["potential_energy_eV"]),
        "temperature_K": float(row["temperature_K"]),
        "max_force_eV_A": float(row["max_force_eV_A"]),
        "rms_force_eV_A": float(row["rms_force_eV_A"]),
        "min_distance_A": float(row["min_distance_A"]),
        "worst_force_atom_index": worst,
        "worst_force_atom_species": atoms.get_chemical_symbols()[worst] if worst >= 0 else "",
        "worst_force_norm_eV_A": float(force_norm[worst]) if worst >= 0 else float("nan"),
    }
    return out


def load_soap():
    from dscribe.descriptors import SOAP

    return SOAP(
        species=["Cr", "Sb", "Te"],
        periodic=True,
        r_cut=5.0,
        n_max=4,
        l_max=4,
        average="inner",
        sigma=0.5,
    )


def sample_extxyz(path: Path, limit: int) -> list[Atoms]:
    frames = list(iread(path, format="extxyz"))
    if len(frames) <= limit:
        return frames
    indices = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[i] for i in np.unique(indices)]


def reference_soap(data_root: Path, neb_root: Path, soap, limit: int) -> tuple[np.ndarray, list[dict]]:
    vectors: list[np.ndarray] = []
    meta: list[dict] = []
    for label in ("train", "valid", "test"):
        path = data_root / f"{label}.xyz"
        if not path.exists():
            continue
        for index, atoms in enumerate(sample_extxyz(path, limit)):
            vectors.append(np.asarray(soap.create(atoms), dtype=float))
            meta.append({"reference": label, "source": str(path), "index": index})
    for path in sorted(neb_root.glob("*/neb_soap_features.csv")):
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
    return np.asarray(vectors), meta


def fit_distribution(reference: np.ndarray, meta: list[dict], n_md: int) -> tuple[np.ndarray, list[dict], PCA, dict]:
    ref_labels = np.asarray([row["reference"] for row in meta])
    train = reference[ref_labels == "train"]
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
    gates = {
        "train_knn_q95": float(np.quantile(baseline, 0.95)),
        "train_knn_q99": float(np.quantile(baseline, 0.99)),
        "pca_components": int(pca.n_components_),
        "n_train": int(len(train)),
    }
    return mean, scale, pca, gates


def add_distances(rows: list[dict], vectors: np.ndarray, reference: np.ndarray, meta: list[dict], mean: np.ndarray, scale: np.ndarray, pca: PCA, gates: dict) -> None:
    ref_labels = np.asarray([row["reference"] for row in meta])
    ref_pca = pca.transform((reference - mean) / scale)
    md_pca = pca.transform((vectors - mean) / scale)
    by_label = {label: ref_pca[ref_labels == label] for label in sorted(set(ref_labels))}
    for row, point in zip(rows, md_pca):
        for label, cloud in by_label.items():
            if len(cloud):
                row[f"distance_to_{label}"] = float(np.min(np.linalg.norm(cloud - point[None, :], axis=1)))
            else:
                row[f"distance_to_{label}"] = float("nan")
        row["pca1"] = float(point[0])
        row["pca2"] = float(point[1]) if len(point) > 1 else 0.0
        row["outside_train_q95"] = bool(row.get("distance_to_train", np.nan) > gates["train_knn_q95"])


def iter_run_frames(run: Path, chosen_steps: set[int]) -> Iterable[tuple[int, Atoms]]:
    for atoms in iread(run / "trajectory.extxyz", format="extxyz"):
        step = int(atoms.info.get("step", -1))
        if step in chosen_steps:
            yield step, atoms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--reproduce-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--neb-root", type=Path, required=True)
    parser.add_argument("--frames-per-reference", type=int, default=250)
    parser.add_argument("--force-disagreement", action="store_true")
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    soap = load_soap()
    reference, ref_meta = reference_soap(args.data_root, args.neb_root, soap, args.frames_per_reference)
    mean, scale, pca, gates = fit_distribution(reference, ref_meta, 0)

    run_specs = []
    for source_name, root in (("original", args.original_root), ("reproduce", args.reproduce_root)):
        for run in sorted((root / "runs").glob("*")):
            manifest_path = run / "run_manifest.json"
            metrics_path = run / "metrics.csv"
            trajectory_path = run / "trajectory.extxyz"
            if not (manifest_path.exists() and metrics_path.exists() and trajectory_path.exists()):
                continue
            manifest = json.loads(manifest_path.read_text())
            if int(manifest.get("training_seed", -1)) not in (123, 234, 345):
                continue
            metrics = pd.read_csv(metrics_path)
            onset_index, trigger = first_bad(metrics)
            if onset_index is None:
                onset_step = int(metrics.iloc[-1]["step"])
                trigger = "not_observed"
            else:
                onset_step = int(metrics.iloc[onset_index]["step"])
            onset_ps = onset_step * float(manifest.get("timestep_fs", 1.0)) * 1.0e-3
            run_specs.append({
                "source": source_name,
                "run": run,
                "manifest": manifest,
                "metrics": metrics,
                "onset_step": onset_step,
                "onset_ps": onset_ps,
                "trigger": trigger,
            })

    frame_rows: list[dict] = []
    frame_vectors: list[np.ndarray] = []
    force_frames: list[tuple[dict, Atoms]] = []
    run_summary: list[dict] = []
    for spec in run_specs:
        manifest = spec["manifest"]
        metrics = spec["metrics"]
        selected = selected_steps(metrics, spec["onset_ps"])
        force_targets = {
            int(step)
            for window in ("stable_pre", "precursor", "threshold")
            for step in evenly_spaced(selected[window], 3)
        }
        row_by_step = {
            int(step): {"step": int(step), **values}
            for step, values in metrics.set_index("step").to_dict("index").items()
        }
        wanted = {int(step) for values in selected.values() for step in values}
        window_by_step = {}
        for window, values in selected.items():
            for step in values:
                window_by_step[int(step)] = window
        matched = 0
        for step, atoms in iter_run_frames(spec["run"], wanted):
            metric = row_by_step[step]
            scalar = scalar_features(atoms, metric)
            structural = structural_features(atoms)
            row = {
                "source": spec["source"],
                "run": spec["run"].name,
                "family": manifest.get("family"),
                "training_seed": manifest.get("training_seed"),
                "md_seed": manifest.get("md_seed"),
                "model_sha256": manifest.get("model_sha256"),
                "window": window_by_step[step],
                "onset_step": spec["onset_step"],
                "onset_ps": spec["onset_ps"],
                "trigger": spec["trigger"],
                **scalar,
                **structural,
            }
            frame_rows.append(row)
            frame_vectors.append(np.asarray(soap.create(atoms), dtype=float))
            snapshot_dir = out / "snapshots" / f"{spec['source']}__{spec['run'].name}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"{window_by_step[step]}__step_{step:07d}.extxyz"
            if not snapshot_path.exists():
                atoms.write(snapshot_path, format="extxyz")
            if args.force_disagreement and step in force_targets:
                force_frames.append((row.copy(), atoms.copy()))
            matched += 1
        run_summary.append({
            "source": spec["source"],
            "run": spec["run"].name,
            "family": manifest.get("family"),
            "training_seed": manifest.get("training_seed"),
            "md_seed": manifest.get("md_seed"),
            "model_sha256": manifest.get("model_sha256"),
            "input_sha256": manifest.get("input_sha256"),
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
    with (out / "run_summary.csv").open("w", newline="") as handle:
        fields = sorted({key for row in run_summary for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(run_summary)

    ref_pca = pca.transform((reference - mean) / scale)
    ref_rows = []
    for item, point in zip(ref_meta, ref_pca):
        ref_rows.append({"reference": item["reference"], "source": item["source"], "index": item["index"], "pca1": float(point[0]), "pca2": float(point[1]) if len(point) > 1 else 0.0})
    with (out / "reference_embedding.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reference", "source", "index", "pca1", "pca2"])
        writer.writeheader()
        writer.writerows(ref_rows)

    if args.force_disagreement:
        from mace.calculators import MACECalculator

        model_paths = {}
        for spec in run_specs:
            model_paths[spec["manifest"]["model_sha256"]] = Path(spec["manifest"]["model"])
        calculators = {
            model_sha: MACECalculator(model_paths=str(model_path), device="cpu", default_dtype="float64")
            for model_sha, model_path in sorted(model_paths.items())
        }
        disagreement_rows = []
        for row, atoms in force_frames:
            predictions = {}
            for model_sha, calculator in calculators.items():
                atoms.calc = calculator
                predictions[model_sha] = np.asarray(atoms.get_forces(), dtype=float)
            model_keys = sorted(predictions)
            stack = np.stack([predictions[key] for key in model_keys])
            ensemble_std = np.std(stack, axis=0)
            result = {
                "source": row["source"],
                "run": row["run"],
                "training_seed": row["training_seed"],
                "window": row["window"],
                "step": row["step"],
                "time_ps": row["time_ps"],
                "onset_ps": row["onset_ps"],
                "current_model_sha256": row["model_sha256"],
                "ensemble_force_rms_std_eV_A": float(np.sqrt(np.mean(ensemble_std * ensemble_std))),
                "ensemble_force_max_std_eV_A": float(np.max(np.linalg.norm(ensemble_std, axis=1))),
            }
            for left_index, left in enumerate(model_keys):
                for right in model_keys[left_index + 1 :]:
                    delta = predictions[left] - predictions[right]
                    result[f"pair_rms_{left[:8]}_{right[:8]}_eV_A"] = float(np.sqrt(np.mean(delta * delta)))
                    result[f"pair_max_{left[:8]}_{right[:8]}_eV_A"] = float(np.max(np.linalg.norm(delta, axis=1)))
            disagreement_rows.append(result)
        if disagreement_rows:
            fields = sorted({key for row in disagreement_rows for key in row})
            with (out / "force_disagreement.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(disagreement_rows)

    manifest = {
        "schema": "migrationbench_from_scratch_failure_attribution_v1",
        "original_root": str(args.original_root),
        "reproduce_root": str(args.reproduce_root),
        "data_root": str(args.data_root),
        "neb_root": str(args.neb_root),
        "soap": {"species": ["Cr", "Sb", "Te"], "r_cut_A": 5.0, "n_max": 4, "l_max": 4, "average": "inner", "sigma_A": 0.5},
        "thresholds": THRESHOLDS,
        "distribution_fit": gates,
        "script_sha256": sha256_file(Path(__file__)),
        "n_runs": len(run_summary),
        "n_frame_rows": len(frame_rows),
        "n_reference_rows": len(ref_rows),
        "force_disagreement": bool(args.force_disagreement),
        "note": "Distances are descriptor-space novelty diagnostics, not calibrated model uncertainty or proof of physical causality.",
    }
    (out / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"runs": len(run_summary), "frame_rows": len(frame_rows), "references": len(ref_rows), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
