#!/usr/bin/env python3
"""Prepare a 15-model Figure 6 representation-analysis input package.

The historical Figure 6 used 101 sampled frames per model and a feature vector
consisting of energy, flattened forces, a neighbour-distance histogram, a
10-component SOAP PCA summary, and the historical zero force-sensitivity
placeholder.  This script applies that same feature grammar to the current
15-model MD matrix.  Raw trajectories remain on Rockfish; only deterministic
sampled frame subsets and derived feature arrays are materialized locally.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent.parent / "analysis" / "md_1ns_updated_publication_figure_20261025"
INPUTS = CAMPAIGN / "md15_inputs.tsv"
OUT = ROOT / "updated_campaign_20261102"
SAMPLES = OUT / "source" / "trajectory_samples"
FEATURES = OUT / "source" / "fig6_full_campaign_features.npz"
FEATURE_META = OUT / "source" / "feature_metadata.tsv"
SAMPLE_META = OUT / "source" / "sample_manifest.tsv"
PREP_MANIFEST = OUT / "source" / "preparation_manifest.json"

REMOTE = "rockfish"
MAX_FRAMES = 101
NATOMS = 2050
BLOCK = NATOMS + 2

REMOTE_EXTRACT = r"""
import sys

paths = sys.argv[1:]
if not paths:
    raise SystemExit("no paths")
block = int(open(paths[0], "rb").readline().strip()) + 2
counts = []
for path in paths:
    with open(path, "rb") as handle:
        counts.append(sum(1 for _ in handle) // block)
total = sum(counts)
k = min(101, total)
selected = set()
if k == 1:
    selected.add(0)
else:
    for j in range(k):
        selected.add(round(j * (total - 1) / (k - 1)))
base = 0
out = sys.stdout.buffer
for path, count in zip(paths, counts):
    with open(path, "rb") as handle:
        for i in range(count):
            lines = [handle.readline() for _ in range(block)]
            if base + i in selected:
                out.write(b"".join(lines))
    base += count
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inputs() -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    with INPUTS.open(newline="") as handle:
        lines = iter(handle)
        header = next(lines).lstrip("# ")
        rows = (line for line in (header, *lines) if line.strip())
        for row in csv.DictReader(rows, delimiter="\t"):
            item = grouped.setdefault(row["model_id"], {"family": row["family"], "paths": []})
            item["paths"].append(row["trajectory_path"])
    return grouped


def remote_sample(model_id: str, item: dict[str, object], output: Path) -> dict[str, object]:
    code = base64.b64encode(REMOTE_EXTRACT.encode()).decode()
    paths = [str(path) for path in item["paths"]]
    command = "python3 -c " + shlex.quote(
        "import base64; exec(base64.b64decode(" + repr(code) + "))"
    )
    command += " " + " ".join(shlex.quote(path) for path in paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", REMOTE, command],
            stdout=handle,
            stderr=subprocess.PIPE,
            check=True,
            text=False,
        )
    return {
        "model_id": model_id,
        "family": item["family"],
        "remote_paths": paths,
        "sample_path": str(output),
        "sample_sha256": sha256(output),
        "sample_bytes": output.stat().st_size,
    }


def extract() -> list[dict[str, object]]:
    items = read_inputs()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(remote_sample, model_id, item, SAMPLES / f"{model_id}.extxyz"): model_id
            for model_id, item in items.items()
        }
        for future in as_completed(futures):
            model_id = futures[future]
            record = future.result()
            print(f"sampled {model_id}: {record['sample_bytes']} bytes")
            records.append(record)
    records.sort(key=lambda record: str(record["model_id"]))
    SAMPLE_META.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_META.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_id", "family", "remote_paths", "sample_path", "sample_bytes", "sample_sha256"],
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "remote_paths": "|".join(record["remote_paths"])})
    return records


def load_sample_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with SAMPLE_META.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row["remote_paths"] = row["remote_paths"].split("|")
            row["sample_bytes"] = int(row["sample_bytes"])
            records.append(row)
    return records


def get_energy(atoms) -> float:
    for key in ("energy", "MACE_energy", "free_energy", "potential_energy"):
        if key in atoms.info:
            return float(atoms.info[key])
    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:
        raise KeyError("no energy field in sampled frame") from exc


def compute_features(records: list[dict[str, object]]) -> dict[str, object]:
    from ase.io import read
    from ase.neighborlist import neighbor_list
    from dscribe.descriptors import SOAP
    from sklearn.decomposition import PCA

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []
    rows: list[dict[str, object]] = []
    model_order = [str(record["model_id"]) for record in records]
    for record in records:
        model_id = str(record["model_id"])
        sample_path = Path(str(record["sample_path"]))
        atoms_list = read(sample_path, index=":")
        if not atoms_list:
            raise ValueError(f"no frames in {sample_path}")
        species = sorted(set(atoms_list[0].get_chemical_symbols()))
        soap = SOAP(species=species, periodic=True, r_cut=5.0, n_max=4, l_max=4, average="inner")
        energies = np.array([get_energy(atoms) for atoms in atoms_list], dtype=float)
        forces = np.array([atoms.get_forces().reshape(-1) for atoms in atoms_list], dtype=float)
        neighbours = []
        soap_rows = []
        for atoms in atoms_list:
            _, _, distances = neighbor_list("ijd", atoms, cutoff=5.0)
            hist, _ = np.histogram(distances, bins=30, range=(0.0, 5.0))
            neighbours.append(hist.astype(float))
            soap_rows.append(np.asarray(soap.create(atoms), dtype=float))
        soap_matrix = np.asarray(soap_rows, dtype=float)
        n_components = min(10, soap_matrix.shape[0], soap_matrix.shape[1])
        soap_pca = PCA(n_components=n_components).fit_transform(soap_matrix)
        if n_components < 10:
            soap_pca = np.hstack([soap_pca, np.zeros((soap_pca.shape[0], 10 - n_components))])
        sensitivity = np.zeros((len(atoms_list), 1), dtype=float)
        model_features = np.hstack([
            energies[:, None],
            forces,
            np.asarray(neighbours, dtype=float),
            soap_pca,
            sensitivity,
        ])
        if model_features.shape[1] != 6192:
            raise ValueError(f"{model_id}: expected 6192 features, got {model_features.shape[1]}")
        all_features.append(model_features)
        all_labels.extend([model_id] * len(model_features))
        rows.append({
            "model_id": model_id,
            "family": record["family"],
            "n_frames": len(model_features),
            "n_features": model_features.shape[1],
            "sample_sha256": record["sample_sha256"],
            "feature_energy_min": float(np.min(energies)),
            "feature_energy_max": float(np.max(energies)),
        })
        print(f"features {model_id}: {model_features.shape}")
    matrix = np.vstack(all_features)
    np.savez_compressed(
        FEATURES,
        features=matrix,
        model_ids=np.asarray(all_labels, dtype=object),
        model_order=np.asarray(model_order, dtype=object),
    )
    with FEATURE_META.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "shape": list(matrix.shape),
        "feature_sha256": sha256(FEATURES),
        "model_order": model_order,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--compute", action="store_true")
    args = parser.parse_args()
    if not args.extract and not args.compute:
        args.extract = True
        args.compute = True
    records = extract() if args.extract else load_sample_records()
    if args.compute:
        feature_summary = compute_features(records)
    else:
        feature_summary = None
    manifest = {
        "artifact": "Figure 6 full-campaign 15-model representation input",
        "status": "RECONSTRUCTION",
        "model_matrix": "3 Foundation checkpoints + 3 seeds x FT-600K + 3 seeds x FT-MultiT + 3 seeds x Scratch-600K + 3 seeds x Scratch-MultiT",
        "feature_grammar": "legacy Figure 6: energy + flattened forces + 30-bin neighbor histogram + 10-component SOAP PCA + zero sensitivity placeholder",
        "sampling": "deterministic evenly spaced frame indices across the listed trajectory segments; maximum 101 frames per model; short Scratch trajectories retain their actual frame count",
        "input_manifest": str(INPUTS),
        "input_manifest_sha256": sha256(INPUTS),
        "sample_records": records,
        "feature_summary": feature_summary,
    }
    PREP_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {PREP_MANIFEST}")


if __name__ == "__main__":
    main()
