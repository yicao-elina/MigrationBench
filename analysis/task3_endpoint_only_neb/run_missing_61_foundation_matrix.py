#!/usr/bin/env python3
"""Reproduce the missing 61-atom Foundation fixed-path matrix for 1-6 and 1-8."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

from ase.io import read, write


RAW = Path("/Users/alina/Project/26MIgrationBench/data_raw/historical_neb/legacy_61")
MODELS = Path("/Users/alina/Documents/Codex/foundation_grouped_benchmark_20261001/models.tsv")
EVALUATOR = Path("/Users/alina/Documents/Codex/foundation_grouped_benchmark_20261001/evaluate_neb_foundation.py")
OUT = Path(__file__).resolve().parent / "exploratory_energy_audit" / "cr_61_81_three_class_20261003"
INPUTS = OUT / "inputs_61"
RESULTS = OUT / "foundation_61_missing"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_cell(qe_input: Path):
    lines = qe_input.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            return [[float(x) for x in lines[i + j].split("!")[0].split()[:3]] for j in (1, 2, 3)]
    raise ValueError(f"No CELL_PARAMETERS in {qe_input}")


def prepare_path(path_id: str) -> Path:
    source = RAW / path_id / "sb2te3.xyz"
    qe_input = RAW / path_id / "pw_1.in"
    frames = read(source, index=":")
    cell = read_cell(qe_input)
    for frame in frames:
        frame.set_cell(cell)
        frame.set_pbc((True, True, True))
    target = INPUTS / f"{path_id}.extxyz"
    target.parent.mkdir(parents=True, exist_ok=True)
    write(target, frames, format="extxyz")
    return target


def load_models():
    rows = []
    with MODELS.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            model = row["local_checkpoint"]
            if not model and row["model_id"] == "mpa0_medium":
                model = "/Users/alina/Downloads/mace-mpa-0-medium.model"
            if not model or not Path(model).exists():
                raise FileNotFoundError(f"Missing local checkpoint for {row['model_id']}: {model}")
            row["resolved_local_checkpoint"] = model
            row["resolved_sha256"] = sha256(Path(model))
            rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {path_id: prepare_path(path_id) for path_id in ("1-6", "1-8")}
    models = load_models()
    manifest = {
        "protocol": "fixed-path single-point Foundation energy/force evaluation",
        "paths": {k: {"input": str(v), "input_sha256": sha256(v), "source_xyz": str(RAW / k / "sb2te3.xyz"), "source_xyz_sha256": sha256(RAW / k / "sb2te3.xyz"), "source_qe_input": str(RAW / k / "pw_1.in"), "source_qe_input_sha256": sha256(RAW / k / "pw_1.in")} for k, v in paths.items()},
        "models": [{k: row[k] for k in ("model_id", "family", "variant", "head", "resolved_local_checkpoint", "resolved_sha256")} for row in models],
        "results": [],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    for row in models:
        for path_id, xyz in paths.items():
            out = RESULTS / f"{row['model_id']}__{path_id}.json"
            args = ["python", str(EVALUATOR), "--model-id", row["model_id"], "--model", row["resolved_local_checkpoint"], "--path-id", path_id, "--xyz", str(xyz), "--out", str(out), "--device", "cpu"]
            if row.get("head"):
                args += ["--head", row["head"]]
            print("RUN", row["model_id"], path_id, flush=True)
            subprocess.run(args, check=True)
            payload = json.loads(out.read_text())
            manifest["results"].append({"model_id": row["model_id"], "path_id": path_id, "result": str(out), "result_sha256": sha256(out), "finite": payload["finite"], "barrier_eV": payload["barrier_eV"], "n_images": payload["n_images"]})
    (OUT / "foundation_61_missing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"n_models": len(models), "n_paths": len(paths), "n_results": len(manifest["results"]), "all_finite": all(x["finite"] for x in manifest["results"])}, indent=2))


if __name__ == "__main__":
    main()
