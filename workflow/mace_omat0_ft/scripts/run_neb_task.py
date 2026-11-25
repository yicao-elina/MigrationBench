#!/usr/bin/env python3
"""Run one provenance-checked fixed-path or MLFF-NEB task.

The endpoint/path manifest is deliberately external to the model campaign.
This keeps DFT endpoint provenance separate from the trained-model matrix and
prevents a missing path from being silently reconstructed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from ase.io import read, write


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise SystemExit(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if expected and actual != expected:
        raise SystemExit(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def software_versions() -> dict[str, str | None]:
    values = {}
    for distribution in ("ase", "mace-torch", "torch"):
        try:
            values[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            values[distribution] = None
    return values


def load_images(spec: dict):
    input_spec = spec["input"]
    kind = input_spec["kind"]
    if kind == "images":
        path = Path(input_spec["images"]).resolve()
        actual = require_hash(path, input_spec.get("images_sha256"), "NEB images")
        images = read(path, index=":")
        source = {"kind": kind, "path": str(path), "sha256": actual}
    elif kind == "endpoints":
        initial_path = Path(input_spec["initial"]).resolve()
        final_path = Path(input_spec["final"]).resolve()
        initial_hash = require_hash(initial_path, input_spec.get("initial_sha256"), "NEB initial")
        final_hash = require_hash(final_path, input_spec.get("final_sha256"), "NEB final")
        initial = read(initial_path)
        final = read(final_path)
        if len(initial) != len(final):
            raise SystemExit("NEB endpoints have different atom counts")
        n_images = int(spec["protocol"]["self_relaxed_mlff_neb"].get("n_images", spec["expected_nimages"]))
        if n_images < 4:
            raise SystemExit("NEB requires at least four images")
        from ase.mep import NEB

        images = [initial.copy()] + [initial.copy() for _ in range(n_images - 2)] + [final.copy()]
        neb = NEB(images, climb=False, method="improvedtangent")
        neb.interpolate(method=input_spec.get("interpolation", "idpp"), mic=True)
        source = {
            "kind": kind,
            "initial": str(initial_path),
            "initial_sha256": initial_hash,
            "final": str(final_path),
            "final_sha256": final_hash,
        }
    else:
        raise SystemExit(f"Unsupported NEB input kind: {kind}")

    if len(images) != int(spec["expected_nimages"]):
        raise SystemExit(f"Expected {spec['expected_nimages']} images, got {len(images)}")
    expected_natoms = spec.get("expected_natoms")
    if expected_natoms is not None and any(len(atoms) != int(expected_natoms) for atoms in images):
        raise SystemExit("NEB image atom count does not match expected_natoms")
    ref = images[0]
    for index, atoms in enumerate(images):
        if len(atoms) != len(ref) or tuple(atoms.get_chemical_symbols()) != tuple(ref.get_chemical_symbols()):
            raise SystemExit(f"NEB image {index} has inconsistent species/order")
        if tuple(bool(x) for x in atoms.pbc) != tuple(bool(x) for x in spec["cell_pbc"]):
            raise SystemExit(f"NEB image {index} PBC does not match manifest")
        if not np.isfinite(atoms.positions).all():
            raise SystemExit(f"NEB image {index} contains non-finite positions")
    return images, source


def load_model(config: dict, family: str, seed: int) -> tuple[Path, dict]:
    training_manifest = Path(config["remote_root"]) / "training" / family / f"seed{seed}" / "training_manifest.json"
    if not training_manifest.is_file():
        raise SystemExit(f"Missing training manifest: {training_manifest}")
    payload = json.loads(training_manifest.read_text())
    artifact = Path(payload["artifact_path"])
    artifact_hash = require_hash(artifact, payload.get("artifact_sha256"), "trained model artifact")
    return artifact, {"path": str(training_manifest), "sha256": sha256_file(training_manifest), "artifact_sha256": artifact_hash}


def mace_calculator(model_path: Path, device: str, dtype: str):
    from mace.calculators import MACECalculator

    try:
        return MACECalculator(model_paths=str(model_path), device=device, default_dtype=dtype)
    except TypeError:
        try:
            return MACECalculator(model_path=str(model_path), device=device, default_dtype=dtype)
        except TypeError:
            return MACECalculator(model_paths=str(model_path), device=device)


def fixed_path(images, model_path: Path, protocol: dict, out_dir: Path) -> dict:
    calculator = mace_calculator(model_path, protocol.get("device", "cpu"), protocol.get("default_dtype", "float64"))
    rows = []
    outputs = []
    for index, atoms in enumerate(images):
        frame = atoms.copy()
        frame.calc = calculator
        energy = float(frame.get_potential_energy())
        forces = np.asarray(frame.get_forces(), dtype=float)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise SystemExit(f"Non-finite MACE result at fixed-path image {index}")
        output = atoms.copy()
        output.info.update({"neb_image": index, "energy": energy, "mlff_fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()), "neb_protocol": "fixed_path_single_point"})
        output.arrays["forces"] = forces
        outputs.append(output)
        rows.append({"image_index": index, "energy_eV": energy, "relative_energy_eV": 0.0, "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max())})
    minimum = min(row["energy_eV"] for row in rows)
    for row in rows:
        row["relative_energy_eV"] = row["energy_eV"] - minimum
    write(out_dir / "fixed_path_images.extxyz", outputs)
    with (out_dir / "fixed_path_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return {"profile": rows, "barrier_proxy_eV": max(row["relative_energy_eV"] for row in rows), "outputs": {"images": str(out_dir / "fixed_path_images.extxyz"), "profile": str(out_dir / "fixed_path_profile.csv")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--neb-manifest", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest_spec = yaml.safe_load(args.neb_manifest.read_text())
    if manifest_spec.get("status") not in {"approved", "ready"}:
        raise SystemExit("NEB input manifest must have status: approved or ready")
    seeds = [int(x) for x in config["training"]["seeds"]]
    jobs = [(family, seed, mode) for family in config["families"] for seed in seeds for mode in config["neb"]["modes"] if manifest_spec["protocol"].get(mode, {}).get("enabled", False)]
    if args.index < 0 or args.index >= len(jobs):
        raise SystemExit(f"NEB array index {args.index} outside 0..{len(jobs)-1}")
    family, seed, mode = jobs[args.index]
    model_path, model_record = load_model(config, family, seed)
    images, input_record = load_images(manifest_spec)
    out_dir = Path(config["remote_root"]) / config["neb"].get("output_root", "neb") / family / f"seed{seed}" / mode
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty NEB output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    input_manifest_hash = sha256_file(args.neb_manifest)
    if mode == "fixed_path_single_point":
        result = fixed_path(images, model_path, manifest_spec["protocol"][mode], out_dir)
    elif mode == "self_relaxed_mlff_neb":
        protocol = manifest_spec["protocol"][mode]
        runner = Path(config["neb"].get("runner_script", "src/migrationbench/run_mlff_neb.py"))
        if not runner.is_absolute():
            runner = args.config.resolve().parents[3] / runner
        command = [sys.executable, str(runner)]
        input_spec = manifest_spec["input"]
        if input_spec["kind"] == "images":
            command += ["--images", input_spec["images"]]
        else:
            command += ["--initial", input_spec["initial"], "--final", input_spec["final"], "--n-images", str(protocol.get("n_images", manifest_spec["expected_nimages"])), "--interpolate-method", input_spec.get("interpolation", "idpp")]
        command += ["--out-dir", str(out_dir), "--calculator", "mace-model", "--model-path", str(model_path), "--device", protocol.get("device", "cuda"), "--default-dtype", protocol.get("default_dtype", "float64"), "--fmax", str(protocol.get("fmax_eV_A", 0.03)), "--steps", str(protocol.get("max_steps", 500)), "--spring-constant", str(protocol.get("spring_constant", 0.1)), "--seed", str(protocol.get("seed", 42)), "--migrant-element", protocol.get("migrant_element", "Cr"), "--migrant-tether-k", str(protocol.get("migrant_tether_k", 0.0)), "--host-tether-k", str(protocol.get("host_tether_k", 0.0)), "--run-role", "production_preconditioner", "--source-candidate-manifest", str(args.neb_manifest)]
        if protocol.get("climb", False):
            command.append("--climb")
        subprocess.run(command, check=True)
        result = {"outputs": {"runner_manifest": str(out_dir / "mlff_neb_manifest.json")}}
    else:
        raise SystemExit(f"Unsupported NEB mode: {mode}")
    record = {"schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(), "family": family, "training_seed": seed, "mode": mode, "model": model_record, "model_path": str(model_path), "model_sha256": sha256_file(model_path), "input_manifest": str(args.neb_manifest.resolve()), "input_manifest_sha256": input_manifest_hash, "input": input_record, "software_versions": software_versions(), "result": result, "scientific_boundary": manifest_spec["scientific_boundary"]}
    (out_dir / "neb_task_manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
