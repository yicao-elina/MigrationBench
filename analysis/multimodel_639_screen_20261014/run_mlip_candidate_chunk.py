from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def software() -> dict:
    try:
        import ase
        ase_version = ase.__version__
    except Exception:
        ase_version = "unavailable"
    try:
        import mace
        mace_version = getattr(mace, "__version__", "unknown")
    except Exception:
        mace_version = "unavailable"
    try:
        import torch
        torch_info = {"version": torch.__version__, "cuda": bool(torch.cuda.is_available()),
                      "device_count": int(torch.cuda.device_count())}
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    return {"python": sys.version.split()[0], "ase": ase_version, "mace": mace_version,
            "torch": torch_info, "hostname": socket.gethostname(),
            "platform": platform.platform(), "pid": os.getpid()}


def min_distance_and_pair(atoms) -> tuple[float, str]:
    try:
        i, j, d = neighbor_list("ijd", atoms, cutoff=4.0, self_interaction=False)
        if len(d) == 0:
            return float("nan"), "none"
        k = int(np.nanargmin(d))
        syms = atoms.get_chemical_symbols()
        pair = "-".join(sorted((syms[int(i[k])], syms[int(j[k])])))
        return float(d[k]), pair
    except Exception:
        return float("nan"), "unavailable"


def geometry(atoms) -> dict:
    finite = bool(np.isfinite(atoms.positions).all())
    min_d, pair = min_distance_and_pair(atoms) if finite else (float("nan"), "nonfinite")
    syms = np.asarray(atoms.get_chemical_symbols())
    cr = np.where(syms == "Cr")[0]
    return {
        "natoms": int(len(atoms)),
        "species": ";".join(f"{s}:{int(np.sum(syms == s))}" for s in sorted(set(syms))),
        "pbc": json.dumps(np.asarray(atoms.pbc, dtype=bool).tolist()),
        "cell_volume_A3": float(abs(atoms.get_volume())),
        "geometry_finite": finite,
        "min_interatomic_distance_A": min_d,
        "min_distance_pair": pair,
        "n_cr": int(len(cr)),
    }


def force_summary(forces: np.ndarray, symbols: list[str]) -> dict:
    norms = np.linalg.norm(forces, axis=1)
    finite = bool(np.isfinite(forces).all())
    if not finite:
        return {"forces_finite": False, "max_force_eV_A": float("nan"), "rms_force_eV_A": float("nan"),
                "p95_force_eV_A": float("nan"), "p99_force_eV_A": float("nan"), "top_force_species": "nonfinite",
                "top_force_atom_index": -1}
    k = int(np.argmax(norms))
    return {"forces_finite": True, "max_force_eV_A": float(norms[k]),
            "rms_force_eV_A": float(np.sqrt(np.mean(forces * forces))),
            "p95_force_eV_A": float(np.quantile(norms, 0.95)),
            "p99_force_eV_A": float(np.quantile(norms, 0.99)),
            "top_force_species": str(symbols[k]), "top_force_atom_index": k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--chunk-index", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    models = pd.read_csv(args.models, sep="\t")
    start = args.chunk_index * args.chunk_size
    stop = min(start + args.chunk_size, len(manifest))
    candidates = manifest.iloc[start:stop].copy()
    if candidates.empty:
        raise SystemExit(f"empty chunk {args.chunk_index} start={start}")
    chunk_tag = f"chunk_{args.chunk_index:04d}"
    rows: list[dict] = []
    force_arrays: list[np.ndarray] = []
    force_candidate_ids: list[str] = []
    force_model_ids: list[str] = []
    started = time.time()
    sw = json.dumps(software(), sort_keys=True)
    model_records = models.to_dict("records")
    calculators = {}
    calculator_errors = {}
    try:
        from mace.calculators import MACECalculator
        for model in model_records:
            try:
                model_path = Path(str(model["model_path"]))
                if not model_path.is_file():
                    raise FileNotFoundError(str(model_path))
                calculators[str(model["model_id"])] = MACECalculator(
                    model_paths=str(model_path), device=args.device, default_dtype="float64"
                )
            except Exception as exc:
                calculator_errors[str(model["model_id"])] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        calculator_errors["__import__"] = f"{type(exc).__name__}: {exc}"
    for candidate in candidates.to_dict("records"):
        input_path = Path(str(candidate["prepared_extxyz"]))
        base = {k: candidate.get(k, "") for k in candidate}
        base["source_model_sha256"] = candidate.get("model_sha256", "")
        base["source_training_seed"] = candidate.get("training_seed", "")
        base["source_window"] = candidate.get("window", "")
        base["source_step"] = candidate.get("step", "")
        base["source_time_ps"] = candidate.get("time_ps", "")
        base.update({"chunk_index": args.chunk_index, "chunk_tag": chunk_tag,
                     "input_path": str(input_path), "input_sha256": candidate.get("candidate_file_sha256", "remote_only")})
        try:
            if input_path.is_file():
                base["input_sha256"] = sha256(input_path)
            atoms = read(input_path, index=0)
            geom = geometry(atoms)
            symbols = atoms.get_chemical_symbols()
        except Exception as exc:
            for model in model_records:
                row = {**base, **model}
                row.update({"evaluation_status": "input_failure", "runtime_error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=6), "software_json": sw})
                rows.append(row)
            continue
        for model in model_records:
            row = {**base, **model}
            row.update(geom)
            row.update({"evaluation_status": "runtime_failure", "runtime_error": "", "traceback": "",
                        "energy_eV": float("nan"), "energy_per_atom_eV": float("nan"),
                        "forces_finite": False, "max_force_eV_A": float("nan"), "rms_force_eV_A": float("nan"),
                        "p95_force_eV_A": float("nan"), "p99_force_eV_A": float("nan"),
                        "top_force_species": "", "top_force_atom_index": -1,
                        "software_json": sw})
            try:
                model_id = str(model["model_id"])
                if model_id not in calculators:
                    raise RuntimeError(calculator_errors.get(model_id, "calculator initialization failed"))
                atoms.calc = calculators[model_id]
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)
                row.update({"energy_eV": energy, "energy_per_atom_eV": energy / len(atoms),
                            "evaluation_status": "success"})
                row.update(force_summary(forces, symbols))
                if not np.isfinite(energy) or not bool(row["forces_finite"]):
                    row["evaluation_status"] = "nonfinite_output"
                force_arrays.append(forces.astype(np.float32))
                force_candidate_ids.append(str(candidate["candidate_id"]))
                force_model_ids.append(str(model["model_id"]))
            except Exception as exc:
                row.update({"runtime_error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=8)})
            rows.append(row)
    results_path = out / f"{chunk_tag}_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    if force_arrays:
        np.savez_compressed(out / f"{chunk_tag}_forces.npz",
                            forces=np.stack(force_arrays),
                            candidate_ids=np.asarray(force_candidate_ids),
                            model_ids=np.asarray(force_model_ids))
    meta = {"chunk_index": args.chunk_index, "chunk_size": args.chunk_size,
            "row_start": start, "row_stop_exclusive": stop, "n_candidates": len(candidates),
            "n_result_rows": len(rows), "n_force_arrays": len(force_arrays),
            "manifest": str(args.manifest), "models": str(args.models), "device": args.device,
            "software": json.loads(sw), "calculator_errors": calculator_errors,
            "elapsed_s": time.time() - started,
            "status": "COMPLETE" if len(rows) == len(candidates) * len(models) else "PARTIAL"}
    (out / f"{chunk_tag}_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
