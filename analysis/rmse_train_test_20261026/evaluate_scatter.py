#!/usr/bin/env python3
"""Hash-checked MACE evaluation with full energy/force parity arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    rows = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    row = rows[idx]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {**row, "started": time.time(), "errors": []}
    try:
        observed_model = sha256_file(row["model_path"])
        observed_data = sha256_file(row["data_path"])
        rec.update(
            model_sha256_observed=observed_model,
            data_sha256_observed=observed_data,
            model_sha_ok=observed_model == row["model_sha256"],
            data_sha_ok=observed_data == row["data_sha256"],
        )
        if not rec["model_sha_ok"] or not rec["data_sha_ok"]:
            raise RuntimeError("HASH_MISMATCH")

        from ase.io import read
        from mace.calculators import MACECalculator

        frames = read(row["data_path"], index=":")
        calc = MACECalculator(model_paths=[row["model_path"]], device="cpu", default_dtype="float64")
        energy_ref, energy_pred = [], []
        force_ref, force_pred, force_frame = [], [], []
        frame_n_atoms = []
        frame_ids = []
        for frame_idx, atoms in enumerate(frames):
            e_ref = float(atoms.info.get("energy", atoms.get_potential_energy() if atoms.calc else np.nan))
            f_ref = np.asarray(atoms.arrays["forces"] if "forces" in atoms.arrays else atoms.get_forces(), dtype=np.float64)
            atoms.calc = calc
            e_pred = float(atoms.get_potential_energy())
            f_pred = np.asarray(atoms.get_forces(), dtype=np.float64)
            n = len(atoms)
            energy_ref.append(e_ref / n)
            energy_pred.append(e_pred / n)
            force_ref.append(f_ref.reshape(-1))
            force_pred.append(f_pred.reshape(-1))
            force_frame.extend([frame_idx] * f_ref.size)
            frame_n_atoms.append(n)
            frame_ids.append(str(atoms.info.get("source_trajectory", "NA")))

        e_ref = np.asarray(energy_ref, dtype=np.float64)
        e_pred = np.asarray(energy_pred, dtype=np.float64)
        f_ref = np.concatenate(force_ref).astype(np.float64)
        f_pred = np.concatenate(force_pred).astype(np.float64)
        frame_force_rmse = np.array(
            [np.sqrt(np.mean((a - b) ** 2)) for a, b in zip(force_pred, force_ref)], dtype=np.float64
        )
        finite = bool(np.isfinite(e_ref).all() and np.isfinite(e_pred).all() and np.isfinite(f_ref).all() and np.isfinite(f_pred).all())
        if not finite:
            raise RuntimeError("NONFINITE_PREDICTION_OR_REFERENCE")
        e_err = e_pred - e_ref
        f_err = f_pred - f_ref
        rec["summary"] = {
            "n_frames": int(len(e_ref)),
            "n_force_components": int(len(f_ref)),
            "energy_rmse_meV_per_atom": 1e3 * float(np.sqrt(np.mean(e_err**2))),
            "force_rmse_meV_A": 1e3 * float(np.sqrt(np.mean(f_err**2))),
            "energy_mae_meV_per_atom": 1e3 * float(np.mean(np.abs(e_err))),
            "force_mae_meV_A": 1e3 * float(np.mean(np.abs(f_err))),
            "energy_r2": r2_score(e_ref, e_pred),
            "force_r2": r2_score(f_ref, f_pred),
            "energy_bias_meV_per_atom": 1e3 * float(np.mean(e_err)),
            "force_bias_meV_A": 1e3 * float(np.mean(f_err)),
        }
        np.savez_compressed(
            out / f"{row['eval_id']}__parity.npz",
            energy_ref_eV_per_atom=e_ref.astype(np.float32),
            energy_pred_eV_per_atom=e_pred.astype(np.float32),
            force_ref_eV_A=f_ref.astype(np.float32),
            force_pred_eV_A=f_pred.astype(np.float32),
            force_frame_index=np.asarray(force_frame, dtype=np.int32),
            frame_force_rmse_meV_A=(1e3 * frame_force_rmse).astype(np.float32),
            frame_n_atoms=np.asarray(frame_n_atoms, dtype=np.int32),
        )
        with open(out / f"{row['eval_id']}__per_frame.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "trajectory", "n_atoms", "energy_ref_eV_per_atom", "energy_pred_eV_per_atom", "energy_abs_error_meV_per_atom", "force_rmse_meV_A"])
            for i, (traj, n, fr, fp, frmse) in enumerate(zip(frame_ids, frame_n_atoms, e_ref, e_pred, frame_force_rmse)):
                w.writerow([i, traj, n, fr, fp, 1e3 * abs(fp - fr), 1e3 * frmse])
        rec["status"] = "PARITY_EVAL_OK"
    except Exception as exc:
        rec["status"] = "PARITY_EVAL_FAILED"
        rec["errors"].append(repr(exc))
    rec["finished"] = time.time()
    with open(out / f"{row['eval_id']}.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(rec["status"], row["eval_id"], rec.get("summary", {}), flush=True)
    return 0 if rec["status"] == "PARITY_EVAL_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
