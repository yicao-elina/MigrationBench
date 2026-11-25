from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result-dir", default="results_production")
    ap.add_argument("--expected-chunks", type=int, default=32)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    result_root = root / args.result_dir
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidate_manifest = pd.read_csv(args.manifest)
    model_registry = pd.read_csv(args.models, sep="\t")
    expected_ids = set(candidate_manifest["candidate_id"].astype(str))
    expected_models = set(model_registry["model_id"].astype(str))
    chunk_manifests = []
    result_frames = []
    missing = []
    for idx in range(args.expected_chunks):
        tag = f"chunk_{idx:04d}"
        mp = result_root / f"{tag}_manifest.json"
        rp = result_root / f"{tag}_results.csv"
        if not mp.is_file() or not rp.is_file():
            missing.append(idx)
            continue
        chunk_manifests.append(json.loads(mp.read_text()))
        result_frames.append(pd.read_csv(rp))
    if missing and not args.allow_partial:
        raise SystemExit(f"missing production chunks: {missing}")
    if not result_frames:
        raise SystemExit("no chunk result CSVs found")
    results = pd.concat(result_frames, ignore_index=True)
    results["candidate_id"] = results["candidate_id"].astype(str)
    results["model_id"] = results["model_id"].astype(str)
    results["candidate_index"] = pd.to_numeric(results["candidate_index"], errors="coerce")
    source_meta = candidate_manifest[["candidate_id", "model_sha256", "training_seed", "window", "step", "time_ps", "source", "run_key", "onset_ps", "trigger", "candidate_group"]].copy()
    source_meta = source_meta.rename(columns={
        "model_sha256": "source_model_sha256", "training_seed": "source_training_seed",
        "window": "source_window", "step": "source_step", "time_ps": "source_time_ps",
        "source": "source_trajectory", "run_key": "source_run_key",
        "onset_ps": "source_onset_ps", "trigger": "source_trigger",
    })
    results = results.merge(source_meta, on="candidate_id", how="left", validate="many_to_one")
    results = results.sort_values(["candidate_index", "model_id"]).reset_index(drop=True)
    results.to_csv(out / "mlip_candidate_results_all.csv", index=False)
    failures = results[results["evaluation_status"] != "success"].copy()
    failures.to_csv(out / "failure_ledger.csv", index=False)

    force_map: dict[tuple[str, str], np.ndarray] = {}
    force_files = []
    for idx in range(args.expected_chunks):
        fp = result_root / f"chunk_{idx:04d}_forces.npz"
        if not fp.is_file():
            continue
        with np.load(fp, allow_pickle=False) as data:
            forces = np.asarray(data["forces"], dtype=float)
            candidates = data["candidate_ids"].astype(str)
            model_ids = data["model_ids"].astype(str)
            for candidate_id, model_id, force in zip(candidates, model_ids, forces):
                force_map[(candidate_id, model_id)] = force
        force_files.append(str(fp))

    candidate_rows = []
    pair_rows = []
    for candidate_id, group in results.groupby("candidate_id", sort=False):
        group = group[group["evaluation_status"] == "success"]
        present = {str(row.model_id): force_map[(candidate_id, str(row.model_id))]
                   for row in group.itertuples() if (candidate_id, str(row.model_id)) in force_map}
        base = {"candidate_id": candidate_id,
                "candidate_index": float(group["candidate_index"].iloc[0]) if len(group) else np.nan,
                "candidate_group": group["candidate_group"].iloc[0] if len(group) else "",
                "window": group["window"].iloc[0] if len(group) else "",
                "source": group["source"].iloc[0] if len(group) else "",
                "training_seed": group["training_seed"].iloc[0] if len(group) else np.nan,
                "n_models_success": len(present)}
        max_forces = pd.to_numeric(group["max_force_eV_A"], errors="coerce").dropna().to_numpy(float)
        energies = pd.to_numeric(group["energy_eV"], errors="coerce").dropna().to_numpy(float)
        base.update({"max_force_min_eV_A": float(np.min(max_forces)) if len(max_forces) else np.nan,
                     "max_force_max_eV_A": float(np.max(max_forces)) if len(max_forces) else np.nan,
                     "max_force_range_eV_A": float(np.ptp(max_forces)) if len(max_forces) else np.nan,
                     "energy_spread_eV": float(np.ptp(energies)) if len(energies) else np.nan})
        if len(present) >= 2:
            stack = np.stack(list(present.values()), axis=0)
            base["ensemble_force_std_rms_eV_A"] = float(np.sqrt(np.mean(np.var(stack, axis=0))))
            base["ensemble_force_max_std_eV_A"] = float(np.max(np.std(stack, axis=0)))
            pairwise = []
            for left, left_force in present.items():
                for right, right_force in present.items():
                    if left >= right:
                        continue
                    diff = left_force - right_force
                    pairwise.append((left, right, float(np.sqrt(np.mean(diff * diff))),
                                     float(np.max(np.linalg.norm(diff, axis=1)))))
            base["pairwise_force_rms_max_eV_A"] = max(x[2] for x in pairwise)
            base["pairwise_force_max_max_eV_A"] = max(x[3] for x in pairwise)
            base["pairwise_force_rms_median_eV_A"] = float(np.median([x[2] for x in pairwise]))
            for left, right, rms, max_diff in pairwise:
                pair_rows.append({**base, "model_left": left, "model_right": right,
                                  "force_rms_difference_eV_A": rms,
                                  "force_max_difference_eV_A": max_diff})
        else:
            base.update({"ensemble_force_std_rms_eV_A": np.nan, "ensemble_force_max_std_eV_A": np.nan,
                         "pairwise_force_rms_max_eV_A": np.nan, "pairwise_force_max_max_eV_A": np.nan,
                         "pairwise_force_rms_median_eV_A": np.nan})
        candidate_rows.append(base)
    candidate_summary = pd.DataFrame(candidate_rows)
    candidate_summary.to_csv(out / "candidate_model_spread.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "pairwise_force_disagreement.csv", index=False)

    numeric = ["max_force_eV_A", "rms_force_eV_A", "p95_force_eV_A", "p99_force_eV_A",
               "energy_per_atom_eV"]
    model_group_rows = []
    for (candidate_group, window, model_id), group in results.groupby(["candidate_group", "window", "model_id"], dropna=False):
        successes = group[group["evaluation_status"] == "success"]
        row = {"candidate_group": candidate_group, "window": window, "model_id": model_id,
               "model_label": group["model_label"].iloc[0], "model_family": group["model_family"].iloc[0],
               "training_seed": group["training_seed"].iloc[0], "n_candidates": len(group),
               "n_success": len(successes), "success_fraction": len(successes) / len(group)}
        for column in numeric:
            row[f"{column}_median"] = quantile(successes[column], 0.50)
            row[f"{column}_q25"] = quantile(successes[column], 0.25)
            row[f"{column}_q75"] = quantile(successes[column], 0.75)
            row[f"{column}_q95"] = quantile(successes[column], 0.95)
        model_group_rows.append(row)
    pd.DataFrame(model_group_rows).to_csv(out / "model_stage_summary.csv", index=False)

    if not candidate_summary.empty:
        stage = candidate_summary.groupby(["candidate_group", "window"], dropna=False).agg(
            n_candidates=("candidate_id", "nunique"),
            n_all_three_models=("n_models_success", lambda x: int(np.sum(np.asarray(x) == 3))),
            max_force_range_median_eV_A=("max_force_range_eV_A", "median"),
            max_force_range_q95_eV_A=("max_force_range_eV_A", lambda x: quantile(x, .95)),
            energy_spread_median_eV=("energy_spread_eV", "median"),
            ensemble_force_spread_median_eV_A=("ensemble_force_std_rms_eV_A", "median"),
            ensemble_force_spread_q95_eV_A=("ensemble_force_std_rms_eV_A", lambda x: quantile(x, .95)),
            pairwise_force_rms_q95_eV_A=("pairwise_force_rms_max_eV_A", lambda x: quantile(x, .95)),
        ).reset_index()
    else:
        stage = pd.DataFrame()
    stage.to_csv(out / "candidate_group_summary.csv", index=False)

    expected_rows = len(candidate_manifest) * len(model_registry)
    actual_pairs = results[["candidate_id", "model_id"]].drop_duplicates()
    duplicate_pairs = len(results) - len(actual_pairs)
    checks = {
        "status": "VERIFIED" if not missing and len(results) == expected_rows and duplicate_pairs == 0 else "PROVENANCE_PARTIAL",
        "root": str(root), "candidate_manifest": str(args.manifest), "candidate_manifest_sha256": sha256(Path(args.manifest)),
        "model_registry": str(args.models), "model_registry_sha256": sha256(Path(args.models)),
        "n_candidates_expected": len(candidate_manifest), "n_models_expected": len(model_registry),
        "n_rows_expected": expected_rows, "n_rows_observed": len(results), "duplicate_candidate_model_rows": duplicate_pairs,
        "missing_chunks": missing, "chunk_manifests": chunk_manifests, "force_files": force_files,
        "candidate_ids_missing": sorted(expected_ids - set(results["candidate_id"])),
        "model_ids_missing": sorted(expected_models - set(results["model_id"])),
        "input_hashes_observed": sorted(set(results["input_sha256"].dropna().astype(str))),
        "model_hashes_observed": sorted(set(results["model_sha256"].dropna().astype(str))),
        "software": {"hostname": socket.gethostname(), "platform": platform.platform()},
    }
    (out / "analysis_manifest.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    print(json.dumps({k: checks[k] for k in ["status", "n_rows_expected", "n_rows_observed", "missing_chunks", "duplicate_candidate_model_rows"]}, indent=2))


if __name__ == "__main__":
    main()
