from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def q(series: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(quantile)) if len(values) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--canonical-manifest", required=True)
    ap.add_argument("--model-registry", default=str(Path(__file__).with_name("models_rockfish.tsv")))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = Path(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw_results = pd.read_csv(source / "mlip_candidate_results_all.csv")
    canonical = pd.read_csv(args.canonical_manifest)
    model_registry = pd.read_csv(args.model_registry, sep="\t")
    raw_manifest = json.loads((source / "analysis_manifest.json").read_text())

    required = {"candidate_id", "candidate_index", "prepared_extxyz", "candidate_group", "window", "source", "training_seed"}
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise SystemExit(f"canonical manifest missing columns: {missing}")
    if canonical["prepared_extxyz"].duplicated().any():
        raise SystemExit("canonical manifest has duplicate prepared_extxyz paths")

    if "execution_candidate_id" in raw_results:
        exec_ids = raw_results[["execution_candidate_id", "execution_candidate_index", "prepared_extxyz"]].drop_duplicates().rename(columns={"execution_candidate_id": "candidate_id", "execution_candidate_index": "candidate_index"})
    else:
        exec_ids = raw_results[["candidate_id", "candidate_index", "prepared_extxyz"]].drop_duplicates()
    if exec_ids["candidate_id"].duplicated().any() or exec_ids["prepared_extxyz"].duplicated().any():
        raise SystemExit("raw results do not have a one-to-one execution candidate/path mapping")
    mapping = exec_ids.merge(
        canonical[["candidate_id", "candidate_index", "prepared_extxyz", "candidate_group", "window", "source", "training_seed", "source_snapshot", "step", "time_ps", "run_key", "onset_ps", "trigger"]],
        on="prepared_extxyz", how="outer", validate="one_to_one", suffixes=("_execution", "_canonical"), indicator=True,
    )
    if not bool((mapping["_merge"] == "both").all()):
        raise SystemExit("execution/canonical source-frame mapping is not complete")
    mapping = mapping.drop(columns=["_merge"])
    mapping.to_csv(output / "candidate_execution_mapping.csv", index=False)

    map_cols = ["prepared_extxyz", "candidate_id_canonical", "candidate_index_canonical", "candidate_group", "window_canonical", "source_canonical", "training_seed_canonical", "source_snapshot", "step_canonical", "time_ps_canonical", "run_key", "onset_ps", "trigger"]
    mapping = mapping.rename(columns={
        "candidate_id": "candidate_id_canonical", "candidate_index": "candidate_index_canonical",
        "window": "window_canonical", "source": "source_canonical", "training_seed": "training_seed_canonical",
        "step": "step_canonical", "time_ps": "time_ps_canonical",
    })
    if "execution_candidate_id" not in raw_results:
        raw_results = raw_results.rename(columns={"candidate_id": "execution_candidate_id", "candidate_index": "execution_candidate_index"})
    if "candidate_group" in raw_results:
        raw_results = raw_results.rename(columns={"candidate_group": "candidate_group_execution"})
    results = raw_results.merge(mapping[map_cols], on="prepared_extxyz", how="left", validate="many_to_one")
    if results["candidate_id_canonical"].isna().any():
        raise SystemExit("unmatched canonical candidate IDs after result merge")
    results["candidate_id"] = results["candidate_id_canonical"]
    results["candidate_index"] = results["candidate_index_canonical"]
    results["candidate_group"] = results["candidate_group"]
    results["window"] = results["window_canonical"]
    results["source"] = results["source_canonical"]
    results["training_seed"] = results["training_seed_canonical"]
    results["source_training_seed"] = results["training_seed_canonical"]
    results = results.drop(columns=["candidate_id_canonical", "candidate_index_canonical", "window_canonical", "source_canonical", "training_seed_canonical"])
    results = results.merge(model_registry[["model_id", "training_seed"]].rename(columns={"training_seed": "model_training_seed"}), on="model_id", how="left", validate="many_to_one")
    if results["model_training_seed"].isna().any():
        raise SystemExit("missing model training seed in model registry merge")
    results = results.sort_values(["candidate_index", "model_id"]).reset_index(drop=True)
    results.to_csv(output / "mlip_candidate_results_all.csv", index=False)
    results[results["evaluation_status"] != "success"].to_csv(output / "failure_ledger.csv", index=False)

    def remap_compact(name: str) -> pd.DataFrame:
        frame = pd.read_csv(source / name)
        if "execution_candidate_id" not in frame:
            frame = frame.rename(columns={"candidate_id": "execution_candidate_id"})
        if "candidate_group" in frame:
            frame = frame.rename(columns={"candidate_group": "candidate_group_execution"})
        frame = frame.merge(mapping[["candidate_id_execution", "candidate_id_canonical", "candidate_index_canonical", "candidate_group", "window_canonical", "source_canonical", "training_seed_canonical"]], left_on="execution_candidate_id", right_on="candidate_id_execution", how="left", validate="one_to_one")
        frame["candidate_id"] = frame["candidate_id_canonical"]
        frame["candidate_index"] = frame["candidate_index_canonical"]
        frame["candidate_group"] = frame["candidate_group"]
        frame["window"] = frame["window_canonical"]
        frame["source"] = frame["source_canonical"]
        frame["training_seed"] = frame["training_seed_canonical"]
        return frame.drop(columns=["candidate_id_execution", "candidate_id_canonical", "candidate_index_canonical", "window_canonical", "source_canonical", "training_seed_canonical"])

    spread = remap_compact("candidate_model_spread.csv")
    spread.to_csv(output / "candidate_model_spread.csv", index=False)
    pairwise = pd.read_csv(source / "pairwise_force_disagreement.csv")
    if "execution_candidate_id" not in pairwise:
        pairwise = pairwise.rename(columns={"candidate_id": "execution_candidate_id"})
    if "candidate_group" in pairwise:
        pairwise = pairwise.rename(columns={"candidate_group": "candidate_group_execution"})
    pairwise = pairwise.merge(mapping[["candidate_id_execution", "candidate_id_canonical", "candidate_index_canonical", "candidate_group", "window_canonical", "source_canonical", "training_seed_canonical"]], left_on="execution_candidate_id", right_on="candidate_id_execution", how="left", validate="many_to_one")
    pairwise["candidate_id"] = pairwise["candidate_id_canonical"]
    pairwise["candidate_index"] = pairwise["candidate_index_canonical"]
    pairwise["candidate_group"] = pairwise["candidate_group"]
    pairwise["window"] = pairwise["window_canonical"]
    pairwise["source"] = pairwise["source_canonical"]
    pairwise["training_seed"] = pairwise["training_seed_canonical"]
    pairwise = pairwise.drop(columns=["candidate_id_execution", "candidate_id_canonical", "candidate_index_canonical", "window_canonical", "source_canonical", "training_seed_canonical"])
    pairwise.to_csv(output / "pairwise_force_disagreement.csv", index=False)

    numeric = ["max_force_eV_A", "rms_force_eV_A", "p95_force_eV_A", "p99_force_eV_A", "energy_per_atom_eV"]
    stage_rows = []
    for (group_name, window, model_id), frame in results.groupby(["candidate_group", "window", "model_id"], dropna=False):
        success = frame[frame["evaluation_status"] == "success"]
        row = {"candidate_group": group_name, "window": window, "model_id": model_id,
               "model_label": frame["model_label"].iloc[0], "model_family": frame["model_family"].iloc[0],
               "source_training_seed": ";".join(sorted(set(frame["source_training_seed"].astype(str)))),
               "model_training_seed": frame["model_training_seed"].iloc[0], "n_candidates": len(frame),
               "n_success": len(success), "success_fraction": len(success) / len(frame)}
        for col in numeric:
            row[f"{col}_median"] = q(success[col], .50)
            row[f"{col}_q25"] = q(success[col], .25)
            row[f"{col}_q75"] = q(success[col], .75)
            row[f"{col}_q95"] = q(success[col], .95)
        stage_rows.append(row)
    stage = pd.DataFrame(stage_rows)
    stage.to_csv(output / "model_stage_summary.csv", index=False)
    group_summary = spread.groupby(["candidate_group", "window"], dropna=False).agg(
        n_candidates=("candidate_id", "nunique"),
        n_all_three_models=("n_models_success", lambda x: int(np.sum(np.asarray(x) == 3))),
        max_force_range_median_eV_A=("max_force_range_eV_A", "median"),
        max_force_range_q95_eV_A=("max_force_range_eV_A", lambda x: q(x, .95)),
        energy_spread_median_eV=("energy_spread_eV", "median"),
        ensemble_force_spread_median_eV_A=("ensemble_force_std_rms_eV_A", "median"),
        ensemble_force_spread_q95_eV_A=("ensemble_force_std_rms_eV_A", lambda x: q(x, .95)),
        pairwise_force_rms_q95_eV_A=("pairwise_force_rms_max_eV_A", lambda x: q(x, .95)),
    ).reset_index()
    group_summary.to_csv(output / "candidate_group_summary.csv", index=False)

    expected_ids = set(canonical["candidate_id"].astype(str))
    actual_ids = set(results["candidate_id"].astype(str))
    pairs = results[["candidate_id", "model_id"]].drop_duplicates()
    manifest = {
        "schema": "migrationbench_mlip_candidate_analysis_reconciled_v1",
        "status": "VERIFIED" if len(mapping) == 639 and expected_ids == actual_ids and len(results) == 1917 and len(pairs) == len(results) and (results["evaluation_status"] == "success").all() else "PROVENANCE_PARTIAL",
        "raw_collector_analysis_manifest_sha256": sha256(source / "analysis_manifest.json"),
        "canonical_candidate_manifest": str(Path(args.canonical_manifest)),
        "canonical_candidate_manifest_sha256": sha256(Path(args.canonical_manifest)),
        "model_registry": str(Path(args.model_registry)),
        "model_registry_sha256": sha256(Path(args.model_registry)),
        "execution_to_canonical_mapping": str(output / "candidate_execution_mapping.csv"),
        "n_execution_candidates": int(len(mapping)), "n_canonical_candidates": int(len(canonical)),
        "n_candidates_expected": 639, "n_models_expected": 3, "n_rows_expected": 1917, "n_rows_observed": int(len(results)),
        "duplicate_candidate_model_rows": int(len(results) - len(pairs)),
        "candidate_ids_missing": sorted(expected_ids - actual_ids), "candidate_ids_unexpected": sorted(actual_ids - expected_ids),
        "execution_ids_unique": int(mapping["candidate_id_execution"].nunique()),
        "canonical_ids_unique": int(mapping["candidate_id_canonical"].nunique()),
        "source_paths_one_to_one": True,
        "evaluation_status_counts": results["evaluation_status"].value_counts(dropna=False).to_dict(),
        "input_hashes_observed": sorted(set(results["input_sha256"].dropna().astype(str))),
        "model_hashes_observed": sorted(set(results["model_sha256"].dropna().astype(str))),
        "scientific_boundary": "candidate IDs were reconciled by immutable prepared_extxyz source path; static MLIP force spread is descriptive and is not DFT correctness or causal proof",
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ["status", "n_execution_candidates", "n_canonical_candidates", "n_rows_expected", "n_rows_observed", "duplicate_candidate_model_rows", "candidate_ids_missing"]}, indent=2))


if __name__ == "__main__":
    main()
