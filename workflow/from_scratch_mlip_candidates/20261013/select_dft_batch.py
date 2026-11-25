from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--candidate-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    analysis = Path(args.analysis)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    spread = pd.read_csv(analysis / "candidate_model_spread.csv")
    results = pd.read_csv(analysis / "mlip_candidate_results_all.csv")
    manifest = pd.read_csv(args.candidate_manifest)
    ok = spread[spread["n_models_success"] == 3].copy()
    ok = ok[ok["candidate_group"].notna()].copy()
    if ok.empty:
        raise SystemExit("no all-three-model candidates are available")
    best_model = results[results["evaluation_status"] == "success"].sort_values(["candidate_id", "max_force_eV_A"], ascending=[True, False]).drop_duplicates("candidate_id")
    ok = ok.merge(best_model[["candidate_id", "top_force_atom_index", "top_force_species", "max_force_eV_A", "model_id"]], on="candidate_id", how="left", suffixes=("", "_top_model"))
    ok = ok.merge(manifest[["candidate_id", "prepared_extxyz", "local_extxyz", "remote_extxyz", "source", "source_snapshot", "step", "time_ps", "training_seed", "window", "candidate_group", "local_extxyz_sha256", "min_distance_A", "temperature_K", "force_max_norm_eV_A", "rms_force_eV_A"]], on="candidate_id", how="left", suffixes=("", "_manifest"))
    ok["min_distance_A"] = pd.to_numeric(ok["min_distance_A"], errors="coerce")
    geometry_excluded = ok[~ok["min_distance_A"].ge(0.8)].copy()
    ok = ok[ok["min_distance_A"].ge(0.8)].copy()
    if ok.empty:
        raise SystemExit("no all-three-model candidates pass the DFT-ready minimum-distance gate")
    ok["priority_score"] = pd.to_numeric(ok["ensemble_force_std_rms_eV_A"], errors="coerce").fillna(-1.0)
    ok["pairwise_priority"] = pd.to_numeric(ok["pairwise_force_max_max_eV_A"], errors="coerce").fillna(-1.0)
    quotas = [
        ("threshold_or_terminal", max(1, args.n // 5)),
        ("precursor_near_onset", max(1, args.n * 2 // 5)),
        ("stable_pre_onset", max(1, args.n // 5)),
        ("stable_early_control", max(1, args.n // 10)),
        ("terminal_stable_target", 1),
    ]
    selected = []
    used = set()
    for group, quota in quotas:
        candidates = ok[ok["candidate_group"] == group].sort_values(["priority_score", "pairwise_priority", "candidate_id"], ascending=[False, False, True])
        for row in candidates.to_dict("records"):
            if row["candidate_id"] in used:
                continue
            selected.append(row)
            used.add(row["candidate_id"])
            if sum(x["candidate_group"] == group for x in selected) >= quota:
                break
    if len(selected) < args.n:
        rest = ok[~ok["candidate_id"].isin(used)].sort_values(["priority_score", "pairwise_priority", "candidate_id"], ascending=[False, False, True])
        selected.extend(rest.head(args.n - len(selected)).to_dict("records"))
    selected = selected[:args.n]
    selected_df = pd.DataFrame(selected)
    selected_df.insert(0, "batch_order", range(len(selected_df)))
    selected_df["selection_role"] = selected_df["candidate_group"].map({
        "threshold_or_terminal": "high_force_disagreement_threshold",
        "precursor_near_onset": "high_force_disagreement_precursor",
        "stable_pre_onset": "stable_pre_control",
        "stable_early_control": "stable_early_control",
        "terminal_stable_target": "stable_terminal_control",
    }).fillna("high_force_disagreement_screen")
    selected_df["dft_status"] = "prepared_only_mlip_selected"
    selected_df["selection_basis"] = "all-three-model success; rank by ensemble_force_std_rms_eV_A then pairwise_force_max_max_eV_A within stage quota"
    selected_df.to_csv(output / "dft_batch_selection.csv", index=False)
    meta = {
        "schema": "migrationbench_mlip_driven_dft_selection_v1",
        "status": "PREPARED_ONLY",
        "n_selected": len(selected_df), "requested": args.n,
        "analysis_manifest": str(analysis / "analysis_manifest.json"),
        "analysis_manifest_sha256": sha256(analysis / "analysis_manifest.json"),
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": sha256(Path(args.candidate_manifest)),
        "selection_rules": quotas,
        "groups_selected": selected_df["candidate_group"].value_counts().to_dict(),
        "geometry_gate": "min_distance_A >= 0.8 on the source 2050-atom candidate before PBC-preserving reconstruction",
        "n_geometry_excluded_from_selection": int(len(geometry_excluded)),
        "geometry_excluded_candidate_ids": geometry_excluded["candidate_id"].astype(str).tolist(),
        "no_dft_submitted": True,
        "boundary": "MLIP force disagreement selects informative candidates; it does not establish DFT correctness or causal mechanism",
    }
    (output / "dft_batch_selection_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
