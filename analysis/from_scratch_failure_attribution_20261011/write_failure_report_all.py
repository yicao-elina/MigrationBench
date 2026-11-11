#!/usr/bin/env python3
"""Write evidence-bounded report and manuscript-ready tables for all replicas."""

import argparse
from pathlib import Path

import pandas as pd


def fmt(value, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def trigger_text(group: pd.DataFrame) -> str:
    values = sorted({str(value) for value in group["trigger"].dropna() if str(value) not in {"", "not_observed"}})
    return "; ".join(values) if values else "not observed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = pd.read_csv(args.analysis / "run_summary.csv")
    ledger = pd.read_csv(args.analysis / "replica_ledger.csv") if (args.analysis / "replica_ledger.csv").exists() else pd.DataFrame()
    frame = pd.read_csv(args.analysis / "frame_features.csv") if (args.analysis / "frame_features.csv").exists() else pd.DataFrame()
    initial_audit = pd.read_csv(args.analysis / "initial_state_audit.csv") if (args.analysis / "initial_state_audit.csv").exists() else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# From-scratch failure attribution: audited Results and Discussion",
        "",
        "## Scope and protocol",
        "",
        "This audit preserves the original from-scratch campaign and evaluates exact-protocol reproductions in separate roots. The protocol uses the same 2050-atom fully periodic input, model-family/training-seed pairing, initial momenta, `md_seed=20261005`, 600 K Langevin NVT, 1 fs timestep, friction 0.01 fs^-1, 200-step saved-frame interval, and 1,000,000-step target. The scientific gate is `T > 5000 K` or recorded `max |F| > 100 eV Å^-1` or `d_min < 0.5 Å` or a non-finite value.",
        "",
        "Threshold onset is the first saved frame that crosses a recorded gate. It is not automatically a physical phase transition, and a trajectory stopped at that frame is classified as unstable rather than assigned a diffusion coefficient.",
        "",
        "## Replica ledger",
        "",
    ]
    if not ledger.empty:
        for seed in (123, 234, 345):
            group = ledger[ledger["training_seed"].astype(int) == seed]
            sources = ", ".join(f"{row.source}: {row.acceptance}" for row in group.itertuples())
            lines.append(f"- training seed {seed}: {sources}")
    lines += ["", "## Threshold-onset results", "", "| source | family | training seed | onset (ps) | trigger | max T (K) | max recorded |F| (eV Å^-1) | min distance (Å) |", "|---|---|---:|---:|---|---:|---:|---:|"]
    if not summary.empty:
        for row in summary.sort_values(["training_seed", "source"]).itertuples(index=False):
            lines.append(f"| {row.source} | {row.family} | {int(row.training_seed)} | {fmt(row.onset_ps)} | {row.trigger} | {fmt(row.max_temperature_K)} | {fmt(row.max_force_eV_A)} | {fmt(row.min_distance_A, 3)} |")
    lines += ["", "## Reproducibility and trigger classification", ""]
    if not initial_audit.empty:
        numeric_pass = (initial_audit.get("position_max_abs_A", pd.Series(dtype=float)).fillna(float("inf")) <= 1.0e-12) & (initial_audit.get("momentum_max_abs", pd.Series(dtype=float)).fillna(float("inf")) <= 1.0e-12) & (initial_audit.get("cell_max_abs_A", pd.Series(dtype=float)).fillna(float("inf")) <= 1.0e-12)
        categorical_pass = initial_audit.get("species_equal", pd.Series(False, index=initial_audit.index)).astype(bool) & initial_audit.get("pbc_equal", pd.Series(False, index=initial_audit.index)).astype(bool) & initial_audit.get("natoms_equal", pd.Series(False, index=initial_audit.index)).astype(bool)
        lines.append(f"The initial-state audit passed exact positions/momenta/cell/species/PBC checks for {int((numeric_pass & categorical_pass).sum())}/{len(initial_audit)} available trajectories; this separates onset differences from an input-state mismatch.")
    if summary.empty:
        lines.append("No terminal run summary was available; no scientific conclusion is made.")
    else:
        for seed in (123, 234, 345):
            group = summary[summary["training_seed"].astype(int) == seed]
            if group.empty:
                lines.append(f"- Seed {seed}: no run summary was available.")
                continue
            onset = group["onset_ps"].dropna()
            spread = float(onset.max() - onset.min()) if len(onset) else float("nan")
            family = ", ".join(sorted(set(group["family"].astype(str))))
            lines.append(f"- Seed {seed} ({family}) has {len(group)} audited run(s), onset range {fmt(onset.min())}–{fmt(onset.max())} ps and saved-frame spread {fmt(spread)} ps; observed triggers are `{trigger_text(group)}`.")
        lines += ["", "These rows test repeatability of the model/input/protocol combination. They do not merge the baseline seed-345 model with the two `ablation_lmax_2` model branches as if they were one neural-network parameterization."]

    lines += ["", "## Precursor and configuration-distribution evidence", ""]
    if frame.empty:
        lines.append("Selected-frame features were not available, so precursor and descriptor claims are blocked.")
    else:
        feature_names = [
            "Cr_Te_coord_3p2_mean",
            "Cr_Sb_coord_3p2_mean",
            "CrTe_bond_median_A",
            "CrTe_angle_mean_deg",
            "min_CrTe_A",
            "Cr_velocity_median_A_ps",
            "force_max_norm_eV_A",
            "distance_to_train",
            "distance_to_test",
            "distance_to_neb",
        ]
        summary_rows = []
        for seed in (123, 234, 345):
            group = frame[frame["training_seed"].astype(int) == seed]
            precursor = group[group["window"].isin(["precursor", "threshold"])]
            stable = group[group["window"] == "stable_pre"]
            q99 = float(precursor["outside_train_q99"].astype(str).str.lower().eq("true").mean()) if len(precursor) and "outside_train_q99" in precursor else float("nan")
            dist_train = float(precursor["distance_to_train"].median()) if len(precursor) and "distance_to_train" in precursor else float("nan")
            dist_neb = float(precursor["distance_to_neb"].median()) if len(precursor) and "distance_to_neb" in precursor and precursor["distance_to_neb"].notna().any() else float("nan")
            stable_crte = float(stable["Cr_Te_coord_3p2_mean"].median()) if len(stable) else float("nan")
            precursor_crte = float(precursor["Cr_Te_coord_3p2_mean"].median()) if len(precursor) else float("nan")
            lines.append(f"- Seed {seed}: precursor/threshold selected frames outside the training q99 gate = {fmt(100*q99, 1)}%; median distance to train = {fmt(dist_train, 3)}; median distance to NEB = {fmt(dist_neb, 3)}; Cr–Te coordination median changed from {fmt(stable_crte, 2)} in stable-pre to {fmt(precursor_crte, 2)} near threshold.")
            for feature in feature_names:
                if feature not in group:
                    continue
                for window, values in group.groupby("window"):
                    numeric = pd.to_numeric(values[feature], errors="coerce").dropna()
                    if len(numeric):
                        summary_rows.append({"training_seed": seed, "source": "all", "window": window, "feature": feature, "n": len(numeric), "median": numeric.median(), "q25": numeric.quantile(0.25), "q75": numeric.quantile(0.75)})
            direction_lines = []
            for feature in ("Cr_Te_coord_3p2_mean", "CrTe_bond_median_A", "CrTe_angle_mean_deg", "min_CrTe_A", "Cr_velocity_median_A_ps"):
                if feature not in group:
                    continue
                stable_value = pd.to_numeric(stable[feature], errors="coerce").median()
                precursor_value = pd.to_numeric(precursor[feature], errors="coerce").median()
                if pd.notna(stable_value) and pd.notna(precursor_value):
                    direction_lines.append(f"{feature}: {precursor_value - stable_value:+.3g}")
            if direction_lines:
                lines.append(f"  Observed stable-pre to precursor/threshold median shifts: {'; '.join(direction_lines)}. These are descriptive recurrence checks, not causal effects.")
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(args.output.parent / "window_feature_summary.csv", index=False)
            tex_lines = ["% Median and interquartile ranges from selected stable/precursor/threshold snapshots.", "\\begin{tabular}{lllrrrr}", "seed & window & feature & n & median & q25 & q75 " + "\\\\", "\\hline"]
            for row in summary_rows:
                tex_lines.append(f"{int(row['training_seed'])} & {row['window']} & {row['feature']} & {int(row['n'])} & {fmt(row['median'], 3)} & {fmt(row['q25'], 3)} & {fmt(row['q75'], 3)} " + "\\\\")
            tex_lines.append("\\end{tabular}")
            (args.output.parent / "window_feature_summary.tex").write_text("\n".join(tex_lines) + "\n")
        lines += ["", "The descriptor analysis is a standardized PCA nearest-neighbour diagnostic fitted on training descriptors. A frame outside a training q95/q99 gate is evidence of observed distribution novelty, not calibrated uncertainty and not proof that the training set caused the runaway. NEB proximity is also a geometric comparison, not evidence that the MD trajectory follows a physical NEB mechanism."]

    lines += ["", "## Evidence boundary and limitations", "", "The strongest supported conclusion is limited to whether the same recorded model/input/initial-state/protocol combination repeatedly crosses the operational instability gate. A common precursor may be reported only when the same feature direction is present across independent runs; a final threshold frame alone is insufficient. The threshold frame is saved at the configured 0.2 ps resolution and can combine force, temperature, and short-distance criteria.", "", "No Dxy is reported for unstable branches. HFACF, thermal conductivity, and physical phase-transition claims require separate heat-flux/DFT/convergence evidence. Optional DFT snapshots are prepared as inputs only and are not silently submitted or interpreted here."]
    args.output.write_text("\n".join(lines) + "\n")

    line_break = "\\\\"
    table_lines = [
        "% Auto-generated from run_summary.csv; unstable branches intentionally have no Dxy.",
        "\\begin{tabular}{llrrrrl}",
        "source & family & seed & onset (ps) & trigger & max $T$ (K) & max $|F|$ (eV/\\AA) " + line_break,
        "\\hline",
    ]
    for row in summary.sort_values(["training_seed", "source"]).itertuples(index=False):
        table_lines.append(f"{row.source} & {row.family} & {int(row.training_seed)} & {fmt(row.onset_ps)} & {row.trigger} & {fmt(row.max_temperature_K)} & {fmt(row.max_force_eV_A)} " + line_break)
    table_lines.append("\\end{tabular}")
    (args.output.parent / "from_scratch_failure_summary_table.tex").write_text("\n".join(table_lines) + "\n")

    manuscript = [
        "The exact-protocol from-scratch branches were evaluated with the original periodic 2050-atom input, initial momenta, model/input pairings, and Langevin-NVT protocol. We define onset as the first saved frame crossing the operational temperature, recorded-force, minimum-distance, or finite-value gate. Because the baseline seed-345 model and the two seed-123/234 `ablation_lmax_2` models are distinct parameterizations, recurrence is interpreted within branch rather than across all three rows. Descriptor-space distances and NEB proximity are reported as distribution-novelty diagnostics; without matched DFT forces they do not establish a causal training-data defect or a unique physical tipping point. Unstable branches are reported as unstable — no Dxy reported.",
    ]
    (args.output.parent / "from_scratch_failure_manuscript_paragraph.md").write_text("\n\n".join(manuscript) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
