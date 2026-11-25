#!/usr/bin/env python3
"""Build the four-tier, two-PES task-conditioned distribution atlas.

The figure is deliberately an atlas rather than a single scalar OOD ranking.
The upper row contains DFT/reference-PES evidence where it exists; the lower
row contains MLIP-generated observables, model error, or committee spread.
Fixed-path and self-relaxed NEB are kept as separate operators. Failure/CAR
rows without a full-cell DFT label are shown as MLIP-only diagnostics.

The script writes compact derived tables, a source inventory, a claim-boundary
matrix, and PDF/PNG/SVG renders. It does not modify source data or submit jobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OTHER = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TABLES = OUT / "tables"
DERIVED = OUT / "derived"

PLOT_ATLAS = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_ATLAS) not in sys.path:
    sys.path.insert(0, str(PLOT_ATLAS))
from plot_atlas import apply_publication_style, get_palette  # noqa: E402


PAL = get_palette("jhu_mace_models")
MODEL_COLORS = {
    "FT-600K": PAL["roles"]["ft_600k"],
    "FT-MultiT": PAL["roles"]["ft_multit"],
    "Foundation-OMAT": PAL["roles"]["foundation_omat"],
    "From-scratch": PAL["roles"]["from_scratch"],
    "Scratch-5%": "#F3B35D",
}
TASK_COLORS = {
    "train": "#A9ADB3",
    "valid": "#5E6C7C",
    "test": "#2D638E",
    "neb": "#7A3E8E",
    "DFT recovery": "#3D3D3D",
}
INK = "#202020"
LIGHT = "#F5F6F7"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_paths() -> dict[str, Path]:
    return {
        "historical_reference_embedding": OTHER / "analysis/multimodel_639_screen_20261014/inputs/reference_embedding.csv",
        "historical_failure_features": OTHER / "analysis/multimodel_639_screen_20261014/inputs/frame_features.csv",
        "historical_descriptor_manifest": OTHER / "analysis/multimodel_639_screen_20261014/inputs/diagnostic_manifest.json",
        "grouped_rmse": REPO / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/grouped_seed_comparison.csv",
        "grouped_overlap_full": REPO / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/overlap_full_leakage_audit.json",
        "grouped_overlap_scratch5": REPO / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/overlap_scratch5_leakage_audit.json",
        "md_results": OTHER / "analysis/mace_1ns_merged_20261010/release/merged_results.csv",
        "md_uncertainty": OTHER / "analysis/mace_1ns_merged_20261010/release/replica_uncertainty.csv",
        "md_failure_ledger": OTHER / "analysis/mace_1ns_merged_20261016/FAILURE_EXCLUSION.tsv",
        "neb_matrix": REPO / "analysis/neb_protocol_matrix_20261018/derived/neb_matrix_status.csv",
        "neb_barrier_atlas": REPO / "analysis/neb_protocol_matrix_20261018/raw/barrier_atlas.csv",
        "dft_recovery": REPO / "analysis/phase3_dft_recovery_20261004/results_reconstructed/dft_vs_mace_within_method.csv",
        "dft_recovery_manifest": REPO / "analysis/phase3_dft_recovery_20261004/qe_inputs_force/reconstruction_manifest.json",
        "failure_mlip_results": OTHER / "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/results_reconciled/mlip_candidate_results_all.csv",
        "failure_mlip_spread": OTHER / "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/results_reconciled/candidate_model_spread.csv",
        "failure_stage_summary": OTHER / "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/results_reconciled/model_stage_summary.csv",
        "endpoint19_mace_dft": OTHER / "analysis/discovery_dft_audit_20261016/mace_dft_endpoint_19_20261017/MACE_DFT_ENDPOINT_19_COMPARISON.tsv",
        "endpoint19_status": OTHER / "analysis/discovery_dft_audit_20261016/mace_dft_endpoint_19_20261017/MACE_ENDPOINT_19_STATUS.tsv",
        "endpoint47_audit": OTHER / "analysis/discovery_dft_audit_20261016/DISCOVERY_PROGRESS_AUDIT_20261018.md",
        "singlepoint_dft_results": OTHER / "analysis/dft_single_point_validation_20261001/dft_single_point_results.csv",
    }


def require_sources(paths: dict[str, Path]) -> None:
    missing = [f"{k}: {p}" for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing sources:\n" + "\n".join(missing))


def style_axis(ax, *, xlabel: str | None = None, ylabel: str | None = None,
               xsize: float = 11.5, ysize: float = 11.5) -> None:
    ax.grid(False)
    ax.tick_params(direction="out", width=1.05, length=4.4, labelsize=11.0)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.05)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xsize, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ysize, fontweight="bold")


def panel_label(ax, text: str) -> None:
    ax.text(-0.13, 1.04, text, transform=ax.transAxes, fontsize=15,
            fontweight="bold", ha="left", va="bottom", color=INK)


def load(paths: dict[str, Path]) -> dict[str, pd.DataFrame | dict]:
    data: dict[str, pd.DataFrame | dict] = {}
    data["reference"] = pd.read_csv(paths["historical_reference_embedding"])
    data["failure_features"] = pd.read_csv(paths["historical_failure_features"])
    data["descriptor_manifest"] = json.loads(paths["historical_descriptor_manifest"].read_text())
    data["rmse"] = pd.read_csv(paths["grouped_rmse"])
    data["md"] = pd.read_csv(paths["md_results"])
    data["md_uncertainty"] = pd.read_csv(paths["md_uncertainty"])
    data["md_failure"] = pd.read_csv(paths["md_failure_ledger"], sep="\t")
    data["neb"] = pd.read_csv(paths["neb_matrix"])
    data["dft_recovery"] = pd.read_csv(paths["dft_recovery"])
    data["failure_results"] = pd.read_csv(paths["failure_mlip_results"])
    data["failure_spread"] = pd.read_csv(paths["failure_mlip_spread"])
    data["failure_stage"] = pd.read_csv(paths["failure_stage_summary"])
    data["endpoint19"] = pd.read_csv(paths["endpoint19_mace_dft"], sep="\t")
    data["endpoint19_status"] = pd.read_csv(paths["endpoint19_status"], sep="\t")
    data["singlepoint_dft"] = pd.read_csv(paths["singlepoint_dft_results"])
    return data


def add_reference_distances(reference: pd.DataFrame) -> pd.DataFrame:
    out = reference.copy()
    train = out.loc[out["reference"].eq("train"), ["pca1", "pca2"]].to_numpy(float)
    points = out[["pca1", "pca2"]].to_numpy(float)
    out["pca_distance_to_train"] = np.array([
        np.sqrt(((train - point) ** 2).sum(axis=1)).min() for point in points
    ])
    return out


def build_long_tables(data: dict[str, pd.DataFrame | dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    ref = data["reference"]
    assert isinstance(ref, pd.DataFrame)
    for _, r in ref.iterrows():
        rows.append({
            "task_layer": "static" if r["reference"] in {"train", "valid", "test"} else "fixed_path_NEB",
            "lane": "DFT_reference",
            "evidence_id": f"historical_reference_{r['reference']}_{int(r['index'])}",
            "group": r["reference"],
            "model_family": "",
            "training_seed": "",
            "protocol": "historical_SOAP_PCA",
            "x": r["pca1"], "y": r["pca2"],
            "error": np.nan, "uncertainty": np.nan,
            "status": "DESCRIPTIVE_HISTORICAL_COMMON_SPACE",
            "source": str(Path(data_source_path(data, "reference"))),
        })
    ff = data["failure_features"]
    assert isinstance(ff, pd.DataFrame)
    for i, r in ff.iterrows():
        rows.append({
            "task_layer": "failure_CAR",
            "lane": "MLIP_generated",
            "evidence_id": f"failure_frame_{i}",
            "group": r.get("window", "unknown"),
            "model_family": str(r.get("family", "")),
            "training_seed": r.get("training_seed", ""),
            "protocol": "stage_balanced_639_frame_diagnostic",
            "x": r.get("distance_to_train", np.nan), "y": r.get("max_force_eV_A", np.nan),
            "error": np.nan, "uncertainty": np.nan,
            "status": "DESCRIPTIVE_MLIP_ONLY",
            "source": str(data_source_path(data, "failure_features")),
        })
    rmse = data["rmse"]
    assert isinstance(rmse, pd.DataFrame)
    for _, r in rmse.iterrows():
        rows.append({
            "task_layer": "static_RMSE",
            "lane": "MLIP_on_DFT_reference",
            "evidence_id": f"{r['family']}_{int(r['seed'])}",
            "group": r["family"], "model_family": r["family"],
            "training_seed": r["seed"], "protocol": "grouped_split",
            "x": r["energy_rmse_mev_atom"], "y": r["force_rmse_mev_A"],
            "error": r["pathway_1_4_abs_error_eV"], "uncertainty": np.nan,
            "status": "VERIFIED_GROUPED_MULTI_SEED" if r["family"] != "MPA0" else "DESCRIPTIVE_SINGLE_SEED",
            "source": str(data_source_path(data, "rmse")),
        })
    md = data["md"]
    assert isinstance(md, pd.DataFrame)
    for _, r in md.iterrows():
        rows.append({
            "task_layer": "equilibrium_MD",
            "lane": "MLIP_generated",
            "evidence_id": str(r["model"]),
            "group": r["stability"], "model_family": r["family"],
            "training_seed": r.get("training_seed", ""), "protocol": "600K_1fs_2050atom",
            "x": r.get("time_ps", np.nan), "y": r.get("min_distance_A", np.nan),
            "error": np.nan, "uncertainty": np.nan,
            "status": "VERIFIED_STABLE" if "stable" in str(r["stability"]) else "FAILURE_MODE",
            "source": str(data_source_path(data, "md")),
        })
    neb = data["neb"]
    assert isinstance(neb, pd.DataFrame)
    for _, r in neb.iterrows():
        rows.append({
            "task_layer": "NEB",
            "lane": "MLIP_on_DFT_path" if r["protocol"] == "fixed_path" else "MLIP_self_relaxed",
            "evidence_id": f"{r['protocol']}_{r['cell_atoms']}_{r['path_id']}_{r['model_label']}",
            "group": r["path_id"], "model_family": r["family"],
            "training_seed": r.get("seed", ""), "protocol": r["protocol"],
            "x": r["barrier_eV"], "y": abs(r["barrier_error_eV"]),
            "error": abs(r["barrier_error_eV"]), "uncertainty": np.nan,
            "status": r["publication_status"], "source": str(data_source_path(data, "neb")),
        })
    rec = data["dft_recovery"]
    assert isinstance(rec, pd.DataFrame)
    for _, r in rec.iterrows():
        rows.append({
            "task_layer": "failure_CAR",
            "lane": "MLIP_on_DFT_reconstruction",
            "evidence_id": f"{r['case']}_{r['model']}",
            "group": r["case"], "model_family": r["model"], "training_seed": "",
            "protocol": "81atom_local_reconstruction",
            "x": r["dft_max_force_eV_A"], "y": r["force_rmse_eV_A"],
            "error": r["force_rmse_eV_A"], "uncertainty": np.nan,
            "status": r["scope"], "source": str(data_source_path(data, "dft_recovery")),
        })
    long = pd.DataFrame(rows)
    summary_rows = []
    summary_rows += [{"task_layer": "static", "lane": "DFT_reference", "n": int((ref["reference"].isin(["train", "valid", "test"])).sum()), "status": "DESCRIPTIVE_HISTORICAL_COMMON_SPACE", "note": "250 train, 122 valid, 124 test"}]
    summary_rows += [{"task_layer": "equilibrium_MD", "lane": "DFT_reference", "n": int((ref["reference"].isin(["valid", "test"])).sum()), "status": "DESCRIPTIVE_HELD_OUT_AIMD_REFERENCE", "note": "DFT labels are held-out AIMD frames, not the long MLIP trajectories"}]
    summary_rows += [{"task_layer": "fixed_path_NEB", "lane": "DFT_reference", "n": int((ref["reference"] == "neb").sum()), "status": "DESCRIPTIVE_PROVISIONAL_PATH_REFERENCE", "note": "25 historical fixed-path images"}]
    summary_rows += [{"task_layer": "failure_CAR", "lane": "DFT_reference", "n": int(rec["case"].nunique()), "status": "RECONSTRUCTION", "note": "four 81-atom local-cell DFT recovery cases; not full-cell validation"}]
    summary_rows += [{"task_layer": "static_RMSE", "lane": "MLIP_on_DFT_reference", "n": len(rmse), "status": "VERIFIED_GROUPED_MULTI_SEED", "note": "three FT-600K, three Scratch, three Scratch-5%, one MPA0"}]
    summary_rows += [{"task_layer": "equilibrium_MD", "lane": "MLIP_generated", "n": len(md), "status": "VERIFIED_PLUS_FAILURE_MODE", "note": "10 audited rows; common MD velocity seed is not an independent replica"}]
    summary_rows += [{"task_layer": "NEB", "lane": "MLIP_fixed_and_self_relaxed", "n": len(neb), "status": "PROTOCOL_SEPARATED_PROVISIONAL", "note": "100 fixed-path finite rows; 40 self-relaxed rows, with convergence statuses retained"}]
    summary_rows += [{"task_layer": "failure_CAR", "lane": "MLIP_static_committee", "n": len(data["failure_results"]), "status": "VERIFIED_STATIC_DIAGNOSTIC", "note": "639 structures x 3 checkpoints; disagreement is not calibrated uncertainty"}]
    summary = pd.DataFrame(summary_rows)
    return long, summary


def data_source_path(data: dict, key: str) -> Path:
    # Filled by main through a private map; kept separate to make table code readable.
    aliases = {
        "reference": "historical_reference_embedding",
        "failure_features": "historical_failure_features",
        "rmse": "grouped_rmse",
        "md": "md_results",
        "neb": "neb_matrix",
        "dft_recovery": "dft_recovery",
    }
    return data["_source_paths"][aliases.get(key, key)]


def make_failure_stage_summary(spread: pd.DataFrame) -> pd.DataFrame:
    d = spread.copy()
    stage = d.get("window", pd.Series(index=d.index, dtype=object)).astype(str)
    d["stage"] = stage.map({
        "stable_early": "early stable",
        "stable_pre": "pre-onset",
        "precursor": "precursor",
        "threshold": "threshold",
    }).fillna(stage)
    value = "pairwise_force_rms_median_eV_A"
    if value not in d:
        value = "ensemble_force_std_rms_eV_A"
    out = d.groupby("stage", as_index=False)[value].median().rename(columns={value: "median_disagreement_eV_A"})
    order = ["early stable", "pre-onset", "precursor", "threshold"]
    out["order"] = out["stage"].map({s: i for i, s in enumerate(order)})
    return out.sort_values("order").drop(columns="order")


def draw_figure(data: dict[str, pd.DataFrame | dict], ref: pd.DataFrame, stage_summary: pd.DataFrame) -> Path:
    apply_publication_style(font_size=12)
    fig, axes = plt.subplots(2, 4, figsize=(16.2, 8.9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    # Upper row: DFT/reference PES lane.
    # A: static DFT split in the historical common SOAP/PCA view.
    ax = axes[0, 0]
    for group in ["train", "valid", "test"]:
        part = ref.loc[ref["reference"].eq(group)]
        ax.scatter(part["pca1"], part["pca2"], s=20 if group != "train" else 15,
                   color=TASK_COLORS[group], alpha=0.62, edgecolor="white", linewidth=0.25,
                   rasterized=True, label=group)
    ax.set_xlim(-4.0, 3.2); ax.set_ylim(-4.2, 2.0)
    ax.set_xlabel("SOAP PCA 1"); ax.set_ylabel("SOAP PCA 2")
    ax.legend(frameon=True, facecolor="white", edgecolor="#C8C8C8", fontsize=10.2,
              loc="upper right", borderpad=0.35)
    ax.set_title("Static / RMSE", fontsize=12.5, fontweight="bold", pad=7)
    ax.text(0.03, 0.05, "DFT-labeled\ntrain 250 · valid 122 · test 124",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK,
            ha="left", va="bottom")
    style_axis(ax); panel_label(ax, "a")

    # B: near-equilibrium DFT reference distribution by distance to the train envelope.
    ax = axes[0, 1]
    held = ref.loc[ref["reference"].isin(["valid", "test"])].copy()
    groups = ["valid", "test"]
    vals = [held.loc[held["reference"].eq(g), "pca_distance_to_train"].to_numpy() for g in groups]
    vp = ax.violinplot(vals, positions=[0, 1], showmeans=False, showmedians=True,
                       showextrema=False, widths=0.65)
    for body, g in zip(vp["bodies"], groups):
        body.set_facecolor(TASK_COLORS[g]); body.set_edgecolor(INK); body.set_alpha(0.70); body.set_linewidth(0.65)
    rng = np.random.default_rng(20261018)
    for i, (g, v) in enumerate(zip(groups, vals)):
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.07, 0.07, len(v)), v,
                   s=11, color=TASK_COLORS[g], edgecolor="white", linewidth=0.2, alpha=0.55, rasterized=True)
    ax.set_xticks([0, 1], ["valid", "test"])
    ax.set_ylabel("nearest distance in retained SOAP/PCA space")
    ax.set_title("Equilibrium / near-equilibrium", fontsize=12.5, fontweight="bold", pad=7)
    ax.text(0.03, 0.95, "DFT reference: held-out AIMD frames",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    style_axis(ax); panel_label(ax, "b")

    # C: fixed-path NEB reference distribution; keep the large PCA separation visible.
    ax = axes[0, 2]
    held_vals = held["pca_distance_to_train"].to_numpy()
    neb_vals = ref.loc[ref["reference"].eq("neb"), "pca_distance_to_train"].to_numpy()
    ax.boxplot([held_vals, neb_vals], positions=[0, 1], widths=0.46, patch_artist=True,
               boxprops={"facecolor": "#E6E8EA", "edgecolor": INK, "linewidth": 1.0},
               medianprops={"color": INK, "linewidth": 1.5},
               whiskerprops={"color": INK, "linewidth": 1.0}, capprops={"color": INK, "linewidth": 1.0},
               flierprops={"marker": "o", "markersize": 3, "markerfacecolor": TASK_COLORS["neb"], "markeredgecolor": "white"})
    ax.scatter(np.full(len(held_vals), 0) + rng.uniform(-0.08, 0.08, len(held_vals)), held_vals,
               color="#777E87", alpha=0.24, s=10, rasterized=True)
    ax.scatter(np.full(len(neb_vals), 1) + rng.uniform(-0.08, 0.08, len(neb_vals)), neb_vals,
               color=TASK_COLORS["neb"], alpha=0.88, edgecolor="white", linewidth=0.25, s=27, rasterized=True)
    ax.set_xticks([0, 1], ["held-out\nAIMD", "fixed-path\nNEB"])
    ax.set_ylabel("distance to DFT training envelope")
    ax.set_title("Fixed-path NEB", fontsize=12.5, fontweight="bold", pad=7)
    ax.text(0.03, 0.95, "25 DFT-path images\nprovisional candidate reference",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    style_axis(ax); panel_label(ax, "c")

    # D: DFT-labelled stress/failure cases are a separate local-cell reconstruction.
    ax = axes[0, 3]
    rec = data["dft_recovery"]
    assert isinstance(rec, pd.DataFrame)
    unique_rec = rec.drop_duplicates("case").copy()
    labels = ["normal", "bounded", "runaway\nFoundation", "runaway\nScratch"]
    order = ["normal_461800_initial", "bounded_multiT_462000_step464000",
             "runaway_foundation_461800_step462120", "runaway_scratch_461800_step462060"]
    values = [float(unique_rec.loc[unique_rec["case"].eq(k), "dft_max_force_eV_A"].iloc[0]) for k in order]
    ax.bar(np.arange(4), values, color="#7E858C", edgecolor=INK, linewidth=0.65, width=0.64)
    ax.scatter(np.arange(4), values, color=TASK_COLORS["DFT recovery"], s=28, zorder=3,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("DFT max force (eV Å$^{-1}$)")
    ax.set_title("Failure / CAR", fontsize=12.5, fontweight="bold", pad=7)
    ax.text(0.03, 0.95, "DFT labels: 4 reconstructed\n81-atom local cells",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    style_axis(ax); panel_label(ax, "d")

    # Lower row: MLIP-generated PES / error / uncertainty lane.
    # A: grouped static RMSE and path-error points.
    ax = axes[1, 0]
    rmse = data["rmse"]
    assert isinstance(rmse, pd.DataFrame)
    markers = {"OMAT-FT": "o", "Scratch": "s", "Scratch-5%": "D", "MPA0": "P"}
    cmap = {"OMAT-FT": MODEL_COLORS["FT-600K"], "Scratch": MODEL_COLORS["From-scratch"],
            "Scratch-5%": MODEL_COLORS["Scratch-5%"], "MPA0": MODEL_COLORS["FT-600K"]}
    for fam, part in rmse.groupby("family", sort=False):
        ax.scatter(part["energy_rmse_mev_atom"], part["force_rmse_mev_A"],
                   s=58, color=cmap.get(fam, "#666666"), marker=markers.get(fam, "o"),
                   edgecolor=INK, linewidth=0.5, alpha=0.92, label=fam)
    ax.set_xlabel("test energy RMSE (meV atom$^{-1}$)")
    ax.set_ylabel("test force RMSE (meV Å$^{-1}$)")
    ax.set_title("Static pointwise accuracy", fontsize=12.5, fontweight="bold", pad=7)
    ax.legend(frameon=True, facecolor="white", edgecolor="#C8C8C8", fontsize=9.7, loc="upper left", borderpad=0.3)
    ax.text(0.98, 0.05, "color = model lineage\nmarker = split/seed scope",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom", fontweight="bold")
    style_axis(ax); panel_label(ax, "e")

    # B: equilibrium stability and uncertainty; one point per audited trajectory.
    ax = axes[1, 1]
    md = data["md"]
    assert isinstance(md, pd.DataFrame)
    fail = data["md_failure"]
    assert isinstance(fail, pd.DataFrame)
    family_y = {"Foundation-OMAT": 3, "FT-MultiT": 2, "FT-600K": 1, "From-scratch": 0}
    rows = []
    for _, r in md.iterrows():
        fam = {"foundation": "Foundation-OMAT", "scratch": "From-scratch", "OMAT-FT": "FT-600K", "Multi-T FT": "FT-MultiT", "foundation representative": "Foundation-OMAT"}.get(str(r["family"]), str(r["family"]))
        if pd.notna(r.get("time_ps")):
            rows.append((fam, float(r["time_ps"]), "stable"))
    for _, r in fail.iterrows():
        fam = str(r["family"])
        rows.append((fam, float(r["first_instability_ps"]), "failure"))
    for fam, x, state in rows:
        if fam not in family_y: continue
        y = family_y[fam] + (0.10 if state == "failure" else -0.10)
        ax.scatter(x, y, s=57 if state == "stable" else 68,
                   color=MODEL_COLORS[fam], marker="o" if state == "stable" else "X",
                   edgecolor=INK if state == "stable" else "white", linewidth=0.55, zorder=4)
    ax.axvline(1000, color="#666666", linestyle=(0, (3, 2)), linewidth=1.0)
    ax.set_xscale("log"); ax.set_xlim(1.0, 1300); ax.set_ylim(-0.6, 3.6)
    ax.set_yticks(list(family_y.values()), list(family_y.keys()))
    ax.set_xlabel("last stable / terminal time (ps)")
    ax.set_title("Equilibrium MD stability", fontsize=12.5, fontweight="bold", pad=7)
    ax.text(0.03, 0.05, "filled circle = stable release\n× = failure-mode branch",
            transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    inset = ax.inset_axes([0.57, 0.06, 0.39, 0.36])
    unc = data["md_uncertainty"]
    assert isinstance(unc, pd.DataFrame)
    unc = unc.loc[unc["window"].eq("500_900_ps")].copy()
    ux = np.arange(len(unc))
    inset.errorbar(ux, unc["mean_D_cm2_s"] * 1e6, yerr=unc["sd_D_cm2_s"] * 1e6,
                   fmt="o", color=MODEL_COLORS["FT-600K"], ecolor="#4E5660", capsize=2.2,
                   markersize=4.5, markeredgecolor=INK, linewidth=0.85)
    inset.set_xticks(ux, ["FT", "MultiT"], fontsize=8.7, fontweight="bold")
    inset.set_ylabel(r"$D_{xy}$ ($10^{-6}$ cm$^2$ s$^{-1}$)", fontsize=8.8, fontweight="bold")
    inset.tick_params(labelsize=8.5, width=0.8, length=2.5)
    inset.set_title("seed spread", fontsize=9.3, fontweight="bold", pad=2)
    for spine in inset.spines.values(): spine.set_linewidth(0.8)
    style_axis(ax); panel_label(ax, "f")

    # C: fixed-path and self-relaxed NEB are separated by marker and x-offset.
    ax = axes[1, 2]
    neb = data["neb"]
    assert isinstance(neb, pd.DataFrame)
    fam_order = ["Foundation", "FT-600K", "FT-MultiT", "Scratch-MultiT"]
    fam_display = {"Foundation": "Foundation", "FT-600K": "FT-600K", "FT-MultiT": "FT-MultiT", "Scratch-MultiT": "Scratch"}
    fam_color = {"Foundation": MODEL_COLORS["Foundation-OMAT"], "FT-600K": MODEL_COLORS["FT-600K"],
                 "FT-MultiT": MODEL_COLORS["FT-MultiT"], "Scratch-MultiT": MODEL_COLORS["From-scratch"]}
    rng2 = np.random.default_rng(12345)
    for i, fam in enumerate(fam_order):
        part = neb.loc[(neb["family"] == fam) & neb["barrier_error_eV"].notna()].copy()
        for protocol, marker, dx in [("fixed_path", "o", -0.13), ("self_relaxed", "^", 0.13)]:
            vals = np.abs(part.loc[part["protocol"].eq(protocol), "barrier_error_eV"].to_numpy(float))
            if not len(vals): continue
            xs = np.full(len(vals), i + dx) + rng2.uniform(-0.035, 0.035, len(vals))
            ax.scatter(xs, vals, s=25 if protocol == "fixed_path" else 31,
                       color=fam_color[fam], marker=marker, edgecolor="white", linewidth=0.30,
                       alpha=0.82, rasterized=True)
            ax.plot([i + dx - 0.09, i + dx + 0.09], [np.median(vals)] * 2,
                    color=INK, linewidth=1.5, zorder=5)
    ax.set_yscale("log"); ax.set_ylim(0.02, 45)
    ax.set_xticks(np.arange(4), [fam_display[f] for f in fam_order])
    ax.set_ylabel("absolute barrier error to provisional DFT (eV)")
    ax.set_title("NEB process fidelity", fontsize=12.5, fontweight="bold", pad=7)
    legend_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#6B7280", markeredgecolor=INK, markersize=6, label="fixed path"),
                      Line2D([0], [0], marker="^", color="none", markerfacecolor="#6B7280", markeredgecolor=INK, markersize=6, label="self-relaxed")]
    ax.legend(handles=legend_handles, frameon=True, facecolor="white", edgecolor="#C8C8C8", fontsize=9.2,
              loc="upper left", borderpad=0.32)
    ax.text(0.98, 0.05, "family colors follow JHU palette\nprotocols are not pooled",
            transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
    style_axis(ax); panel_label(ax, "g")

    # D: DFT-labelled recovery error, with MLIP-only committee disagreement inset.
    ax = axes[1, 3]
    rec = data["dft_recovery"]
    assert isinstance(rec, pd.DataFrame)
    model_map = {"foundation-omat": "Foundation-OMAT", "from-scratch": "From-scratch",
                 "naive-fine-tuning": "FT-600K", "multi_T-fine-tuning": "FT-MultiT"}
    xlabels = ["normal", "bounded", "runaway\nFnd", "runaway\nScr"]
    cases = list(rec["case"].drop_duplicates())
    xmap = {case: i for i, case in enumerate(cases)}
    for model, part in rec.groupby("model", sort=False):
        fam = model_map.get(model, model)
        if fam not in MODEL_COLORS: continue
        ax.scatter([xmap[c] for c in part["case"]], part["force_rmse_eV_A"],
                   s=34, color=MODEL_COLORS[fam], edgecolor=INK, linewidth=0.45,
                   label=fam, alpha=0.90)
    ax.set_xticks(np.arange(len(cases)), xlabels)
    ax.set_ylabel("force RMSE vs DFT (eV Å$^{-1}$)")
    ax.set_title("Failure / CAR diagnostics", fontsize=12.5, fontweight="bold", pad=7)
    # The model-family legend is already explicit in panel e/g; omitting a
    # second legend here keeps the committee-spread inset unobstructed.
    ax.text(0.98, 0.05, "DFT-labeled reconstruction\nnot full-cell validation",
            transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
    inset = ax.inset_axes([0.56, 0.55, 0.40, 0.34])
    if not stage_summary.empty:
        sx = np.arange(len(stage_summary))
        inset.plot(sx, stage_summary["median_disagreement_eV_A"], color="#5A5A5A", linewidth=1.4, marker="o", markersize=4.2)
        inset.set_xticks(sx, ["early", "pre", "prec.", "thr."][:len(sx)], rotation=0, fontsize=8.0, fontweight="bold")
        inset.set_ylabel("committee force spread", fontsize=8.3, fontweight="bold")
        inset.tick_params(labelsize=8.0, width=0.7, length=2.2)
        inset.set_title("MLIP-only 639-frame diagnostic", fontsize=8.5, fontweight="bold", pad=2)
        inset.grid(False)
        for spine in inset.spines.values(): spine.set_linewidth(0.75)
    style_axis(ax); panel_label(ax, "h")

    fig.text(0.5, 1.015, "DFT/reference potential-energy surface", ha="center", va="bottom",
             fontsize=13.5, fontweight="bold", color="#3D3D3D")
    fig.text(0.5, 0.485, "MLIP-generated potential-energy surface and task observables", ha="center", va="bottom",
             fontsize=13.5, fontweight="bold", color="#3D3D3D")
    fig.text(0.5, -0.015,
             "Configuration-space panels use a retained historical SOAP/PCA fit for descriptive comparison; error and uncertainty are shown only where the corresponding label or paired-model diagnostic exists.",
             ha="center", va="top", fontsize=10.5, color="#3D3D3D")
    for ext, dpi in [("pdf", 300), ("png", 450), ("svg", 300)]:
        path = FIG / f"fig_task_conditioned_distribution_atlas.{ext}"
        fig.savefig(path, format=ext, dpi=dpi, bbox_inches="tight", facecolor="white",
                    metadata={"Creator": "Plot Atlas / MigrationBench evidence atlas"})
    plt.close(fig)
    return FIG / "fig_task_conditioned_distribution_atlas.pdf"


def write_inventory(paths: dict[str, Path], data: dict[str, pd.DataFrame | dict]) -> None:
    rows = []
    for key, path in paths.items():
        if not path.exists():
            status = "MISSING"
            digest = ""
            size = ""
        else:
            status = "PRESENT"
            digest = sha256(path) if path.is_file() else "directory"
            size = path.stat().st_size if path.is_file() else ""
        rows.append({"source_id": key, "path": str(path), "sha256": digest, "bytes": size, "status": status})
    pd.DataFrame(rows).to_csv(TABLES / "source_inventory.tsv", sep="\t", index=False)


def write_claims(data: dict[str, pd.DataFrame | dict], paths: dict[str, Path], stage_summary: pd.DataFrame) -> None:
    claims = [
        ["A2-01", "FT-600K fine-tuning source contains AIMD trajectories rather than NEB output", "VERIFIED", "grouped provenance manifest and split audit", "No NEB-adjacent segment was present by source design; this is source exclusion, not a post-hoc deletion"],
        ["A2-02", "No exact or near-duplicate match between 25 fixed-path images and grouped FT-600K training frames", "VERIFIED", "overlap_full_leakage_audit.json; overlap_scratch5_leakage_audit.json", "SOAP/RMSD thresholds are audit cutoffs, not physical accuracy thresholds"],
        ["A2-03", "The retained common SOAP/PCA map separates held-out AIMD and fixed-path NEB directions", "DESCRIPTIVE", "reference_embedding.csv; frame_features.csv", "Historical common fit; not a recalculated current grouped-train embedding"],
        ["A2-04", "Failure/CAR samples are outside the historical train envelope but do not form a monotonic scalar OOD ladder", "DESCRIPTIVE", "639-frame distance_to_train and stage summary", "Stage-balanced large-cell diagnostic; no causal instability claim"],
        ["A2-05", "Grouped FT-600K fixed-path error is 0.47 +/- 0.12 eV on pathway 1-4", "VERIFIED_GROUPED_MULTI_SEED", "grouped_seed_comparison.csv", "Current DFT candidate is provisional; historical 0.16 eV is not reproduced"],
        ["A2-06", "Stable MD trajectories and failure branches are distinct MLIP-generated observables", "VERIFIED_PLUS_FAILURE_MODE", "mace_1ns_merged_20261010/release and FIG-004 additive release", "Shared MD velocity seed is not an independent velocity replica"],
        ["A2-07", "Fixed-path and self-relaxed NEB are different process operators", "PROTOCOL_SEPARATED_PROVISIONAL", "neb_matrix_status.csv", "Do not pool into a universal ranking; self-relaxed DFT validation remains open"],
        ["A2-08", "Four reconstructed 81-atom DFT cases provide local force-error checks", "RECONSTRUCTION", "dft_vs_mace_within_method.csv", "Not full-cell validation; cross-model absolute energies are not compared"],
        ["A2-09", "639 static failure candidates have paired MLIP force-disagreement diagnostics", "VERIFIED_STATIC_DIAGNOSTIC", "mlip_candidate_results_all.csv; candidate_model_spread.csv", "Disagreement is not calibrated uncertainty and is not DFT error"],
    ]
    pd.DataFrame(claims, columns=["claim_id", "claim", "status", "evidence", "boundary"]).to_csv(TABLES / "claim_boundary_matrix.tsv", sep="\t", index=False)
    pending = [
        ["DFT-linked endpoint-19", "MLIP single-point/relaxation", "57/57 rows across four families", "COMPLETE_NO_SUBMISSION", "MACE_DFT_ENDPOINT_19_COMPARISON.tsv", "DFT/MLIP absolute energy comparison excluded because references/geometries differ"],
        ["47-endpoint campaign", "MLIP endpoint relaxation", "564 endpoint x start x model rows; three Scratch failures retained", "COMPLETE_NO_SUBMISSION", "analysis/mace_endpoint_four_model", "DFT endpoint acceptance still pending for the full 47-row set"],
        ["Current NEB matrix", "fixed-path and self-relaxed MLIP", "140/140 finite rows; convergence/status retained", "COMPLETE_NO_SUBMISSION", "analysis/neb_protocol_matrix_20261018/derived/neb_matrix_status.csv", "No duplicate MLIP scoring job is justified"],
        ["Phase-3 recovery", "DFT-linked MLIP static scoring", "16/16 model-case rows", "COMPLETE_NO_SUBMISSION", "analysis/phase3_dft_recovery_20261004/results_reconstructed/dft_vs_mace_within_method.csv", "RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED"],
        ["20 single-point DFT candidates", "MLIP-on-DFT comparison", "0/20 accepted DFT outputs", "HOLD_DFT_GATE", "analysis/dft_single_point_validation_20261001/dft_single_point_results.csv", "No accepted DFT label exists; do not submit duplicate MLIP scoring as if it closed the DFT gap"],
        ["47 endpoint DFT retries", "MLIP follow-up after DFT acceptance", "42 operational, 5 running, 0 publication-verified", "HOLD_DFT_IDENTITY_GATE", "DISCOVERY_PROGRESS_AUDIT_20261018.md", "Prepare keyed join only after final DFT identity/scientific gate"],
    ]
    pd.DataFrame(pending, columns=["scope", "calculation", "coverage", "submission_disposition", "evidence", "boundary"]).to_csv(TABLES / "pending_mlip_evaluations.tsv", sep="\t", index=False)


def main() -> None:
    for directory in [FIG, TABLES, DERIVED]:
        directory.mkdir(parents=True, exist_ok=True)
    paths = source_paths()
    require_sources(paths)
    data = load(paths)
    data["_source_paths"] = paths
    ref = data["reference"]
    assert isinstance(ref, pd.DataFrame)
    ref = add_reference_distances(ref)
    data["reference"] = ref
    long, summary = build_long_tables(data)
    long.to_csv(DERIVED / "task_distribution_long.csv", index=False)
    summary.to_csv(DERIVED / "task_distribution_summary.csv", index=False)
    spread = data["failure_spread"]
    assert isinstance(spread, pd.DataFrame)
    stage_summary = make_failure_stage_summary(spread)
    stage_summary.to_csv(DERIVED / "failure_stage_committee_disagreement_summary.csv", index=False)
    failure_features = data["failure_features"]
    assert isinstance(failure_features, pd.DataFrame)
    failure_distance = failure_features.copy()
    failure_distance["stage"] = failure_distance["window"].map({
        "stable_early": "early stable", "stable_pre": "pre-onset",
        "precursor": "precursor", "threshold": "threshold",
    }).fillna(failure_distance["window"])
    q99 = float(data["descriptor_manifest"]["distribution_fit"]["train_knn_q99"])
    failure_distance_summary = failure_distance.groupby("stage", as_index=False).agg(
        n=("distance_to_train", "size"),
        median_distance_to_train=("distance_to_train", "median"),
        q25_distance_to_train=("distance_to_train", lambda x: x.quantile(0.25)),
        q75_distance_to_train=("distance_to_train", lambda x: x.quantile(0.75)),
        fraction_outside_train_q99=("distance_to_train", lambda x: float((x > q99).mean())),
    )
    failure_distance_summary.to_csv(DERIVED / "failure_stage_distance_summary.csv", index=False)
    write_inventory(paths, data)
    write_claims(data, paths, stage_summary)
    figure_path = draw_figure(data, ref, stage_summary)

    manifest = {
        "schema_version": "1.0",
        "package": "task_conditioned_distribution_atlas_20261018",
        "status": "EVIDENCE_BOUNDED_ATLAS",
        "figure": str(figure_path),
        "figure_outputs": [str(p) for p in FIG.glob("fig_task_conditioned_distribution_atlas.*")],
        "derived_tables": [str(p) for p in DERIVED.glob("*.csv")],
        "source_inventory": str(TABLES / "source_inventory.tsv"),
        "claim_boundary_matrix": str(TABLES / "claim_boundary_matrix.tsv"),
        "pending_mlip_evaluations": str(TABLES / "pending_mlip_evaluations.tsv"),
        "descriptor_fit": data["descriptor_manifest"],
        "counts": {
            "historical_reference_frames": int(len(ref)),
            "failure_frames": int(len(data["failure_features"])),
            "grouped_rmse_rows": int(len(data["rmse"])),
            "md_rows": int(len(data["md"])),
            "neb_rows": int(len(data["neb"])),
            "dft_recovery_rows": int(len(data["dft_recovery"])),
            "failure_mlip_rows": int(len(data["failure_results"])),
        },
        "claim_boundaries": [
            "The upper row separates DFT/reference-PES observations from MLIP-generated trajectories and paths.",
            "The historical SOAP/PCA map is descriptive and is not a current grouped-FT-600K retraining embedding.",
            "Fixed-path and self-relaxed NEB are shown separately; self-relaxed MLIP paths are not DFT-validated here.",
            "DFT-referenced failure points are local-cell reconstructions; the 639-frame committee spread is MLIP-only and not calibrated uncertainty.",
            "The atlas is a task-conditioned comparison, not a universal model ranking.",
        ],
    }
    for p in list(FIG.glob("*")) + list(DERIVED.glob("*.csv")) + list(TABLES.glob("*.tsv")):
        if p.is_file():
            manifest.setdefault("sha256", {})[str(p)] = sha256(p)
    (OUT / "task_conditioned_distribution_atlas_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figure": str(figure_path), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()

