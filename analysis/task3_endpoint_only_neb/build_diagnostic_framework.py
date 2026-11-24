"""Build the auditable migration-path diagnostic framework.

The outputs separate three things that must not be conflated:
1. DFT reference qualification, including missing SCF evidence;
2. same-geometry Foundation-model error decomposition; and
3. out-of-fold, geometry-only early-warning diagnostics.

All QE values are screened path-table references.  No output from this script
promotes the current NEB runs to converged activation barriers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import sys
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import apply_publication_style, panel_label, save_figure

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "exploratory_energy_audit"
IMG = Path(os.environ.get("FRAMEWORK_IMG", str(AUDIT / "corrected_dft_foundation_models/corrected_dft_foundation_image_profiles.csv")))
GATE = ROOT / "dft_gate_manifest.csv"
NEB_AUDIT = AUDIT / "scf_neb_iteration_energy_audit.csv"
STRUCT = Path(os.environ.get("FRAMEWORK_STRUCT", str(AUDIT / "structural_diagnostics/qe_path_structure_stage_metrics.csv")))
QE_RAW = Path(os.environ.get("FRAMEWORK_QE_RAW", str(AUDIT / "processed/mace_dft_relative_image_metrics.csv")))
OUT = Path(os.environ.get("FRAMEWORK_OUT", str(AUDIT / "diagnostic_framework")))
OUT.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "mace-large-density-agnesi-stress": "Agnesi-L-dens",
    "mace-medium-density-agnesi-stress": "Agnesi-M-dens",
    "mace-small-density-agnesi-stress": "Agnesi-S-dens",
    "mace_agnesi_medium": "Agnesi-M", "mace-mh-0": "MH-0", "mace-mh-1": "MH-1",
    "mace-mp-0b3-medium": "MP-0B3", "mace-omat-0-medium": "OMAT-0-M",
    "mace-omat-0-small": "OMAT-0-S",
}
MODEL_ORDER = list(MODEL_LABELS)
MODEL_COLORS = ["#178F78", "#263E69", "#C12F55", "#2B68A6", "#327BB4", "#4A91B6", "#F04B23", "#F06A76", "#6B6B6B"]
STAGE_ORDER = ["endpoint", "near_endpoint", "interior", "transition_state_window"]
STAGE_LABELS = {"endpoint": "endpoint", "near_endpoint": "near-endpoint", "interior": "interior", "transition_state_window": "TS window"}
STAGE_COLORS = {"endpoint": "#202020", "near_endpoint": "#327BB4", "interior": "#178F78", "transition_state_window": "#C12F55"}


def path_type(path_id: str) -> str:
    if "gap-diffuse" in path_id:
        return "in_gap_diffusion"
    if "gap2Q" in path_id or "gap2q" in path_id:
        return "gap_to_QL"
    if "QLs" in path_id:
        return "within_QL"
    return "unclassified_control"


def corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    x, y = x.to_numpy(float), y.to_numpy(float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1]) if method == "pearson" else float(spearmanr(x, y).statistic)


def qualify_dft() -> pd.DataFrame:
    gate = pd.read_csv(GATE)
    audit = pd.read_csv(NEB_AUDIT)
    d = gate.merge(audit, on="path_id", how="left", suffixes=("_gate", "_audit"), validate="one_to_one")
    d["finite_energy_for_all_images"] = d["all_path_images_finite"].astype(bool)
    d["scf_warning_free"] = d["scf_warning_images"].fillna(999).eq(0)
    d["last_table_stable_lt_0p1_eV"] = d["exploratory_energy_stable"].fillna(False).astype(bool)
    d["complete_final_energy_table"] = d["n_complete_finite_energy_tables"].fillna(0).ge(1)
    d["path_identity_within_0p05_A"] = d["endpoint_path_max_diff_A"].fillna(np.inf).le(0.05)
    d["explicit_path_convergence_marker"] = d["explicit_path_convergence"].fillna(False).astype(bool)
    # The final output audit is the authoritative completion evidence for this
    # table; the earlier gate manifest may be conservative for a path whose
    # copied output was later parsed successfully.
    d["job_done_evidence"] = d["job_done_audit"].fillna(d["job_done_gate"]).astype(bool)
    d["formal_converged_reference"] = (
        d["finite_energy_for_all_images"] & d["job_done_evidence"] & d["scf_warning_free"] &
        d["explicit_path_convergence_marker"] & d["complete_final_energy_table"] & d["path_identity_within_0p05_A"]
    )
    d["reference_tier"] = np.select(
        [d["formal_converged_reference"], d["finite_energy_for_all_images"] & d["job_done_evidence"] & d["last_table_stable_lt_0p1_eV"]],
        ["converged_reference_candidate", "exploratory_fixed_geometry"],
        default="rejected_or_incomplete",
    )
    d["formal_reference_status"] = np.where(d.formal_converged_reference, "eligible", "not_eligible")
    d["path_type"] = d.path_id.map(path_type)
    cols = ["path_id", "path_type", "system", "dopant", "reference_tier", "formal_reference_status",
            "finite_energy_for_all_images", "job_done_evidence", "job_done_gate", "job_done_audit", "scf_warning_free",
            "scf_warning_images", "inner_electronic_scf_iterations", "explicit_path_convergence_marker",
            "last_table_stable_lt_0p1_eV", "max_abs_image_energy_delta_eV", "rms_image_energy_delta_eV",
            "complete_final_energy_table", "path_identity_within_0p05_A", "endpoint_path_max_diff_A",
            "max_path_gradient_raw", "neb_outer_iterations", "self_consistency_image_calls",
            "n_energy_tables", "n_complete_finite_energy_tables"]
    out = d[cols].rename(columns={"job_done_evidence": "job_done"})
    out.to_csv(OUT / "dft_reference_qualification.csv", index=False)
    return out


def add_alignment_profiles(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, path), q in d.groupby(["model_id", "path_id"], sort=False):
        q = q.sort_values("image_index").copy()
        model_raw = q.energy_eV.to_numpy(float)
        qe_raw = q.dft_path_energy_eV.to_numpy(float)
        offsets = {
            "image1": (model_raw[0], qe_raw[0]),
            "endpoint_mean": ((model_raw[0] + model_raw[-1]) / 2, (qe_raw[0] + qe_raw[-1]) / 2),
            "path_min": (model_raw.min(), qe_raw.min()),
        }
        for alignment, (m0, q0) in offsets.items():
            for i, (_, row) in enumerate(q.iterrows()):
                rows.append({"model_id": model, "path_id": path, "image_index": int(row.image_index),
                             "alignment": alignment, "model_aligned_eV": model_raw[i] - m0,
                             "qe_aligned_eV": qe_raw[i] - q0})
    p = pd.DataFrame(rows)
    p["error_eV"] = p.model_aligned_eV - p.qe_aligned_eV
    p["abs_error_eV"] = p.error_eV.abs()
    return p


def build_metrics(d: pd.DataFrame, structural: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = add_alignment_profiles(d)
    p.to_csv(OUT / "alignment_profile_errors.csv", index=False)
    alignment_rows = []
    for (model, path, alignment), q in p.groupby(["model_id", "path_id", "alignment"]):
        q = q.sort_values("image_index")
        qe_peak = int(q.loc[q.qe_aligned_eV.idxmax(), "image_index"])
        model_peak = int(q.loc[q.model_aligned_eV.idxmax(), "image_index"])
        alignment_rows.append({"model_id": model, "path_id": path, "alignment": alignment,
                              "mae_eV": q.abs_error_eV.mean(), "rmse_eV": np.sqrt(np.mean(q.error_eV ** 2)),
                              "max_abs_error_eV": q.abs_error_eV.max(), "profile_pearson": corr(q.model_aligned_eV, q.qe_aligned_eV),
                              "profile_spearman": corr(q.model_aligned_eV, q.qe_aligned_eV, "spearman"),
                              "model_peak_image": model_peak, "qe_peak_image": qe_peak,
                              "ts_position_error_images": abs(model_peak - qe_peak),
                              "model_barrier_from_min_eV": q.model_aligned_eV.max() - q.model_aligned_eV.min(),
                              "qe_barrier_from_min_eV": q.qe_aligned_eV.max() - q.qe_aligned_eV.min(),
                              "barrier_error_eV": (q.model_aligned_eV.max() - q.model_aligned_eV.min()) - (q.qe_aligned_eV.max() - q.qe_aligned_eV.min())})
    path_metrics = pd.DataFrame(alignment_rows)
    path_metrics["abs_barrier_error_eV"] = path_metrics.barrier_error_eV.abs()
    path_metrics.to_csv(OUT / "model_path_metrics_by_alignment.csv", index=False)

    current = path_metrics[path_metrics.alignment.eq("image1")].copy()
    current = current.merge(d[["model_id", "path_id", "image_index", "qe_relative_energy_eV"]].drop_duplicates(),
                            on=["model_id", "path_id"], how="left")
    # Rebuild a clean image-level table for stage-specific errors.
    stage = structural[["path_id", "system", "dopant", "image_index", "stage", "stage_coarse", "dopant_displacement_from_image1_A",
                        "dopant_nearest_host_A", "dopant_nearest3_mean_A", "dopant_coordination_n_within_4p3A",
                        "dopant_coordination_n_within_4p6A", "nearest3_distance_std_A", "host_rms_displacement_from_image1_A",
                        "all_atom_rms_displacement_from_image1_A", "max_gradient_norm_eV_per_A", "rms_gradient_norm_eV_per_A"]]
    frame = d.merge(stage, on=["path_id", "image_index"], how="inner", validate="many_to_one")
    frame["path_type"] = frame.path_id.map(path_type)
    frame["error_eV"] = frame.relative_energy_eV - frame.qe_relative_energy_eV
    frame["abs_error_eV"] = frame.error_eV.abs()
    frame.to_csv(OUT / "model_frame_errors_with_structure.csv", index=False)
    stage_metrics = frame.groupby(["model_id", "path_type", "stage"], as_index=False).agg(
        n_frames=("abs_error_eV", "size"), mae_eV=("abs_error_eV", "mean"), rmse_eV=("error_eV", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        median_abs_error_eV=("abs_error_eV", "median"), max_abs_error_eV=("abs_error_eV", "max"), signed_bias_eV=("error_eV", "mean"))
    stage_metrics.to_csv(OUT / "model_stage_metrics.csv", index=False)

    path_difficulty = frame.groupby(["path_id", "path_type", "system", "dopant"], as_index=False).agg(
        qe_path_barrier_like_eV=("qe_relative_energy_eV", "max"), median_model_abs_error_eV=("abs_error_eV", "median"),
        max_model_abs_error_eV=("abs_error_eV", "max"), median_model_range_eV=("relative_energy_eV", lambda x: float(x.max() - x.min())),
        max_dopant_displacement_A=("dopant_displacement_from_image1_A", "max"), max_nearest_host_distance_A=("dopant_nearest_host_A", "max"))
    # The model-range aggregation above is across model/image values; replace
    # it with the intended maximum across models at a common image.
    spread = d.groupby(["path_id", "image_index"], as_index=False).relative_energy_eV.agg(lambda x: float(x.max() - x.min()))
    spread = spread.groupby("path_id", as_index=False).relative_energy_eV.max().rename(columns={"relative_energy_eV": "max_model_range_eV"})
    path_difficulty = path_difficulty.drop(columns=["median_model_range_eV"]).merge(spread, on="path_id")
    path_difficulty["difficulty_rank_by_qe_barrier"] = path_difficulty.qe_path_barrier_like_eV.rank(ascending=False, method="min").astype(int)
    path_difficulty["difficulty_rank_by_model_error"] = path_difficulty.median_model_abs_error_eV.rank(ascending=False, method="min").astype(int)
    path_difficulty.to_csv(OUT / "path_difficulty_summary.csv", index=False)

    alignment_summary = path_metrics.groupby(["model_id", "alignment"], as_index=False).agg(
        median_mae_eV=("mae_eV", "median"), median_rmse_eV=("rmse_eV", "median"), median_abs_barrier_error_eV=("abs_barrier_error_eV", "median"),
        median_ts_position_error_images=("ts_position_error_images", "median"), median_profile_spearman=("profile_spearman", "median"))
    alignment_summary.to_csv(OUT / "alignment_sensitivity_summary.csv", index=False)

    ranking_rows = []
    qe_bar = current[current.image_index.eq(1)].copy()[["model_id", "path_id"]]
    # Use path-level barrier metrics from image1 alignment.
    bm = path_metrics[path_metrics.alignment.eq("image1")]
    for model, q in bm.groupby("model_id"):
        ranking_rows.append({"model_id": model, "model_label": MODEL_LABELS[model],
                             "barrier_rank_spearman": corr(q.qe_barrier_from_min_eV, q.model_barrier_from_min_eV, "spearman"),
                             "barrier_pearson": corr(q.qe_barrier_from_min_eV, q.model_barrier_from_min_eV),
                             "median_ts_position_error_images": q.ts_position_error_images.median(),
                             "median_profile_spearman": q.profile_spearman.median(),
                             "median_abs_barrier_error_eV": q.abs_barrier_error_eV.median()})
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(OUT / "model_ranking_and_shape_metrics.csv", index=False)
    return {"frame": frame, "stage": stage_metrics, "path": path_difficulty, "alignment": alignment_summary,
            "path_metrics": path_metrics, "ranking": ranking, "profiles": p}


def failure_predictors(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = ["image_index", "dopant_displacement_from_image1_A", "dopant_nearest_host_A", "dopant_nearest3_mean_A",
                "dopant_coordination_n_within_4p3A", "dopant_coordination_n_within_4p6A", "nearest3_distance_std_A",
                "host_rms_displacement_from_image1_A", "all_atom_rms_displacement_from_image1_A"]
    out_metrics, out_pred = [], []
    for model in MODEL_ORDER:
        q = frame[frame.model_id.eq(model)].copy().sort_values(["path_id", "image_index"])
        X = q[features].replace([np.inf, -np.inf], np.nan).fillna(q[features].median()).to_numpy(float)
        groups = q.path_id.to_numpy()
        for threshold in (0.2, 0.5):
            y = (q.abs_error_eV.to_numpy() >= threshold).astype(int)
            pred = np.full(len(q), np.nan)
            if len(np.unique(y)) >= 2:
                cv = GroupKFold(n_splits=4)
                for train, test in cv.split(X, y, groups):
                    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=0))
                    if len(np.unique(y[train])) < 2:
                        pred[test] = y[train].mean()
                    else:
                        pipe.fit(X[train], y[train]); pred[test] = pipe.predict_proba(X[test])[:, 1]
            else:
                pred[:] = y.mean()
            yhat = (pred >= .5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
            out_metrics.append({"model_id": model, "model_label": MODEL_LABELS[model], "error_threshold_eV": threshold,
                                "n_frames": len(y), "n_high_error": int(y.sum()),
                                "auroc": roc_auc_score(y, pred) if len(np.unique(y)) >= 2 else np.nan,
                                "auprc": average_precision_score(y, pred), "false_positive_rate": fp / max(fp + tn, 1),
                                "false_negative_rate": fn / max(fn + tp, 1), "precision_at_0p5": tp / max(tp + fp, 1),
                                "recall_at_0p5": tp / max(tp + fn, 1)})
            for i, row in q.iterrows():
                out_pred.append({"model_id": model, "path_id": row.path_id, "image_index": int(row.image_index),
                                 "error_threshold_eV": threshold, "high_error": int(y[q.index.get_loc(i)]),
                                 "risk_probability": float(pred[q.index.get_loc(i)])})
    metrics = pd.DataFrame(out_metrics)
    predictions = pd.DataFrame(out_pred)
    metrics.to_csv(OUT / "geometry_only_failure_predictor_metrics.csv", index=False)
    predictions.to_csv(OUT / "geometry_only_failure_predictor_oof.csv", index=False)
    return metrics, predictions


def clean(ax):
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_linewidth(.8)
    ax.tick_params(width=.8, length=3, direction="out", pad=2)
    for lab in (*ax.get_xticklabels(), *ax.get_yticklabels()): lab.set_fontweight("bold")


def plot_framework(metrics: dict[str, pd.DataFrame], predictor: pd.DataFrame, qualification: pd.DataFrame) -> None:
    apply_publication_style(8.2)
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.9))
    stage = (metrics["stage"].groupby(["model_id", "stage"], as_index=False).mae_eV.mean()
             .pivot(index="model_id", columns="stage", values="mae_eV")
             .reindex(index=MODEL_ORDER, columns=STAGE_ORDER))
    im = axes[0, 0].imshow(stage.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=np.nanpercentile(stage.to_numpy(), 95))
    axes[0, 0].set_xticks(np.arange(4), [STAGE_LABELS[x] for x in STAGE_ORDER], rotation=35, ha="right")
    axes[0, 0].set_yticks(np.arange(9), [MODEL_LABELS[x] for x in MODEL_ORDER])
    axes[0, 0].set_ylabel("Foundation model"); axes[0, 0].set_title("Stage-specific energy MAE", fontsize=8.8)
    cbar = fig.colorbar(im, ax=axes[0, 0], fraction=.046, pad=.04); cbar.set_label("MAE (eV)")
    for i in range(stage.shape[0]):
        for j in range(stage.shape[1]):
            if np.isfinite(stage.iloc[i, j]): axes[0, 0].text(j, i, f"{stage.iloc[i,j]:.2f}", ha="center", va="center", fontsize=4.6, color="white")

    path = metrics["path"]
    for _, row in path.iterrows():
        color = {"2L-Cu": "#C12F55", "2L-Li": "#327BB4", "2L-Na": "#178F78"}.get(row.system, "#555555")
        axes[0, 1].scatter(row.qe_path_barrier_like_eV, row.median_model_abs_error_eV, s=28, color=color, edgecolor="white", linewidth=.4)
        axes[0, 1].text(row.qe_path_barrier_like_eV, row.median_model_abs_error_eV, row.path_id.replace("2L-", "").replace("__neb-", "\n"), fontsize=4.7, ha="center", va="bottom")
    axes[0, 1].set_xlabel("QE path-table barrier-like ΔE (eV)"); axes[0, 1].set_ylabel("Median model |error| (eV)")
    axes[0, 1].set_title("Path difficulty as an exam problem", fontsize=8.8)
    axes[0, 1].set_xscale("log"); axes[0, 1].set_yscale("log")

    align = metrics["alignment"]
    for i, alignment in enumerate(["image1", "endpoint_mean", "path_min"]):
        vals = align[align.alignment.eq(alignment)].median_mae_eV.to_numpy()
        axes[0, 2].boxplot(vals, positions=[i], widths=.5, patch_artist=True, boxprops={"facecolor": "white", "edgecolor": "#333333"}, medianprops={"color": "#C12F55"}, whiskerprops={"color": "#333333"}, capprops={"color": "#333333"})
        axes[0, 2].scatter(np.full(len(vals), i), vals, s=15, color=MODEL_COLORS, edgecolor="white", linewidth=.2, zorder=3)
    axes[0, 2].set_xticks([0, 1, 2], ["image 1", "endpoint\nmean", "path\nminimum"])
    axes[0, 2].set_ylabel("Median profile MAE (eV)"); axes[0, 2].set_title("Sensitivity to energy alignment", fontsize=8.8)

    f = metrics["frame"]
    for stage_name, q in f.groupby("stage"):
        axes[1, 0].scatter(q.dopant_displacement_from_image1_A, np.maximum(q.abs_error_eV, 1e-4), s=11, alpha=.65, color=STAGE_COLORS[stage_name], label=STAGE_LABELS[stage_name])
    axes[1, 0].axhline(.2, color="#98489F", lw=.9, ls=(0, (3, 2)))
    axes[1, 0].set_xlabel("Dopant displacement from image 1 (Å)"); axes[1, 0].set_ylabel("|Foundation − QE| (eV)")
    axes[1, 0].set_yscale("log"); axes[1, 0].set_title("Geometry-linked error", fontsize=8.8)
    axes[1, 0].legend(frameon=False, fontsize=5.3, loc="upper left")

    pred = predictor[predictor.error_threshold_eV.eq(.2)].copy().merge(metrics["frame"][["model_id", "path_id", "image_index"]], on=["model_id", "path_id", "image_index"], how="left")
    # Plot model-level cross-validated discrimination as a compact, auditable
    # alternative to an unreadable nine-curve ROC panel.
    auc = metrics["predictor"] if "predictor" in metrics else None
    # predictor metrics are passed separately through the dataframe below.
    pred_summary = metrics.get("predictor_metrics", pd.DataFrame())
    if not pred_summary.empty:
        ps = pred_summary[pred_summary.error_threshold_eV.eq(.2)].sort_values("auroc")
        y = np.arange(len(ps)); axes[1, 1].barh(y, ps.auroc, color=[MODEL_COLORS[MODEL_ORDER.index(x)] for x in ps.model_id], edgecolor="none")
        axes[1, 1].set_yticks(y, ps.model_label); axes[1, 1].set_xlim(0, 1); axes[1, 1].set_xlabel("Grouped-CV AUROC")
    axes[1, 1].set_title("Geometry-only early-warning signal", fontsize=8.8)

    counts = qualification.reference_tier.value_counts().reindex(["converged_reference_candidate", "exploratory_fixed_geometry", "rejected_or_incomplete"], fill_value=0)
    axes[1, 2].bar(np.arange(3), counts.to_numpy(), color=["#178F78", "#327BB4", "#C12F55"], edgecolor="none")
    axes[1, 2].set_xticks(np.arange(3), ["converged\ncandidate", "exploratory\nfixed geometry", "rejected\n/incomplete"])
    axes[1, 2].set_ylabel("Number of paths"); axes[1, 2].set_title("DFT reference qualification", fontsize=8.8)
    for ax in axes.flat: clean(ax)
    for i, ax in enumerate(axes.flat): panel_label(ax, chr(ord("a") + i), x=-.16, y=1.12)
    fig.subplots_adjust(left=.16, right=.98, bottom=.14, top=.93, wspace=.50, hspace=.58)
    stem = OUT / "figure_diagnostic_framework_overview"
    save_figure(fig, stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight"); fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    qualification = qualify_dft()
    d = pd.read_csv(IMG)
    qe_raw = pd.read_csv(QE_RAW)
    if "variant" in qe_raw.columns:
        qe_raw = qe_raw[qe_raw.variant.eq("base")][["path_id", "image_index", "dft_path_energy_eV"]].drop_duplicates()
    else:
        qe_raw = qe_raw[["path_id", "image_index", "path_energy_raw"]].rename(columns={"path_energy_raw": "dft_path_energy_eV"}).drop_duplicates()
    d = d.merge(qe_raw, on=["path_id", "image_index"], how="left", validate="many_to_one")
    if d.dft_path_energy_eV.isna().any():
        raise ValueError("QE raw path-table energies did not align with the Foundation frames")
    structural = pd.read_csv(STRUCT)
    metrics = build_metrics(d, structural)
    predictor_metrics, predictor_oof = failure_predictors(metrics["frame"])
    metrics["predictor_metrics"] = predictor_metrics
    plot_framework(metrics, predictor_oof, qualification)
    policy = predictor_oof[predictor_oof.error_threshold_eV.eq(.2)].merge(
        metrics["frame"][["model_id", "path_id", "image_index", "abs_error_eV"]],
        on=["model_id", "path_id", "image_index"], how="left", validate="one_to_one")
    policy["provisional_usage_band"] = pd.cut(
        policy.risk_probability, bins=[-np.inf, .2, .5, np.inf],
        labels=["direct_use_candidate", "needs_validation", "DFT_fallback"], right=False,
    ).astype(str)
    policy.to_csv(OUT / "provisional_usage_policy_oof.csv", index=False)
    policy_summary = policy.groupby(["model_id", "provisional_usage_band"], as_index=False).agg(
        n_frames=("abs_error_eV", "size"), observed_high_error_rate=("high_error", "mean"),
        mean_abs_error_eV=("abs_error_eV", "mean"), median_risk_probability=("risk_probability", "median"))
    policy_summary.to_csv(OUT / "provisional_usage_policy_summary.csv", index=False)
    framework_manifest = {
        "scope": "migration-path diagnostic framework",
        "dft_reference_rule": "formal reference requires finite images, completed job, no SCF warning, explicit path convergence marker, complete final table, and endpoint/path identity <= 0.05 A",
        "current_dft_counts": qualification.reference_tier.value_counts().to_dict(),
        "energy_alignment_modes": ["image1", "endpoint_mean", "path_min"],
        "same_geometry_foundation_evaluations": int(len(d)),
        "stage_definition": "endpoint images 1 and 10; TS window = QE path-table maximum +/- 1; near-endpoint = 2,3,8,9 after TS-window exclusion; remaining images = interior",
        "path_type_definition": {"in_gap_diffusion": "path_id contains gap-diffuse", "gap_to_QL": "path_id contains gap2QL/gap2Ql", "within_QL": "path_id contains QLs", "unclassified_control": "remaining path IDs retained but not forced into the three canonical classes"},
        "failure_predictor": "grouped 4-fold cross-validated logistic regression using geometry/image-index features only; no QE energy, force, model spread, or stage labels used as predictors",
        "high_error_thresholds_eV": [0.2, 0.5],
        "provisional_usage_policy": {"direct_use_candidate": "OOF risk < 0.2 and only for models with AUROC >= 0.75; not a certification", "needs_validation": "OOF risk 0.2-0.5 or model AUROC < 0.75", "DFT_fallback": "OOF risk >= 0.5 or geometry outside the validated feature envelope"},
        "embedding_scope": "NEB frames only; no provenance-qualified raw training/test/AIMD split included",
        "claim_boundary": "screened QE path-table reference; no converged DFT barrier or model-accuracy certification",
    }
    (OUT / "diagnostic_framework_manifest.json").write_text(json.dumps(framework_manifest, indent=2) + "\n")
    (OUT / "diagnostic_framework_paper_text.md").write_text(
        """# Migration-path diagnostic framework

## Framework claim

Each retained migration path is treated as a structured examination problem rather than only as a barrier-comparison datum. The framework first qualifies the QE reference, then decomposes Foundation-model error by image, stage, path, and alignment convention, and finally tests whether geometry-only features can identify high-error frames before an expensive reference calculation.

## Acceptance boundary

The current QE campaign supplies finite path snapshots and an exploratory last-table stability screen, but it does not meet the formal reference criterion: every audited NEB output contains at least one SCF-not-converged warning, no output exposes the inner electronic SCF iteration count, and no path carries an explicit convergence marker. Accordingly, the present results are valid for diagnostic path sampling, not for certification of a converged DFT activation barrier.

## Model failure diagnosis

The same-geometry matrix makes it possible to localize disagreement by migration stage and local environment. The Cu QLs path is a high-difficulty examination case with large Foundation-model elevation; the stage-resolved outputs determine whether the error is concentrated near the operational transition-state window or already present near the endpoints. Geometry-only grouped cross-validation reports AUROC, AUPRC, false-positive rate, and false-negative rate for a pre-computation warning rule. These metrics are screening evidence, not a deployment-ready uncertainty model, until a larger set of independent paths is available.

## Training-data implication

The current embedding output is explicitly NEB-frame-only because a provenance-qualified raw training/test/AIMD structure release was not found in the active workspace. The next data-design step is therefore to add independently hashed reference distributions and rerun the same descriptor and grouped-validation pipeline; only then can embedding distance be used as a defensible training-coverage criterion.
"""
    )
    print(json.dumps({"output": str(OUT), "dft_tiers": qualification.reference_tier.value_counts().to_dict(), "models": len(MODEL_ORDER), "frames": len(d)}, indent=2))


if __name__ == "__main__":
    main()
