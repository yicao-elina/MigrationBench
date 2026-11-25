#!/usr/bin/env python3
"""Build the evidence-bounded A2 task-distribution figures.

The two figures deliberately separate two questions that are often conflated.

1. The grouped A2 audit asks whether the 25 fixed-path DFT images were copied
   or near-duplicated in the declared FT-600K training file.
2. The historical common SOAP/PCA diagnostic asks how held-out AIMD, fixed-path
   NEB, and the large-cell MD failure-attribution sample occupy a *shared,
   descriptive* representation space.

The second analysis is not substituted for the first: its reference data are a
historical 521-frame descriptor fit, whereas A2's leakage audit is performed
against the current 1,786-frame trajectory-grouped FT-600K training file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
HISTORICAL = Path(
    os.environ.get("MIGRATIONBENCH_HISTORICAL_ROOT", "../Wiley-MigrationBench")
).expanduser().resolve()
ATLAS = Path(
    os.environ.get("MIGRATIONBENCH_PLOT_ATLAS", "plot_library")
) / "src"
if str(ATLAS) not in sys.path:
    sys.path.insert(0, str(ATLAS))

from plot_atlas import apply_publication_style, joint_scatter  # noqa: E402


# These colours encode task/configuration roles, not MLIP families.  The JHU
# MACE palette is reserved for model-family comparisons elsewhere in the paper.
TASK_COLORS = {
    "AIMD training": "#4C4C4C",
    "Held-out AIMD": "#9A9A9A",
    "Fixed-path NEB": "#6E4C9B",
    "Large-cell MD failure sample": "#C65447",
}
FAILURE_COLORS = {
    "early stable": "#6C8EBF",
    "pre-onset": "#8AA7C7",
    "precursor": "#D39A4B",
    "threshold": "#B54B4B",
}
INK = "#202020"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths() -> dict[str, Path]:
    return {
        "grouped_overlap_full": REPO / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/overlap_full_leakage_audit.json",
        "grouped_overlap_scratch5": REPO / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/overlap_scratch5_leakage_audit.json",
        "historical_reference_embedding": HISTORICAL / "analysis/multimodel_639_screen_20261014/inputs/reference_embedding.csv",
        "historical_frame_features": HISTORICAL / "analysis/multimodel_639_screen_20261014/inputs/frame_features.csv",
        "historical_descriptor_manifest": HISTORICAL / "analysis/multimodel_639_screen_20261014/inputs/diagnostic_manifest.json",
    }


def require_sources(paths: dict[str, Path]) -> None:
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))


def set_axes_style(ax, *, labelsize: float = 12.0, ticksize: float = 11.5) -> None:
    ax.grid(False)
    ax.tick_params(width=1.05, length=4.5, labelsize=ticksize, direction="out")
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.05)
    ax.xaxis.label.set_fontsize(labelsize)
    ax.yaxis.label.set_fontsize(labelsize)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    output = PACKAGE / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext, dpi in (("pdf", 300), ("png", 450), ("svg", 300)):
        final = output / f"{stem}.{ext}"
        temporary = output / f".{stem}.{ext}.tmp"
        fig.savefig(temporary, format=ext, dpi=dpi, bbox_inches="tight",
                    facecolor="white", metadata={"Creator": "Plot Atlas"})
        os.replace(temporary, final)
        paths.append(final)
    plt.close(fig)
    return paths


def load_inputs(paths: dict[str, Path]):
    with paths["grouped_overlap_full"].open() as handle:
        full = json.load(handle)
    with paths["grouped_overlap_scratch5"].open() as handle:
        scratch5 = json.load(handle)
    with paths["historical_descriptor_manifest"].open() as handle:
        descriptor_manifest = json.load(handle)
    reference = pd.read_csv(paths["historical_reference_embedding"])
    features = pd.read_csv(paths["historical_frame_features"])
    return full, scratch5, descriptor_manifest, reference, features


def map_data(reference: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    train = reference.loc[reference["reference"].eq("train"), ["pca1", "pca2"]].copy()
    train["task"] = "AIMD training"
    held = reference.loc[reference["reference"].isin(["test", "valid"]), ["pca1", "pca2"]].copy()
    held["task"] = "Held-out AIMD"
    neb = reference.loc[reference["reference"].eq("neb"), ["pca1", "pca2"]].copy()
    neb["task"] = "Fixed-path NEB"
    failed = features.loc[:, ["pca1", "pca2"]].copy()
    failed["task"] = "Large-cell MD failure sample"
    return pd.concat([train, held, neb, failed], ignore_index=True)


def draw_task_map(map_frame: pd.DataFrame) -> list[Path]:
    apply_publication_style(font_size=12)
    fig = plt.figure(figsize=(8.8, 8.9), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[0.66, 1.35], hspace=0.15)
    order = list(TASK_COLORS)
    ordered = map_frame.copy()
    ordered["task"] = pd.Categorical(ordered["task"], categories=order, ordered=True)
    ordered = ordered.sort_values("task")

    # Global view: the full vertical separation of the historical fixed-path
    # NEB reference is intentionally retained here rather than rescaled.
    ax_global = fig.add_subplot(outer[0, 0])
    for task in ("AIMD training", "Held-out AIMD", "Fixed-path NEB"):
        part = ordered.loc[ordered["task"].eq(task)]
        ax_global.scatter(part["pca1"], part["pca2"], s=28 if task == "Fixed-path NEB" else 18,
                          alpha=0.63, color=TASK_COLORS[task], edgecolor="white",
                          linewidth=0.25, rasterized=True, label=task)
    ax_global.set_xlim(-23.0, 3.5)
    ax_global.set_ylim(-20.0, 235.0)
    ax_global.set_xticks([-20, -10, 0])
    ax_global.set_yticks([0, 100, 200])
    ax_global.set_xlabel("SOAP PCA 1")
    ax_global.set_ylabel("SOAP PCA 2")
    set_axes_style(ax_global)
    ax_global.annotate("fixed-path NEB", xy=(-9.0, 214), xytext=(-15.7, 175),
                       fontsize=11.7, fontweight="bold", color=TASK_COLORS["Fixed-path NEB"],
                       ha="center", va="center",
                       arrowprops={"arrowstyle": "-", "color": TASK_COLORS["Fixed-path NEB"], "lw": 1.1})
    ax_global.annotate("AIMD reference", xy=(0.1, 0.1), xytext=(-14.2, 42),
                       fontsize=11.7, fontweight="bold", color=TASK_COLORS["AIMD training"],
                       ha="center", va="center",
                       arrowprops={"arrowstyle": "-", "color": TASK_COLORS["AIMD training"], "lw": 1.1})
    ax_global.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#C8C8C8",
                     framealpha=1, fontsize=11.2, borderpad=0.45, handletextpad=0.45)
    ax_global.text(-0.09, 1.02, "a", transform=ax_global.transAxes, fontsize=15, fontweight="bold",
                   ha="left", va="bottom")

    # Detail view: the failure-attribution sample shares the stored historical
    # fit with AIMD references, and the top/right KDEs make the projected
    # distributions explicit.  NEB is intentionally omitted: panel a shows it
    # on its full, common coordinate scale.
    detail_order = ["AIMD training", "Held-out AIMD", "Large-cell MD failure sample"]
    detail = ordered.loc[ordered["task"].isin(detail_order)].copy()
    detail["task"] = pd.Categorical(detail["task"].astype(str), categories=detail_order, ordered=True)
    panel = joint_scatter(
        fig,
        outer[1, 0],
        detail["pca1"], detail["pca2"],
        {"categorical": [TASK_COLORS[key] for key in detail_order], "sequential": ["#F1F1F1", "#303030"]},
        groups=detail["task"].astype(str), marginal="kde", regression=False,
        point_size=20, alpha=0.56, annotate=False, marginal_axes=False,
        axis_linewidth=1.05, tick_width=1.05, tick_length=4.5,
        point_edgecolor="white", point_linewidth=0.22, legend_frame=True,
        marginal_ratio=0.19, marginal_gap=0.035,
    )
    ax, top, right = panel["main"], panel["top"], panel["right"]
    ax.set_xlim(-2.1, 3.35)
    ax.set_ylim(-5.2, 0.85)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-4, -2, 0])
    ax.set_xlabel("SOAP PCA 1")
    ax.set_ylabel("SOAP PCA 2")
    set_axes_style(ax)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((0.02, 0.98))
        legend._loc = 2
        legend.set_frame_on(True)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#C8C8C8")
        legend.get_frame().set_alpha(1)
        for text in legend.get_texts():
            text.set_fontsize(11.0)
            text.set_fontweight("normal")
    for marginal in (top, right):
        marginal.tick_params(labelsize=11.0, width=1.0, length=3.5)
        for spine in marginal.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color(INK)
    ax.text(-0.10, 1.02, "b", transform=ax.transAxes, fontsize=15, fontweight="bold",
            ha="left", va="bottom")
    return save_figure(fig, "fig_a2_task_distribution_map")


def draw_audit_panel(full: dict, scratch5: dict, features: pd.DataFrame, descriptor_manifest: dict) -> list[Path]:
    apply_publication_style(font_size=12)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), layout="constrained")
    ax = axes[0]
    rows = []
    for label, audit in (("Full grouped", full), ("Scratch-5%", scratch5)):
        summary = audit["summaries"]["train"]
        rows.append((label, "SOAP", summary["soap_cosine_distance"]["min"] / audit["thresholds"]["near_soap_cosine"]))
        rows.append((label, "Cr-local RMSD", summary["cr_local_species_rmsd_A"]["min"] / audit["thresholds"]["near_rmsd_A"]))
    audit_frame = pd.DataFrame(rows, columns=["split", "metric", "ratio"])
    x = np.arange(2)
    width = 0.35
    colors = {"SOAP": "#6E4C9B", "Cr-local RMSD": "#4C4C4C"}
    for index, metric in enumerate(["SOAP", "Cr-local RMSD"]):
        values = audit_frame.loc[audit_frame.metric.eq(metric), "ratio"].to_numpy()
        bars = ax.bar(x + (index - 0.5) * width, values, width=width,
                      color=colors[metric], edgecolor=INK, linewidth=0.65, label=metric)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.14, f"{value:.1f}$\\times$",
                    ha="center", va="bottom", fontsize=11.2, fontweight="bold", color=INK)
    ax.axhline(1, color="#666666", linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.55, 85)
    ax.set_yticks([1, 10, 40])
    ax.set_yticklabels(["1", "10", "40"])
    ax.set_xticks(x, ["Full grouped", "Scratch-5%"])
    ax.set_ylabel("nearest distance / near-duplicate cutoff")
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, 0.995), ncol=2,
              frameon=False, fontsize=11.2, handlelength=1.5, columnspacing=1.1)
    set_axes_style(ax)
    ax.text(-0.14, 1.02, "a", transform=ax.transAxes, fontsize=15, fontweight="bold",
            ha="left", va="bottom")

    ax = axes[1]
    stage_map = {
        "stable_early": "early stable",
        "stable_pre": "pre-onset",
        "precursor": "precursor",
        "threshold": "threshold",
    }
    d = features.loc[:, ["window", "distance_to_train"]].copy()
    d["stage"] = d["window"].map(stage_map)
    order = list(stage_map.values())
    position = np.arange(len(order))
    rng = np.random.default_rng(20261018)
    for i, stage in enumerate(order):
        values = d.loc[d.stage.eq(stage), "distance_to_train"].to_numpy(float)
        jitter = rng.uniform(-0.09, 0.09, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=18, alpha=0.52,
                   color=FAILURE_COLORS[stage], edgecolor="white", linewidth=0.22,
                   rasterized=True, zorder=2)
        median = np.median(values)
        ax.plot([i - 0.18, i + 0.18], [median, median], color=INK, lw=2.0, zorder=3)
    q99 = descriptor_manifest["distribution_fit"]["train_knn_q99"]
    ax.axhline(q99, color="#666666", linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xticks(position, ["early\nstable", "pre-\nonset", "precursor", "threshold"])
    ax.set_ylim(0, 15.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_ylabel("20-PC SOAP kNN distance to training")
    set_axes_style(ax)
    ax.text(-0.14, 1.02, "b", transform=ax.transAxes, fontsize=15, fontweight="bold",
            ha="left", va="bottom")
    return save_figure(fig, "fig_a2_distance_audits")


def write_derived_tables(full: dict, scratch5: dict, descriptor_manifest: dict,
                         reference: pd.DataFrame, features: pd.DataFrame,
                         paths: dict[str, Path]) -> list[Path]:
    tables = PACKAGE / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for key, path in paths.items():
        source_rows.append({"input_key": key, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    sources = pd.DataFrame(source_rows)
    source_path = tables / "source_manifest.csv"
    sources.to_csv(source_path, index=False)

    train_summary = full["summaries"]["train"]
    scratch_summary = scratch5["summaries"]["train"]
    summary_rows = [
        {
            "evidence_layer": "A2 direct leakage audit", "configuration_group": "Full grouped FT-600K training",
            "n": train_summary["total_frames"], "metric": "minimum SOAP cosine distance to 25 DFT-path images",
            "value": train_summary["soap_cosine_distance"]["min"], "reference": "near-duplicate cutoff 1e-4",
            "status": "VERIFIED", "interpretation": "No SOAP-near image under the declared cutoff.",
        },
        {
            "evidence_layer": "A2 direct leakage audit", "configuration_group": "Full grouped FT-600K training",
            "n": train_summary["local_rmsd_comparable_frames"], "metric": "minimum comparable Cr-local RMSD (A)",
            "value": train_summary["cr_local_species_rmsd_A"]["min"], "reference": "near-duplicate cutoff 0.05 A",
            "status": "VERIFIED", "interpretation": "No comparable local environment is RMSD-near under the declared cutoff.",
        },
        {
            "evidence_layer": "A2 direct leakage audit", "configuration_group": "Scratch-5% grouped subset",
            "n": scratch_summary["total_frames"], "metric": "minimum SOAP cosine distance to 25 DFT-path images",
            "value": scratch_summary["soap_cosine_distance"]["min"], "reference": "near-duplicate cutoff 1e-4",
            "status": "VERIFIED", "interpretation": "No SOAP-near image under the declared cutoff.",
        },
        {
            "evidence_layer": "A2 direct leakage audit", "configuration_group": "Scratch-5% grouped subset",
            "n": scratch_summary["local_rmsd_comparable_frames"], "metric": "minimum comparable Cr-local RMSD (A)",
            "value": scratch_summary["cr_local_species_rmsd_A"]["min"], "reference": "near-duplicate cutoff 0.05 A",
            "status": "VERIFIED", "interpretation": "No comparable local environment is RMSD-near under the declared cutoff.",
        },
        {
            "evidence_layer": "Historical common SOAP/PCA map", "configuration_group": "AIMD training reference",
            "n": int((reference.reference == "train").sum()), "metric": "shared 20-PC SOAP reference fit",
            "value": np.nan, "reference": "historical descriptor fit", "status": "DESCRIPTIVE",
            "interpretation": "Visualization reference only; not the grouped FT-600K training file.",
        },
        {
            "evidence_layer": "Historical common SOAP/PCA map", "configuration_group": "Held-out AIMD reference",
            "n": int(reference.reference.isin(["test", "valid"]).sum()), "metric": "shared 20-PC SOAP reference fit",
            "value": np.nan, "reference": "historical descriptor fit", "status": "DESCRIPTIVE",
            "interpretation": "Held-out AIMD overlaps the reference manifold in the plotted projection.",
        },
        {
            "evidence_layer": "Historical common SOAP/PCA map", "configuration_group": "Fixed-path NEB reference",
            "n": int((reference.reference == "neb").sum()), "metric": "shared 20-PC SOAP reference fit",
            "value": np.nan, "reference": "historical descriptor fit", "status": "DESCRIPTIVE",
            "interpretation": "A distinct projected region; a 2D projection is not a calibrated OOD distance.",
        },
        {
            "evidence_layer": "Historical failure attribution", "configuration_group": "Large-cell MD failure-attribution sample",
            "n": len(features), "metric": "outside historical training kNN q99 fraction",
            "value": float(features.outside_train_q99.mean()), "reference": "q99 = %.6f" % descriptor_manifest["distribution_fit"]["train_knn_q99"],
            "status": "DESCRIPTIVE", "interpretation": "Stage-balanced sample; not a time-weighted MD population or a proof of causal failure.",
        },
        {
            "evidence_layer": "Self-relaxed NEB", "configuration_group": "Current model-in-the-loop paths",
            "n": np.nan, "metric": "common grouped-training SOAP map",
            "value": np.nan, "reference": "not yet generated", "status": "VALIDATION_GAP",
            "interpretation": "Raw self-relaxed paths exist, but a common descriptor transform with the grouped FT-600K train set is absent locally.",
        },
    ]
    summary = pd.DataFrame(summary_rows)
    summary_path = tables / "task_distribution_evidence_summary.csv"
    summary.to_csv(summary_path, index=False)

    stage_map = {"stable_early": "early stable", "stable_pre": "pre-onset", "precursor": "precursor", "threshold": "threshold"}
    stage = features.assign(stage=features.window.map(stage_map)).groupby("stage", observed=True).agg(
        n=("distance_to_train", "size"),
        kNN_distance_min=("distance_to_train", "min"),
        kNN_distance_median=("distance_to_train", "median"),
        kNN_distance_max=("distance_to_train", "max"),
        time_ps_median=("time_ps", "median"),
        max_force_median_eV_A=("max_force_eV_A", "median"),
    ).reindex(list(stage_map.values())).reset_index()
    stage_path = tables / "failure_sample_distance_by_stage.csv"
    stage.to_csv(stage_path, index=False)
    return [source_path, summary_path, stage_path]


def write_manifest(paths: dict[str, Path], figures: list[Path], tables: list[Path]) -> Path:
    manifest = {
        "schema_version": "1.0",
        "package": "reviewer_a2_distribution_shift_task_ladder_20261018",
        "purpose": "Separate A2 direct leakage evidence from descriptive task-distribution diagnostics.",
        "figures": [{"path": str(path), "sha256": sha256(path)} for path in figures],
        "tables": [{"path": str(path), "sha256": sha256(path)} for path in tables],
        "inputs": [{"key": key, "path": str(path), "sha256": sha256(path)} for key, path in paths.items()],
        "claim_boundaries": [
            "The grouped audit verifies no exact or near-duplicate target image under the declared fingerprint, SOAP, and comparable local-RMSD metrics.",
            "The historical SOAP/PCA map is descriptive and uses a different, 521-frame historical AIMD reference fit; it is not a replacement for the grouped A2 leakage audit.",
            "The 639-frame failure-attribution set is stage-balanced and comes from large-cell MD branches; it is not a time-weighted trajectory population and does not establish a causal instability mechanism.",
            "Self-relaxed NEB structures are not placed in the common grouped-training descriptor space here; this remains a validation gap.",
        ],
    }
    target = PACKAGE / "distribution_task_ladder_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate inputs and print output paths without rendering figures.")
    args = parser.parse_args()
    paths = source_paths()
    require_sources(paths)
    full, scratch5, descriptor_manifest, reference, features = load_inputs(paths)
    if args.check:
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return
    map_frame = map_data(reference, features)
    figures = draw_task_map(map_frame)
    figures += draw_audit_panel(full, scratch5, features, descriptor_manifest)
    tables = write_derived_tables(full, scratch5, descriptor_manifest, reference, features, paths)
    manifest = write_manifest(paths, figures, tables)
    print(json.dumps({"figures": [str(path) for path in figures], "tables": [str(path) for path in tables], "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
