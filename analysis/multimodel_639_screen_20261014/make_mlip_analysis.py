from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from plot_atlas import apply_publication_style, bold_tick_labels, get_palette, save_figure


MODEL_ORDER = [
    "ablation_lmax_2_seed123",
    "ablation_lmax_2_seed234",
    "baseline_seed345",
]
MODEL_LABELS = {
    "ablation_lmax_2_seed123": "lmax2 / seed 123",
    "ablation_lmax_2_seed234": "lmax2 / seed 234",
    "baseline_seed345": "baseline / seed 345",
}
GROUP_ORDER = [
    "stable_early_control",
    "stable_pre_onset",
    "precursor_near_onset",
    "threshold_or_terminal",
    "terminal_stable_target",
]
WINDOW_ORDER = ["stable_early", "stable_pre", "precursor", "threshold"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def style(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#222222")
    ax.tick_params(width=1.1, length=5)
    bold_tick_labels(ax)


def colors() -> dict[str, str]:
    return dict(zip(MODEL_ORDER, get_palette("nature_steered_dynamics")["categorical"][:3]))


def model_label(value: str) -> str:
    return MODEL_LABELS.get(value, value)


def plot_model_force_distribution(results: pd.DataFrame, output: Path) -> None:
    palette = colors()
    groups = [g for g in GROUP_ORDER if g in set(results["candidate_group"].dropna())]
    fig, ax = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
    positions = []
    labels = []
    for gi, group_name in enumerate(groups):
        for mi, model_id in enumerate(MODEL_ORDER):
            values = pd.to_numeric(results.loc[(results["candidate_group"] == group_name) & (results["model_id"] == model_id) & (results["evaluation_status"] == "success"), "max_force_eV_A"], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                continue
            pos = gi * 4 + mi + 1
            positions.append(pos)
            labels.append(model_label(model_id))
            bp = ax.boxplot(values, positions=[pos], widths=.72, patch_artist=True, showfliers=False,
                            boxprops={"facecolor": palette[model_id], "edgecolor": "#222222", "linewidth": .8},
                            medianprops={"color": "#111111", "linewidth": 1.2},
                            whiskerprops={"color": "#222222", "linewidth": .8},
                            capprops={"color": "#222222", "linewidth": .8})
            jitter = np.linspace(-.20, .20, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), pos) + jitter, values, s=7, color="#222222", alpha=.27, linewidths=0, zorder=3, rasterized=True)
    centers = [gi * 4 + 2 for gi in range(len(groups))]
    ax.set_xticks(centers, [g.replace("_", "\n") for g in groups])
    ax.set_ylabel("MLIP max |F| (eV Å$^{-1}$)")
    ax.set_xlabel("Source trajectory stage / candidate group")
    ax.axhline(100.0, color="#444444", linestyle=(0, (3, 2)), linewidth=1.0, label="MD instability force gate")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0)
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=palette[m], markeredgecolor="#222222", markersize=8, label=model_label(m)) for m in MODEL_ORDER]
    handles.append(Line2D([0], [0], color="#444444", linestyle=(0, (3, 2)), label="100 eV Å$^{-1}$ gate"))
    ax.legend(handles=handles, frameon=True, fontsize=8, ncol=2, loc="upper left")
    ax.set_title("Static force response across all 639 candidates", fontweight="bold")
    style(ax)
    save_figure(fig, output / "fig_mlip_force_by_group.png", dpi=350)
    fig.savefig(output / "fig_mlip_force_by_group.pdf", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    fig.savefig(output / "fig_mlip_force_by_group.svg", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    plt.close(fig)


def plot_spread_relationship(spread: pd.DataFrame, output: Path) -> None:
    palette = get_palette("descriptor_reduction")
    window_colors = {"stable_early": palette["total"], "stable_pre": palette["energy"], "precursor": palette["target"], "threshold": "#222222"}
    fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
    for window in WINDOW_ORDER:
        group = spread[spread["window"] == window]
        if group.empty:
            continue
        ax.scatter(group["max_force_range_eV_A"], group["ensemble_force_std_rms_eV_A"], s=19, alpha=.65,
                   color=window_colors[window], edgecolor="white", linewidth=.3, label=window.replace("_", " "), rasterized=True)
    ax.set_xlabel("Cross-model range of max |F| (eV Å$^{-1}$)")
    ax.set_ylabel("Cross-model force spread RMS (eV Å$^{-1}$)")
    ax.set_title("Same-configuration force disagreement", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, fontsize=8, loc="best")
    style(ax)
    save_figure(fig, output / "fig_mlip_force_spread_relationship.png", dpi=350)
    fig.savefig(output / "fig_mlip_force_spread_relationship.pdf", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    fig.savefig(output / "fig_mlip_force_spread_relationship.svg", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    plt.close(fig)


def plot_stage_heatmap(results: pd.DataFrame, output: Path) -> None:
    groups = [g for g in GROUP_ORDER if g in set(results["candidate_group"].dropna())]
    matrix = np.full((len(groups), len(MODEL_ORDER)), np.nan)
    for i, group in enumerate(groups):
        for j, model_id in enumerate(MODEL_ORDER):
            values = pd.to_numeric(results.loc[(results["candidate_group"] == group) & (results["model_id"] == model_id) & (results["evaluation_status"] == "success"), "max_force_eV_A"], errors="coerce").dropna()
            if len(values):
                matrix[i, j] = values.median()
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    finite = matrix[np.isfinite(matrix)]
    norm = Normalize(vmin=0, vmax=max(100.0, float(np.nanpercentile(finite, 95))) if len(finite) else 100.0)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", norm=norm)
    ax.set_xticks(np.arange(len(MODEL_ORDER)), [model_label(x) for x in MODEL_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(groups)), [x.replace("_", " ") for x in groups])
    for i in range(len(groups)):
        for j in range(len(MODEL_ORDER)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white" if matrix[i, j] < norm.vmax * .55 else "#111111", fontsize=8)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Source candidate group")
    ax.set_title("Median static max force by stage and model", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, pad=.02)
    cbar.set_label("Median max |F| (eV Å$^{-1}$)")
    style(ax)
    save_figure(fig, output / "fig_mlip_stage_heatmap.png", dpi=350)
    fig.savefig(output / "fig_mlip_stage_heatmap.pdf", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    fig.savefig(output / "fig_mlip_stage_heatmap.svg", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    plt.close(fig)


def write_text(results: pd.DataFrame, spread: pd.DataFrame, model_summary: pd.DataFrame, group_summary: pd.DataFrame, out: Path, status: str) -> None:
    lines = [
        "# Three-model MLIP candidate evaluation: audited Results and Discussion",
        "",
        f"Campaign status: `{status}`. This text is generated from the compact MLIP result tables and must not be upgraded beyond the analysis manifest.",
        "",
        "## Scope and protocol",
        "",
        "The 639 saved 2050-atom fully periodic configurations were evaluated as static single points with the three registered checkpoint files: ablation_lmax_2 seed 123, ablation_lmax_2 seed 234, and baseline seed 345. The same candidate structure is therefore paired across models. No geometry relaxation or MD continuation was performed. Energies are retained for provenance but are not used for absolute cross-family ranking; the primary comparison is the force field, force tails, geometry, and paired cross-model disagreement.",
        "",
        "## Coverage and numerical status",
        "",
        f"The compact result table contains {len(results)} model-candidate rows for {results['candidate_id'].nunique()} unique candidates and {results['model_id'].nunique()} model IDs. {int((results['evaluation_status'] == 'success').sum())} rows are finite successful evaluations; {int((results['evaluation_status'] != 'success').sum())} rows remain in the failure ledger. These are numerical evaluation statuses, not evidence that any model is physically correct.",
        "",
        "## Stage comparison",
        "",
        "The following table reports the model-wise median static maximum force for each source stage/group. `stable_early` and `stable_pre` are source-trajectory labels; the MLIP evaluation itself does not redefine the onset gate.",
        "",
        "| candidate group | window | model | n | success | median max |F| (eV Å^-1) | q95 max |F| (eV Å^-1) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in model_summary.sort_values(["candidate_group", "window", "model_id"]).itertuples():
        lines.append(f"| {row.candidate_group} | {row.window} | {model_label(row.model_id)} | {row.n_candidates} | {row.n_success} | {row.max_force_eV_A_median:.3g} | {row.max_force_eV_A_q95:.3g} |")
    lines += [
        "",
        "## Mechanistic interpretation and boundary",
        "",
        "The paired same-configuration force spread is the direct MLIP diagnostic for this campaign. If the spread and model-specific force ranking increase from stable-pre to precursor/threshold, the result supports a model-dependent amplification of forces in the extrapolative region. If the three models remain close while all show the same geometric deterioration, the evidence instead supports a shared geometry-driven trigger. Neither pattern establishes which force is DFT-correct: the models share training lineage and the 2050-atom DFT comparison remains unavailable.",
        "",
        "Descriptor novelty remains descriptive distributional evidence. A high force spread is not a calibrated uncertainty and is not a causal proof that the lmax reduction caused the runaway. Threshold-stopped MD branches remain `FAILURE_MODE`/`unstable — no Dxy`; this static campaign does not create a transport result.",
        "",
        "## Reproducible claim status",
        "",
        "- `VERIFIED` for: candidate/model row coverage, checkpoint hashes, finite MLIP energy/force outputs, paired force-spread calculations, and stage-wise summaries when the analysis manifest passes.",
        "- `FAILURE_MODE` for: any model-candidate runtime or nonfinite rows, retained in `failure_ledger.csv`.",
        "- `PROVENANCE_PARTIAL` for: physical correctness relative to DFT, DFT parity of the local reconstruction, and any causal statement about why the MD runaway occurs.",
        "",
        "The next DFT step should select configurations where (i) the same candidate has a large paired force spread, (ii) the spread rises before the source onset, and (iii) the local periodic reconstruction passes the finite/minimum-distance/PBC gate. Those inputs remain DFT-ready candidates, not DFT results, until QE convergence and provenance gates pass.",
    ]
    (out / "MLIP_RESULTS_DISCUSSION.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    analysis = Path(args.analysis)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(analysis / "mlip_candidate_results_all.csv")
    spread = pd.read_csv(analysis / "candidate_model_spread.csv")
    model_summary = pd.read_csv(analysis / "model_stage_summary.csv")
    group_summary = pd.read_csv(analysis / "candidate_group_summary.csv")
    status = json.loads((analysis / "analysis_manifest.json").read_text()).get("status", "UNKNOWN")
    plot_model_force_distribution(results, out)
    if not spread.empty:
        plot_spread_relationship(spread, out)
        plot_stage_heatmap(results, out)
    model_summary.to_csv(out / "table_mlip_model_stage_summary.csv", index=False)
    group_summary.to_csv(out / "table_mlip_candidate_group_summary.csv", index=False)
    results.groupby(["candidate_group", "window"], dropna=False).size().rename("n_model_rows").reset_index().to_csv(out / "table_mlip_candidate_counts.csv", index=False)
    write_text(results, spread, model_summary, group_summary, out, status)
    figure_files = [p for p in sorted(out.iterdir()) if p.suffix in {".png", ".svg", ".pdf"}]
    manifest = {"schema": "migrationbench_mlip_candidate_analysis_v1", "status": status,
                "analysis_manifest_sha256": sha256(analysis / "analysis_manifest.json"),
                "inputs": {p.name: sha256(p) for p in [analysis / "mlip_candidate_results_all.csv", analysis / "candidate_model_spread.csv", analysis / "model_stage_summary.csv", analysis / "candidate_group_summary.csv"] if p.is_file()},
                "plot_script_sha256": sha256(Path(__file__)),
                "outputs": {p.name: sha256(p) for p in figure_files + [out / "MLIP_RESULTS_DISCUSSION.md", out / "table_mlip_model_stage_summary.csv", out / "table_mlip_candidate_group_summary.csv", out / "table_mlip_candidate_counts.csv"] if p.is_file()},
                "interpretation_boundary": "paired static MLIP evidence; not DFT correctness or causal proof"}
    (out / "figure_and_text_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": status, "n_rows": len(results), "n_figures": len(figure_files)}, indent=2))


if __name__ == "__main__":
    main()
