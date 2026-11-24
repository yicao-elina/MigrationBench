#!/usr/bin/env python3
"""Render publication-style endpoint-47 figures using the local Plot Atlas style."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from scipy.stats import spearmanr

from plot_atlas import apply_publication_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
FIGURES = ROOT / "figures"
INPUTS = ROOT / "raw" / "rockfish_mace_inputs"

COLORS = {
    "within_vdW_gap": "#2B6CB0",
    "gap_QL_interfacial": "#DD6B20",
    "within_QL": "#2F855A",
    "failure": "#A61B1B",
    "neutral": "#6B7280",
}
REGION_LABELS = {
    "within_vdW_gap": "within gap",
    "gap_QL_interfacial": "gap–QL interface",
    "within_QL": "within QL",
}
MODEL_ORDER = [
    "FT-600K",
    "FT-MultiT",
    "OMAT-derived fine-tuned legacy Foundation",
    "Scratch",
]
MODEL_SHORT = {
    "FT-600K": "FT-600K",
    "FT-MultiT": "FT-MultiT",
    "OMAT-derived fine-tuned legacy Foundation": "OMAT-derived\nlegacy",
    "Scratch": "Scratch",
}


def load_endpoint() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "DFT_ENDPOINT_47_CLASSIFICATION.tsv", sep="\t")


def letter(ax, value: str) -> None:
    ax.text(
        -0.11,
        1.035,
        value,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def legend_regions(ax, *, loc: str = "best", fontsize: float = 7.5) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[key],
            markeredgecolor="white",
            markersize=5.8,
            label=REGION_LABELS[key],
        )
        for key in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]
    ]
    ax.legend(handles=handles, frameon=False, loc=loc, fontsize=fontsize, handletextpad=0.35)


def save_all(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIGURES / f"{stem}.png")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def fig1_lattice_distribution(endpoint: pd.DataFrame) -> None:
    apply_publication_style(8.5)
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    reference = read(INPUTS / "epmlff_001_unique_020_base" / "input.extxyz")
    symbols = np.array(reference.get_chemical_symbols())
    positions = reference.get_positions()
    cell = np.asarray(reference.cell)
    candidate_cart = endpoint[["cr_frac_x", "cr_frac_y", "cr_frac_z"]].to_numpy() @ cell
    fig = plt.figure(figsize=(7.35, 6.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 0.92], hspace=0.46, wspace=0.34)
    ax_xz = fig.add_subplot(grid[0, 0])
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_z = fig.add_subplot(grid[1, :])

    for symbol, color, size, label in [
        ("Te", "#9DBBC9", 14, "host Te"),
        ("Sb", "#59636C", 12, "host Sb"),
    ]:
        subset = positions[symbols == symbol]
        ax_xz.scatter(subset[:, 0], subset[:, 2], s=size, color=color, alpha=0.62, linewidths=0, label=label)
    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        mask = endpoint["vertical_region_geometric"].eq(region).to_numpy()
        ax_xz.scatter(
            candidate_cart[mask, 0],
            candidate_cart[mask, 2],
            s=38,
            color=COLORS[region],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
    ax_xz.set_xlabel("x (Å)", fontsize=9)
    ax_xz.set_ylabel("z (Å)", fontsize=9)
    ax_xz.set_title("Projected host lattice and Cr candidate sites", loc="left", pad=5, fontsize=10, fontweight="bold")
    ax_xz.legend(frameon=False, fontsize=6.8, loc="lower right")
    letter(ax_xz, "a")

    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        subset = endpoint.loc[endpoint["vertical_region_geometric"].eq(region)]
        ax_xy.scatter(
            subset["cr_frac_x"],
            subset["cr_frac_y"],
            s=44,
            color=COLORS[region],
            edgecolor="white",
            linewidth=0.65,
            label="_nolegend_",
        )
    gap_to_ql = endpoint["route_families_multilabel"].str.contains("gap-to-QL penetration", na=False)
    ax_xy.scatter(
        endpoint.loc[gap_to_ql, "cr_frac_x"],
        endpoint.loc[gap_to_ql, "cr_frac_y"],
        s=82,
        facecolor="none",
        edgecolor="#111827",
        linewidth=0.9,
        label="gap-to-QL route member",
    )
    ax_xy.set_xlabel("fractional x", fontsize=9)
    ax_xy.set_ylabel("fractional y", fontsize=9)
    ax_xy.set_xlim(-0.035, 1.035)
    ax_xy.set_ylim(-0.035, 1.035)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title("In-plane source-site coverage", loc="left", pad=5, fontsize=10, fontweight="bold")
    ax_xy.legend(frameon=False, fontsize=6.5, loc="lower left", handletextpad=0.25)
    letter(ax_xy, "b")

    mean_lower = endpoint["central_lower_te_surface_frac_z"].mean()
    mean_upper = endpoint["central_upper_te_surface_frac_z"].mean()
    margin = 0.040
    ax_z.axhspan(mean_lower, mean_upper, color=COLORS["within_vdW_gap"], alpha=0.11, lw=0)
    ax_z.axhspan(mean_lower - margin, mean_lower, color=COLORS["gap_QL_interfacial"], alpha=0.10, lw=0)
    ax_z.axhspan(mean_upper, mean_upper + margin, color=COLORS["gap_QL_interfacial"], alpha=0.10, lw=0)
    sort = endpoint.sort_values("cr_frac_z").reset_index(drop=True)
    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        subset = sort.loc[sort["vertical_region_geometric"].eq(region)]
        ax_z.scatter(
            subset.index + 1,
            subset["cr_frac_z"],
            s=38,
            color=COLORS[region],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    ax_z.axhline(mean_lower, color="#6B7280", lw=0.85, ls="--")
    ax_z.axhline(mean_upper, color="#6B7280", lw=0.85, ls="--")
    ax_z.set_xlim(0, 48)
    ax_z.set_xlabel("candidate site rank along c", fontsize=9)
    ax_z.set_ylabel("Cr fractional z", fontsize=9)
    ax_z.set_title("Vertical classification from the two central Te surfaces", loc="left", pad=5, fontsize=10, fontweight="bold")
    legend_regions(ax_z, loc="upper left", fontsize=6.8)
    letter(ax_z, "c")
    fig.text(
        0.5,
        0.006,
        "All coordinates are immutable MACE source inputs; colors are geometric regions, not stability labels.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#4B5563",
    )
    save_all(fig, "FIG1_ENDPOINT47_LATTICE_SITE_DISTRIBUTION")


def fig2_retry_energy(endpoint: pd.DataFrame) -> None:
    apply_publication_style(8.5)
    valid = endpoint.loc[endpoint["relaxation_class"].eq("DFT_LOCAL_MINIMUM_CANDIDATE")].copy()
    valid = valid.sort_values("dft_relative_energy_eV_retry21").reset_index(drop=True)
    fig = plt.figure(figsize=(7.35, 5.95))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 0.9], height_ratios=[1.0, 0.94], hspace=0.46, wspace=0.42)
    ax_rank = fig.add_subplot(grid[:, 0])
    ax_feature = fig.add_subplot(grid[0, 1])
    ax_region = fig.add_subplot(grid[1, 1])

    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        subset = valid.loc[valid["vertical_region_geometric"].eq(region)]
        y = subset.index + 1
        ax_rank.hlines(y, 0, subset["dft_relative_energy_eV_retry21"], color=COLORS[region], lw=1.05, alpha=0.56)
        ax_rank.scatter(
            subset["dft_relative_energy_eV_retry21"],
            y,
            s=44,
            color=COLORS[region],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax_rank.set_xlabel("DFT ΔE within converged retry-21 (eV)", fontsize=9)
    ax_rank.set_ylabel("endpoint rank", fontsize=9)
    ax_rank.set_yticks([1, 5, 10, 15, 21])
    ax_rank.set_ylim(0.25, 21.75)
    ax_rank.invert_yaxis()
    ax_rank.set_title("Restricted same-campaign relative-energy window", loc="left", pad=5, fontsize=10, fontweight="bold")
    legend_regions(ax_rank, loc="lower right", fontsize=6.8)
    letter(ax_rank, "a")

    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        subset = valid.loc[valid["vertical_region_geometric"].eq(region)]
        ax_feature.scatter(
            subset["cr_to_vdw_gap_center_A"],
            subset["dft_relative_energy_eV_retry21"],
            s=46,
            color=COLORS[region],
            edgecolor="white",
            linewidth=0.65,
        )
    rho, pvalue = spearmanr(valid["cr_to_vdw_gap_center_A"], valid["dft_relative_energy_eV_retry21"])
    ax_feature.text(
        0.04,
        0.96,
        f"Spearman ρ = {rho:.2f}\np = {pvalue:.3f}\nexploratory",
        transform=ax_feature.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        color="#374151",
    )
    ax_feature.set_xlabel("|Cr − gap center| (Å)", fontsize=9)
    ax_feature.set_ylabel("DFT ΔE (eV)", fontsize=9)
    ax_feature.set_title("Energy versus vertical placement", loc="left", pad=5, fontsize=10, fontweight="bold")
    letter(ax_feature, "b")

    region_order = ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]
    positions = np.arange(1, len(region_order) + 1)
    groups = [
        valid.loc[valid["vertical_region_geometric"].eq(region), "dft_relative_energy_eV_retry21"].to_numpy()
        for region in region_order
    ]
    violin = ax_region.violinplot(groups, positions=positions, showmeans=False, showmedians=True, widths=0.75)
    for body, region in zip(violin["bodies"], region_order):
        body.set_facecolor(COLORS[region])
        body.set_alpha(0.42)
        body.set_edgecolor("none")
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        violin[key].set_color("#374151")
        violin[key].set_linewidth(0.8)
    rng = np.random.default_rng(20261023)
    for pos, group, region in zip(positions, groups, region_order):
        ax_region.scatter(
            pos + rng.uniform(-0.10, 0.10, size=len(group)),
            group,
            s=22,
            color=COLORS[region],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax_region.set_xticks(positions, ["within gap", "interface", "within QL"])
    ax_region.set_ylabel("DFT ΔE (eV)", fontsize=9)
    ax_region.set_title("Energy distribution by geometric region", loc="left", pad=5, fontsize=10, fontweight="bold")
    letter(ax_region, "c")
    fig.text(
        0.5,
        0.008,
        "Only 21 BFGS-final retry endpoints are shown. unique_077/078/093 are time-limit nonconverged and excluded.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#4B5563",
    )
    save_all(fig, "FIG2_RETRY21_DFT_ENERGY_STRUCTURE")


def fig3_mace_sensitivity(endpoint: pd.DataFrame) -> None:
    apply_publication_style(8.5)
    summary = pd.read_csv(PROCESSED / "MACE_FOUR_MODEL_ENDPOINT_MODEL_SUMMARY.tsv", sep="\t")
    fig = plt.figure(figsize=(7.35, 6.15))
    grid = fig.add_gridspec(2, 2, hspace=0.47, wspace=0.42)
    ax_span = fig.add_subplot(grid[0, 0])
    ax_disp = fig.add_subplot(grid[0, 1])
    ax_heat = fig.add_subplot(grid[1, 0])
    ax_fail = fig.add_subplot(grid[1, 1])

    rng = np.random.default_rng(20261023)
    for position, model in enumerate(MODEL_ORDER, start=1):
        values = summary.loc[
            summary["model_label_interpreted"].eq(model) & summary["mace_within_model_energy_span_eV"].notna(),
            "mace_within_model_energy_span_eV",
        ].to_numpy()
        x = position + rng.uniform(-0.12, 0.12, size=len(values))
        color = "#4C78A8" if model != "Scratch" else COLORS["failure"]
        ax_span.scatter(x, values, s=19, color=color, alpha=0.65, edgecolor="white", linewidth=0.3)
        ax_span.hlines(np.median(values), position - 0.27, position + 0.27, color="#111827", lw=1.25)
    ax_span.set_yscale("log")
    ax_span.set_xticks(range(1, 5), [MODEL_SHORT[item] for item in MODEL_ORDER])
    ax_span.set_ylabel("three-start energy span (eV)", fontsize=9)
    ax_span.set_title("Within-model local-basin sensitivity", loc="left", pad=5, fontsize=10, fontweight="bold")
    ax_span.tick_params(axis="x", labelsize=7.2)
    letter(ax_span, "a")

    valid = endpoint.loc[endpoint["mace_cross_model_cr_displacement_max_A"].notna()].copy()
    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        subset = valid.loc[valid["vertical_region_geometric"].eq(region)]
        ax_disp.scatter(
            subset["cr_to_vdw_gap_center_A"],
            subset["mace_cross_model_cr_displacement_max_A"],
            s=45,
            color=COLORS[region],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    top = valid.nlargest(3, "mace_cross_model_cr_displacement_max_A")
    for row in top.itertuples():
        ax_disp.annotate(
            row.endpoint.replace("unique_", ""),
            (row.cr_to_vdw_gap_center_A, row.mace_cross_model_cr_displacement_max_A),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.8,
            color="#374151",
        )
    ax_disp.set_xlabel("|Cr − gap center| (Å)", fontsize=9)
    ax_disp.set_ylabel("max cross-model Cr displacement (Å)", fontsize=9)
    ax_disp.set_title("Four-model final-basin disagreement", loc="left", pad=5, fontsize=10, fontweight="bold")
    legend_regions(ax_disp, loc="upper left", fontsize=6.4)
    letter(ax_disp, "b")

    merged = summary.merge(
        endpoint[["endpoint", "vertical_region_geometric"]], on="endpoint", how="left"
    )
    regions = ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]
    matrix = np.empty((len(MODEL_ORDER), len(regions)))
    for i, model in enumerate(MODEL_ORDER):
        for j, region in enumerate(regions):
            vals = merged.loc[
                merged["model_label_interpreted"].eq(model)
                & merged["vertical_region_geometric"].eq(region),
                "mace_within_model_energy_span_eV",
            ]
            matrix[i, j] = vals.median()
    image = ax_heat.imshow(np.log10(matrix), cmap="viridis", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_heat.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6.6, color="white")
    ax_heat.set_xticks(range(3), ["within\ngap", "interface", "within\nQL"])
    ax_heat.set_yticks(range(4), [MODEL_SHORT[item] for item in MODEL_ORDER])
    ax_heat.set_title("Median span by model and vertical region", loc="left", pad=5, fontsize=10, fontweight="bold")
    ax_heat.tick_params(axis="both", labelsize=7.2)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.052, pad=0.04)
    cbar.set_label("log10 span (eV)", fontsize=8)
    letter(ax_heat, "c")

    failures = (
        summary.groupby("model_label_interpreted")["mace_failure_starts"].sum().reindex(MODEL_ORDER).fillna(0)
    )
    bar_colors = ["#9CA3AF", "#9CA3AF", "#9CA3AF", COLORS["failure"]]
    ax_fail.bar(range(4), failures.to_numpy(), color=bar_colors, width=0.65)
    for i, value in enumerate(failures.to_numpy()):
        ax_fail.text(i, value + 0.08, f"{int(value)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_fail.set_xticks(range(4), [MODEL_SHORT[item] for item in MODEL_ORDER])
    ax_fail.set_ylim(0, 3.75)
    ax_fail.set_ylabel("failed relaxation starts", fontsize=9)
    ax_fail.set_title("Failure evidence is retained", loc="left", pad=5, fontsize=10, fontweight="bold")
    ax_fail.tick_params(axis="x", labelsize=7.2)
    ax_fail.text(
        0.96,
        0.93,
        "Scratch: 3/3 failures\nat unique_151",
        transform=ax_fail.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#7F1D1D",
    )
    letter(ax_fail, "d")
    fig.text(
        0.5,
        0.006,
        "All panels show model-internal sensitivity or geometric disagreement; no cross-method total-energy subtraction is used.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#4B5563",
    )
    save_all(fig, "FIG3_FOUR_MODEL_BASIN_SENSITIVITY")


def fig4_source_ensemble() -> None:
    apply_publication_style(8.5)
    endpoint = pd.read_csv(PROCESSED / "DFT_MACE47_INTEGRATED_ENDPOINT_TABLE.tsv", sep="\t")
    records = pd.read_csv(PROCESSED / "MACE12_SOURCE_GEOMETRY_RECORDS.tsv", sep="\t")
    stats = pd.read_csv(PROCESSED / "MACE12_SOURCE_GEOMETRY_ASSOCIATIONS.tsv", sep="\t")
    registry = pd.read_csv(ROOT / "MODEL_REGISTRY.tsv", sep="\t")
    fig = plt.figure(figsize=(7.35, 6.2))
    grid = fig.add_gridspec(2, 2, hspace=0.54, wspace=0.45)
    ax_pca = fig.add_subplot(grid[0, 0])
    ax_gap = fig.add_subplot(grid[0, 1])
    ax_cn = fig.add_subplot(grid[1, 0])
    ax_rank = fig.add_subplot(grid[1, 1])
    metric = "ensemble_cr_force_vector_std_rms_eV_A"
    marker_map = {"within_vdW_gap": "o", "gap_QL_interfacial": "s", "within_QL": "^"}

    scatter = None
    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        subset = endpoint.loc[endpoint["vertical_region_geometric"].eq(region)]
        scatter = ax_pca.scatter(
            subset["descriptor_PC1"],
            subset["descriptor_PC2"],
            c=subset[metric],
            cmap="magma",
            s=42,
            marker=marker_map[region],
            edgecolor="white",
            linewidth=0.5,
            vmin=endpoint[metric].min(),
            vmax=endpoint[metric].max(),
        )
    top = endpoint.nlargest(4, metric)
    for row in top.itertuples():
        ax_pca.annotate(
            row.endpoint.replace("unique_", ""),
            (row.descriptor_PC1, row.descriptor_PC2),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            color="#374151",
        )
    handles = [
        plt.Line2D([0], [0], marker=marker_map[key], color="#374151", markerfacecolor="white",
                   markersize=5.5, linestyle="none", label=REGION_LABELS[key])
        for key in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]
    ]
    ax_pca.legend(handles=handles, frameon=False, fontsize=6.3, loc="upper left", handletextpad=0.25)
    ax_pca.set_xlabel("structural descriptor PC1", fontsize=8.8)
    ax_pca.set_ylabel("structural descriptor PC2", fontsize=8.8)
    ax_pca.set_title("Candidate structural-descriptor projection", loc="left", pad=5, fontsize=9.5, fontweight="bold")
    colorbar = fig.colorbar(scatter, ax=ax_pca, fraction=0.052, pad=0.04)
    colorbar.set_label("12-model Cr-force vector SD (eV Å$^{-1}$)", fontsize=7.4)
    colorbar.ax.tick_params(labelsize=7)
    letter(ax_pca, "a")

    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        subset = endpoint.loc[endpoint["vertical_region_geometric"].eq(region)]
        ax_gap.scatter(
            subset["cr_to_vdw_gap_center_A"],
            subset[metric],
            color=COLORS[region],
            s=43,
            marker=marker_map[region],
            edgecolor="white",
            linewidth=0.5,
        )
    line_x = np.linspace(endpoint["cr_to_vdw_gap_center_A"].min(), endpoint["cr_to_vdw_gap_center_A"].max(), 100)
    slope, intercept = np.polyfit(endpoint["cr_to_vdw_gap_center_A"], endpoint[metric], 1)
    ax_gap.plot(line_x, slope * line_x + intercept, color="#374151", lw=1.0)
    gap_stat = stats.loc[
        stats["outcome"].eq(metric) & stats["feature"].eq("cr_to_vdw_gap_center_A")
    ].iloc[0]
    ax_gap.text(
        0.04,
        0.96,
        f"Spearman ρ = {gap_stat.effect_size:.2f}\nBH q = {gap_stat.p_value_BH_FDR:.1e}",
        transform=ax_gap.transAxes,
        va="top",
        fontsize=7.1,
        color="#374151",
    )
    ax_gap.set_xlabel("|Cr − gap center| (Å)", fontsize=8.8)
    ax_gap.set_ylabel("12-model Cr-force vector SD (eV Å$^{-1}$)", fontsize=8.8)
    ax_gap.set_title("Disagreement versus distance from gap center", loc="left", pad=5, fontsize=9.5, fontweight="bold")
    letter(ax_gap, "b")

    for region in ["within_vdW_gap", "gap_QL_interfacial", "within_QL"]:
        subset = endpoint.loc[endpoint["vertical_region_geometric"].eq(region)]
        ax_cn.scatter(
            subset["n_Te_within_3p2_A"],
            subset[metric],
            color=COLORS[region],
            s=43,
            marker=marker_map[region],
            edgecolor="white",
            linewidth=0.5,
        )
    cn_stat = stats.loc[
        stats["outcome"].eq(metric) & stats["feature"].eq("n_Te_within_3p2_A")
    ].iloc[0]
    ax_cn.text(
        0.04,
        0.96,
        f"Spearman ρ = {cn_stat.effect_size:.2f}\nBH q = {cn_stat.p_value_BH_FDR:.1e}",
        transform=ax_cn.transAxes,
        va="top",
        fontsize=7.1,
        color="#374151",
    )
    ax_cn.set_xticks([3, 4, 5, 6])
    ax_cn.set_xlabel("Te neighbors within 3.2 Å", fontsize=8.8)
    ax_cn.set_ylabel("12-model Cr-force vector SD (eV Å$^{-1}$)", fontsize=8.8)
    ax_cn.set_title("Disagreement versus local Te coordination", loc="left", pad=5, fontsize=9.5, fontweight="bold")
    letter(ax_cn, "c")

    model_order = registry["analysis_model_id"].tolist()
    labels = {
        "foundation_omat_legacy": "OMAT legacy",
        "foundation_mpa0_medium": "MPA-0",
        "foundation_mp0b3_medium": "MP-0b3",
        "ft600k_seed123": "600K-123",
        "ft600k_seed234": "600K-234",
        "ft600k_seed345": "600K-345",
        "ftmultit_seed123": "MultiT-123",
        "ftmultit_seed234": "MultiT-234",
        "ftmultit_seed345": "MultiT-345",
        "scratch_seed123": "Scratch-123",
        "scratch_seed234": "Scratch-234",
        "scratch_seed345": "Scratch-345",
    }
    endpoint_order = endpoint.sort_values(metric)["endpoint"].tolist()
    rank_matrix = (
        records.pivot(index="analysis_model_id", columns="endpoint", values="relative_energy_rank_within_model")
        .reindex(index=model_order, columns=endpoint_order)
        .to_numpy(float)
    )
    image = ax_rank.imshow(rank_matrix, cmap="viridis", aspect="auto", vmin=1, vmax=47)
    for edge in [2.5, 5.5, 8.5]:
        ax_rank.axhline(edge, color="white", lw=1.0)
    ax_rank.set_yticks(range(len(model_order)), [labels[item] for item in model_order])
    ax_rank.tick_params(axis="y", labelsize=6.2)
    ax_rank.set_xticks([0, 23, 46], ["low", "median", "high"])
    ax_rank.tick_params(axis="x", labelsize=6.8)
    ax_rank.set_xlabel("endpoint ordered by force disagreement", fontsize=8.2)
    ax_rank.set_title("Within-model relative-energy rank", loc="left", pad=5, fontsize=9.5, fontweight="bold")
    rank_bar = fig.colorbar(image, ax=ax_rank, fraction=0.052, pad=0.04)
    rank_bar.set_label("rank (low → high)", fontsize=7.2)
    rank_bar.ax.tick_params(labelsize=6.8)
    letter(ax_rank, "d")
    fig.text(
        0.5,
        0.006,
        "Panels a–c are structural correlations of source-geometry model disagreement; the descriptor PCA is not training-data coverage and none is a calibrated DFT error.",
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#4B5563",
    )
    save_all(fig, "FIG4_MACE12_SOURCE_GEOMETRY_DISAGREEMENT")


def main() -> None:
    endpoint = load_endpoint()
    fig1_lattice_distribution(endpoint)
    fig2_retry_energy(endpoint)
    fig3_mace_sensitivity(endpoint)
    fig4_source_ensemble()
    print("Rendered four figure families to", FIGURES)


if __name__ == "__main__":
    main()
