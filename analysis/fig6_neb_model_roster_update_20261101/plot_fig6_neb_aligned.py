#!/usr/bin/env python3
"""Build a Figure 6 update using the model roster shown in the NEB main figure.

The input is the audited NEB dot-matrix v3 table: three Foundation
checkpoints, six Fine-tuning models, and six Scratch models. The four panels
are deliberately task-native rather than SHAP-native:

  (a) all eight NEB/sliding observables as an error matrix;
  (b) fixed-path versus self-relaxed migration error;
  (c) within-QL predicted barrier for fixed and self-relaxed operators;
  (d) sliding predicted peak energy for pristine and Cr-in-gap systems.

The style follows the scientific Plot Atlas contract and the NEB dot-matrix
style spec: Arial first, bold ticks, clean axes, explicit family colors,
status-aware missing/nonconverged rendering, and editable vector exports.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
NEB_SOURCE = SOURCE / "neb_dotmatrix_v3" / "NEB_Dotmatrix_v3"
OUT = ROOT / "figures"

METRICS = NEB_SOURCE / "data" / "NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv"
REFERENCE_PDF = SOURCE / "NEB_main_figure_reference.pdf"
AUDIT_JSON = NEB_SOURCE / "audit.json"
STYLE_JSON = NEB_SOURCE / "style_spec.json"

FAMILY_COLORS = {
    "Foundation": "#95D1B1",
    "Fine-tuning": "#36B3C0",
    "Scratch": "#2F83B8",
}
TASK_COLORS = {
    "migration": "#FABAB9",
    "self": "#F26EA6",
    "sliding": "#74297E",
}
MARKERS = {"foundation_checkpoint": "D", "trained_seed": "o"}

MODEL_ORDER = [
    "MP-0B3",
    "OMAT-0",
    "MPA0",
    "FT-600K / s123",
    "FT-600K / s234",
    "FT-600K / s345",
    "FT-MultiT / s123",
    "FT-MultiT / s234",
    "FT-MultiT / s345",
    "Scratch-600K / s123",
    "Scratch-600K / s234",
    "Scratch-600K / s345",
    "Scratch-MultiT / s123",
    "Scratch-MultiT / s234",
    "Scratch-MultiT / s345",
]

CANONICAL_IDS = {
    "MP-0B3": "foundation_mp0b3",
    "OMAT-0": "foundation_omat0_medium",
    "MPA0": "foundation_mpa0",
    "FT-600K / s123": "finetune_600k_seed123",
    "FT-600K / s234": "finetune_600k_seed234",
    "FT-600K / s345": "finetune_600k_seed345",
    "FT-MultiT / s123": "finetune_multit_seed123",
    "FT-MultiT / s234": "finetune_multit_seed234",
    "FT-MultiT / s345": "finetune_multit_seed345",
    "Scratch-600K / s123": "scratch_600k_seed123",
    "Scratch-600K / s234": "scratch_600k_seed234",
    "Scratch-600K / s345": "scratch_600k_seed345",
    "Scratch-MultiT / s123": "scratch_multit_seed123",
    "Scratch-MultiT / s234": "scratch_multit_seed234",
    "Scratch-MultiT / s345": "scratch_multit_seed345",
}

MATRIX_COLUMNS = [
    "migration_in_gap_fixed",
    "migration_gap_to_ql_fixed",
    "migration_within_ql_fixed",
    "migration_in_gap_self",
    "migration_gap_to_ql_self",
    "migration_within_ql_self",
    "sliding_pristine_fixed",
    "sliding_cr_in_gap_fixed",
]
MATRIX_LABELS = {
    "migration_in_gap_fixed": "in-gap\nfixed",
    "migration_gap_to_ql_fixed": "gap -> QL\nfixed",
    "migration_within_ql_fixed": "within QL\nfixed",
    "migration_in_gap_self": "in-gap\nself",
    "migration_gap_to_ql_self": "gap -> QL\nself",
    "migration_within_ql_self": "within QL\nself",
    "sliding_pristine_fixed": "pristine\nsliding",
    "sliding_cr_in_gap_fixed": "Cr-in-gap\nsliding",
}
MIGRATION_PATHS = [
    ("migration_in_gap", "in-gap"),
    ("migration_gap_to_ql", "gap -> QL"),
    ("migration_within_ql", "within QL"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_atlas_style() -> dict:
    try:
        fm.findfont("Arial", fallback_to_default=False)
        font = "Arial"
    except ValueError:
        font = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return {"font_requested": "Arial", "font_selected": font}


def load_metrics() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(METRICS, sep="\t")
    if len(df) != 120:
        raise ValueError(f"expected 120 NEB dot-matrix records, found {len(df)}")
    if df["row_label"].nunique() != 15:
        raise ValueError("expected 15 model rows")
    if df["column_id"].nunique() != 8:
        raise ValueError("expected 8 task/system columns")
    if df.duplicated(["row_label", "column_id"]).any():
        raise ValueError("duplicate model/task records")
    if set(df["row_label"]) != set(MODEL_ORDER):
        raise ValueError("NEB model roster differs from the referenced main figure")
    df["model_order"] = df["row_label"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    df["column_order"] = df["column_id"].map({name: i for i, name in enumerate(MATRIX_COLUMNS)})
    df["absolute_error_eV"] = pd.to_numeric(df["absolute_error_eV"], errors="coerce")
    df["predicted_value_eV"] = pd.to_numeric(df["predicted_value_eV"], errors="coerce")
    df["is_nc"] = df["scientific_status"].eq("NONCONVERGED") | df["publication_status"].eq("DIAGNOSTIC_ONLY") & df["predicted_value_eV"].isna()
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    if not audit.get("recomputed_errors_match"):
        raise ValueError("NEB source audit does not confirm recomputed errors")
    return df.sort_values(["model_order", "column_order"]).reset_index(drop=True), audit


def apply_axes(ax):
    ax.grid(False)
    ax.tick_params(direction="out", length=4, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")


def title(ax, letter: str, text: str):
    ax.set_title(f"{letter}  {text}", loc="left", pad=8)


def model_meta(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["row_label", "model_kind", "training_seed"]
    return df[cols].drop_duplicates().set_index("row_label").loc[MODEL_ORDER].reset_index()


def panel_a_matrix(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.8, 7.5))
    values = np.full((len(MODEL_ORDER), len(MATRIX_COLUMNS)), np.nan)
    for row in df.itertuples():
        if not row.is_nc and np.isfinite(row.absolute_error_eV):
            values[row.model_order, row.column_order] = row.absolute_error_eV
    cmap = mpl.colormaps["Blues"].copy()
    cmap.set_bad("#F2F2F2")
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=PowerNorm(gamma=0.55, vmin=0, vmax=25), interpolation="nearest")
    ax.set_xticks(np.arange(len(MATRIX_COLUMNS)))
    ax.set_xticklabels([MATRIX_LABELS[c] for c in MATRIX_COLUMNS])
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_xlabel("NEB pathway or sliding system")
    ax.set_ylabel("Model used in NEB main figure")
    title(ax, "(a)", "Absolute error across the 15-model NEB campaign")
    for i, row in enumerate(MODEL_ORDER):
        family = df.loc[df["row_label"].eq(row), "model_kind"].iloc[0]
        color = FAMILY_COLORS["Foundation" if family == "foundation_checkpoint" else ("Fine-tuning" if row.startswith("FT-") else "Scratch")]
        ax.get_yticklabels()[i].set_color(color)
    for row in df.itertuples():
        x, y = row.column_order, row.model_order
        if row.is_nc:
            ax.text(x, y, "NC", ha="center", va="center", color="#777777", fontsize=8, fontweight="bold")
        elif np.isfinite(row.absolute_error_eV):
            text_color = "white" if row.absolute_error_eV > 9 else "#111111"
            ax.text(x, y, f"{row.absolute_error_eV:.2g}", ha="center", va="center", color=text_color, fontsize=7.4, fontweight="bold")
    for y in [2.5, 8.5]:
        ax.axhline(y, color="white", linewidth=1.7)
    for x in [2.5, 5.5]:
        ax.axvline(x, color="white", linewidth=1.7)
    cbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Absolute error (eV)", fontweight="bold")
    cbar.ax.tick_params(labelsize=8.5)
    family_handles = [Patch(facecolor=FAMILY_COLORS[k], edgecolor="none", label=k) for k in ["Foundation", "Fine-tuning", "Scratch"]]
    ax.legend(handles=family_handles, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3, fontsize=9)
    apply_axes(ax)
    fig.subplots_adjust(left=0.24, right=0.91, top=0.90, bottom=0.23)
    return fig


def family_for(row_label: str) -> str:
    if row_label in ["MP-0B3", "OMAT-0", "MPA0"]:
        return "Foundation"
    if row_label.startswith("FT-"):
        return "Fine-tuning"
    return "Scratch"


def operator_error_points(df: pd.DataFrame, path_prefix: str):
    points = []
    for path_id, path_label in MIGRATION_PATHS:
        for operator, suffix in [("Fixed path", "_fixed"), ("Self-relaxed", "_self")]:
            sub = df[df["column_id"].eq(path_id + suffix)].copy()
            sub["path_label"] = path_label
            sub["operator"] = operator
            points.append(sub)
    return pd.concat(points, ignore_index=True)


def panel_b_migration_error(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    points = operator_error_points(df, "migration")
    path_labels = [label for _, label in MIGRATION_PATHS]
    operator_offsets = {"Fixed path": -0.16, "Self-relaxed": 0.16}
    for ipath, path_label in enumerate(path_labels):
        for operator in ["Fixed path", "Self-relaxed"]:
            sub = points[points["path_label"].eq(path_label) & points["operator"].eq(operator)]
            for row in sub.itertuples():
                if row.is_nc or not np.isfinite(row.absolute_error_eV):
                    ax.scatter(ipath + operator_offsets[operator], 0.06, marker="x", s=42, color="#777777", linewidth=1.2, zorder=5)
                else:
                    ax.scatter(ipath + operator_offsets[operator], row.absolute_error_eV, marker=MARKERS[row.model_kind], s=36, facecolor=FAMILY_COLORS[family_for(row.row_label)], edgecolor="white", linewidth=0.45, alpha=0.9, zorder=4)
            good = sub.loc[~sub["is_nc"], "absolute_error_eV"].dropna()
            if len(good):
                median = float(good.median())
                ax.plot([ipath + operator_offsets[operator] - 0.11, ipath + operator_offsets[operator] + 0.11], [median, median], color="#111111", linewidth=1.7, zorder=6)
    ax.set_xticks(range(len(path_labels)))
    ax.set_xticklabels(path_labels)
    ax.set_ylabel("Absolute barrier error (eV)")
    ax.set_xlabel("Migration pathway")
    ax.set_ylim(bottom=0)
    title(ax, "(b)", "Fixed-path and self-relaxed migration error")
    handles = [
        Patch(facecolor=FAMILY_COLORS["Foundation"], edgecolor="none", label="Foundation"),
        Patch(facecolor=FAMILY_COLORS["Fine-tuning"], edgecolor="none", label="Fine-tuning"),
        Patch(facecolor=FAMILY_COLORS["Scratch"], edgecolor="none", label="Scratch"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#888888", markeredgecolor="white", markersize=6, label="Foundation checkpoint"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888", markeredgecolor="white", markersize=6, label="Training seed"),
        Line2D([0], [0], marker="_", color="#111111", linewidth=1.7, label="Median"),
        Line2D([0], [0], marker="x", color="#777777", linestyle="None", markersize=6, label="NC / diagnostic"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.20), ncol=3, fontsize=8.6, handlelength=1.2, columnspacing=1.0)
    apply_axes(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.30)
    return fig


def panel_c_within_ql(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    fixed = df[df["column_id"].eq("migration_within_ql_fixed")].copy()
    self_relaxed = df[df["column_id"].eq("migration_within_ql_self")].copy()
    family_order = ["Foundation", "Fine-tuning", "Scratch"]
    family_centers = {name: i for i, name in enumerate(family_order)}
    model_offset = {
        "Foundation": np.linspace(-0.20, 0.20, 3),
        "Fine-tuning": np.linspace(-0.24, 0.24, 6),
        "Scratch": np.linspace(-0.24, 0.24, 6),
    }
    for family in family_order:
        rows = [m for m in MODEL_ORDER if family_for(m) == family]
        offsets = model_offset[family]
        for i, model in enumerate(rows):
            x = family_centers[family] + offsets[i]
            f = fixed[fixed["row_label"].eq(model)].iloc[0]
            s = self_relaxed[self_relaxed["row_label"].eq(model)].iloc[0]
            color = FAMILY_COLORS[family]
            ax.scatter(x - 0.018, f.predicted_value_eV, marker=MARKERS[f.model_kind], s=42, facecolor=color, edgecolor="white", linewidth=0.5, zorder=4)
            if s.is_nc or not np.isfinite(s.predicted_value_eV):
                ax.scatter(x + 0.018, 0.10, marker="x", s=44, color="#777777", linewidth=1.2, zorder=5)
            else:
                ax.scatter(x + 0.018, s.predicted_value_eV, marker=MARKERS[s.model_kind], s=42, facecolor="white", edgecolor=color, linewidth=1.4, zorder=4)
    ax.axhline(1.8449, color="#333333", linewidth=1.0, linestyle=(0, (4, 2)), label="DFT reference 1.845 eV")
    ax.set_xticks(range(3))
    ax.set_xticklabels(family_order)
    ax.set_ylabel("Predicted within-QL barrier (eV)")
    ax.set_xlabel("Training strategy")
    ax.set_ylim(bottom=0)
    title(ax, "(c)", "Within-QL barrier: fixed path versus self-relaxed")
    handles = [
        Patch(facecolor=FAMILY_COLORS["Foundation"], edgecolor="none", label="Foundation"),
        Patch(facecolor=FAMILY_COLORS["Fine-tuning"], edgecolor="none", label="Fine-tuning"),
        Patch(facecolor=FAMILY_COLORS["Scratch"], edgecolor="none", label="Scratch"),
        Line2D([0], [0], marker="D", color="#555555", markerfacecolor="#999999", linestyle="None", markersize=6, label="Foundation checkpoint"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#999999", linestyle="None", markersize=6, label="Training seed"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#FFFFFF", linestyle="None", markersize=6, label="Self-relaxed"),
        Line2D([0], [0], marker="x", color="#777777", linestyle="None", markersize=6, label="NC / diagnostic"),
        Line2D([0], [0], color="#333333", linestyle=(0, (4, 2)), linewidth=1.0, label="DFT reference"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.20), ncol=3, fontsize=8.4, handlelength=1.3, columnspacing=1.0)
    apply_axes(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.31)
    return fig


def panel_d_sliding(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    systems = [("sliding_pristine_fixed", "Pristine"), ("sliding_cr_in_gap_fixed", "Cr-in-gap")]
    family_order = ["Foundation", "Fine-tuning", "Scratch"]
    centers = {name: i for i, name in enumerate([s[1] for s in systems])}
    family_offsets = {"Foundation": -0.20, "Fine-tuning": 0.0, "Scratch": 0.20}
    for col, label in systems:
        sub = df[df["column_id"].eq(col)]
        for family in family_order:
            rows = sub[sub["row_label"].map(family_for).eq(family)]
            jitter = np.linspace(-0.09, 0.09, len(rows))
            for j, row in enumerate(rows.itertuples()):
                x = centers[label] + family_offsets[family] + jitter[j]
                ax.scatter(x, row.predicted_value_eV, marker=MARKERS[row.model_kind], s=42, facecolor=FAMILY_COLORS[family], edgecolor="white", linewidth=0.5, zorder=4)
    ax.axhline(1.653834726, color="#333333", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.axhline(2.5708301675, color="#333333", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.text(0.02, 1.653834726 + 0.07, "DFT 1.654", transform=ax.get_yaxis_transform(), fontsize=8.5, color="#333333")
    ax.text(0.52, 2.5708301675 + 0.07, "DFT 2.571", transform=ax.get_yaxis_transform(), fontsize=8.5, color="#333333")
    ax.set_xticks(range(2))
    ax.set_xticklabels(["Pristine", "Cr-in-gap"])
    ax.set_ylabel("Predicted sliding peak (eV)")
    ax.set_xlabel("Sliding system")
    ax.set_ylim(bottom=0)
    title(ax, "(d)", "Sliding peak energy across the NEB model roster")
    handles = [
        Patch(facecolor=FAMILY_COLORS["Foundation"], edgecolor="none", label="Foundation"),
        Patch(facecolor=FAMILY_COLORS["Fine-tuning"], edgecolor="none", label="Fine-tuning"),
        Patch(facecolor=FAMILY_COLORS["Scratch"], edgecolor="none", label="Scratch"),
        Line2D([0], [0], marker="D", color="#555555", markerfacecolor="#999999", linestyle="None", markersize=6, label="Foundation checkpoint"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#999999", linestyle="None", markersize=6, label="Training seed"),
        Line2D([0], [0], color="#333333", linestyle=(0, (4, 2)), linewidth=1.0, label="DFT reference"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.20), ncol=3, fontsize=8.4, handlelength=1.3, columnspacing=1.0)
    apply_axes(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.31)
    return fig


def composite(df: pd.DataFrame):
    # Standalone panel functions are the source of truth; composite is a layout wrapper.
    figs = [panel_a_matrix(df), panel_b_migration_error(df), panel_c_within_ql(df), panel_d_sliding(df)]

    def trim_white(image):
        rgb = image[..., :3]
        mask = np.any(rgb < 248, axis=2)
        ys, xs = np.where(mask)
        if not len(xs):
            return image
        pad = 10
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(image.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(image.shape[1], int(xs.max()) + pad + 1)
        return image[y0:y1, x0:x1]

    rendered = []
    for fig in figs:
        fig.canvas.draw()
        rendered.append(trim_white(np.asarray(fig.canvas.buffer_rgba())))
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.8), constrained_layout=False)
    for ax, image in zip(axes.flat, rendered):
        ax.imshow(image)
        ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015, wspace=0.01, hspace=0.03)
    return fig


def save(fig, stem: str):
    paths = []
    for ext in ["png", "pdf", "svg"]:
        path = OUT / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def write_registry(df: pd.DataFrame):
    rows = []
    for label in MODEL_ORDER:
        sub = df[df["row_label"].eq(label)]
        rows.append(
            {
                "canonical_model_id": CANONICAL_IDS[label],
                "plot_label": label,
                "family": family_for(label),
                "model_kind": sub["model_kind"].iloc[0],
                "training_seed": "" if pd.isna(sub["training_seed"].iloc[0]) else int(sub["training_seed"].iloc[0]),
                "model_sha256": sub["model_sha256"].dropna().iloc[0] if sub["model_sha256"].notna().any() else "",
            }
        )
    path = ROOT / "MODEL_REGISTRY.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style = apply_atlas_style()
    df, audit = load_metrics()
    registry_path = write_registry(df)
    outputs = []
    for stem, builder in [
        ("fig6_neb_panel_a_error_matrix", lambda: panel_a_matrix(df)),
        ("fig6_neb_panel_b_migration_operator_error", lambda: panel_b_migration_error(df)),
        ("fig6_neb_panel_c_within_ql_barriers", lambda: panel_c_within_ql(df)),
        ("fig6_neb_panel_d_sliding_peaks", lambda: panel_d_sliding(df)),
    ]:
        outputs.extend(save(builder(), stem))
    outputs.extend(save(composite(df), "fig6_neb_roster_aligned_v1"))

    source_hashes = {}
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file():
            source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    output_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in outputs}
    manifest = {
        "artifact": "Figure 6 NEB-roster-aligned four-panel update",
        "status": "RECONSTRUCTION",
        "model_roster": {
            "n_models": len(MODEL_ORDER),
            "models": MODEL_ORDER,
            "family_counts": {family: sum(family_for(label) == family for label in MODEL_ORDER) for family in ["Foundation", "Fine-tuning", "Scratch"]},
            "source_reference": "NEB_main_figure_reference.pdf",
        },
        "data_scope": {
            "records": int(len(df)),
            "observables": ["migration fixed", "migration self-relaxed", "sliding fixed"],
            "source_metrics": str(METRICS.relative_to(ROOT)),
            "audit_input_sha256": audit.get("input_sha256"),
            "nonconverged_records": int(df["is_nc"].sum()),
        },
        "scientific_boundaries": [
            "Fixed-path migration and sliding are supporting diagnostics.",
            "Self-relaxed migration is primary only where the source status is CONVERGED/PRIMARY_EVIDENCE.",
            "The three nonconverged within-QL self-relaxed records remain NC/diagnostic and are not converted to numeric evidence.",
            "DFT references in the source package carry the source warning boundary; no validated DFT MEP claim is inferred.",
            "This update is not tied to the SHAP campaign; it follows the model roster and metrics used by the NEB main figure.",
        ],
        "style": {
            "contract": "scientific-plot-atlas",
            "runtime": "local-contract fallback; recovered Plot Atlas package was not hydrated",
            **style,
            "dpi_png": 350,
            "editable_exports": ["pdf", "svg"],
            "family_colors": FAMILY_COLORS,
        },
        "source_sha256": source_hashes,
        "model_registry_sha256": sha256(registry_path),
        "outputs_sha256": output_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "matplotlib": mpl.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    manifest_path = ROOT / "figure_manifest_neb_roster_aligned_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "outputs": [str(p) for p in outputs], "records": len(df), "models": len(MODEL_ORDER)}, indent=2))


if __name__ == "__main__":
    main()
