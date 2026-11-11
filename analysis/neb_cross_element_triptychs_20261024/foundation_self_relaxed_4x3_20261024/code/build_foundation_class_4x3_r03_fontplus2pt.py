#!/usr/bin/env python3
"""Render an additive typography revision of the Foundation 4×3 R03 figure.

The scientific inputs and plotted values are inherited from R03.  This
revision increases every visible text size by 2 pt and writes to a new
directory so that the archived R03 products remain unchanged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


THIS = Path(__file__).resolve()
CAMPAIGN = THIS.parents[1]
ANALYSIS_ROOT = CAMPAIGN.parents[1]
ARCHIVE = CAMPAIGN / "archive" / "legacy_figure_products_20261024"
LEGACY_R03 = ARCHIVE / "build_foundation_class_4x3_r03.py"
REVISION = CAMPAIGN / "figure_revisions" / "foundation_class_4x3_R03_fontplus2pt"
FIG_OUT = REVISION / "figures" / "foundation_class_4x3_R03_fontplus2pt"
DATA_OUT = REVISION / "data" / "foundation_class_4x3_R03_fontplus2pt"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R03 = load_module("foundation_r03_fontplus2pt", LEGACY_R03)
R02 = R03.R02
R01 = R03.R01

# The legacy scripts were archived after their original run.  Rebind their
# source paths to the live checkout before reusing their data loaders.
LIVE_OUT = ANALYSIS_ROOT / "neb_cross_element_triptychs_20261024"
R01.OUT = LIVE_OUT
R01.RAW_DATA_OUT = LIVE_OUT / "data"
R01.INPUT_DIR = ANALYSIS_ROOT / "task3_endpoint_only_neb" / "interpolations"
R01.CR_INPUT_DIR = (
    ANALYSIS_ROOT
    / "task3_endpoint_only_neb"
    / "neb_fixed_vs_self_matrix_20261019"
    / "inputs"
    / "fixed_r01"
)
R01.CR_TIER1 = (
    ANALYSIS_ROOT
    / "task3_endpoint_only_neb"
    / "neb_fixed_vs_self_matrix_20261019"
    / "derived_r01"
    / "TIER1_IMAGE_PROFILES.tsv"
)
R01.CR_QE_TABLE = LIVE_OUT / "data" / "Cr_fixed_path_representative_profiles.tsv"
R01.QE_COMBINED = LIVE_OUT / "data" / "foundation_model_qe_reference_profiles_3x3.tsv"
R02.OUT = LIVE_OUT
R02.DATA_OUT = ARCHIVE / "data" / "foundation_class_4x3_R02"
R02.SUPPLEMENT = R02.DATA_OUT / "Cr_mace_mh1_fixed_path_profiles_R02.tsv"
R03.OUT = REVISION
R03.FIG_OUT = FIG_OUT
R03.DATA_OUT = DATA_OUT
R03.CR_REFERENCE = LIVE_OUT / "data" / "Cr_fixed_path_representative_profiles.tsv"
R03.__file__ = str(THIS)


FONT_BOOST_PT = 2.0


def _apply_typography() -> None:
    """Apply the R03 font recipe plus the requested 2 pt increase."""
    R01.plt.rcParams.update(
        {
            "axes.labelsize": 11.5 + FONT_BOOST_PT,
            "xtick.labelsize": 10.0 + FONT_BOOST_PT,
            "ytick.labelsize": 10.0 + FONT_BOOST_PT,
            "legend.fontsize": 10.0 + FONT_BOOST_PT,
            "font.size": 10.0 + FONT_BOOST_PT,
        }
    )


def _plot_panel(ax, element, path_key, model_rows, grouped, qe_rows, panel_label):
    R01.plot_panel(ax, element, path_key, model_rows, grouped, qe_rows, panel_label)
    if panel_label and ax.texts:
        ax.texts[-1].set_fontsize(13.0 + FONT_BOOST_PT)


def render(model_rows, grouped, qe_rows):
    """Reuse the R03 plot grammar with the typography revision applied."""
    _apply_typography()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    plt = R01.plt
    fig, axes = plt.subplots(4, 3, figsize=(10.8, 11.8), sharex=False, sharey=False)
    panel_qa = []
    labels = iter("abcdefghijkl")
    for i, element in enumerate(R01.ELEMENTS):
        for j, (path_key, display) in enumerate(R01.PATHS):
            ax = axes[i, j]
            _plot_panel(ax, element, path_key, model_rows, grouped, qe_rows, next(labels))
            if i == 0:
                ax.set_title(display, fontsize=12.0 + FONT_BOOST_PT, fontweight="bold", pad=8)
            if j == 0:
                ax.set_ylabel(
                    f"{element}\nΔE (eV)",
                    fontsize=11.0 + FONT_BOOST_PT,
                    fontweight="bold",
                    labelpad=8,
                )
            if i == len(R01.ELEMENTS) - 1:
                ax.set_xlabel(
                    "normalized image coordinate, $s$",
                    fontsize=10.8 + FONT_BOOST_PT,
                    labelpad=7,
                )
            panel_qa.append(
                {"element": element, "path_key": path_key, "y_limits": [float(v) for v in ax.get_ylim()]}
            )

    legend_handles = [
        R01.Line2D([], [], color=R01.CLASS_COLORS[c], lw=2.0, label=c)
        for c in R01.CLASS_ORDER
    ]
    legend_handles.append(
        R01.Line2D(
            [],
            [],
            color="#222222",
            lw=1.45,
            ls=(0, (4, 2)),
            marker="D",
            markersize=3.6,
            markerfacecolor="white",
            label="screened QE path table",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.102, right=0.985, bottom=0.070, top=0.918, wspace=0.36, hspace=0.46)
    stem = FIG_OUT / "foundation_model_class_4x3_R03_fontplus2pt"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for element in R01.ELEMENTS:
        for path_key, display in R01.PATHS:
            fig1, ax = plt.subplots(figsize=(3.35, 2.85))
            _plot_panel(ax, element, path_key, model_rows, grouped, qe_rows, "")
            ax.set_title(
                f"{element} · {display}",
                fontsize=11.5 + FONT_BOOST_PT,
                fontweight="bold",
                pad=7,
            )
            ax.set_xlabel(
                "normalized image coordinate, $s$",
                fontsize=10.3 + FONT_BOOST_PT,
                labelpad=6,
            )
            ax.set_ylabel("ΔE (eV)", fontsize=10.3 + FONT_BOOST_PT, labelpad=6)
            fig1.subplots_adjust(left=0.21, right=0.98, bottom=0.22, top=0.84)
            panel_stem = FIG_OUT / f"panel_{element}_{path_key}"
            fig1.savefig(panel_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
            fig1.savefig(panel_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
            fig1.savefig(panel_stem.with_suffix(".pdf"), dpi=400, bbox_inches="tight", facecolor="white")
            plt.close(fig1)
    return {"figure": str(stem), "panels": panel_qa}


def write_evidence_package() -> None:
    generated_manifest = REVISION / "figure_data_manifest_foundation_class_4x3_R03.json"
    generated_qa = REVISION / "QA_REPORT_foundation_class_4x3_R03.json"
    manifest_path = REVISION / "figure_data_manifest_foundation_class_4x3_R03_fontplus2pt.json"
    qa_path = REVISION / "QA_REPORT_foundation_class_4x3_R03_fontplus2pt.json"
    index_path = REVISION / "DELIVERY_INDEX_foundation_class_4x3_R03_fontplus2pt.md"
    figure_stem = FIG_OUT / "foundation_model_class_4x3_R03_fontplus2pt"
    manifest = json.loads(generated_manifest.read_text())
    manifest.update(
        {
            "figure_id": "foundation_model_class_4x3_R03_fontplus2pt",
            "revision_of": "foundation_model_class_4x3_R03",
            "revision_type": "typography_only",
            "font_size_delta_pt": FONT_BOOST_PT,
            "font_size_policy": "all visible text sizes from the archived R03 recipe increased by 2 pt",
            "figure_outputs": {
                "png": str(figure_stem.with_suffix(".png")),
                "svg": str(figure_stem.with_suffix(".svg")),
                "pdf": str(figure_stem.with_suffix(".pdf")),
            },
            "source_r03_script": str(ARCHIVE / "build_foundation_class_4x3_r03.py"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    qa = json.loads(generated_qa.read_text())
    qa.update(
        {
            "figure_id": "foundation_model_class_4x3_R03_fontplus2pt",
            "revision_of": "foundation_model_class_4x3_R03",
            "typography_only": True,
            "font_size_delta_pt": FONT_BOOST_PT,
            "figure_files_exist": all(Path(path).exists() for path in manifest["figure_outputs"].values()),
        }
    )
    qa_path.write_text(json.dumps(qa, indent=2) + "\n")

    index_path.write_text(
        "# Foundation-model class ensemble 4×3 delivery — R03 typography revision\n\n"
        "This additive revision keeps the R03 data and scientific scope unchanged and increases all visible text sizes by 2 pt.\n\n"
        f"- Figure PNG: `{figure_stem.with_suffix('.png')}`\n"
        f"- Figure SVG: `{figure_stem.with_suffix('.svg')}`\n"
        f"- Figure PDF: `{figure_stem.with_suffix('.pdf')}`\n"
        f"- R03 source script: `{ARCHIVE / 'build_foundation_class_4x3_r03.py'}`\n"
        f"- R03 figure manifest: `{manifest_path}`\n"
        f"- QA report: `{qa_path}`\n\n"
        "Scientific status: unchanged from R03. This is a typography-only revision; no data, model identity, path selection, or scientific label was changed.\n"
    )

    (REVISION / "EVIDENCE_PACKAGE_foundation_class_4x3_R03_fontplus2pt.md").write_text(
        "# Evidence package — Foundation-model class ensemble 4×3, R03 +2 pt typography\n\n"
        "This package is an additive typography revision of the archived R03 release. The plotted data and scientific boundaries are unchanged; only visible text sizes were increased by 2 pt, with small layout-margin adjustments to prevent clipping.\n\n"
        f"- Delivery index: `{index_path}`\n"
        f"- Figure manifest: `{manifest_path}`\n"
        f"- QA report: `{qa_path}`\n"
        f"- Reproduction script: `{THIS}`\n"
        f"- Original R03 script: `{ARCHIVE / 'build_foundation_class_4x3_r03.py'}`\n"
        f"- Original R03 evidence package: `{ARCHIVE / 'EVIDENCE_PACKAGE_foundation_class_4x3_R03.md'}`\n"
        f"- Data source: `{LIVE_OUT / 'data' / 'Cr_fixed_path_representative_profiles.tsv'}`\n\n"
        "QA scope: 1080 model records, 360 class profiles, and 120 QE/DFT reference records are inherited from R03; the typography revision adds a file-existence check for PNG, SVG, and PDF outputs.\n"
    )

    # R03.main uses its historical filenames internally.  Remove only those
    # generated intermediates from this new revision directory after the
    # revision-specific copies have been written.
    for intermediate in (
        generated_manifest,
        generated_qa,
        REVISION / "DELIVERY_INDEX_foundation_class_4x3_R03.md",
        REVISION / "CAPTIONS_foundation_class_4x3_R03.md",
    ):
        intermediate.unlink(missing_ok=True)


def main() -> None:
    R03.render = render
    R03.main()
    write_evidence_package()


if __name__ == "__main__":
    main()
