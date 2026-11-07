#!/usr/bin/env python3
"""Write the additive source inventory and evidence-boundary tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DERIVED = HERE / "derived"
TABLES = HERE / "tables"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    sources = {
        "grouped_train_xyz": HERE / "raw_cache/grouped_a2/train.xyz",
        "grouped_valid_xyz": HERE / "raw_cache/grouped_a2/valid.xyz",
        "grouped_test_xyz": HERE / "raw_cache/grouped_a2/test.xyz",
        "grouped_neb_1_4": HERE / "raw_cache/grouped_a2/neb_1-4.extxyz",
        "grouped_split_summary": HERE / "raw_cache/grouped_a2/grouped_split_summary.json",
        "grouped_reference_embedding": DERIVED / "grouped_reference_embedding.csv",
        "soap_pca_fit": DERIVED / "soap_pca_fit.npz",
        "static_error_points": DERIVED / "static_error_points.csv",
        "md_distribution_points": DERIVED / "md_distribution_points.csv",
        "md_distribution_summary": DERIVED / "md_distribution_summary.csv",
        "rockfish_training_rmse_eval": DERIVED / "rockfish_training_rmse_eval.csv",
        "rockfish_training_rmse_manifest": DERIVED / "rockfish_training_rmse_manifest.json",
        "rmse_parser_script": HERE / "parse_rockfish_training_rmse.py",
        "rmse_plot_script": HERE / "plot_original_training_rmse_four_model.py",
        "rockfish_foundation_log": HERE / "raw_cache/rockfish_original_training_20261019/logs/foundation_omat_run123_train.txt",
        "rockfish_scratch_log": HERE / "raw_cache/rockfish_original_training_20261019/logs/scratch_mace01_run123_train.txt",
        "rockfish_ft600k_log": HERE / "raw_cache/rockfish_original_training_20261019/logs/ft600k_run123_train.txt",
        "rockfish_ft_multit_log": HERE / "raw_cache/rockfish_original_training_20261019/logs/ft_multit_run123_train.txt",
        "rockfish_foundation_native_png": HERE / "raw_cache/rockfish_original_training_20261019/figures/foundation_omat_run123_stage_two.png",
        "rockfish_scratch_native_png": HERE / "raw_cache/rockfish_original_training_20261019/figures/scratch_mace01_run123_stage_two.png",
        "rockfish_ft600k_native_png": HERE / "raw_cache/rockfish_original_training_20261019/figures/ft600k_run123_stage_two.png",
        "rockfish_ft_multit_native_png": HERE / "raw_cache/rockfish_original_training_20261019/figures/ft_multit_run123_stage_two.png",
        "compute_script": HERE / "compute_soap_pca.py",
        "panel_common_script": HERE / "panel_common.py",
        "panel_a_script": HERE / "plot_task_conditioned_panel_a.py",
        "archived_static_panel_b_script": HERE / "plot_task_conditioned_panel_b.py",
        "panel_c_script": HERE / "plot_task_conditioned_panel_c.py",
        "composition_script": HERE / "compose_task_conditioned_distribution_atlas.py",
        "atlas_plot_script": HERE / "plot_distribution_atlas_v2.py",
        "failure_plot_script": HERE / "plot_failure_stage_grouped.py",
        "atlas_v2_manifest": HERE / "task_conditioned_distribution_atlas_v2_manifest.json",
        "atlas_caption": HERE / "fig_task_conditioned_distribution_atlas_caption.tex",
        "panel_a_caption": HERE / "fig_task_conditioned_panel_a_caption.tex",
        "archived_static_panel_b_caption": HERE / "fig_task_conditioned_panel_b_caption.tex",
        "panel_c_caption": HERE / "fig_task_conditioned_panel_c_caption.tex",
        "failure_caption": HERE / "fig_failure_stage_grouped_caption.tex",
        "main_atlas_pdf": HERE / "figures/fig_task_conditioned_distribution_atlas.pdf",
        "main_atlas_png": HERE / "figures/fig_task_conditioned_distribution_atlas.png",
        "main_atlas_svg": HERE / "figures/fig_task_conditioned_distribution_atlas.svg",
        "panel_a_pdf": HERE / "figures/fig_task_conditioned_panel_a.pdf",
        "panel_a_png": HERE / "figures/fig_task_conditioned_panel_a.png",
        "panel_a_svg": HERE / "figures/fig_task_conditioned_panel_a.svg",
        "archived_static_panel_b_pdf": HERE / "figures/fig_task_conditioned_panel_b.pdf",
        "archived_static_panel_b_png": HERE / "figures/fig_task_conditioned_panel_b.png",
        "archived_static_panel_b_svg": HERE / "figures/fig_task_conditioned_panel_b.svg",
        "panel_c_pdf": HERE / "figures/fig_task_conditioned_panel_c.pdf",
        "panel_c_png": HERE / "figures/fig_task_conditioned_panel_c.png",
        "panel_c_svg": HERE / "figures/fig_task_conditioned_panel_c.svg",
        "original_training_rmse_pdf": HERE / "figures/fig_original_training_rmse_four_model.pdf",
        "original_training_rmse_png": HERE / "figures/fig_original_training_rmse_four_model.png",
        "original_training_rmse_svg": HERE / "figures/fig_original_training_rmse_four_model.svg",
        "original_training_rmse_caption": HERE / "fig_original_training_rmse_four_model_caption.tex",
        "failure_pdf": HERE / "figures/fig_failure_stage_grouped.pdf",
        "failure_png": HERE / "figures/fig_failure_stage_grouped.png",
        "failure_svg": HERE / "figures/fig_failure_stage_grouped.svg",
    }
    rows = []
    for source_id, path in sources.items():
        rows.append({
            "source_id": source_id,
            "path": str(path),
            "sha256": digest(path) if path.is_file() else "",
            "bytes": path.stat().st_size if path.is_file() else "",
            "status": "PRESENT" if path.exists() else "MISSING",
        })
    pd.DataFrame(rows).to_csv(TABLES / "source_inventory.tsv", sep="\t", index=False)
    claims = [
        ["DIST-01", "The grouped split is trajectory/group level", "VERIFIED", "grouped_split_summary.json; grouped reference files", "Whole AIMD trajectories are assigned to one partition; counts are 1786/311/267."],
        ["DIST-02", "Grouped train/valid/test distributions are shown with marginals", "VERIFIED_DESCRIPTIVE", "grouped_reference_embedding.csv", "PCA is fit on grouped train only; this is a representation diagnostic, not a scalar OOD score."],
        ["DIST-03", "Fixed-path pathway-1--4 images are separated from grouped AIMD cells", "VERIFIED_SCOPE_LIMITED", "neb_1-4.extxyz; grouped_reference_embedding.csv", "The fixed-path images are 61 atoms; grouped AIMD cells are 81--86 atoms."],
        ["DIST-04", "Archived static test-error points are retained for audit", "AUDIT_ONLY_MIXED_EVIDENCE", "static_error_points.csv", "The manuscript-facing RMSE figure now uses the original Rockfish validation histories; the archived point table is not used as the main RMSE comparison."],
        ["DIST-05", "MLIP-generated MD clouds show configuration-space occupancy", "DESCRIPTIVE_MLIP_ONLY", "md_distribution_points.csv; FIG-004 continuation manifests", "These are not DFT errors; the common MD velocity seed means model/checkpoint branches are not independent velocity replicas."],
        ["DIST-06", "Short-contact collapse frames are retained as status, not forced into SOAP", "VERIFIED_FAILURE_BOUNDARY", "md_distribution_points.csv", "A 1.2 Angstrom precheck records short_contact_skipped rows before DScribe allocation."],
        ["DIST-07", "Failure/CAR is a separate diagnostic", "VERIFIED_STATIC_DIAGNOSTIC", "fig_failure_stage_grouped.pdf; 639-candidate MLIP outputs", "Committee force spread is not calibrated uncertainty or DFT error."],
        ["DIST-08", "Original Rockfish RMSE histories are available for all four model families", "DESCRIPTIVE_SINGLE_RUN", "rockfish_training_rmse_eval.csv; rockfish_training_rmse_manifest.json", "87 default-head mode=eval checkpoints from run 123 logs; these are validation histories, not grouped held-out test statistics or seed uncertainty."],
    ]
    pd.DataFrame(claims, columns=["claim_id", "claim", "status", "evidence", "boundary"]).to_csv(TABLES / "claim_boundary_matrix.tsv", sep="\t", index=False)
    md_summary_path = DERIVED / "md_distribution_summary.csv"
    md_summary = pd.read_csv(md_summary_path).to_dict(orient="records") if md_summary_path.is_file() else []
    qc = {
        "figure_model_order": ["Foundation", "Scratch", "FT-600K", "FT-MultiT"],
        "no_in_figure_explanatory_text": True,
        "minimum_final_placement_font_pt": 11,
        "target_final_placement_font_pt": 12,
        "typography_rule": "Re-plan margins, panel widths, legends, and insets after enlarging fonts; no overlap or clipping is acceptable.",
        "composition_rule": "Render panels independently before composition; standalone panels are the manuscript-facing assets when a stacked composite would reduce readability.",
        "no_free_form_in_figure_text": "All explanatory or provenance notes belong in captions or audit tables, not inside the plot area.",
        "distribution_surface_rule": "Use soft filled density surfaces with at most one boundary per group; trim negligible marginal KDE tails so peaks stay within calibrated joint support.",
        "rmse_figure_rule": "Use the original Rockfish mode=eval default-head histories in a separate four-column, two-row figure; do not call them grouped test errors or seed uncertainty.",
        "reference_pca_fit": "grouped train only",
        "panel_c_axis_limits": {"x": [-28.0, 5.0], "y": [-25.0, 10.0], "reason": "full combined support of grouped reference and MLIP-generated MD coordinates"},
        "md_distribution_summary": md_summary,
    }
    (DERIVED / "plot_qc.json").write_text(json.dumps(qc, indent=2) + "\n")


if __name__ == "__main__":
    main()
