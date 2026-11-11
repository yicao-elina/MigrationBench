#!/usr/bin/env python3
"""Add conservative group, article, and evidence-chain annotations."""

from __future__ import annotations

import csv
import json
from pathlib import Path


AUDIT_ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/figure_evidence_audit_20261008")


def classify(root_id: str, logical: str) -> tuple[str, str, str, str, str]:
    """Return group, role, evidence status, article target, and gap template."""
    p = logical.lower()
    if root_id in {"wiley-migrationbench", "wiley-migrationbench-ablation-worktree"}:
        if "from_scratch_ablation_md_20261005" in p:
            return ("MD-ABLATION-20PS", "from-scratch ablation MD screening/transport diagnostic", "traceable", "Results/SI: ablation stability and transport-screen boundary", "20 ps/1 ns acceptance and formal transport gates remain incomplete; preserve failed branches and velocity/training-seed separation")
        if "mace_1ns_publication_figure" in p:
            return ("MD-1NS", "current 1-ns MACE benchmark/variant", "traceable", "Results: dynamic stability and diffusion; SI transport candidate", "Article integration and heat-flux absence must remain explicit")
        if "mace_md_time_convergence" in p:
            return ("MD-CONV", "time-window convergence diagnostic", "partial", "SI transport-method validation", "Plot script is not co-located with the image; map exact source generator")
        if "md_continuation_20260930" in p:
            return ("MD-1NS-LEGACY", "legacy/continuation MD panel", "partial", "SI transport or audit appendix only", "Retain as legacy until its source trajectory and renderer are matched to the 1-ns package")
        if "figure1_context_method_result" in p:
            return ("FIG-1-DIFFUSION", "context-method-result composite", "traceable", "Introduction/Results transition; diffusion-aware evaluation", "Not yet mapped to the revision1 TeX figure labels")
        if "phase1_static_force_comparison" in p:
            return ("P1-FORCE", "same-geometry force/geometry diagnostic", "traceable", "Results: model disagreement and trajectory-integrity gate", "Not yet integrated into the paper source")
        if "phase2_handoff_md" in p:
            return ("P2-HANDOFF", "short-handoff stability/gate diagnostic", "traceable", "Results: dynamic stability mechanism; possibly SI", "Keep duplicate retry/audit renders linked and designate one canonical output")
        if "task2_cross_element" in p:
            return ("TASK2-ROSTER", "cross-element profile/seed diagnostic", "traceable", "Results: cross-element transfer and seed stability", "DFT gate is still open; do not write barrier-accuracy claims")
        if "task3_endpoint_only_neb" in p:
            return ("TASK3-NEB", "endpoint-only/Cr path diagnostic", "partial", "Results or SI: path-family diagnostics", "Many per-path exports have no co-located generator; use the parent renderer and record the exact input tables")
        if "mace_endpoint_four_model" in p:
            return ("ENDPOINT-MACE", "four-model endpoint relaxation diagnostic", "traceable", "Results: endpoint robustness", "Confirm current article section and whether endpoint task remains in the revision")
        return ("WILEY-OTHER", "unclassified current output", "partial", "unmapped", "Needs manual family assignment")

    if "paper/revision1/appendix" in p:
        return ("REV1-APPENDIX", "revision1 endpoint appendix figure", "traceable", "SI endpoint-relaxation appendix", "Caption is in endpoint_perturbation_appendix.tex; confirm final compiled placement")
    if "paper/revision1/figures/foundation_grouped" in p:
        return ("REV1-FOUNDATION", "revision1 fixed-path foundation profile", "traceable", "SI Detailed Migration Pathway Analysis / In-Gap Diffusion", "DFT path is provisional; no self-relaxed NEB interpretation")
    if "paper/revision1/figures" in p:
        return ("REV1-ARTICLE", "revision1 article/SI figure asset", "traceable", "Mapped by sn-article-final.tex or sn-article-SI.tex", "Check figure source ledger and exact generator; some assets are compiled from older workflows")
    if p.startswith("paper/revision1/") and (p.endswith("sn-article-final") or p.endswith("sn-article-si")):
        return ("REV1-COMPILED", "compiled manuscript PDF", "compiled document", "Entire manuscript/SI", "Not an independent figure; audit included figure assets instead")
    if "analysis/phase3_dft" in p:
        return ("P3-DFT", "DFT recovery/validation diagnostic", "partial", "Results/SI: DFT gate and model-force validation", "No accepted formal DFT target is available in the audited state")
    if "analysis/phase1_static_force" in p:
        return ("P1-FORCE", "same-geometry force/geometry diagnostic", "traceable", "Results: model disagreement and trajectory-integrity gate", "Duplicate checkout; use current package as canonical")
    if "analysis/phase2_handoff" in p:
        return ("P2-HANDOFF", "short-handoff stability/gate diagnostic", "traceable", "Results: dynamic stability mechanism", "Duplicate checkout; reconcile against current package")
    if "data/published/neb_barrier_atlas" in p:
        return ("ARCHIVE-NEB-BARRIER", "archived barrier-atlas derivative", "partial", "SI DFT NEB barrier inventory", "No co-located plotting script; treat as archived derivative until source is recovered")
    if "data/published/neb_energy_profiles" in p:
        return ("ARCHIVE-NEB-PROFILE", "archived path-profile derivative", "partial", "SI DFT NEB profiles / fixed-path benchmark", "No co-located plotting script; path-table provenance is present")
    if "representative_trajectory_portfolio" in p:
        return ("ARCHIVE-TRAJ-PORTFOLIO", "trajectory coverage/similarity derivative", "partial", "Methods/SI data coverage audit", "Figure generator is not co-located with the archived image")
    if "endpoint_relax_comparison" in p or "qe_geometry_transform_relax" in p:
        return ("ARCHIVE-ENDPOINT", "archived endpoint-relaxation curve", "partial", "SI endpoint-relaxation evidence", "Recover generator and exact raw log lineage")
    if "training_leakage_audit" in p:
        return ("ARCHIVE-LEAKAGE", "training leakage audit derivative", "partial", "Methods/Results data provenance", "Generator is outside the figure directory; link exact source script")
    if "historical_selfneb_foundation_audit" in p:
        return ("ARCHIVE-SELFNEB", "historical self-NEB derivative", "partial", "SI NEB stability audit", "Source input and renderer are not co-located")
    if "mlff_preconditioner_comparison" in p:
        return ("ARCHIVE-PRECOND", "MLFF preconditioner residual derivative", "partial", "SI NEB methods/audit", "Recover source path and plotting command")
    if p.startswith("paper/fig/shap") or "work/rockfish_shap_candidates" in p:
        return ("LEGACY-SHAP", "legacy SHAP/descriptor diagnostic", "partial", "Legacy Fig. 6/interpretability audit", "Some images lack direct source-data and renderer linkage")
    if p.startswith("paper/fig/si"):
        return ("LEGACY-SI", "legacy supplementary image", "partial", "Legacy SI", "Image-only or historical source chain not closed")
    if p.startswith("paper/fig/"):
        return ("LEGACY-NEURIPS", "legacy manuscript figure asset", "partial", "Original main/SI manuscript", "Source-data and plotting provenance generally not co-located")
    if p.startswith("figures/publication/"):
        return ("LEGACY-PUBLISHED", "duplicate/packaged revision figure", "partial", "Revision1 manuscript asset", "Link to canonical revision1 asset and generator")
    if p.startswith("paper/response_to_reviewers") or p.startswith("paper/sn-article"):
        return ("COMPILED-DOCUMENT", "compiled response/manuscript PDF", "compiled document", "Response or manuscript", "Not an independent figure")
    return ("PAPER-OTHER", "unclassified paper-repro output", "partial", "unmapped", "Needs manual family assignment")


def split_candidates(record: dict) -> dict[str, str]:
    candidates = [Path(item) for item in record["nearby_candidates"]]
    def join(suffixes: set[str]) -> str:
        return " | ".join(str(path) for path in candidates if path.suffix.lower() in suffixes)
    return {
        "plot_script_candidates": join({".py", ".sh", ".slurm"}),
        "data_or_table_candidates": join({".csv", ".tsv", ".json", ".jsonl", ".xyz", ".extxyz", ".path"}),
        "text_or_caption_candidates": join({".md", ".tex", ".txt", ".out", ".err", ".log"}),
    }


def main() -> None:
    records = json.loads((AUDIT_ROOT / "figure_inventory.json").read_text(encoding="utf-8"))
    out = []
    counters: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        group, role, evidence, article, gaps = classify(record["root_id"], record["logical_figure"])
        counters[group] = counters.get(group, 0) + 1
        candidate_fields = split_candidates(record)
        out.append(
            {
                "audit_id": f"FIG-{index:03d}",
                "group_id": group,
                "root_id": record["root_id"],
                "logical_figure": record["logical_figure"],
                "logical_path": record["logical_path"],
                "role": role,
                "evidence_chain_status": evidence,
                "article_target": article,
                "caption_discussion_source_candidates": candidate_fields["text_or_caption_candidates"],
                "raw_or_processed_data_candidates": candidate_fields["data_or_table_candidates"],
                "plot_script_candidates": candidate_fields["plot_script_candidates"],
                "figure_outputs": " | ".join(item["path"] for item in record["outputs"]),
                "figure_sha256": " | ".join(item["sha256"] for item in record["outputs"]),
                "latest_mtime_utc": record["latest_mtime_utc"],
                "recent_since_2026_08_08": record["recent_since_2026_08_08"],
                "status_tags": record["status_tags"],
                "gaps_to_resolve": gaps,
            }
        )

    fields = list(out[0])
    with (AUDIT_ROOT / "figure_evidence_matrix.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out)
    (AUDIT_ROOT / "figure_group_summary.json").write_text(
        json.dumps({"group_count": len(counters), "counts": dict(sorted(counters.items()))}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(out), "groups": len(counters), "counts": dict(sorted(counters.items()))}, indent=2))


if __name__ == "__main__":
    main()
