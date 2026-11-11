#!/usr/bin/env python3
"""Write a readable register for evidence-bearing but not-yet-figured analyses."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/figure_evidence_audit_20261008")


def recommendation(relative: str) -> str:
    value = relative.lower()
    if "from_scratch_ablation_md" in value:
        return "Use existing MD-ABLATION-20PS package; verify branch/source-of-truth and 1 ns gate."
    if "phase2_integrator_sensitivity" in value:
        return "Use generated MD-SENSITIVITY-48RUN package."
    if "mace_candidate_eval" in value:
        return "Use generated MACE-CANDIDATE-EVAL package; runtime audit only."
    if "dft_validation_phase3" in value:
        return "Use generated DFT-PHASE3-INPUT-AUDIT package; MACE input audit only."
    if "dft_single_point_validation" in value:
        return "Use generated DFT-SINGLE-POINT-GATE package; no accepted DFT values."
    if "task1_foundation_path" in value:
        return "Use generated FOUNDATION-PATH-GATE package; numerical profile panels remain blocked."
    if "victor_dft_reference_benchmark" in value:
        return "No figure yet: run matrix/manifest only; wait for accepted outputs."
    if "grouped_retrain" in value or "generalization_repair" in value:
        return "No scientific plot until model/result artifacts and validation rows are present."
    if "table_schemas" in value:
        return "Schema/table support only; link to the actual rendered parent package."
    return "Manual review: decide whether a diagnostic figure is scientifically meaningful before rendering."


def main() -> None:
    payload = json.loads((ROOT / "figure_candidate_inventory.json").read_text(encoding="utf-8"))
    all_candidates = payload["all_evidence_bearing_candidate_directories"]
    unrendered = payload["table_backed_unrendered_candidates"]
    lines = [
        "# Figure candidate register (2026-10-08)",
        "",
        "This register closes the gap between rendered-output inventory and analyses that have tables, raw data, manifests, or scripts but no direct figure in the same package. A candidate is not automatically a publication result. The 102 evidence-bearing directories include nested package directories and duplicate worktree checkouts; the 52 truly unrendered candidates are the priority queue.",
        "",
        f"- Evidence-bearing candidate directories: **{len(all_candidates)}**",
        f"- Table-backed/unrendered candidates: **{len(unrendered)}**",
        "- Generated packages: `MD-SENSITIVITY-48RUN`, `MACE-CANDIDATE-EVAL`, `DFT-PHASE3-INPUT-AUDIT`, `DFT-SINGLE-POINT-GATE`, `FOUNDATION-PATH-GATE`",
        "",
        "## Table-backed or data-backed candidates without a direct rendered output",
        "",
        "| ID | Root | Directory | Data | Scripts | Text | Recommended action |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for index, record in enumerate(unrendered, start=1):
        lines.append(
            f"| CAND-U-{index:03d} | `{record['root_id']}` | `{record['relative_directory']}` | {record['data_file_count']} | {record['script_file_count']} | {record['text_file_count']} | {recommendation(record['relative_directory'])} |"
        )
    lines.extend([
        "",
        "## All evidence-bearing candidate directories",
        "",
        "The full machine-readable evidence file list is in `figure_candidate_inventory.json` and `.tsv`; this section records the rendered descendant status so nested figure packages are not falsely counted as unrendered.",
        "",
        "| ID | Root | Directory | Status | Descendant renders | Data | Scripts | Text |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for index, record in enumerate(all_candidates, start=1):
        lines.append(
            f"| CAND-{index:03d} | `{record['root_id']}` | `{record['relative_directory']}` | `{record['candidate_status']}` | {record['render_file_count']} | {record['data_file_count']} | {record['script_file_count']} | {record['text_file_count']} |"
        )
    (ROOT / "figure_candidate_register.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {ROOT / 'figure_candidate_register.md'} ({len(all_candidates)} candidates)")


if __name__ == "__main__":
    main()
