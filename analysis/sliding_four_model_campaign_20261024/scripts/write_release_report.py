#!/usr/bin/env python3
"""Write the bounded Stage-1 fixed-path release report after collection and plotting."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, root_path, sha256_file, write_json


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def yes_no(value: object) -> str:
    return "PASS" if value is True or value == "PASS" else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    gate_path = root / "manifests/STAGE1_GATE.json"
    gate = read_json(gate_path)
    expected = read_table(root / "results/tables/EXPECTED_vs_ACTUAL.tsv")
    metrics = read_table(root / "results/tables/fixed_metrics_per_model.csv")
    summaries = read_table(root / "results/tables/fixed_family_summary.csv")
    run_log = read_table(root / "manifests/RUN_LOG.tsv")
    figure_manifest_path = root / "results/figures/FIG4_FIXED_PATH_PLOT_MANIFEST.json"
    figure_manifest = read_json(figure_manifest_path)
    text_manifest_path = root / "results/FIXED_PATH_TEXT_MANIFEST.json"
    text_manifest = read_json(text_manifest_path)

    fixed_expected = next((row for row in expected if row.get("stage") == "fixed" and row.get("scope") == "Tier 1 current matrix"), {})
    dense_expected = next((row for row in expected if row.get("stage") == "dense"), {})
    fixed_expected_count = int(fixed_expected.get("expected_rows", 0) or 0)
    fixed_actual_pass = int(fixed_expected.get("actual_pass", 0) or 0)
    dense_expected_count = int(dense_expected.get("expected_rows", 0) or 0)
    dense_actual_pass = int(dense_expected.get("actual_pass", 0) or 0)
    failed_rows = [row for row in expected if row.get("status") != "PASS"]
    pass_metrics = [row for row in metrics if row.get("run_status") == "PASS"]
    jobs = ", ".join(f"{row.get('stage')} `{row.get('job_id')}`" for row in run_log) or "not recorded"

    lines = [
        "# Stage 1 fixed-path release report — interlayer sliding Fig. 4b/c",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Campaign: `{config['campaign_id']}`",
        "",
        "## Release decision",
        "",
        f"- M0 input validation: **{yes_no(gate.get('m0_status'))}**",
        f"- A0 legacy runner self-check: **{yes_no(gate.get('a0_status'))}**",
        f"- Tier-1 fixed-path collection: **{yes_no(gate.get('fixed_status'))}**",
        f"- Tier-1 dense diagnostic collection: **{yes_no(gate.get('dense_status'))}**",
        f"- Fixed-path current rows passed: **{fixed_actual_pass}/{fixed_expected_count}**",
        f"- Dense diagnostic rows passed: **{dense_actual_pass}/{dense_expected_count}**",
        f"- Fixed-path finite model profiles collected: **{len(pass_metrics)}**",
        "",
        "The two panels are released as fixed-path relative-energy diagnostics. They are not self-relaxed NEB barriers and are not presented as validated migration barriers.",
        "",
        "## Reproducibility anchors",
        "",
        f"- Input hashes: [`manifests/INPUT_HASHES.tsv`]({root / 'manifests/INPUT_HASHES.tsv'})",
        f"- Model registry used: [`manifests/MODEL_REGISTRY_USED.tsv`]({root / 'manifests/MODEL_REGISTRY_USED.tsv'})",
        f"- DFT reference: `{config['protocol']['dft_reference_status']}`; `{config['protocol']['dft_reference_method']}`",
        f"- Slurm submissions: {jobs}",
        f"- Runner: `scripts/eval_fixed_path.py`; no geometry optimization is performed",
        "",
        "## Model accounting",
        "",
        f"- Current profiles in the panels: `{len(pass_metrics)}` rows across `{len(config['systems'])}` systems.",
        "- Foundation checkpoints are shown individually because they are checkpoint variants, not three random-seed replicates.",
        "- FT-600K and FT-MultiT are summarized with three training seeds each.",
        "- Scratch-600K and Scratch-MultiT remain separate max_L=2 subfamilies; they are not collapsed into one Scratch mean.",
        "- The legacy `dc41a989...` Foundation checkpoint is retained for A0 self-check only and is excluded from the current panels and current conclusions.",
        "",
        "## Panel artifacts",
        "",
    ]
    for output in figure_manifest.get("outputs", []):
        path = Path(output["pdf"])
        if not path.is_file():
            path = root / "results/figures" / path.name
        lines.append(f"- `{output['panel']}` PDF: [{path.name}]({path}) — SHA256 `{output.get('pdf_sha256', sha256_file(path) if path.is_file() else 'missing')}`")
        png = Path(output.get("png", ""))
        if not png.is_file() and png.name:
            png = root / "results/figures" / png.name
        if png.name:
            lines.append(f"  PNG QA render: `{png}`")
    if not figure_manifest:
        lines.append("- Figure manifest not yet present.")
    lines.extend([
        "",
        "## Scientific boundary and open items",
        "",
        "- DFT/QE reference records remain `COMPLETED_BUT_SCF_WARNING`; the warning-bearing reference is preserved rather than silently upgraded.",
        "- FR pseudopotential filenames are present in the retained inputs, but explicit noncollinear/spin-orbit flags were not independently confirmed in the archived input/output; no verified SOC claim is made here.",
        "- The fixed-path comparison tests the same retained geometries for every model. A future self-relaxed NEB campaign must be reported as a separate observable and must not be pooled with these profiles.",
        "",
        "## Linked machine-readable evidence",
        "",
        f"- Gate: [`manifests/STAGE1_GATE.json`]({gate_path})",
        f"- Expected/actual audit: [`results/tables/EXPECTED_vs_ACTUAL.tsv`]({root / 'results/tables/EXPECTED_vs_ACTUAL.tsv'})",
        f"- Per-model metrics: [`results/tables/fixed_metrics_per_model.csv`]({root / 'results/tables/fixed_metrics_per_model.csv'})",
        f"- Family summary: [`results/tables/fixed_family_summary.csv`]({root / 'results/tables/fixed_family_summary.csv'})",
        f"- Discussion: [`results/FIXED_PATH_DISCUSSION.md`]({root / 'results/FIXED_PATH_DISCUSSION.md'})",
        f"- Caption suggestion: [`results/FIXED_PATH_CAPTION_SUGGESTION.md`]({root / 'results/FIXED_PATH_CAPTION_SUGGESTION.md'})",
        "",
    ])
    if failed_rows:
        lines.extend(["## Failed or incomplete rows", "", "The following rows are not release-passing and must remain visible:", ""])
        for row in failed_rows:
            lines.append(f"- `{row.get('stage')}` / `{row.get('model_id')}` / `{row.get('system_id')}`: expected `{row.get('expected_model_sha256')}`; actual status `{row.get('actual_status')}`")
        lines.append("")

    report_path = root / "results/STAGE1_FIXED_RELEASE_REPORT.md"
    report_path.write_text("\n".join(lines))
    write_json(root / "results/STAGE1_FIXED_RELEASE_MANIFEST.json", {
        "campaign_id": config["campaign_id"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "gate": str(gate_path),
        "expected_rows": len(expected),
        "expected_rows_pass": fixed_actual_pass,
        "fixed_expected_rows": fixed_expected_count,
        "dense_rows_pass": dense_actual_pass,
        "dense_expected_rows": dense_expected_count,
        "fixed_metrics_pass": len(pass_metrics),
        "figure_manifest": str(figure_manifest_path),
        "text_manifest": str(text_manifest_path),
        "current_summary_rows": len(summaries),
        "fixed_path_only": True,
    })
    print(report_path)


if __name__ == "__main__":
    main()
