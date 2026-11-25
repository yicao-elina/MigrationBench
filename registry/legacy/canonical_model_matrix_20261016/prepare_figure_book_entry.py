#!/usr/bin/env python3
"""Create one additive revision-1 figure-registry entry without overwriting an ID."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-id", default="FIG-219")
    parser.add_argument("--figure-manifest", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--handoff-root", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--source-export", required=True)
    parser.add_argument("--target-root", required=True)
    args = parser.parse_args()

    figure_manifest_path = Path(args.figure_manifest)
    analysis_manifest_path = Path(args.analysis_manifest)
    handoff = Path(args.handoff_root)
    asset = Path(args.asset)
    source_export = Path(args.source_export)
    target_root = Path(args.target_root)
    target = target_root / "figures" / args.figure_id
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing figure registry entry: {target}")
    for required in (figure_manifest_path, analysis_manifest_path, handoff / "canonical_639_caption.tex", asset, source_export):
        if not required.is_file():
            raise SystemExit(f"missing required figure-book input: {required}")
    target.mkdir(parents=True)
    shutil.copy2(asset, target / "asset.pdf")
    analysis = json.loads(analysis_manifest_path.read_text())
    figure = json.loads(figure_manifest_path.read_text())
    caption = (handoff / "canonical_639_caption.tex").read_text()
    n_candidates = analysis["n_candidates_expected"]
    n_models = analysis["n_models_expected"]
    n_rows = analysis["n_rows_observed"]
    n_success = analysis["n_success_rows"]

    (target / "figure.tex").write_text(
        f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width=\\linewidth,height=0.58\\textheight,keepaspectratio]{{figure_registry/figures/{args.figure_id}/asset.pdf}}\n\\caption{{\\protect\\input{{figure_registry/figures/{args.figure_id}/caption.tex}}}}\n\\label{{fig:canonical_639_overview}}\n\\end{{figure}}\n"
    )
    (target / "caption.tex").write_text(caption)
    (target / "results.tex").write_text(
        f"The canonical screen contains {n_candidates} candidates and {n_models} accepted model entries, with {n_rows} observed model--candidate rows and {n_success} finite successful evaluations. Detailed family and stage summaries are retained in the machine-readable analysis package.\n"
    )
    (target / "discussion.tex").write_text(
        "The figure reports paired static MLIP evidence on fixed saved configurations. Force disagreement is model dependence, not calibrated uncertainty, DFT correctness, or causal proof; the event-driven candidate set is not an unbiased random test set.\n"
    )
    (target / "reviewer.tex").write_text(
        "\\textbf{Reviewer linkage:} PENDING. Independent DFT reference-force validation and reduced-cell convergence/parity checks remain required before any DFT correctness claim.\n"
    )
    (target / "evidence.tex").write_text(
        "\\begin{itemize}[leftmargin=1.4em,noitemsep]\n"
        f"\\item Plot output manifest: \\begingroup\\scriptsize\\url{{{figure_manifest_path}}}\\endgroup\n"
        f"\\item Canonical analysis manifest: \\begingroup\\scriptsize\\url{{{analysis_manifest_path}}}\\endgroup\n"
        f"\\item Source export: \\begingroup\\scriptsize\\url{{{source_export}}}\\endgroup\n"
        f"\\item Asset SHA256: \\texttt{{{sha256(target / 'asset.pdf')}}}\n"
        "\\end{itemize}\n"
    )
    (target / "evidence_auto.tex").write_text(
        "\\emph{Machine-collected evidence; generated from the canonical figure and analysis manifests.}\n"
        f"\\begin{{itemize}}\\item Analysis status: \\texttt{{{analysis.get('status', 'UNKNOWN')}}}\\item Figure status: \\texttt{{{figure.get('status', 'UNKNOWN')}}}\\end{{itemize}}\n"
    )
    (target / "article.tex").write_text("\\textbf{Article placement.} Proposed placement: Supplementary Information / canonical accepted-model 639-candidate MLIP diagnostic.\\n")
    (target / "gap.tex").write_text("\\statusline{\\textbf{Audit gap / next action:} DFT reference-force validation and reduced-cell convergence/parity remain separate gates.}\n")
    manifest = {
        "schema_version": 1,
        "figure_id": args.figure_id,
        "logical_figure": "canonical_model_matrix_20261016/canonical_639_screen_v1",
        "title": "Canonical accepted-model 639-candidate MLIP overview",
        "group_id": "MD-CANONICAL-MODEL-MATRIX",
        "parent_group": "MD-CANONICAL-MODEL-MATRIX",
        "sibling_ids": [],
        "article_targets": ["Results/SI: accepted-model 639-candidate static MLIP evaluation"],
        "reviewer_status": "PENDING",
        "scientific_status": "paired static MLIP evidence; no DFT correctness or causal claim",
        "lock_status": "draft",
        "promotion_status": "review_book_only",
        "asset": "asset.pdf",
        "asset_sha256": sha256(target / "asset.pdf"),
        "source_export": str(source_export),
        "evidence_candidates": {
            "caption_discussion_source_candidates": [str(handoff / "canonical_639_results_discussion.md")],
            "raw_or_processed_data_candidates": [str(analysis_manifest_path)],
            "plot_script_candidates": [str(source_export), "/Users/alina/Documents/Codex/plot_library"],
        },
    }
    (target / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    authored = ["page.tex", "figure.tex", "caption.tex", "results.tex", "discussion.tex", "reviewer.tex", "evidence.tex", "evidence_auto.tex", "article.tex", "gap.tex"]
    page = (
        f"\\clearpage\n\\section{{{args.figure_id} --- Canonical accepted-model 639-candidate MLIP overview}}\n"
        f"\\label{{figpage:{args.figure_id}}}\n\\textbf{{Group:}} MD-CANONICAL-MODEL-MATRIX\\quad \\textbf{{Evidence chain:}} manifested\\quad \\textbf{{Status:}} paired static MLIP evidence; DFT validation pending\n"
        f"\\input{{figure_registry/figures/{args.figure_id}/figure.tex}}\n\\textbf{{Results text draft.}} \\input{{figure_registry/figures/{args.figure_id}/results.tex}}\n"
        f"\\textbf{{Discussion and boundary.}} \\input{{figure_registry/figures/{args.figure_id}/discussion.tex}}\n"
        f"\\input{{figure_registry/figures/{args.figure_id}/article.tex}}\n\\input{{figure_registry/figures/{args.figure_id}/reviewer.tex}}\n"
        f"\\textbf{{Evidence pointers.}}\\input{{figure_registry/figures/{args.figure_id}/evidence.tex}}\\input{{figure_registry/figures/{args.figure_id}/evidence_auto.tex}}\\input{{figure_registry/figures/{args.figure_id}/gap.tex}}\\vfill\n"
    )
    (target / "page.tex").write_text(page)
    manifest["authored_hashes"] = {name: sha256(target / name) for name in authored + ["page.tex"]}
    manifest["hash_updated_by"] = "prepare_figure_book_entry.py"
    (target / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))

    index_path = target_root / "index.tsv"
    index = pd.read_csv(index_path, sep="\t", dtype=str).fillna("")
    if args.figure_id in set(index["figure_id"]):
        raise SystemExit(f"figure ID already exists in index: {args.figure_id}")
    row = {
        "figure_id": args.figure_id,
        "title": manifest["title"],
        "group_id": manifest["group_id"],
        "article_target": manifest["article_targets"][0],
        "reviewer_ids": "REV-CANONICAL-639",
        "reviewer_status": "PENDING",
        "scientific_status": manifest["scientific_status"],
        "lock_status": "draft",
        "path": f"figures/{args.figure_id}",
        "asset": f"figures/{args.figure_id}/asset.pdf",
        "source_export": str(source_export),
    }
    index = pd.concat([index, pd.DataFrame([row])], ignore_index=True)
    index.to_csv(index_path, sep="\t", index=False)
    print(json.dumps({"figure_id": args.figure_id, "target": str(target), "asset_sha256": manifest["asset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
