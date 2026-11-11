#!/usr/bin/env python3
"""Migrate the frozen 211-page review book into editable per-figure records."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


MAIN = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
AUDIT = MAIN / "analysis/figure_evidence_audit_20261008"
PAPER = Path("/Users/alina/Project/migrationbench-paper-repro/paper/revision1")
LEGACY_TEX = PAPER / "figure_pages.tex"
LEGACY_ROOT = PAPER / "figure_pages"
REGISTRY = PAPER / "figure_registry"
FIGURES = REGISTRY / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "⁻": "-", "¹": "1", "²": "2", "³": "3", "Å": r"\AA{}",
        "±": r"\(\pm\)", "×": r"\(\times\)", "µ": r"\(\mu\)",
        "–": "--", "—": "---", "≤": r"\(\leq\)", "≥": r"\(\geq\)",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def humanize(value: str) -> str:
    text = value.rsplit("/", 1)[-1].replace("_", " ").replace("-", " ")
    return " ".join(text.split()).strip().capitalize()


def balanced(text: str, marker: str) -> str:
    marker_start = text.index(marker)
    open_index = marker_start + len(marker) - 1
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index]
    raise ValueError(f"unbalanced LaTeX marker: {marker}")


def between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    return text[start:end].strip()


def page_blocks(tex: str) -> dict[str, str]:
    matches = list(re.finditer(r"\\section\{(FIG-\d{3})\s+(?:—|---)[^\n]*\}", tex))
    blocks = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tex)
        blocks[match.group(1)] = tex[start:end]
    return blocks


def load_rows() -> tuple[list[dict], dict[str, dict], dict[str, list[dict]]]:
    manifest_rows = list(csv.DictReader((LEGACY_ROOT / "figure_pages_manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    matrix_path = AUDIT / "figure_evidence_matrix.tsv"
    matrix = {}
    for row in csv.DictReader(matrix_path.open(encoding="utf-8"), delimiter="\t"):
        # Both keys are intentionally accepted: the legacy manifest stores the
        # repository-relative logical_figure, while the audit matrix also has
        # the absolute logical_path used by the original scanner.
        matrix[row["logical_figure"]] = row
        matrix[row["logical_path"]] = row
    reviewer_rows = defaultdict(list)
    reviewer_path = LEGACY_ROOT / "figure_reviewer_annotations.tsv"
    for row in csv.DictReader(reviewer_path.open(encoding="utf-8"), delimiter="\t"):
        reviewer_rows[row["figure_id"]].append(row)
    return manifest_rows, matrix, reviewer_rows


def write_registry_readme() -> None:
    text = """# MigrationBench figure registry

This is the long-term editable source for the figure review book. The legacy
`../figure_pages.tex` and `../figure_pages.pdf` are frozen v1 review artifacts;
they are not overwritten by this registry.

Each `figures/FIG-NNN/` directory contains an editable page, caption, Results,
Discussion, reviewer linkage, evidence pointers, gap status, manifest, and a
canonical asset. Edit those fragments directly. `figurectl.py` validates them,
renders one-figure previews, and rebuilds the derived `../figure_book.tex`.

## Lock lifecycle

`draft -> reviewed -> locked -> promoted -> retired`.

`locked` and `promoted` records are never rewritten by a generator. If their
authored hashes change, validation fails until an explicit human runs
`figurectl.py update-hashes FIG-NNN` after reviewing the change.

## Targeted maintenance

```bash
python figurectl.py show FIG-002
python figurectl.py query REV-C9
python figurectl.py validate FIG-002
python figurectl.py render FIG-002
python figurectl.py set-lock FIG-002 reviewed
python figurectl.py set-lock FIG-002 locked
python figurectl.py build-book
```

The default agent workflow is: query the index, open only the requested
`FIG-NNN` directory, validate its evidence/reviewer status, edit the smallest
fragment, render the targeted preview, and leave unrelated figures untouched.

`evidence.tex` is the human-editable evidence note. `evidence_auto.tex` and
`evidence.md` are generated from the audit matrix and should not be hand-edited;
refresh them with the audit repair script when the evidence scan changes.
"""
    (REGISTRY / "README.md").write_text(text, encoding="utf-8")


def write_schema() -> None:
    text = """# Figure registry schema

Required manifest fields:

- `figure_id`: immutable `FIG-NNN` identifier.
- `logical_figure`, `group_id`, `parent_group`, `sibling_ids`: stable semantic identity.
- `article_targets`: proposed main-text/SI locations.
- `reviewer_links`, `reviewer_status`: reviewer question, completed evidence, and required closure.
- `scientific_status`: publication, historical, reconstruction, screening, or gate status.
- `lock_status`: `draft`, `reviewed`, `locked`, `promoted`, or `retired`.
- `asset`, `asset_sha256`: canonical rendered asset.
- `authored_hashes`: hashes of editable fragments used to protect locked records.
- `source_export`, `evidence_candidates`, `legacy_v1_page`: provenance pointers.
- `evidence_auto.tex`: generated evidence candidates; do not hand-edit it.

The per-figure `.tex` fragments are the authored LaTeX surface. The registry
manifest is machine-readable metadata. The book wrapper is derived and may be
regenerated at any time.
"""
    (REGISTRY / "SCHEMA.md").write_text(text, encoding="utf-8")


def main() -> None:
    if FIGURES.exists() and any(FIGURES.glob("FIG-*/manifest.yaml")):
        raise SystemExit("figure registry already exists; refusing to overwrite it")
    if not LEGACY_TEX.exists():
        raise SystemExit(f"missing frozen source: {LEGACY_TEX}")

    rows, matrix, reviewer_rows = load_rows()
    blocks = page_blocks(LEGACY_TEX.read_text(encoding="utf-8"))
    if len(rows) != len(blocks):
        raise SystemExit(f"legacy row/block mismatch: rows={len(rows)} blocks={len(blocks)}")

    REGISTRY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    group_members = defaultdict(list)
    for row in rows:
        group_members[row["group_id"]].append(row["figure_id"])

    index_rows = []
    for row in rows:
        figure_id = row["figure_id"]
        figure_dir = FIGURES / figure_id
        figure_dir.mkdir(parents=True, exist_ok=True)
        block = blocks[figure_id]
        source_asset = PAPER / row["overleaf_asset"]
        asset_name = f"asset{Path(source_asset).suffix.lower()}"
        shutil.copy2(source_asset, figure_dir / asset_name)

        caption = balanced(block, r"\caption{")
        results = between(block, r"\textbf{Results text draft.}", r"\textbf{Discussion and boundary.}")
        discussion = between(block, r"\textbf{Discussion and boundary.}", r"\textbf{Article placement.}")
        article = between(block, r"\textbf{Article placement.}", r"\textbf{Reviewer linkage:}")
        evidence = between(block, r"\textbf{Evidence pointers.}", r"\statusline{")
        gap = row.get("gaps_to_resolve", "") or "No additional gap recorded in the matrix."
        reviewer = reviewer_rows.get(figure_id, [])
        reviewer = [item for item in reviewer if item.get("reviewer_id")]
        matrix_row = matrix.get(row["logical_figure"], {})
        article_target = row.get("article_target", "")
        source_candidates = {
            key: [item.strip() for item in matrix_row.get(key, "").split(" | ") if item.strip()]
            for key in ("caption_discussion_source_candidates", "raw_or_processed_data_candidates", "plot_script_candidates")
        }

        (figure_dir / "caption.tex").write_text(caption.strip() + "\n", encoding="utf-8")
        (figure_dir / "results.tex").write_text(results.strip() + "\n", encoding="utf-8")
        (figure_dir / "discussion.tex").write_text(discussion.strip() + "\n", encoding="utf-8")
        (figure_dir / "article.tex").write_text(r"\textbf{Article placement.} " + article.strip() + "\n", encoding="utf-8")
        (figure_dir / "evidence.tex").write_text(evidence.strip() + "\n", encoding="utf-8")
        (figure_dir / "gap.tex").write_text(r"\statusline{\textbf{Audit gap / next action:} " + latex_escape(gap) + "}\n", encoding="utf-8")

        (figure_dir / "figure.tex").write_text(
            "\n".join([
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=\linewidth,height=0.42\textheight,keepaspectratio]{{figure_registry/figures/{figure_id}/{asset_name}}}",
                rf"\caption{{\protect\input{{figure_registry/figures/{figure_id}/caption.tex}}}}",
                rf"\label{{fig:candidate_{figure_id[4:]}}}",
                r"\end{figure}",
            ]) + "\n",
            encoding="utf-8",
        )

        reviewer_lines = [r"\textbf{Reviewer linkage:} overall status " + latex_escape(row.get("reviewer_status", "")) + "."]
        if reviewer:
            reviewer_lines.append(r"\begin{itemize}[leftmargin=1.4em,noitemsep]")
            for item in reviewer:
                reviewer_lines.append(
                    r"\item \textbf{" + latex_escape(item["reviewer_id"]) + " (" + latex_escape(item["status"]) + r")}: "
                    + latex_escape(item["question"]) + r" \emph{Completed:} "
                    + latex_escape(item["completed_evidence"]) + r" \emph{Still required:} "
                    + latex_escape(item["still_required"])
                )
            reviewer_lines.append(r"\end{itemize}")
        else:
            reviewer_lines.append("No direct reviewer-response figure linkage was identified; retain this page for scientific/article audit.")
        (figure_dir / "reviewer.tex").write_text("\n".join(reviewer_lines) + "\n", encoding="utf-8")

        page_lines = [
            r"\clearpage",
            rf"\section{{{latex_escape(figure_id + ' — ' + humanize(row['logical_figure']))}}}",
            rf"\label{{figpage:{figure_id.lower()}}}",
            rf"\textbf{{Group:}} {latex_escape(row['group_id'])}\quad \textbf{{Evidence chain:}} {latex_escape(matrix_row.get('evidence_chain_status', 'manifested'))}\quad \textbf{{Status:}} {latex_escape(row.get('status_tags', ''))}",
            rf"\input{{figure_registry/figures/{figure_id}/figure.tex}}",
            rf"\textbf{{Results text draft.}} \input{{figure_registry/figures/{figure_id}/results.tex}}",
            rf"\textbf{{Discussion and boundary.}} \input{{figure_registry/figures/{figure_id}/discussion.tex}}",
            rf"\input{{figure_registry/figures/{figure_id}/article.tex}}",
            rf"\input{{figure_registry/figures/{figure_id}/reviewer.tex}}",
            r"\textbf{Evidence pointers.}",
            rf"\input{{figure_registry/figures/{figure_id}/evidence.tex}}",
            rf"\input{{figure_registry/figures/{figure_id}/evidence_auto.tex}}",
            rf"\input{{figure_registry/figures/{figure_id}/gap.tex}}",
            r"\vfill",
        ]
        (figure_dir / "page.tex").write_text("\n".join(page_lines) + "\n", encoding="utf-8")
        (figure_dir / "evidence.md").write_text(
            "# Evidence candidates\n\n" + "\n".join(
                f"## {key}\n\n" + "\n".join(f"- `{value}`" for value in values)
                for key, values in source_candidates.items()
            ) + "\n",
            encoding="utf-8",
        )
        auto_lines = [
            r"\emph{Machine-collected evidence candidates (generated from the audit matrix; do not edit this file).}",
            r"\begin{itemize}[leftmargin=1.4em,noitemsep]",
        ]
        labels = {
            "caption_discussion_source_candidates": "Caption/discussion source",
            "raw_or_processed_data_candidates": "Raw or processed data",
            "plot_script_candidates": "Plot or processing script",
        }
        for key, label in labels.items():
            values = source_candidates.get(key, [])
            if values:
                for value in values:
                    auto_lines.append(rf"\item \textbf{{{label}:}} \url{{{value}}}")
            else:
                auto_lines.append(rf"\item \textbf{{{label}:}} not identified in the audit matrix.")
        auto_lines.extend([r"\end{itemize}"])
        (figure_dir / "evidence_auto.tex").write_text("\n".join(auto_lines) + "\n", encoding="utf-8")

        authored_files = ("page.tex", "figure.tex", "caption.tex", "results.tex", "discussion.tex", "reviewer.tex", "evidence.tex", "article.tex", "gap.tex")
        manifest = {
            "schema_version": 1,
            "figure_id": figure_id,
            "logical_figure": row["logical_figure"],
            "title": humanize(row["logical_figure"]),
            "group_id": row["group_id"],
            "parent_group": row["group_id"],
            "sibling_ids": [item for item in group_members[row["group_id"]] if item != figure_id],
            "article_targets": [item.strip() for item in article_target.split(";") if item.strip()],
            "reviewer_status": row.get("reviewer_status", "NO_DIRECT_REVIEWER_FIGURE_LINK"),
            "reviewer_links": [
                {
                    "id": item["reviewer_id"],
                    "status": item["status"],
                    "question": item["question"],
                    "completed_evidence": item["completed_evidence"],
                    "still_required": item["still_required"],
                }
                for item in reviewer
            ],
            "scientific_status": row.get("status_tags", ""),
            "lock_status": "draft",
            "promotion_status": "review_book_only",
            "asset": asset_name,
            "asset_sha256": sha256(figure_dir / asset_name),
            "source_export": row.get("source_export", ""),
            "legacy_v1_page": f"../figure_pages.tex#{figure_id}",
            "evidence_candidates": source_candidates,
            "authored_hashes": {name: sha256(figure_dir / name) for name in authored_files},
            "migration_note": "Migrated from frozen 211-page review book; authored fragments are now the editable surface.",
        }
        (figure_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
        index_rows.append({
            "figure_id": figure_id,
            "title": manifest["title"],
            "group_id": row["group_id"],
            "article_target": article_target,
            "reviewer_ids": ",".join(item["reviewer_id"] for item in reviewer),
            "reviewer_status": manifest["reviewer_status"],
            "scientific_status": manifest["scientific_status"],
            "lock_status": manifest["lock_status"],
            "path": f"figures/{figure_id}",
            "asset": f"figures/{figure_id}/{asset_name}",
            "source_export": row.get("source_export", ""),
        })

    with (REGISTRY / "index.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(index_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(index_rows)
    (REGISTRY / "registry_version.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "figure_count": len(index_rows), "migrated_from": str(LEGACY_TEX), "migration_source_sha256": sha256(LEGACY_TEX)}, sort_keys=False),
        encoding="utf-8",
    )
    write_registry_readme()
    write_schema()
    print(f"migrated {len(index_rows)} figures to {REGISTRY}")


if __name__ == "__main__":
    main()
