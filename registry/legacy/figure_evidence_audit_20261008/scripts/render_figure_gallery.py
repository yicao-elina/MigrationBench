#!/usr/bin/env python3
"""Write a clickable index of every audited logical figure output."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/figure_evidence_audit_20261008")


def link(path: str, label: str) -> str:
    return f"[{label}]({path})"


def main() -> None:
    inventory = json.loads((ROOT / "figure_inventory.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader((ROOT / "figure_evidence_matrix.tsv").open(encoding="utf-8"), delimiter="\t"))
    by_logical = {row["logical_path"]: row for row in matrix}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in inventory:
        matrix_row = by_logical.get(record["logical_path"], {})
        group = matrix_row.get("group_id", "UNCLASSIFIED")
        grouped[group].append(record)

    lines = [
        "# Figure gallery / clickable completeness index (2026-10-08)",
        "",
        "This index contains every logical rendered output from the three scanned source roots. PNG/SVG/PDF siblings remain one logical figure; the links below expose the available formats so the user can inspect the actual files. The seven audit-generated table-backed packages are listed separately at the end.",
        "",
        f"- Source logical figures: **{len(inventory)}**",
        f"- Source rendered exports: **{sum(len(record['outputs']) for record in inventory)}**",
        f"- Source groups: **{len(grouped)}**",
        "",
    ]
    for group in sorted(grouped):
        records = sorted(grouped[group], key=lambda record: record["logical_figure"])
        lines.extend([f"## {group} ({len(records)} logical figures)", "", "| Logical figure | Root | Recent | Status | Available exports |", "|---|---|---|---|---|"])
        for record in records:
            preferred = sorted(record["outputs"], key=lambda item: {".png": 0, ".svg": 1, ".pdf": 2}.get(item["extension"], 3))[0]
            exports = " ".join(link(item["path"], item["extension"].lstrip(".")) for item in record["outputs"])
            matrix_row = by_logical.get(record["logical_path"], {})
            status = matrix_row.get("evidence_chain_status", "unclassified")
            recent = "yes" if record["recent_since_2026_08_08"] else "no"
            lines.append(f"| {link(preferred['path'], record['logical_figure'])} | `{record['root_id']}` | {recent} | `{status}` | {exports} |")
        lines.append("")

    lines.extend([
        "## Audit-generated table-backed packages",
        "",
        "These are additive figures generated from previously table-backed analyses; they are not included in the source count above.",
        "",
        "| Package | Figure | Evidence package |",
        "|---|---|---|",
    ])
    package_root = ROOT / "generated/table_backed_packages"
    for package in sorted(path for path in package_root.iterdir() if path.is_dir()):
        figure = next(package.glob("*.png"), None)
        manifest = package / "evidence_package_manifest.json"
        if figure is not None:
            lines.append(f"| `{package.name}` | {link(str(figure), 'PNG')} / {link(str(figure.with_suffix('.svg')), 'SVG')} / {link(str(figure.with_suffix('.pdf')), 'PDF')} | {link(str(manifest), 'manifest')} |")

    (ROOT / "figure_gallery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {ROOT / 'figure_gallery.md'} ({len(inventory)} source figures, {len(grouped)} groups)")


if __name__ == "__main__":
    main()
