#!/usr/bin/env python3
"""Find evidence-bearing directories that are not represented by rendered figures.

This complements the rendered-output inventory. It deliberately treats a
table-backed analysis as a candidate, not as a scientific result: the output
records the data/code/text evidence and leaves the render/scientific gate to a
follow-up audit.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


CURRENT_ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
ABLATION_WORKTREE_ROOT = Path("/Users/alina/.codex/worktrees/bf6e/Wiley-MigrationBench")
PAPER_ROOT = Path("/Users/alina/Project/migrationbench-paper-repro")
OUT_ROOT = CURRENT_ROOT / "analysis/figure_evidence_audit_20261008"

DATA_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".jsonl", ".sqlite", ".db", ".npz", ".npy",
    ".parquet", ".extxyz", ".xyz", ".path", ".dat",
}
SCRIPT_EXTENSIONS = {".py", ".sh", ".slurm", ".smk", ".awk", ".r"}
TEXT_EXTENSIONS = {".md", ".tex", ".txt", ".yaml", ".yml", ".toml"}
IMAGE_EXTENSIONS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff"}
SKIP_PARTS = {
    ".git", ".venv", ".venv 2", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".snakemake", "node_modules", "figure_evidence_audit_20261008",
}
SEMANTIC_TERMS = (
    "analysis", "ablation", "benchmark", "validation", "sensitivity", "coverage",
    "retrain", "campaign", "results", "table", "published", "processed", "endpoint",
    "dft", "neb", "md", "trajectory", "shap", "leakage", "convergence", "candidate",
    "force", "handoff", "roster", "path", "profile", "transport", "training",
)


def skipped(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "render"
    if suffix in DATA_EXTENSIONS:
        return "data"
    if suffix in SCRIPT_EXTENSIONS:
        return "script"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    return "other"


def evidence_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    try:
        for path in directory.iterdir():
            if path.is_file() and not skipped(path):
                files.append(path)
    except OSError:
        return []
    return sorted(files)


def semantic_score(directory: Path, counts: dict[str, int], files: list[Path]) -> int:
    value = str(directory).lower()
    score = sum(2 for term in SEMANTIC_TERMS if term in value)
    score += min(counts["data"], 5)
    score += 2 if counts["script"] else 0
    score += 1 if counts["text"] else 0
    names = " ".join(path.name.lower() for path in files)
    if re.search(r"caption|result|discussion|figure|plot|table|manifest|summary", names):
        score += 2
    return score


def collect(root: Path, root_id: str) -> tuple[list[dict], list[dict]]:
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    if not root.exists():
        return [], []
    for path in root.rglob("*"):
        if path.is_file() and not skipped(path):
            by_dir[path.parent].append(path)

    # A package root can contain rendered outputs in a nested `figures/`
    # directory. Count those descendants so a parent campaign is not falsely
    # reported as unrendered merely because its PNGs are one level deeper.
    subtree_render_counts: dict[Path, int] = defaultdict(int)
    for directory, paths in by_dir.items():
        subtree_render_counts[directory] += sum(classify_file(path) == "render" for path in paths)
    for directory in sorted(by_dir, key=lambda item: len(item.parts), reverse=True):
        for parent in directory.parents:
            if parent in by_dir:
                subtree_render_counts[parent] += subtree_render_counts[directory]

    all_candidates: list[dict] = []
    for directory, paths in sorted(by_dir.items(), key=lambda item: str(item[0])):
        counts = defaultdict(int)
        for path in paths:
            counts[classify_file(path)] += 1
        if not counts["data"] or not (counts["script"] or counts["text"]):
            continue
        score = semantic_score(directory, counts, paths)
        if score < 5:
            continue
        relative = str(directory.relative_to(root))
        record = {
            "root_id": root_id,
            "directory": str(directory),
            "relative_directory": relative,
            "render_present": bool(subtree_render_counts[directory]),
            "candidate_status": "rendered_with_evidence" if subtree_render_counts[directory] else "table_backed_unrendered",
            "data_file_count": counts["data"],
            "script_file_count": counts["script"],
            "text_file_count": counts["text"],
            "render_file_count": subtree_render_counts[directory],
            "score": score,
            "evidence_files": [str(path) for path in sorted(paths)],
        }
        all_candidates.append(record)

    unrendered = [record for record in all_candidates if not record["render_present"]]
    return all_candidates, unrendered


def main() -> None:
    all_candidates, unrendered = collect(CURRENT_ROOT, "wiley-migrationbench")
    worktree_candidates, worktree_unrendered = collect(ABLATION_WORKTREE_ROOT, "wiley-migrationbench-ablation-worktree")
    all_candidates.extend(worktree_candidates)
    unrendered.extend(worktree_unrendered)
    paper_candidates, paper_unrendered = collect(PAPER_ROOT, "paper-repro")
    all_candidates.extend(paper_candidates)
    unrendered.extend(paper_unrendered)
    all_candidates.sort(key=lambda record: (-record["score"], record["directory"]))
    unrendered.sort(key=lambda record: (-record["score"], record["directory"]))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": [str(CURRENT_ROOT), str(PAPER_ROOT)],
        "all_evidence_bearing_candidate_directories": all_candidates,
        "table_backed_unrendered_candidates": unrendered,
    }
    (OUT_ROOT / "figure_candidate_inventory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fields = [
        "candidate_id", "root_id", "relative_directory", "candidate_status",
        "data_file_count", "script_file_count", "text_file_count", "render_file_count",
        "score", "evidence_files",
    ]
    with (OUT_ROOT / "figure_candidate_inventory.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, record in enumerate(all_candidates, start=1):
            writer.writerow({
                "candidate_id": f"CAND-{index:03d}",
                "root_id": record["root_id"],
                "relative_directory": record["relative_directory"],
                "candidate_status": record["candidate_status"],
                "data_file_count": record["data_file_count"],
                "script_file_count": record["script_file_count"],
                "text_file_count": record["text_file_count"],
                "render_file_count": record["render_file_count"],
                "score": record["score"],
                "evidence_files": " | ".join(record["evidence_files"]),
            })

    summary = {
        "all_evidence_bearing_candidate_directories": len(all_candidates),
        "table_backed_unrendered_candidates": len(unrendered),
        "by_root": {
            root_id: sum(record["root_id"] == root_id for record in all_candidates)
            for root_id in ("wiley-migrationbench", "wiley-migrationbench-ablation-worktree", "paper-repro")
        },
    }
    (OUT_ROOT / "figure_candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nTop table-backed unrendered candidates:")
    for record in unrendered[:40]:
        print(f"{record['score']:>3}  {record['root_id']:<22}  {record['relative_directory']}")


if __name__ == "__main__":
    main()
