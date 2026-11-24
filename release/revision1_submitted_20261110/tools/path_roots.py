"""Portable path labels shared by build_index, the SQL registry and verify_trace.

Every absolute laptop path is written as `<label>:<relative path>` so the index and the
SQL registry do not depend on one user's home directory. `rev1_path_roots` in the SQL
registry stores the label -> absolute-root table to resolve them again.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
HOME = "/Users/alina"

# (label, absolute root, host, meaning). Longest roots first is enforced in portable().
ROOTS = [
    ("archive", f"{HOME}/Project/migrationbench-archive", "laptop",
     "local archive store for heavy binaries and the Downloads archive; back up to Skipjack"),
    ("chatgpt", f"{HOME}/Documents/ChatGPT/Wiley-MigrationBench", "laptop (iCloud, partly evicted)",
     "ChatGPT/Codex workspace, cold copy; trace-relevant files imported"),
    ("legacy26", f"{HOME}/Project/26MIgrationBench", "laptop", "legacy repo, no remote"),
    ("remotion", f"{HOME}/Project/migrationbench-remotion", "laptop", "Remotion video/schematics project"),
    ("downloads", f"{HOME}/Downloads", "laptop",
     "original ~/Downloads location; MigrationBench files moved to archive:downloads_20261110 on 2026-11-10"),
    ("gdrive", f"{HOME}/Library/CloudStorage/GoogleDrive-yicao.alina@gmail.com/My Drive", "Google Drive",
     "Obsidian project notes"),
    ("codex-worktree", f"{HOME}/.codex/worktrees", "laptop (removed)",
     "Codex worktrees, removed 2026-11-10 after their content was merged"),
    ("git", "", "this repository's history", "git:<commit>:<path>; read with `git show <commit>:<path>`"),
    ("rockfish", "", "Rockfish HPC", "absolute path on Rockfish (/scratch16, /data); see ROCKFISH_MIGRATION_PLAN.tsv"),
]
REPO_ROOTS = [f"{HOME}/Project/migrationbench-paper-repro", f"{HOME}/Project/migrationbench-integrate"]
EXTERNAL_LABELS = tuple(label for label, *_ in ROOTS) + ("skipjack",)
# Older spellings that wrapped an absolute path in a host prefix.
_WRAPPERS = re.compile(r"\b(?:local|icloud|external):(?=/Users/)")
# Repo-relative directories a moved Codex worktree corresponds to.
WORKTREE_IN_REPO = {"canva-figure-provenance": ""}


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


DOWNLOADS_ARCHIVE = f"{HOME}/Project/migrationbench-archive/downloads_20261110"


def import_rewrites() -> list[tuple[str, str]]:
    """Outside-repo sources mapped to where they live now: the repo copy or the archive store."""
    plan = read_tsv(PACKAGE / "imports/import_plan.tsv")
    pairs = [(row["source"].rstrip("/"), row["dest"].rstrip("/")) for row in plan]
    moved = read_tsv(PACKAGE / "imports/DOWNLOADS_ARCHIVE.tsv")
    for row in moved:  # map both the original and the archived top-level name to the archive copy
        original = row["original_path"].removeprefix(f"{HOME}/Downloads/").split("/")[0]
        stored = row["stored_at"].split("/downloads_20261110/", 1)[1].split("/")[0]
        if (Path(DOWNLOADS_ARCHIVE) / stored).exists():
            pairs += [(f"{HOME}/Downloads/{name}", f"{DOWNLOADS_ARCHIVE}/{stored}") for name in {original, stored}]
    pairs = list(dict.fromkeys(pairs))
    chatgpt = f"{HOME}/Documents/ChatGPT/Wiley-MigrationBench"
    pairs += [
        # the bundle's stale name for the foundation checkpoint folder
        (f"{HOME}/Downloads/mace-foundation-models", f"{DOWNLOADS_ARCHIVE}/mace-models"),
        (f"{chatgpt}/analysis/foundation_interlayer_sliding_10model_20261024",
         "reproducibility/canva_palette_vector_bundle_20261105/data/source_archive/analysis/"
         "foundation_interlayer_sliding_10model_20261024"),
        (f"{chatgpt}/output/revision1_camera_ready/reproducibility", "release/revision1_camera_ready/reproducibility"),
    ]
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def portable(text: str) -> str:
    """Replace absolute laptop paths with repo-relative paths or `<label>:` paths."""
    if not text:
        return text
    text = _WRAPPERS.sub("", text).replace("~/Downloads/", f"{HOME}/Downloads/")
    for root in REPO_ROOTS:
        text = text.replace(root + "/", "").replace(root, ".")
    for name, rel in WORKTREE_IN_REPO.items():
        text = text.replace(f"{HOME}/.codex/worktrees/{name}/", rel)
    for label, root, *_ in sorted((r for r in ROOTS if r[1]), key=lambda r: len(r[1]), reverse=True):
        text = text.replace(root + "/", f"{label}:").replace(root, f"{label}:.")
    return text


def localize(text: str, rules: list[tuple[str, str]]) -> str:
    """Point imported sources at their repo copy, then make the rest portable."""
    text = _WRAPPERS.sub("", text or "").replace("~/Downloads/", f"{HOME}/Downloads/")
    for old, new in rules:
        text = text.replace(old, new)
    return portable(text.replace("C/analysis/", "analysis/"))


def resolve(token: str) -> Path | None:
    """Absolute local path for a `<label>:rel` token, or None for remote labels."""
    label, _, rel = token.partition(":")
    for name, root, *_ in ROOTS:
        if name == label and root:
            return Path(root) / rel
    return None


def root_rows() -> list[dict]:
    return [{"label": label, "absolute_root": root, "host": host, "meaning": meaning}
            for label, root, host, meaning in ROOTS]
