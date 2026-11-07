#!/usr/bin/env python3
"""Targeted maintenance CLI for the Revision 1 figure registry.

The registry is the editable source for the figure review book.  The legacy
211-page ``figure_pages.tex`` remains a frozen v1 artifact.  This tool never
rewrites per-figure authored fragments; it only validates them and rebuilds the
derived wrapper/book.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "figure_registry"
FIGURES = REGISTRY / "figures"
INDEX = REGISTRY / "index.tsv"
BOOK = ROOT / "figure_book.tex"
PREVIEW = REGISTRY / "preview"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifests(ids: list[str] | None = None) -> list[tuple[str, Path, dict]]:
    selected = set(ids or [])
    output = []
    for path in sorted(FIGURES.glob("FIG-*/manifest.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        figure_id = data.get("figure_id", path.parent.name)
        if selected and figure_id not in selected:
            continue
        output.append((figure_id, path, data))
    return output


def read_index() -> list[dict]:
    if not INDEX.exists():
        return []
    with INDEX.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latex_preamble() -> list[str]:
    return [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.72in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage[version=4]{mhchem}",
        r"\usepackage{xcolor}",
        r"\usepackage{caption}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\usepackage{url}",
        r"\usepackage{enumitem}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{microtype}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt}",
        r"\setlength{\emergencystretch}{3em}",
        r"\sloppy",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        r"\lhead{MigrationBench figure registry}",
        r"\rhead{Editable evidence-bounded pages}",
        r"\cfoot{\thepage}",
        r"\newcommand{\statusline}[1]{\colorbox{gray!12}{\parbox{0.96\linewidth}{#1}}}",
    ]


def book_lines(ids: list[str] | None = None, title: str = "MigrationBench figure registry") -> list[str]:
    selected = {item[0] for item in manifests(ids)} if ids else None
    lines = latex_preamble() + [r"\begin{document}", f"{{\\LARGE\\bfseries {title}\\par}}", r"\vspace{0.2in}"]
    rows = read_index()
    if selected is not None:
        rows = [row for row in rows if row.get("figure_id") in selected]
    lines.append(f"{{\\large {len(rows)} selected figure pages\\par}}")
    lines.append(r"\vspace{0.2in}")
    lines.append(r"This is a derived view. Edit the per-figure registry fragments, not this wrapper.")
    lines.append(r"\clearpage")
    for row in rows:
        lines.append(rf"\input{{figure_registry/figures/{row['figure_id']}/page.tex}}")
    lines.append(r"\end{document}")
    return lines


def build_book(ids: list[str] | None = None, output: Path = BOOK) -> None:
    output.write_text("\n".join(book_lines(ids)) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def validate(ids: list[str] | None = None) -> int:
    rows = manifests(ids)
    failures: list[str] = []
    seen: set[str] = set()
    for figure_id, manifest_path, data in rows:
        if figure_id in seen:
            failures.append(f"duplicate figure_id: {figure_id}")
        seen.add(figure_id)
        if not figure_id.startswith("FIG-"):
            failures.append(f"invalid figure_id: {figure_id}")
        figure_dir = manifest_path.parent
        for required in ("page.tex", "figure.tex", "caption.tex", "results.tex", "discussion.tex", "reviewer.tex", "evidence.tex", "evidence_auto.tex", "article.tex", "gap.tex"):
            if not (figure_dir / required).exists():
                failures.append(f"{figure_id}: missing {required}")
        asset = figure_dir / data.get("asset", "")
        if not asset.exists():
            failures.append(f"{figure_id}: missing asset {asset}")
        if data.get("lock_status") not in {"draft", "reviewed", "locked", "promoted", "retired"}:
            failures.append(f"{figure_id}: invalid lock_status")
        if data.get("reviewer_status") not in {"DONE", "PARTIAL", "PENDING", "NO_DIRECT_REVIEWER_FIGURE_LINK"}:
            failures.append(f"{figure_id}: invalid reviewer_status")
        authored = data.get("authored_hashes", {})
        if data.get("lock_status") in {"locked", "promoted"}:
            for name, expected in authored.items():
                path = figure_dir / name
                if path.exists() and expected and sha256(path) != expected:
                    failures.append(f"{figure_id}: locked authored file changed: {name}")
    index_rows = read_index()
    index_ids = {row.get("figure_id") for row in index_rows}
    if ids is None and index_ids and index_ids != seen:
        failures.append(f"index mismatch: index={len(index_ids)} manifests={len(seen)}")
    if failures:
        print("VALIDATION FAILED")
        print("\n".join(f"- {item}" for item in failures))
        return 1
    print(f"VALIDATION PASSED: {len(rows)} figure registries")
    return 0


def show(figure_id: str) -> int:
    found = manifests([figure_id])
    if not found:
        print(f"unknown figure: {figure_id}")
        return 1
    _, path, data = found[0]
    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"manifest={path}")
    return 0


def query(args: argparse.Namespace) -> int:
    rows = read_index()
    for row in rows:
        haystack = " ".join(row.values()).lower()
        if args.text.lower() in haystack:
            print("\t".join(row.get(key, "") for key in ("figure_id", "title", "group_id", "reviewer_ids", "reviewer_status", "lock_status", "path")))
    return 0


def render_preview(figure_id: str) -> int:
    if validate([figure_id]) != 0:
        return 1
    PREVIEW.mkdir(parents=True, exist_ok=True)
    tex = PREVIEW / f"{figure_id}.tex"
    tex.write_text("\n".join(latex_preamble() + [r"\begin{document}", rf"\input{{figure_registry/figures/{figure_id}/page.tex}}", r"\end{document}"]) + "\n", encoding="utf-8")
    latexmk = shutil.which("latexmk")
    if not latexmk:
        print(f"wrote preview source {tex}; latexmk unavailable")
        return 0
    with tempfile.TemporaryDirectory(prefix=f"migrationbench-{figure_id}-") as tmp:
        result = subprocess.run([
            latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
            f"-outdir={tmp}", str(tex),
        ], cwd=ROOT, text=True, capture_output=True)
        (PREVIEW / f"{figure_id}.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            print(result.stdout[-4000:])
            print(result.stderr[-4000:])
            return result.returncode
        shutil.copy2(Path(tmp) / f"{figure_id}.pdf", PREVIEW / f"{figure_id}.pdf")
    print(f"wrote {PREVIEW / f'{figure_id}.pdf'}")
    return 0


def update_hashes(figure_id: str) -> int:
    found = manifests([figure_id])
    if not found:
        print(f"unknown figure: {figure_id}")
        return 1
    _, path, data = found[0]
    figure_dir = path.parent
    data["authored_hashes"] = {
        name: sha256(figure_dir / name)
        for name in ("page.tex", "figure.tex", "caption.tex", "results.tex", "discussion.tex", "reviewer.tex", "evidence.tex", "article.tex", "gap.tex")
        if (figure_dir / name).exists()
    }
    data["hash_updated_by"] = "figurectl update-hashes"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"updated authored hashes for {figure_id}; no authored content was changed")
    return 0


def set_lock(figure_id: str, lock_status: str) -> int:
    allowed = {"draft", "reviewed", "locked", "promoted", "retired"}
    if lock_status not in allowed:
        print(f"invalid lock status: {lock_status}; choose from {', '.join(sorted(allowed))}")
        return 1
    found = manifests([figure_id])
    if not found:
        print(f"unknown figure: {figure_id}")
        return 1
    _, path, data = found[0]
    data["lock_status"] = lock_status
    data["lock_changed_by"] = "figurectl set-lock"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"set {figure_id} lock_status={lock_status}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("show")
    p.add_argument("figure_id")

    p = sub.add_parser("validate")
    p.add_argument("figure_ids", nargs="*")

    p = sub.add_parser("query")
    p.add_argument("text")

    p = sub.add_parser("render")
    p.add_argument("figure_id")

    p = sub.add_parser("update-hashes")
    p.add_argument("figure_id")

    p = sub.add_parser("set-lock")
    p.add_argument("figure_id")
    p.add_argument("lock_status", choices=("draft", "reviewed", "locked", "promoted", "retired"))

    p = sub.add_parser("build-book")
    p.add_argument("--only", nargs="*")

    args = parser.parse_args()
    if args.command == "show":
        return show(args.figure_id)
    if args.command == "validate":
        return validate(args.figure_ids or None)
    if args.command == "query":
        return query(args)
    if args.command == "render":
        return render_preview(args.figure_id)
    if args.command == "update-hashes":
        return update_hashes(args.figure_id)
    if args.command == "set-lock":
        return set_lock(args.figure_id, args.lock_status)
    if args.command == "build-book":
        build_book(args.only or None)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
