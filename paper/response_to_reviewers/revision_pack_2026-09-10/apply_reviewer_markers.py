#!/usr/bin/env python3
"""Insert reviewer-map markers into manuscript and SI LaTeX sources.

Default mode is dry-run. Use --apply to edit files in place. The script is
idempotent: if a marker tag is already present near the target anchor, it will
not insert it again.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import shutil
import sys
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class Marker:
    file_name: str
    tag: str
    status: str
    summary: str
    anchor: str
    mode: str = "before"

    @property
    def latex(self) -> str:
        escaped = self.summary.replace("\\", r"\\")
        return rf"\reviewermap{{{self.tag}}}{{{self.status}}}{{{escaped}}}"


MARKERS: tuple[Marker, ...] = (
    Marker(
        "sn-article-final.tex",
        "REV-A4",
        "done-local",
        "Scope generalizable-standard language to a candidate framework demonstrated on Cr-doped Sb2Te3.",
        "This work establishes migration-based non-equilibrium probes as a data-efficient, generalizable standard",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A4",
        "done-local",
        "Frame generalizability as an open question rather than a completed claim.",
        "Can non-equilibrium probes generated from methods, such as NEB, provide a generalizable and efficient benchmark",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A7",
        "done-local",
        "Add nearest-neighbor literature and state the delta relative to the earlier arXiv/workshop report.",
        "diagnostic framework for designing more data-efficient learning loops",
        "after",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A2/REV-C10",
        "pending",
        "Grouped split by trajectory/pathway provenance; pending SOAP-RMSD-overlap-audit and grouped-split-retrained-rmse.",
        "Both FT-600K and FT-Multi",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-C8",
        "pending-release",
        "Confirm reported fine-tuned models used real-force AIMD frames only; pending final code-release hash for zero-force exporter fix.",
        "Energy and forces were read using the keys",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A1",
        "partial",
        "Fig. 3d is fixed-geometry DFT-path scoring; current 1-4 DFT reference candidate is 0.336050 eV; Foundation fixed-path error is +0.688246 eV.",
        "Our DFT calculations serve as the ground-truth reference. This establishes",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A1",
        "partial",
        "Define Fig. 3d and SI Fig. S3 as different NEB operators and report them side-by-side.",
        "NEB Stability as an Additional Robustness Criterion",
        "after",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A3",
        "pending-cluster",
        "Replace unsupported chance explanation with seed-stability evidence over Scratch and FT-600K retraining seeds.",
        "not through physical insight but random chance",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A5",
        "pending-cluster",
        "Add Scratch-5% random-initialization control trained on the same approximately 1000 configurations as FT-600K.",
        "The Critical Role of Task-Specific Fine-Tuning",
        "after",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A6/REV-C11/REV-C12",
        "done-local",
        "Claims softened to consistent-with; original 6191-d euclidean silhouette is 0.333; zero feature stub dropped.",
        "provides a direct mechanistic explanation",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-C11",
        "done-local",
        "Silhouette values are recomputed in original descriptor space; projected values are retained only as a baseline.",
        "Average silhouette scores (from $t$-SNE)",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A6/REV-C13",
        "partial",
        "SHAP explains surrogate error predictions, not MACE internals directly; current 5-fold CV R2 values are 0.9819, 0.8756, and 0.9730.",
        "This approach allows us to interpret the complex MACE potential",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A4",
        "done-local",
        "Broader principles are now framed as case-study hypotheses requiring cross-system validation.",
        "Our findings highlight several broader principles",
    ),
    Marker(
        "sn-article-final.tex",
        "REV-A4",
        "done-local",
        "Conclusion should keep framework generality as future validation rather than a final claim.",
        "Together, these results underscore",
    ),
    Marker(
        "sn-article-SI.tex",
        "REV-A1",
        "partial",
        "DFT NEB references must include per-path convergence status; current path 1-4 reference candidate is 0.336050 eV.",
        "All initial, final, and intermediate configurations were considered to be converged",
    ),
    Marker(
        "sn-article-SI.tex",
        "REV-A2/REV-C10",
        "pending",
        "Grouped split and train/test overlap audit must be documented once SOAP/RMSD values are available.",
        "Both FT-600K and FT-Multi",
    ),
    Marker(
        "sn-article-SI.tex",
        "REV-A1",
        "partial",
        "SI Fig. S3 is self-consistent MLFF NEB; 0.41 eV is a model-reported MLFF barrier, not a fixed-path DFT-reference error.",
        "In contrast, the MACE foundation model, without any system-specific fine-tuning, successfully converged the NEB calculation",
    ),
    Marker(
        "sn-article-SI.tex",
        "REV-A1",
        "partial",
        "Caption must say model-reported MLFF barrier and distinguish self-consistent MLFF NEB from fixed-geometry DFT-path scoring.",
        "yielding a migration barrier of 0.41 eV",
    ),
    Marker(
        "sn-article-SI.tex",
        "REV-A6/REV-C13",
        "partial",
        "Report SHAP surrogate CV R2 and final code-release provenance.",
        "Feature stability analysis across all models",
    ),
)


PREAMBLE = (
    "% Reviewer-map markers are no-op by default and are used only for source review.\n"
    r"\providecommand{\reviewermap}[3]{}" + "\n"
)


def insert_preamble(text: str) -> tuple[str, bool]:
    if r"\providecommand{\reviewermap}[3]" in text:
        return text, False
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(r"\documentclass"):
            lines.insert(idx + 1, PREAMBLE)
            return "".join(lines), True
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(r"\begin{document}"):
            lines.insert(idx, PREAMBLE)
            return "".join(lines), True
    return PREAMBLE + text, True


def marker_already_present(lines: list[str], marker: Marker) -> bool:
    return any(marker.latex in line for line in lines)


def insert_markers(text: str, markers: Iterable[Marker]) -> tuple[str, list[str], list[str]]:
    text, preamble_changed = insert_preamble(text)
    inserted: list[str] = ["PREAMBLE"] if preamble_changed else []
    missing: list[str] = []
    lines = text.splitlines(keepends=True)

    for marker in markers:
        anchor_idx = None
        for idx, line in enumerate(lines):
            if marker.anchor in line:
                anchor_idx = idx
                break
        if anchor_idx is None:
            missing.append(f"{marker.file_name}:{marker.tag}:{marker.anchor}")
            continue
        if marker_already_present(lines, marker):
            continue
        insert_at = anchor_idx if marker.mode == "before" else anchor_idx + 1
        lines.insert(insert_at, marker.latex + "\n")
        inserted.append(f"{marker.file_name}:{marker.tag}")

    return "".join(lines), inserted, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=pathlib.Path,
        help="Directory containing sn-article-final.tex and sn-article-SI.tex.",
    )
    parser.add_argument("--apply", action="store_true", help="Edit files in place.")
    parser.add_argument(
        "--backup-suffix",
        default=".bak-reviewer-markers",
        help="Backup suffix used with --apply.",
    )
    args = parser.parse_args()

    by_file: dict[str, list[Marker]] = {}
    for marker in MARKERS:
        by_file.setdefault(marker.file_name, []).append(marker)

    any_missing = False
    for file_name, markers in by_file.items():
        path = args.root / file_name
        if not path.exists():
            print(f"MISSING FILE: {path}", file=sys.stderr)
            any_missing = True
            continue

        original = path.read_text(encoding="utf-8")
        updated, inserted, missing = insert_markers(original, markers)
        any_missing = any_missing or bool(missing)
        print(f"\n{file_name}")
        print(f"  inserted_or_changed: {len(inserted)}")
        for item in inserted:
            print(f"    + {item}")
        if missing:
            print(f"  missing anchors: {len(missing)}")
            for item in missing:
                print(f"    ! {item}")

        if args.apply and updated != original:
            backup = path.with_name(path.name + args.backup_suffix)
            shutil.copy2(path, backup)
            path.write_text(updated, encoding="utf-8")
            print(f"  wrote: {path}")
            print(f"  backup: {backup}")
        elif updated != original:
            print("  dry-run only; rerun with --apply to write")
        else:
            print("  no changes needed")

    return 1 if any_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
