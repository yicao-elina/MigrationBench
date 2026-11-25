#!/usr/bin/env python3
"""Verify output coverage and native-vector properties, then write manifests."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fitz


PACKAGE = Path(__file__).resolve().parents[1]
FIGURES = PACKAGE / "figures"
MANIFEST = PACKAGE / "CANVA_PALETTE_VECTOR_MANIFEST.json"
QA_TSV = PACKAGE / "manifests/OUTPUT_VECTOR_QA.tsv"

FIGURE_FILES = {
    "CANVA-P03": "CANVA-P03_parity_2x4_palette_updated",
    "CANVA-P04": "CANVA-P04_frame_rmse_palette_updated",
    "CANVA-P05": "CANVA-P05/md_six_panel_palette_updated",
    "CANVA-P06": "CANVA-P06_neb_cr_r05_palette_updated",
    "CANVA-P07": "CANVA-P07_foundation_class_3x3_palette_updated",
    "CANVA-P08": "CANVA-P08_foundation_class_4x3_palette_updated",
    "CANVA-P09": "CANVA-P09_cr_interlayer_sliding_palette_updated",
    "CANVA-P10": "CANVA-P10_foundation_interlayer_sliding_palette_updated",
}

PALETTE = {
    "Foundation": "#8FC7AB",
    "Fine-tuning": "#29A7B6",
    "Scratch": "#327FB1",
    "DFT_reference": "#303840",
    "temperature": {"600 K": "open", "Multi-T": "filled"},
    "seed_marker": {"123": "circle", "234": "triangle", "345": "square"},
    "foundation_checkpoint": "diamond",
    "path_line": {"fixed": "solid", "self-relaxed": "dashed"},
    "foundation_checkpoint_shades": {
        "MACE-MP": "#2F6B4F",
        "MACE-OMAT-0": "#5F9E73",
        "MACE-MH": "#8FC7AB",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_vector(stem: Path) -> dict:
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    for path in (pdf_path, svg_path, png_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    doc = fitz.open(pdf_path)
    if len(doc) != 1:
        raise AssertionError(f"{pdf_path}: expected one page, got {len(doc)}")
    page = doc[0]
    pdf_images = len(page.get_images(full=True))
    pdf_drawings = len(page.get_drawings())
    pdf_text_blocks = len(page.get_text("blocks"))
    svg_text = svg_path.read_text(encoding="utf-8", errors="replace")
    svg_image_tags = len(re.findall(r"<image(?:\s|>)", svg_text))
    if pdf_images != 0 or svg_image_tags != 0:
        raise AssertionError(f"non-vector content in {stem}: pdf_images={pdf_images}, svg_image_tags={svg_image_tags}")
    return {
        "pdf_images": pdf_images,
        "pdf_drawings": pdf_drawings,
        "pdf_text_blocks": pdf_text_blocks,
        "svg_image_tags": svg_image_tags,
        "pdf_bytes": pdf_path.stat().st_size,
        "svg_bytes": svg_path.stat().st_size,
        "png_bytes": png_path.stat().st_size,
    }


def main() -> None:
    outputs = {}
    qa_rows = []
    for figure_id, rel_stem in FIGURE_FILES.items():
        stem = FIGURES / rel_stem
        vector_qa = inspect_vector(stem)
        formats = {}
        for ext in ("pdf", "svg", "png"):
            path = stem.with_suffix(f".{ext}")
            formats[ext] = {
                "path": path.relative_to(PACKAGE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        outputs[figure_id] = formats
        qa_rows.append((figure_id, rel_stem, *[vector_qa[key] for key in ("pdf_images", "pdf_drawings", "pdf_text_blocks", "svg_image_tags")]))

    renderer_hashes = {}
    for path in sorted((PACKAGE / "scripts").glob("render_*.py")):
        renderer_hashes[path.relative_to(PACKAGE).as_posix()] = sha256(path)
    manifest = {
        "bundle_id": "canva_palette_vector_bundle_20261105",
        "status": "REPRODUCED_P03_P10_WITH_LOCAL_SOURCE_ARCHIVE_AND_NATIVE_VECTOR_EXPORTS",
        "scope": list(FIGURE_FILES),
        "palette": PALETTE,
        "reproduction": {
            "entrypoint": "reproduce_all.py",
            "source_archive": "data/source_archive",
            "renderer_sha256": renderer_hashes,
            "source_hash_manifest": "manifests/SOURCE_FILE_HASHES.tsv",
            "source_provenance": "SOURCE_PROVENANCE.tsv",
            "source_pointers": "SOURCE_POINTERS.tsv",
            "failure_exclusion_registry": "FAILURE_EXCLUSION_REGISTRY.tsv",
        },
        "vector_qa": {
            "criterion": "PDF pages=1, embedded PDF images=0, SVG image tags=0",
            "figures": {figure_id: {key: value for key, value in zip(("pdf_images", "pdf_drawings", "pdf_text_blocks", "svg_image_tags"), row[1:])} for figure_id, *row in qa_rows},
        },
        "outputs": outputs,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    QA_TSV.parent.mkdir(parents=True, exist_ok=True)
    with QA_TSV.open("w", encoding="utf-8") as handle:
        handle.write("figure_id\tstem\tpdf_images\tpdf_drawings\tpdf_text_blocks\tsvg_image_tags\n")
        for row in qa_rows:
            handle.write("\t".join(map(str, row)) + "\n")
    print(json.dumps({"manifest": str(MANIFEST), "figures": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
