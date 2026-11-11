#!/usr/bin/env python3
"""Move superseded figure products into an explicit, recoverable archive."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


TRIPTYCH_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = Path(__file__).resolve().parents[1] / "archive/legacy_figure_products_20261024"

FIGURE_DIRS = [
    "Cr_fixed_path",
    "Cr_four_model_neb",
    "Cu_fixed_path",
    "Li_fixed_path",
    "Na_fixed_path",
    "foundation_class_3x3",
    "foundation_class_3x3_R02",
    "foundation_class_3x3_R03",
    "foundation_class_4x3_R01",
    "foundation_class_4x3_R02",
    "foundation_class_4x3_R03",
]
DATA_DIRS = [
    "foundation_class_3x3_R02",
    "foundation_class_3x3_R03",
    "foundation_class_4x3_R01",
    "foundation_class_4x3_R02",
    "foundation_class_4x3_R03",
]
TOP_LEVEL_FILES = [
    "CAPTIONS.md",
    "CAPTIONS_foundation_class_3x3_R01.md",
    "CAPTIONS_foundation_class_3x3_R02.md",
    "CAPTIONS_foundation_class_3x3_R03.md",
    "CAPTIONS_foundation_class_4x3_R01.md",
    "CAPTIONS_foundation_class_4x3_R02.md",
    "CAPTIONS_foundation_class_4x3_R03.md",
    "DELIVERY_INDEX.md",
    "DELIVERY_INDEX_foundation_class_3x3_R01.md",
    "DELIVERY_INDEX_foundation_class_3x3_R02.md",
    "DELIVERY_INDEX_foundation_class_3x3_R03.md",
    "DELIVERY_INDEX_foundation_class_4x3_R01.md",
    "DELIVERY_INDEX_foundation_class_4x3_R02.md",
    "DELIVERY_INDEX_foundation_class_4x3_R03.md",
    "EVIDENCE_PACKAGE_foundation_class_4x3_R03.md",
    "FOUNDATION_CLASS_STYLE.json",
    "QA_REPORT_20261024.json",
    "QA_REPORT_foundation_class_3x3_R01.json",
    "QA_REPORT_foundation_class_3x3_R02.json",
    "QA_REPORT_foundation_class_3x3_R03.json",
    "QA_REPORT_foundation_class_4x3_R01.json",
    "QA_REPORT_foundation_class_4x3_R02.json",
    "QA_REPORT_foundation_class_4x3_R03.json",
    "figure_data_manifest_20261024.json",
    "figure_data_manifest_foundation_class_3x3_R01.json",
    "figure_data_manifest_foundation_class_3x3_R02.json",
    "figure_data_manifest_foundation_class_3x3_R03.json",
    "figure_data_manifest_foundation_class_4x3_R01.json",
    "figure_data_manifest_foundation_class_4x3_R02.json",
    "figure_data_manifest_foundation_class_4x3_R03.json",
    "build_cr_mh1_fixed_path_r01.py",
    "build_foundation_class_3x3_r01.py",
    "build_foundation_class_4x3_r01.py",
    "build_foundation_class_4x3_r02.py",
    "build_foundation_class_4x3_r03.py",
    "build_neb_cross_element_triptychs_r01.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> list[dict[str, object]]:
    if path.is_file():
        return [{"relative_path": path.name, "sha256": None, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}]
    items = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        items.append({"relative_path": str(file_path.relative_to(path)), "sha256": None, "bytes": file_path.stat().st_size, "mtime_ns": file_path.stat().st_mtime_ns})
    return items


def main() -> None:
    sources = [TRIPTYCH_ROOT / "figures" / name for name in FIGURE_DIRS]
    sources += [TRIPTYCH_ROOT / "data" / name for name in DATA_DIRS]
    sources += [TRIPTYCH_ROOT / name for name in TOP_LEVEL_FILES]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise SystemExit("archive scope changed; missing explicit targets:\n" + "\n".join(missing))
    if ARCHIVE_ROOT.exists() and any(path.is_file() for path in ARCHIVE_ROOT.rglob("*")):
        raise SystemExit(f"refusing to merge into non-empty archive: {ARCHIVE_ROOT}")
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sources:
        relative = source.relative_to(TRIPTYCH_ROOT)
        destination = ARCHIVE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = inventory(source)
        shutil.move(str(source), str(destination))
        records.append({"original": str(relative), "archived": str(destination.relative_to(ARCHIVE_ROOT)), "inventory": before})
    manifest = {
        "schema_version": "legacy_figure_archive_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "superseded figure products, derived data, QA/manifests, captions, delivery indexes, and legacy builders",
        "raw_inputs_models_untouched": True,
        "recoverable_archive_root": str(ARCHIVE_ROOT),
        "items": records,
    }
    (ARCHIVE_ROOT / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archived_items": len(records), "archive_root": str(ARCHIVE_ROOT)}, indent=2))


if __name__ == "__main__":
    main()

