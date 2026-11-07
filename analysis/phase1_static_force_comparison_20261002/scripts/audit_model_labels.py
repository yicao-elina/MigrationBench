#!/usr/bin/env python3
"""Acceptance audit for Phase 1 canonical model display labels."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


CANONICAL = {"Scratch", "FT-600K", "FT-MultiT", "Foundation"}
LEGACY = ("from-scratch", "naive-FT", "multi-T FT", "Foundation-OMAT")
MODEL_IDS = {"from_scratch", "naive_ft", "multi_t", "foundation"}


def json_records(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else [value]


def collect_model_labels(root: Path):
    labels = []
    for path in sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonl")):
        for record in json_records(path):
            if isinstance(record, dict) and "model_label" in record:
                labels.append({"file": str(path.relative_to(root)), "model_id": record.get("model_id"), "label": record["model_label"]})
    return labels


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = []
    labels = collect_model_labels(root)
    observed_labels = {item["label"] for item in labels}
    if not observed_labels <= CANONICAL:
        errors.append(f"non-canonical JSON/JSONL model labels: {sorted(observed_labels - CANONICAL)}")
    if not {item["model_id"] for item in labels if item["model_id"]} <= MODEL_IDS:
        errors.append("unexpected machine model ID in model-labelled records")

    display_files = [root / "README.md", root / "Methods.md", root / "Results.md", root / "scripts" / "plot_phase1.py"]
    for path in display_files:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            for legacy in LEGACY:
                if legacy not in line:
                    continue
                # The exact Rockfish path is retained as provenance; it is not a display label.
                provenance_path = legacy == "from-scratch" and ("/1-from-scratch/" in line or "source_path" in line)
                if not provenance_path:
                    errors.append(f"legacy display label remains in {path.name}:{line_number}: {legacy}")

    plot_script = (root / "scripts" / "plot_phase1.py").read_text()
    for label in CANONICAL:
        if f'"{label}"' not in plot_script:
            errors.append(f"canonical label missing from plotting source: {label}")

    figure_checks = {}
    for path in sorted((root / "figures").glob("*.svg")):
        text = path.read_text()
        legacy_hits = [legacy for legacy in LEGACY if legacy in text]
        figure_checks[str(path.relative_to(root))] = {"legacy_labels": legacy_hits, "passed": not legacy_hits}
        if legacy_hits:
            errors.append(f"legacy label remains in SVG: {path.name}")

    pdf_checks = {}
    for path in sorted((root / "figures").glob("*.pdf")):
        result = subprocess.run(["pdftotext", str(path), "-"], check=True, capture_output=True, text=True)
        legacy_hits = [legacy for legacy in LEGACY if legacy in result.stdout]
        pdf_checks[str(path.relative_to(root))] = {"legacy_labels": legacy_hits, "passed": not legacy_hits}
        if legacy_hits:
            errors.append(f"legacy label remains in PDF text: {path.name}")

    report = {
        "schema_version": "phase1-model-label-acceptance-1.0",
        "canonical_display_names": ["Scratch", "FT-600K", "FT-MultiT", "Foundation"],
        "stable_machine_ids": sorted(MODEL_IDS),
        "json_label_values": sorted(observed_labels),
        "json_label_records_checked": len(labels),
        "svg_checks": figure_checks,
        "pdf_checks": pdf_checks,
        "legacy_labels_checked": list(LEGACY),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    output = root / "logs" / "model_label_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
