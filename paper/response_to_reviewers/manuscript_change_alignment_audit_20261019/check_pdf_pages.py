#!/usr/bin/env python3
"""Verify that response-callout and manuscript/SI anchors appear in the compiled PDFs.

The source-level audit (check_alignment.py) checks LaTeX anchors. This companion audit
checks the rendered PDFs in paper/revision1 so that the cited wording is present on an
actual page, and that no internal status marker is visible in any of the three documents.
"""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
REVISION = ROOT / "paper/revision1"

PDFS = {
    "response": REVISION / "response_to_reviewers.pdf",
    "main": REVISION / "sn-article-final.pdf",
    "si": REVISION / "sn-article-SI.pdf",
}

FORBIDDEN_MARKERS = [
    "[INVALID",
    "INVALID-pending",
    "PENDING-CLUSTER",
    "PENDING-data",
    "provisional",
    "validation gap",
    "this revision",
    "(pending)",
    "pending)",
    "not available in",
]


def page_text(path: Path) -> list[str]:
    raw = subprocess.check_output(["pdftotext", str(path), "-"])
    return [re.sub(r"\s+", " ", page.decode("utf-8", errors="replace")) for page in raw.split(b"\f")]


pages = {name: page_text(path) for name, path in PDFS.items()}

# label, PDF, rendered-text anchor
checks = [
    ("A1 response callout", "response", "belonged to a different path (82 atoms, two Cr, twelve images)"),
    ("A1 main", "main", "belonged to a different path (82 atoms, two Cr, twelve images)"),
    ("A1 SI", "si", "belonged to a different path (82 atoms, two Cr, twelve images)"),
    ("A2 main grouped split", "main", "The evaluation split assigns complete source trajectories"),
    ("A2 SI grouped split", "si", "The evaluation split assigns complete source trajectories"),
    ("A3 response callout", "response", "exceeds the seed-to-seed standard deviations"),
    ("A3 main", "main", "exceeds the seed-to-seed standard deviations"),
    ("A3 SI", "si", "Grouped retraining quantifies initialization uncertainty"),
    ("A4 response callout", "response", "This work demonstrates the framework on Cr-doped"),
    ("A4 main", "main", "This work demonstrates the framework on Cr-doped"),
    ("A5 response callout", "response", "not monotonic in training-set size"),
    ("A5 main", "main", "not monotonic in training-set size"),
    ("A5 SI", "si", "not monotonic in training-set size"),
    ("A6 main projection", "main", "The separation in the t-SNE projection"),
    ("A6 SI projection", "si", "The projection audit gives a t-SNE silhouette"),
    ("A7 main related-work delta", "main", "This manuscript extends our earlier non-archival AI4Mat workshop report"),
    ("C8 main provenance", "main", "The reported fine-tuned models are documented as being trained on AIMD frames"),
    ("C9 main protocol", "main", "The FT-600K MD run used a shorter trajectory"),
    ("C9 SI protocol", "si", "The FT-600K MD run used a shorter trajectory"),
    ("C10 response callout", "response", "The Methods split description and Table S2"),
    ("C11 main original-space metric", "main", "Because the raw descriptor vector mixes energy, force, RDF, and SOAP components"),
    ("C11 SI original-space metric", "si", "In the original descriptor space"),
    ("C12 SI feature matrix", "si", "The feature matrix excludes the earlier always-zero"),
    ("C12 main feature matrix", "main", "The feature matrix contains no force-sensitivity channel"),
]

failures = []
for label, pdf_name, anchor in checks:
    hits = [i + 1 for i, text in enumerate(pages[pdf_name]) if anchor in text]
    status = "PASS" if hits else "FAIL"
    print(f"{status}\t{label}\t{pdf_name}\tpages={hits}")
    if not hits:
        failures.append(f"{label}: anchor not found in {pdf_name}")

for pdf_name, texts in pages.items():
    for marker in FORBIDDEN_MARKERS:
        hits = [i + 1 for i, text in enumerate(texts) if marker.lower() in text.lower()]
        status = "FAIL" if hits else "PASS"
        print(f"{status}\tno '{marker}'\t{pdf_name}\tpages={hits}")
        if hits:
            failures.append(f"forbidden marker '{marker}' in {pdf_name} pages {hits}")

if failures:
    print("\nPDF alignment failures:")
    for failure in failures:
        print(f"- {failure}")
    sys.exit(1)

print(f"\nPASS: {len(checks)} anchor checks and {len(FORBIDDEN_MARKERS) * len(pages)} marker checks")
