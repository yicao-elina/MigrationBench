#!/usr/bin/env python3
"""Create the reader-facing, evidence-matched sliding figure derivative.

The approved source is a vector PDF whose two panel titles and first legend
entry are baked into the artwork. This script overlays small white panels and
neutral text while preserving the original curves, markers, axes, and layout.
It does not alter numerical data or imply convergence or training-seed
validation.
"""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import black, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures" / "fig_sliding_revised.pdf"
OUTPUT = ROOT / "figures" / "fig_sliding_neutralized.pdf"
PAGE_W, PAGE_H = 750.0, 439.5


def make_overlay() -> bytes:
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=(PAGE_W, PAGE_H))
    c.setFillColor(white)
    c.setStrokeColor(white)
    # The source titles are immediately above the two lower axes.
    c.rect(62, 210, 305, 28, fill=1, stroke=0)
    c.rect(430, 210, 305, 28, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica", 11)
    c.drawCentredString(214, 217, "Fixed-path relative-energy profile")
    c.drawCentredString(582, 217, "Fixed-path relative-energy profile")

    # Replace only the baked-in first legend label; the marker and all other
    # legend entries remain from the approved source artwork.
    c.setFillColor(white)
    c.rect(284, 202, 104, 14, fill=1, stroke=0)
    c.rect(463, 202, 104, 14, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica", 8.5)
    c.drawString(287, 205, "DFT reference")
    c.drawString(466, 205, "DFT reference")
    c.save()
    return stream.getvalue()


def main() -> None:
    reader = PdfReader(str(SOURCE))
    overlay = PdfReader(BytesIO(make_overlay()))
    page = reader.pages[0]
    page.merge_page(overlay.pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    with OUTPUT.open("wb") as handle:
        writer.write(handle)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
