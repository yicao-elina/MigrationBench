#!/usr/bin/env python3
"""Replace legacy three-decimal labels in checked-in publication PDFs.

The underlying curves and audited data are preserved.  This small overlay
keeps the historical figure geometry while standardising visible result
labels to two decimal places.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import white, black


ROOT = Path(__file__).resolve().parents[2]


def overlay_labels(
    source: Path,
    replacements: list[tuple[float, float, float, float, str, float]],
    legacy_literals: list[bytes],
) -> None:
    """Paint over old text boxes and draw centered replacements.

    Coordinates are PDF points measured from the upper-left corner by
    ``pdftotext -bbox``; the helper converts them to ReportLab's origin.
    """

    reader = PdfReader(str(source))
    if len(reader.pages) != 1:
        raise ValueError(f"expected one-page figure: {source}")
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    tmp = source.with_suffix(".rounded.tmp.pdf")
    canvas = Canvas(str(tmp), pagesize=(width, height))
    for x0, y0, x1, y1, text, font_size in replacements:
        pad_x, pad_y = 1.5, 1.0
        y_bottom = height - y1
        box_w = x1 - x0
        box_h = y1 - y0
        canvas.setFillColor(white)
        canvas.rect(x0 - pad_x, y_bottom - pad_y, box_w + 2 * pad_x, box_h + 2 * pad_y,
                    stroke=0, fill=1)
        canvas.setFillColor(black)
        canvas.setFont("Helvetica", font_size)
        canvas.drawCentredString((x0 + x1) / 2, y_bottom + max(0.0, (box_h - font_size) / 2), text)
    canvas.save()

    overlay = PdfReader(str(tmp)).pages[0]
    writer = PdfWriter()
    page.merge_page(overlay)
    # Remove the old text operators as well as covering them visually.  This
    # keeps text extraction and downstream PDF audits consistent with the
    # visible labels.
    content = page.get_contents().get_data()
    for literal in legacy_literals:
        content = content.replace(literal, b"()")
    writer.add_page(page)
    cleaned = DecodedStreamObject()
    cleaned.set_data(content)
    writer.pages[0].replace_contents(cleaned)
    with source.open("wb") as stream:
        writer.write(stream)
    tmp.unlink()


def main() -> None:
    overlay_labels(
        ROOT / "figures/figS_dft_neb_mep_profiles.pdf",
        [
            (150.309379, 30.201585, 186.522692, 45.346585, "1.76", 11),
            (399.508598, 30.420335, 436.721910, 45.565335, "1.04", 11),
            (640.180473, 30.420335, 677.393785, 45.565335, "1.56", 11),
            (158.836723, 271.615407, 196.050035, 286.760407, "1.69", 11),
            (399.508598, 271.615407, 436.721910, 286.760407, "2.06", 11),
            (639.680473, 271.615407, 676.893785, 286.760407, "1.42", 11),
        ],
        [b"(1\\05676)", b"(1\\05604)", b"(1\\05656)", b"(1\\05669)", b"(2\\05606)", b"(1\\05642)"],
    )
    overlay_labels(
        ROOT / "figures/figS_latent_robustness.pdf",
        [
            (128.025048, 777.147361, 153.045048, 792.668928, "0.33", 11),
            (230.217471, 789.754259, 255.237471, 805.275825, "0.28", 11),
            (434.602317, 742.255913, 459.622317, 757.777480, "0.49", 11),
            (330.745832, 872.145816, 359.095832, 887.667382, "-0.02", 11),
            (535.130678, 873.830708, 563.480678, 889.352275, "-0.03", 11),
            (623.729351, 756.335053, 679.259351, 771.856619, "0.40±0.03", 11),
            (725.921774, 745.170776, 781.451774, 760.692343, "0.47±0.00", 11),
            (843.372009, 759.027751, 868.392009, 774.549318, "0.41", 11),
        ],
        [b"(0.333)", b"(0.277)", b"(0.486)", b"(-0.019)", b"(-0.026)", b"(0.397\xb10.027)", b"(0.473\xb10.000)", b"(0.412)"],
    )
    overlay_labels(
        ROOT / "figures/figS_neb_pes_local_audit.pdf",
        [(672.567544, 179.160780, 695.085544, 193.130190, "0.34", 10)],
        [b"0.336 eV"],
    )


if __name__ == "__main__":
    main()
