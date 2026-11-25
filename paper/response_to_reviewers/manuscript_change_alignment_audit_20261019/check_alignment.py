#!/usr/bin/env python3
"""Check that reviewer-response manuscript-change anchors occur in the files cited.

This is a narrow text-alignment audit, not a substitute for PDF rendering or
scientific validation.  It intentionally checks the exact LaTeX anchors used
inside the blue response-letter callouts and the corresponding manuscript/SI
sections.
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[3]
RESPONSE = ROOT / "paper/revision1/response_to_reviewers.tex"
MAIN = ROOT / "paper/revision1/sn-article-final.tex"
SI = ROOT / "paper/revision1/sn-article-SI.tex"


def normalized(text: str) -> str:
    """Normalize whitespace and remove line comments for source-level matching."""
    text = re.sub(r"(?m)^[ \t]*%.*$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\\pm\s*", r"\\pm", text)
    return text.strip()


texts = {
    "response": normalized(RESPONSE.read_text(encoding="utf-8")),
    "main": normalized(MAIN.read_text(encoding="utf-8")),
    "si": normalized(SI.read_text(encoding="utf-8")),
}


checks = [
    # Each response change-box anchor must occur in the response letter and in the
    # manuscript/SI paragraph it summarizes.
    ("A1 response -> main withdrawn-record scope", "response", "belonged to a different path (82 atoms, two Cr, twelve images)", "main", "belonged to a different path (82 atoms, two Cr, twelve images)"),
    ("A1 response -> SI withdrawn-record scope", "response", "belonged to a different path (82 atoms, two Cr, twelve images)", "si", "belonged to a different path (82 atoms, two Cr, twelve images)"),
    ("A2 response -> main grouped split", "response", "This within-gap probe tests fixed-path energetic fidelity", "main", "The evaluation split assigns complete source trajectories to train, validation, or test"),
    ("A2 response -> SI grouped split", "response", "This within-gap probe tests fixed-path energetic fidelity", "si", "The evaluation split assigns complete source trajectories to train, validation, or test"),
    ("A3 response -> main seed statistics", "response", "The 0.73~eV difference between FT-600K and Scratch exceeds the seed-to-seed standard deviations", "main", "The 0.73~eV difference between FT-600K and Scratch exceeds the seed-to-seed standard deviations"),
    ("A3 response -> SI seed section", "response", "Seed-aware kinetic uncertainty", "si", "Seed-aware kinetic uncertainty"),
    ("A4 response -> main scope", "response", "This work demonstrates the framework on Cr-doped \\ce{Sb2Te3} with the MACE model family", "main", "This work demonstrates the framework on Cr-doped \\ce{Sb2Te3} with the MACE model family"),
    ("A5 response -> main low-data control", "response", "The from-scratch error is therefore not monotonic in training-set size on this path", "main", "The from-scratch error is therefore not monotonic in training-set size on this path"),
    ("A5 response -> SI low-data control", "response", "The from-scratch error is therefore not monotonic in training-set size on this path", "si", "The from-scratch error is not monotonic in training-set size on this path"),
    ("A6 response -> main projection scope", "response", "report the $0.36$--$0.42$ $t$-SNE range", "main", "The separation in the $t$-SNE projection"),
    ("A6 response -> SI projection scope", "response", "report the $0.36$--$0.42$ $t$-SNE range", "si", "The projection audit gives a $t$-SNE silhouette"),
    ("A7 response -> main related-work delta", "response", "carry the related-work comparison and the delta", "main", "This manuscript extends our earlier non-archival AI4Mat workshop report"),
    ("C8 response -> main provenance", "response", "The Methods section states that the fine-tuned models use AIMD frames with real DFT energies and forces and that NEB images remain evaluation data", "main", "The reported fine-tuned models are documented as being trained on AIMD frames with real DFT energies and forces"),
    ("C8 response -> SI provenance", "response", "The Methods section states that the fine-tuned models use AIMD frames with real DFT energies and forces and that NEB images remain evaluation data", "si", "The reported fine-tuned models are documented as being trained on AIMD frames with real DFT energies and forces"),
    ("C9 response -> main protocol", "response", "identify the FT-600K run as a shorter run with a different output interval", "main", "The FT-600K MD run used a shorter trajectory (2,000 steps, output every 5 steps)"),
    ("C9 response -> SI protocol", "response", "identify the FT-600K run as a shorter run with a different output interval", "si", "The FT-600K MD run used a shorter trajectory (2,000 steps, output every 5 steps)"),
    ("C10 response -> main split", "response", "The Methods split description and Table~S2 report the trajectory-grouped evaluation", "main", "The evaluation split assigns complete source trajectories"),
    ("C10 response -> SI split", "response", "The Methods split description and Table~S2 report the trajectory-grouped evaluation", "si", "The evaluation split assigns complete source trajectories"),
    ("C11 response -> main original-space metric", "response", "identify the original-space metric as the primary separation diagnostic", "main", "Because the raw descriptor vector mixes energy, force, RDF, and SOAP components"),
    ("C11 response -> SI original-space metric", "response", "identify the original-space metric as the primary separation diagnostic", "si", "In the original descriptor space, the overall Euclidean silhouette is 0.33"),
    ("C12 response -> SI feature matrix", "response", "remove the inactive feature from the feature matrix", "si", "The feature matrix excludes the earlier always-zero force-sensitivity placeholder"),
    ("C12 response -> main feature matrix", "response", "the main text states that the feature matrix has no force-sensitivity channel", "main", "The feature matrix contains no force-sensitivity channel"),
]


failures = []
for label, source_name, source_needle, target_name, target_needle in checks:
    source_ok = source_needle in texts[source_name]
    target_ok = target_needle in texts[target_name]
    status = "PASS" if source_ok and target_ok else "FAIL"
    print(f"{status}\t{label}")
    if not source_ok:
        failures.append(f"{label}: anchor missing from {source_name}")
    if not target_ok:
        failures.append(f"{label}: anchor missing from {target_name}")


if failures:
    print("\nAlignment failures:")
    for failure in failures:
        print(f"- {failure}")
    sys.exit(1)

print(f"\nPASS: {len(checks)} response/manuscript/SI alignment checks")
