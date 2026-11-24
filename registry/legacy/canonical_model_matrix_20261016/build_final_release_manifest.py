#!/usr/bin/env python3
"""Assemble the final evidence-bounded release status without upgrading claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--screen-analysis", required=True)
    parser.add_argument("--figure-manifest", required=True)
    parser.add_argument("--figure-book", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dft-manifest")
    parser.add_argument("--handoff-root")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    acceptance_path = Path(args.acceptance)
    analysis_path = Path(args.screen_analysis)
    figure_path = Path(args.figure_manifest)
    figure_book_path = Path(args.figure_book)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(registry_path, sep="\t", dtype=str).fillna("")
    acceptance = pd.read_csv(acceptance_path, sep="\t", dtype=str).fillna("")
    screen = json.loads((analysis_path / "CANONICAL_ANALYSIS_MANIFEST.json").read_text())
    figure = json.loads(figure_path.read_text())
    non_foundation = registry[~registry["model_family"].eq("foundation_zero_shot")]
    computational = "COMPLETE" if len(non_foundation) and non_foundation["computational_status"].isin({"COMPLETED_ACCEPTED"}).all() else "INCOMPLETE"
    scientific = "COMPLETE" if screen.get("status") == "VERIFIED" and screen.get("n_nonfinite_or_failed_rows") == 0 else "INCOMPLETE"
    provenance = "COMPLETE" if not registry["provenance_status"].str.contains("PROVISIONAL|UNVERIFIED|FAIL", case=False, regex=True).any() else "PARTIAL_PROVENANCE"
    handoff_root = Path(args.handoff_root) if args.handoff_root else None
    required_handoff = (
        handoff_root
        and handoff_root.is_dir()
        and all(
            (handoff_root / name).is_file()
            for name in (
                "HANDOFF_MANIFEST.json",
                "canonical_639_results_discussion.md",
                "canonical_639_si_methods.tex",
                "canonical_639_caption.tex",
            )
        )
    )
    figure_outputs_exist = all(
        (figure_path.parent / name).is_file()
        for name in figure.get("outputs", {})
    )
    publication = "COMPLETE" if figure.get("status") == "RECONSTRUCTION" and figure_outputs_exist and figure_book_path.is_file() and required_handoff else "INCOMPLETE"
    dft_status = "NOT_PROVIDED"
    if args.dft_manifest:
        dft = json.loads(Path(args.dft_manifest).read_text())
        dft_status = str(dft.get("status", "UNKNOWN"))
    release_status = "COMPLETE" if all(status == "COMPLETE" for status in [computational, scientific, provenance, publication]) else "INCOMPLETE"
    payload = {
        "schema": "migrationbench_canonical_release_manifest_v1",
        "campaign_id": "canonical_model_matrix_20261016",
        "release_status": release_status,
        "status_dimensions": {
            "computational": computational,
            "scientific": scientific,
            "provenance": provenance,
            "publication": publication,
            "dft": dft_status,
        },
        "inputs": {
            "model_registry": {"path": str(registry_path), "sha256": sha256(registry_path)},
            "acceptance_matrix": {"path": str(acceptance_path), "sha256": sha256(acceptance_path)},
            "screen_analysis": {"path": str(analysis_path / "CANONICAL_ANALYSIS_MANIFEST.json"), "sha256": sha256(analysis_path / "CANONICAL_ANALYSIS_MANIFEST.json")},
            "figure_manifest": {"path": str(figure_path), "sha256": sha256(figure_path)},
            "figure_book": {"path": str(figure_book_path), "sha256": sha256(figure_book_path)},
            "dft_manifest": str(args.dft_manifest) if args.dft_manifest else None,
            "handoff_root": str(handoff_root) if handoff_root else None,
        },
        "model_count": len(registry),
        "accepted_model_count": int((registry["computational_status"] == "COMPLETED_ACCEPTED").sum()),
        "registered_comparator_count": int((registry["computational_status"] == "REGISTERED").sum()),
        "held_model_count": int((registry["acceptance_decision"] == "HELD").sum()) if "acceptance_decision" in registry else None,
        "interpretation_boundary": "MLIP static evidence and any DFT results are reported separately; no MLIP disagreement is treated as DFT correctness or causal proof.",
    }
    (output / "FINAL_RELEASE_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n")
    claims = pd.DataFrame(
        [
            {"claim_id": "CANON-COMP-001", "claim": "canonical model files and static screen are computationally complete", "status": computational, "evidence": str(registry_path) + ";" + str(analysis_path / "CANONICAL_ANALYSIS_MANIFEST.json")},
            {"claim_id": "CANON-SCI-001", "claim": "paired static MLIP stage summaries and within-family seed dispersion are reproducible", "status": scientific, "evidence": str(analysis_path)},
            {"claim_id": "CANON-PROV-001", "claim": "model/data lineage is publication-ready without provisional boundaries", "status": provenance, "evidence": str(registry_path)},
            {"claim_id": "CANON-PUB-001", "claim": "figure and manuscript handoff are rebuilt from accepted canonical data", "status": publication, "evidence": str(figure_path) + ";" + str(handoff_root) if handoff_root else str(figure_path)},
            {"claim_id": "CANON-DFT-001", "claim": "DFT-ready local fragments are prepared reconstructions, not exact full-cell parity", "status": dft_status, "evidence": str(args.dft_manifest) if args.dft_manifest else "none"},
        ]
    )
    claims.to_csv(output / "CLAIM_EVIDENCE_MATRIX.tsv", sep="\t", index=False)
    print(json.dumps({"release_status": release_status, "status_dimensions": payload["status_dimensions"]}, indent=2))


if __name__ == "__main__":
    main()
