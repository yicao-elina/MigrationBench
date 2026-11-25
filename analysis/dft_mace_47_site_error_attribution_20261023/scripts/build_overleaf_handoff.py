#!/usr/bin/env python3
"""Build deterministic, manuscript-facing handoff manifests for this package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "paper_handoff"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = [
        ("H1", "figures/FIG1_ENDPOINT47_LATTICE_SITE_DISTRIBUTION.pdf", "VERIFIED", "Main Results", "Discovery-set distribution; cite before grouped-split/model comparisons", "processed/DFT_ENDPOINT_47_CLASSIFICATION.tsv", "Defines the denominator for all later endpoint/model comparisons", "Use geometric coverage language; no stability labels"),
        ("H2", "figures/FIG2_RETRY21_DFT_ENERGY_STRUCTURE.pdf", "LINEAGE_SUPPORTED", "Main Results or SI", "Restricted retry-subset energy context", "analysis/discovery_dft_audit_20261016/DISCOVERY_PROGRESS_AUDIT_20261018.md", "Connects DFT endpoint audit to endpoint/NEB discussion", "21-endpoint retry subset only; 077/078/093 excluded"),
        ("H3", "figures/FIG3_FOUR_MODEL_BASIN_SENSITIVITY.pdf", "VERIFIED", "Model diagnostics or SI", "Legacy four-model local-basin sensitivity", "processed/MACE_FOUR_MODEL_RECORDS.tsv", "Motivates seed/model ensemble interpretation", "Not DFT error, barrier, or transport validation"),
        ("H4", "figures/FIG4_MACE12_SOURCE_GEOMETRY_DISAGREEMENT.pdf", "LINEAGE_SUPPORTED", "Model robustness and SI", "12-model source-geometry disagreement", "processed/MACE12_SOURCE_GEOMETRY_ASSOCIATIONS.tsv", "Feeds uncertainty-aware interpretation of downstream NEB/MD", "Exploratory model spread, not calibrated uncertainty"),
        ("T1", "tables/TABLE1_DFT47_GEOMETRY_ENERGY_AND_ENSEMBLE_SUMMARY.tsv", "LINEAGE_SUPPORTED", "Main Results/SI", "DFT geometry, restricted energy, and ensemble summary", "processed/DFT_MACE47_INTEGRATED_ENDPOINT_TABLE.tsv", "Single numeric table backing FIG1/FIG2", "Keep failed/nonconverged rows visible"),
        ("T2", "tables/TABLE2_LEGACY_FOUR_MODEL_SENSITIVITY.tsv", "VERIFIED", "SI", "Four-model model-internal sensitivity", "processed/MACE_FOUR_MODEL_ENDPOINT_MODEL_SUMMARY.tsv", "Supports model-selection and robustness narrative", "No absolute accuracy claim"),
        ("T3", "tables/TABLE3_MACE12_STRUCTURAL_ASSOCIATIONS.tsv", "LINEAGE_SUPPORTED", "Main Results/SI", "Structural correlates of model disagreement", "processed/MACE12_SOURCE_GEOMETRY_ASSOCIATIONS.tsv", "Supports feature-linked disagreement discussion", "BH-corrected exploratory associations"),
        ("D1", "processed/DFT_MACE_ERROR_ELIGIBILITY.tsv", "EXCLUDE_FROM_PUBLICATION", "SI audit/limitations", "DFT–MACE error eligibility ledger", "DFT final XYZ archive status", "Defines the next reproducibility extension", "Do not subtract unmatched total energies"),
        ("R1", "NATURE_READY_RESULTS_AND_DISCUSSION_20261023.md", "LINEAGE_SUPPORTED", "Internal writing source", "Results/discussion language and caveats", "All package manifests and audits", "Seed text for manuscript revision", "Copy only after claim matrix review"),
    ]
    out_tsv = HANDOFF / "OVERLEAF_HANDOFF_MANIFEST.tsv"
    with out_tsv.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["artifact_id", "source_path", "sha256", "status", "suggested_location", "manuscript_role", "upstream_evidence", "downstream_connection", "handoff_caveat"])
        for artifact_id, rel, status, location, role, upstream, downstream, caveat in rows:
            path = ROOT / rel
            writer.writerow([artifact_id, rel, sha256(path), status, location, role, upstream, downstream, caveat])

    release = {
        "package": "dft_mace_47_site_error_attribution_20261023",
        "bundle_commit": "TO_BE_FILLED_AFTER_FIRST_BUNDLE_COMMIT",
        "overleaf_version": "PENDING_USER_HANDOFF",
        "release_status": "READY_FOR_REVIEW_NOT_YET_MERGED",
        "repository": "https://github.com/yicao-elina/MigrationBench.git",
        "handoff_manifest": "paper_handoff/OVERLEAF_HANDOFF_MANIFEST.tsv",
        "claim_matrix": "paper_handoff/PAPER_CLAIM_EVIDENCE_MATRIX.tsv",
        "rebuild_commands": [
            "python scripts/build_attribution_package.py",
            "python scripts/integrate_source_geometry_ensemble.py",
            "python scripts/build_publication_tables.py",
            "python scripts/render_publication_figures.py",
        ],
        "required_overleaf_checks": [
            "Compile from a clean checkout",
            "Visual-inspect every figure panel at journal scale",
            "Compare rendered table values with TSV sources",
            "Record the Overleaf Git commit/version beside bundle_commit",
        ],
        "scientific_boundaries": [
            "No stable/metastable/unstable labels",
            "No migration barrier or transport claim",
            "No calibrated MACE-DFT absolute error until matched DFT final XYZ is archived",
            "Foundation label means OMAT-derived fine-tuned legacy Foundation",
        ],
        "key_artifact_hashes": {rel: sha256(ROOT / rel) for _, rel, *_ in rows},
    }
    (HANDOFF / "PAPER_RELEASE_MANIFEST.json").write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
