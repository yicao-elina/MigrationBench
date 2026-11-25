#!/usr/bin/env python3
"""Write bounded SI/main-text handoff files from the canonical analysis tables."""

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
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analysis = Path(args.analysis)
    registry_path = Path(args.registry)
    figure_path = Path(args.figure)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(analysis / "canonical_model_stage_summary.tsv", sep="\t")
    groups = pd.read_csv(analysis / "canonical_candidate_group_counts.tsv", sep="\t")
    registry = pd.read_csv(registry_path, sep="\t", dtype=str).fillna("")
    manifest = json.loads((analysis / "CANONICAL_ANALYSIS_MANIFEST.json").read_text())
    n_candidates = manifest["n_candidates_expected"]
    n_models = manifest["n_models_expected"]
    n_rows = manifest["n_rows_observed"]
    n_success = manifest["n_success_rows"]
    families = ", ".join(sorted(registry["model_family"].unique()))
    status = manifest["status"]

    results_md = f"""# Canonical 639-candidate MLIP results and discussion

Analysis status: `{status}`. This handoff is generated from the accepted canonical screen and must not be upgraded beyond the release manifest.

## Scope

The screen contains {n_candidates} mechanism-focused candidate configurations evaluated against {n_models} accepted model entries, producing {n_rows} expected paired model/candidate rows and {n_success} finite successful rows. The model families are: {families}. The same candidate geometry is evaluated by every checkpoint, so force disagreement is paired at fixed configuration.

No geometry relaxation, MD continuation, transport calculation, or DFT calculation is performed by this screen. Energies are retained for provenance; the primary diagnostic is the force response and its stage-wise and within-family seed dispersion.

## Stage-wise summaries

The machine-readable table `canonical_model_stage_summary.tsv` contains model- and family-specific medians, q95 tails, maxima, and energy-per-atom medians. `canonical_within_family_seed_dispersion.tsv` reports only same-family seed dispersion and excludes foundation comparators and historical heterogeneous models.

## Interpretation boundary

Large paired force disagreement indicates model dependence on the same saved configuration. It is not a calibrated uncertainty, a DFT-correctness test, or proof that one model caused the MD runaway. The candidate set is event-driven and mechanism-focused, not an unbiased random test set. Any DFT-ready local fragment is labelled a `RECONSTRUCTION` and remains `PREPARED_ONLY` until QE convergence, input hashes, and parity checks pass.

## Reproducibility

The accepted model registry, candidate manifest, screen analysis manifest, Plot Atlas figure manifest, and figure book path/hash are the controlling evidence. Historical heterogeneous models remain diagnostic comparators and are excluded from formal same-family seed variance.
"""
    (output / "canonical_639_results_discussion.md").write_text(results_md)
    si_tex = f"""% Canonical 639-candidate SI methods handoff
\\section{{Canonical static MLIP screen}}
The accepted canonical screen evaluates {n_candidates} saved, fully periodic candidate configurations with {n_models} model entries. The same candidate geometry is paired across models. The result table has {n_rows} expected rows and {n_success} finite successful rows at the time of this handoff. No geometry relaxation, MD continuation, or DFT calculation is implied by the static evaluation.

The model families are recorded in the canonical registry; foundation checkpoints are comparator entries and are not treated as random-seed replicas. Within-family seed dispersion is computed only for the explicitly matched training-seed families. Historical heterogeneous checkpoints are retained as diagnostics and excluded from strict seed statistics.

All numerical status, model hashes, source split hashes, and rebuild commands are controlled by the machine-readable release manifest. The evidence is MLIP-only unless a separate QE record passes its own convergence and provenance gates.
"""
    (output / "canonical_639_si_methods.tex").write_text(si_tex)
    caption = """Canonical 639-candidate static MLIP diagnostic. Each saved periodic candidate was evaluated at fixed geometry by the accepted model matrix. The panels summarize stage-wise force response, family-level distributions with seed medians, descriptor-space coverage with a local inset, and candidate novelty versus paired model force range. This is MLIP-only evidence; force disagreement is not calibrated uncertainty or DFT correctness, and the candidate set is mechanism-focused rather than an unbiased random test set."""
    (output / "canonical_639_caption.tex").write_text(caption + "\n")
    handoff = {
        "schema": "migrationbench_canonical_639_handoff_v1",
        "status": status,
        "analysis_manifest_sha256": sha256(analysis / "CANONICAL_ANALYSIS_MANIFEST.json"),
        "registry_sha256": sha256(registry_path),
        "figure_manifest_sha256": sha256(figure_path),
        "outputs": {},
        "interpretation_boundary": "MLIP-only static evidence; not DFT correctness or causal proof",
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "HANDOFF_MANIFEST.json":
            handoff["outputs"][path.name] = sha256(path)
    (output / "HANDOFF_MANIFEST.json").write_text(json.dumps(handoff, indent=2) + "\n")
    print(json.dumps({"status": status, "outputs": sorted(handoff["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
