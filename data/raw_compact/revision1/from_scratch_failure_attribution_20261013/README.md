# Compact provenance capsule: 639 from-scratch candidates

This directory contains the compact, publication-audit evidence for the
639-candidate three-model MLIP screen and its two distinct 20-point selection
layers. The CSV/TSV/JSON files are frozen copies of the campaign manifests;
their SHA256 values are recorded in
`registry/from_scratch_mlip_candidate_selection_20261013.json`.

The repository intentionally excludes the 2050-atom extxyz files, model
checkpoints, QE outdirs, wavefunctions, and full Slurm result trees. The
compact ledgers preserve their approved external paths, model identities,
source-frame hashes, execution-to-canonical mapping, and scientific status.
The absence of those heavyweight files from Git is a storage boundary, not a
claim that the underlying data are absent.

The repeatability audit is retained in `run_summary.csv`,
`initial_state_audit.csv`, `diagnostic_manifest.json` and
`failure_attribution_results_discussion.md`. The manuscript-facing composite
is regenerated from the compact MLIP tables by
`paper/revision1/scripts/render_from_scratch_failure_attribution.py`; its
output manifest is stored beside the rendered figure.

Run from the repository root:

```bash
python scripts/validate_from_scratch_mlip_candidate_selection.py
```

The validator checks the 639-row inventory, five category counts, the original
20-row `wave_01` coverage method, the later 20-row MLIP-informed DFT subset,
the 639-row execution reconciliation, and all three model hashes.
