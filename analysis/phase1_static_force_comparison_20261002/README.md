# Phase 1 artifact: static same-configuration force comparison

This directory is an isolated, additive artifact. It contains the exact common
inputs, source/model provenance, 16 Rockfish single-point outputs, analysis CSVs,
and Plot Atlas-style publication figures.

Canonical model display names used consistently in figures, tables, and prose:

| Stable model ID | Canonical display name | Provenance meaning |
|---|---|---|
| `from_scratch` | Scratch | model trained from scratch |
| `naive_ft` | FT-600K | single-temperature 600 K fine-tuning |
| `multi_t` | FT-MultiT | multi-temperature fine-tuning |
| `foundation` | Foundation | OMAT foundation checkpoint |

The stable IDs remain unchanged in machine-readable records; only human-facing
labels use the canonical names above.

Main entry points:

- `inputs/snapshot_manifest.json`: exact step selection, species/cell/PBC/key-atom checks.
- `results/evaluation_summary.json`: 16/16 completion gate.
- `analysis/`: force, stress, key-atom, local-disagreement, and geometry tables.
- `figures/`: PNG, PDF, and SVG figures.
- `Methods.md` and `Results.md`: paper-ready text with evidence boundaries.
- `scripts/`: extraction, evaluation, analysis, plotting, inventory, and Slurm scripts.

`results_local_diagnostic/` is retained separately as environment evidence: the
local macOS runtime completed only the Foundation model and recorded CUDA-backend
serialization failures for the three Cr-specific models. It is not substituted
for the formal Rockfish result.

The source trajectory and model weights remain on Rockfish; this artifact stores
their paths and hashes plus the processed static outputs, not a duplicate of the
multi-gigabyte continuation trajectory.
