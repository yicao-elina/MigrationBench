# Independent review: Figures 1–4

Review mode: read-only source/table audit plus rendered-output inspection.
Review date: 2026-10-31.

## Figure 1

Result: `PARTIAL`.

The comprehensive map has 2,724 rows and the structure-universe table has 2,652
rows; the current plot is therefore based on real local data. The current
metadata do not yet provide a canonical geometry key with a complete
trajectory-group/task-family/dopant/site join. The 72 thermal points have
temperature metadata, but that does not establish the full benchmark task
matrix. The schematic is visually usable but is not evidence for the counts.

Action before publication: freeze the canonical IDs, task joins, embedding
representation/normalization, and deterministic seed; re-run count checks.

## Figure 2

Result: `PROVISIONAL` for parity/seed panels; `PARTIAL` for distribution shift.

The parity package contains 24 summary rows and paired per-frame/NPZ sources.
The direct-evaluation lineage is traceable and the force/RMSE plots are
recomputable. The source energy field requires an explicit unit audit before a
final energy-axis claim. Cross-temperature/task stratification is not fully
joined, so the distribution-shift panel cannot support a generalization claim.

Action before publication: freeze units, denominators, split, seed, held-out
trajectory group, and metadata join; recompute summaries from raw rows.

## Figure 3

Result: `G1_WARNING / DIAGNOSTIC`.

The three fixed paths and ten-image curves are plot-ready from the local TSVs,
but the QE rows are `COMPLETED_BUT_SCF_WARNING`. The current Scratch line is a
legacy/endpoint-only diagnostic and must not be promoted to accepted current
Scratch evidence. The metric labels corresponding to 0.47, 0.68, and 1.20 are
not defined by the current source contract.

Action before publication: obtain a converged/reference-qualified QE record,
freeze the metric definition, and replace the legacy line or keep it
diagnostic-only.

## Figure 4

Result: `DIAGNOSTIC`.

Same-path profiles support a visual transfer diagnostic and a paired residual
plot. The paired `Scratch - FT` sign convention, seed pairing, and environment
labels must be frozen before the green/negative-gain annotation becomes a
claim. The QE warning remains applicable.

Action before publication: publish the paired error table with image keys,
model keys, metric formula, and source status; then independently recompute.
