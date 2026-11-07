# Independent reviewer brief

## Purpose

Audit the Atomic Migration blueprint figures using the local evidence catalog
and the acceptance criteria. The reviewer is independent of the plotting pass:
read the source tables and manifests first, then inspect the rendered outputs.
Do not infer missing evidence from a polished Canva page.

## Inputs

1. `BLUEPRINT_FIGURE_TODO.md`
2. `DATA_AND_VISUAL_ACCEPTANCE_CRITERIA.md`
3. `CANVA_FIGURE_REGISTRY.tsv` and `DOWNLOAD_EVIDENCE.tsv`
4. `manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json`
5. all individual PNG/PDF/SVG files under `figures/`
6. the exact absolute source paths recorded in the TODO and manifest

## Review procedure

For each panel:

1. Confirm source path, SHA256, schema, expected row/image count, model/seed
   identity, and scientific status.
2. Recompute at least one numeric check from the source table: row count,
   finite-value count, RMSE/MAE, profile endpoint, observed MD horizon, or
   class sample count as appropriate.
3. Compare fixed-image versus generated/self-relaxed data and ensure they are
   not pooled.
4. Inspect PNG at native size and PDF/SVG for clipping, empty content, font
   substitution, legend overlap, misleading colors, and warning visibility.
5. Record `PASS`, `FAIL`, `PROVISIONAL`, `DIAGNOSTIC`, or `BLOCKED` with exact
   evidence. A red placeholder is a correct outcome when the input gate is
   missing.

## Mandatory reviewer questions

- Figure 1: Is every point a real configuration with a documented task and
  embedding provenance? Is any OOD/error claim being inferred from PCA alone?
- Figure 2: Are units, per-atom denominators, split, seeds, and held-out groups
  frozen and reproducible? Does the frame panel use actual metadata?
- Figure 3: Are the ten fixed images paired exactly? Is the G1 warning visible?
  Is the legacy Scratch line kept out of a publication acceptance claim?
- Figure 4: Is `Scratch error - FT error` a paired calculation on the same images
  with a frozen sign convention?
- Figure 5: Do all generated paths have ten finite images and convergence logs?
  If not, is the output clearly blocked and free of barrier/success claims?
- Figure 6: Do curves stop at recorded horizons? Are failed runs excluded from
  diffusion ranking and 1 ns completion language?
- Figure 7: Are fixed/self paths separate? Are QE SCF warnings and model-identity
  SD labels visible? Are endpoint-only proxies identified?
- Figure 8: Is there a validated geometry-keyed error join and held-out
  selection/calibration? If not, are B–D placeholders retained?
- Tables 1–2: Can every status be traced to a figure/source row without mixing
  Foundation, Scratch, legacy, proxy, or QE-warning lineages?

## Required report format

Write one row per panel:

```text
figure | panel | data result | visual result | evidence path | blocker/action | reviewer/date
```

Use `PUBLICATION_PASS` only when all applicable gates pass and no open warning,
proxy, legacy, endpoint-only, or unit ambiguity conflicts with the intended
claim. Otherwise retain the most conservative state.

## Current independent audit summary

The first independent read-only review found:

- Figure 1: partial; configuration counts are real, but canonical geometry/task
  joins are incomplete.
- Figure 2: parity and seed panels are data-backed/provisional; distribution
  shift and source-unit decisions remain open.
- Figure 3: G1 warning/legacy diagnostic; publication barrier claim blocked.
- Figure 4: transfer diagnostic; paired error metric and seed contract open.
- Figure 5: blocked; current campaign is endpoint-only/precanonical and stopped
  for repair.
- Figure 6: observed-horizon diagnostic; only 4/15 current rows reach 1,000 ps.
- Figure 7: SCF-warning/proxy diagnostic; fixed and generated paths are retained
  separately.
- Figure 8: descriptor map only; B–D blocked pending geometry-keyed validation.
- Tables 1–2: conservative audit/matrix only; final unified acceptance blocked.

The detailed independent notes are in `review/`.
