# MigrationBench figure — frozen visual/scientific acceptance standard v3.0

Independent visual reviewer: `01a170d5-ebcd-7ae2-bff3-ce45f4c2a1be`  
Independent visual-reference search: `01a170dc-1b78-7c81-9316-6fc50c2c4a76`

This v3 standard incorporates the user's latest correction and the independent online visual-reference report. It supersedes the v2 composition requirement while preserving the v2 two-distribution / three-model semantics.

## Non-negotiable preservation of the original scientific visual elements

The supplied CS-style reference is not a loose mood board. Its actual visual modules must be retained as real vector/scientific elements, not replaced by generic placeholder rectangles:

1. `Coordinates`: a Cartesian 3-D axis tripod with colored atom/point markers. The revised figure must contain **two** such coordinate-distribution panels: a broad/diffuse foundation-pretraining cloud and a compact in-house migration cloud.
2. `Descriptors`: a mathematical per-atom descriptor representation, with a bracketed/matrix-like vector or row structure and explicit `G_i` notation. A generic “descriptor” box without mathematical structure fails this gate.
3. `Neural Networks for atomic species`: a real layered node-and-connection diagram with input nodes, hidden nodes, output node, and `E_i`-type atomic-energy readout. A generic `MACE` rectangle fails this gate.
4. `Total Energy`: an explicit atomic-energy aggregation, including the visual equation `\(\sum_i E_i\)` or an equivalent `\(E_\mathrm{tot}=\sum_i E_i\)` readout. Omitting the total-energy element fails this gate.

The new training/model semantics may be added around or after these modules, but they may not erase, flatten, or replace the original four-part scientific pipeline.

## Reference-derived visual standard

The independent search report identifies the following primary visual grammars:

- Behler's coordinate → per-atom descriptor → element-specific atomic NN → atomic energies → total-energy sequence is the closest structural reference for this figure.
- OVITO-style Cartesian axis tripods and colored particle markers are the correct coordinate visual language.
- Bracketed per-atom descriptor vectors/matrices are appropriate; the figure must not fabricate exact descriptor dimensions or values.
- Behler–Parrinello / ANI-style node-layer diagrams are appropriate for atomic-species networks; circles and weighted connections should be visible.
- The total-energy readout should visibly aggregate atomic contributions.
- Graph/message-passing references may inform node/edge vocabulary, but a generic graph diagram must not be mislabeled as the internal MACE implementation. Do not add unsupported tensor-product, freezing, or layer-specific claims.

Reference report: `visual_reference_search_report_20261024.md`.

## Two-distribution / three-model semantic hard gates

The first glance must communicate:

```text
large, broad foundation / pretraining coordinates ─┐
                                                    ├→ descriptors → shared atomic-species NN → total energy interface
small, compact in-house task coordinates ──────────┘
```

The model/training extension must explicitly show:

| Route | Required source and process | Required result |
|---|---|---|
| Generalist | foundation/pretraining data → pretrain; resulting foundation checkpoint is `use as-is / no update` | Foundation / Generalist model |
| Specialist | in-house migration data → train from scratch; no foundation checkpoint input | Specialist model |
| Fine-tuned | foundation checkpoint + in-house migration data → fine-tune | Fine-tuned model |

- `foundation/pretraining data` and `foundation checkpoint` must be visually and textually distinct.
- Fine-tuning must show two independent inputs entering the update step: a purple checkpoint input and a blue in-house-data input.
- The same atomic-species NN/MACE architecture should be shown as the shared architecture; route differences belong to provenance, initialization, and update labels unless three full network insets are explicitly needed for readability.
- Do not invent layer freezing, head-only updates, model sizes, data counts, performance rankings, or quantitative overlap.

## Frozen style and layout rules

- White background; deep navy structure lines; teal process arrows.
- Blue = data input; orange = training/update operation; purple = checkpoint/model/result.
- Use mathematical typography and Helvetica-like sans-serif labels consistent with the supplied reference.
- No gradients, shadows, 3-D UI cards, green/cyan canvas, or decorative placeholder panels.
- The two coordinate clouds are qualitative and must be labeled `not to scale`; no unsupported sample counts.
- Position, size, arrow weight, or brightness must not imply a model-performance ranking.
- Solid arrows carry the main flow; thin dashed connectors may show checkpoint lineage/context only.
- No arrow crossings or ambiguous merges; the fine-tuning two-input merge must be legible.

## FAIL conditions

Fail if any original visual module is absent or replaced by a generic placeholder; if either coordinate distribution is not visibly a 3-D coordinate sketch; if the descriptor lacks mathematical structure; if the network lacks visible layers/connections and an atomic-energy readout; if `\(\sum_i E_i\)` is missing; if any model route lacks its required source/process/result; if Generalist appears fine-tuned; if Specialist receives a foundation checkpoint; if Fine-tuning lacks either input; if the figure implies performance; or if any label clips/overlaps at delivered PDF size.
