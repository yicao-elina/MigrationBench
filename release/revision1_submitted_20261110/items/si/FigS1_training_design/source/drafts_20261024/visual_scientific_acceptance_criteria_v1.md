# MigrationBench training-design figure — frozen visual and scientific review standard v1.0

Independent reviewer: `01a170d5-ebcd-7ae2-bff3-ce45f4c2a1be`

## Content hard gates

| Route | Data / provenance | Initialization | Current process | Result |
|---|---|---|---|---|
| Specialist | only in-house data; small / curated | scratch / untrained initialization | train from scratch | Specialist model |
| Generalist | only foundation / pretraining data; large / broad | foundation checkpoint | no current update; use as-is | Foundation model |
| Fine-tuning | foundation checkpoint + in-house data | foundation checkpoint | fine-tuning on in-house data | Fine-tuned model |

- `foundation/pretraining data` and `foundation checkpoint` must be visually and textually distinct.
- Generalist must explicitly say `no update` or `use as-is`.
- Fine-tuning must show both checkpoint and in-house data entering the update step.
- Specialist must not receive a foundation checkpoint or pretraining data.
- Do not invent layer freezing, head-only updates, model sizes, data counts, or performance claims.

## Parallel-comparison gates

- Three equal horizontal lanes: Specialist, Generalist, Fine-tuning.
- Four aligned stages: `Data / provenance → Initialization → Current update or use → Resulting model`.
- All lanes read left to right with equal visual weight and equal output-node size.
- Lane order must not imply a ranking.
- Only Fine-tuning has a two-input merge before the update step.

## Frozen visual language

- White background; deep navy structure lines; teal process arrows.
- Blue = data input; orange = training/update operation; purple = checkpoint/model/result.
- Labels must carry the science; color alone is insufficient.
- No gradients, shadows, 3D effects, UI-style colored background, or decorative cards.
- Use the previous reproduction's Helvetica-like sans titles, mathematical typography, stable line weights, and generous whitespace.

## Arrow and provenance rules

- Solid teal arrows represent the current flow.
- Thin dashed navy/teal connectors represent upstream provenance/context only.
- Generalist upstream pretraining, if shown, must be labeled `upstream / context only` and must not look like current MigrationBench training.
- Arrows may not cross, reverse, or ambiguously merge.

## Nature-level readability and FAIL conditions

The first glance must reveal three routes; the second glance must distinguish data, checkpoint, update, and output. Fail if any route lacks its required source/initialization/process/output, if `foundation` is ambiguous, if Generalist appears fine-tuned, if Fine-tuning lacks either input, if Specialist starts from a foundation checkpoint, if color/size/position/arrow weight implies performance, if text overlaps or clips, or if unverified parameter or performance claims are added.

## Recommended composition

```text
                 Data / provenance
                 Initialization
                 Current update or use
                 Resulting model

Specialist       in-house data → scratch init → train from scratch → Specialist model
Generalist       foundation data → foundation checkpoint → no update/use as-is → Foundation model
Fine-tuning      in-house data ───────────────┐
                                               ├→ fine-tuning → Fine-tuned model
                 foundation checkpoint ───────┘
```
