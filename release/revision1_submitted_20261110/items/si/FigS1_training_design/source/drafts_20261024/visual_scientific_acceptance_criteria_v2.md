# MigrationBench two-distribution / three-model figure — frozen review standard v2.0

Independent reviewer: `01a170d5-ebcd-7ae2-bff3-ce45f4c2a1be`

This v2 standard supersedes the v1 layout requirement. It keeps the v1 semantic hard gates, but replaces the three-row composition with a single left-to-right composition that begins from two coordinate distributions.

## Core narrative hard gates

The first glance must communicate:

```text
large, broad foundation / pretraining coordinates ─┐
                                                    ├→ shared MACE architecture → three training routes → three models
small, compact in-house task coordinates ──────────┘
```

The two coordinate panels must be visibly three-dimensional axis sketches, not merely colored dataset cards:

- Foundation / pretraining data: a visibly broader, more diffuse qualitative coordinate cloud; labeled `large · broad · upstream`.
- In-house migration data: a visibly smaller, compact qualitative coordinate cloud; labeled `small · curated · target-specific`.
- A note must state that the coordinate clouds are qualitative and not to scale; no unsupported sample counts may be introduced.

## Route and model hard gates

| Route | Required source and process | Required result |
|---|---|---|
| Generalist | foundation / pretraining data → `pretrain`; the resulting checkpoint is used `as-is / no update` for the current task | `Foundation / Generalist model` |
| Specialist | in-house migration data → `train from scratch`; no foundation checkpoint input | `Specialist model` |
| Fine-tuned | foundation checkpoint + in-house migration data → `fine-tune` | `Fine-tuned model` |

- `foundation / pretraining data` and `foundation checkpoint` must be visually and textually distinct.
- The fine-tuning route must show two inputs entering the update step: one purple checkpoint input and one blue in-house-data input.
- The Generalist route must not visually look fine-tuned. It must explicitly say `use as-is` or `no update`.
- The Specialist route must not receive the foundation checkpoint or broad pretraining data.
- The same MACE architecture must be represented once as a shared architecture; route differences belong to the data, initialization, and training/update labels.
- Do not invent layer freezing, head-only updates, model sizes, data counts, rankings, or performance claims.

## Frozen visual language

- White background; deep navy structure lines; teal process arrows.
- Blue = data input; orange = training/update operation; purple = checkpoint/model/result.
- Labels must carry the science; color alone is insufficient.
- No gradients, shadows, 3D effects, UI-style colored background, or decorative cards.
- Use the previous reproduction's Helvetica-like sans titles, mathematical typography, stable line weights, and generous whitespace.
- The route/model boxes must have equal visual weight. Position, size, arrow thickness, or brightness must not imply a performance ranking.

## Arrow and provenance rules

- Solid teal arrows represent the main left-to-right processing flow.
- Thin dashed navy connectors may indicate checkpoint lineage or context, but must not be mistaken for a current data-update step.
- The two coordinate distributions may feed the shared descriptor/architecture stage through separate arrows that visibly converge on the same MACE architecture.
- Avoid crossing arrows, reverse arrows, ambiguous merges, and unlabeled duplicated `foundation` concepts.

## Readability and FAIL conditions

Fail if any of the following occurs: the two panels are not recognizable as coordinate distributions; broad and compact distributions are not distinguishable; the shared MACE architecture is absent or duplicated confusingly; any route lacks its required data/checkpoint/process/result; Generalist appears fine-tuned; Fine-tuning lacks either input; Specialist starts from a foundation checkpoint; the figure implies a ranking; the qualitative/not-to-scale caveat is absent; or text overlaps, clips, or becomes unreadable at the delivered PDF size.
