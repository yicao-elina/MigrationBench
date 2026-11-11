# MigrationBench figure — clean conference schematic acceptance standard v4.0

Independent visual reviewer: `01a170d5-ebcd-7ae2-bff3-ce45f4c2a1be`  
Reference-search agent: `01a170dc-1b78-7c81-9316-6fc50c2c4a76`

This v4 standard adds the user's latest visual correction: the figure must be clean and schematic, with the low text density and compact panel grammar of the supplied CatBench graphical abstract. It preserves all v3 scientific-element gates.

## Visual reference and composition

Primary style reference supplied by the user: [CatBench graphical abstract / ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S2666386425005673), whose figure uses compact panel headings, short labels, visual glyphs, arrows, and minimal explanatory prose.

The figure should read like a paper graphical abstract:

- one short title or no title inside the artboard;
- concise stage labels (`Coordinates`, `Descriptors`, `Neural Networks`, `Total Energy`);
- short route labels (`Pretrain`, `Scratch`, `Fine-tune`);
- mathematical notation and visual glyphs carry the explanation;
- no paragraph-length subtitles, repeated explanatory sentences, or long prose inside the figure;
- a single compact legend/caveat is acceptable when needed for color semantics or `not to scale`.

## Original visual modules remain mandatory

The clean redesign must still retain these real vector/scientific elements from the supplied CS-style reference, not generic placeholder blocks:

1. two Cartesian 3-D coordinate tripods with colored point/atom clouds;
2. a bracketed mathematical per-atom descriptor vector/matrix representation with `G_i` notation;
3. a visible atomic-species neural network with input, hidden, and output nodes plus connections and `E_i` readout;
4. atomic contributions and explicit `\(\sum_i E_i\)` total-energy aggregation.

The lower training routes may use short rounded operation labels, but their source, network, model, and energy elements must remain real glyphs rather than placeholder `MACE`/model cards.

## Semantic hard gates

- Foundation / pretraining coordinate cloud: visibly broad, labeled with short tags equivalent to `foundation`, `large`, `broad`, `upstream`.
- In-house coordinate cloud: visibly compact, labeled with short tags equivalent to `in-house`, `small`, `curated`, `target-specific`.
- The two coordinate panels must each have an independent, visually traceable arrow into the descriptor stage; a shared orphan junction fails.
- Generalist route: broad data → `pretrain` → Foundation checkpoint/model → `as-is` or `no update`.
- Specialist route: in-house data → `scratch` → Specialist model; no foundation checkpoint input.
- Fine-tuning route: purple foundation checkpoint + blue in-house data → `fine-tune` → Fine-tuned model.
- `foundation / pretraining data` and `foundation checkpoint` must remain distinct.
- Do not imply freezing, head-only training, data counts, performance ranking, or quantitative distribution overlap.

## Clean-layout hard gates

- No label, subtitle, or arrow may overlap any model name, network glyph, equation, or coordinate axis.
- Route arrows must pass through whitespace or terminate at glyphs; they may not run through text.
- Equal route weight; no position/size/brightness ranking.
- White background, navy structure, teal arrows, blue data, orange update, purple model/checkpoint.
- No gradients, shadows, dashboard-like cards, or dense explanatory paragraphs.
- `not to scale` may appear once in a compact footer; it must not dominate the artboard.

## FAIL conditions

Fail if any original scientific visual module is missing; if either coordinate source lacks an independent input arrow; if prose density makes the figure read like a methods paragraph; if any arrow/text overlap or clipping occurs; if the three route semantics are ambiguous; or if the clean graphical-abstract style is not achieved.
