# Literature and design notes

## What the searched literature contributes

- The MACE-freeze work explicitly separates a transfer-learning workflow from the foundation model and shows controlled parameter freezing as a separate concept. This supports keeping `foundation checkpoint` distinct from `in-house data`, while leaving the exact update scope unspecified here unless the protocol confirms it: [npj Computational Materials — Fine-tuning foundation models of materials interatomic potentials with frozen transfer learning](https://www.nature.com/articles/s41524-025-01727-x).
- Materials pretraining figures commonly show an upstream broad corpus, representation/model pretraining, and a downstream fine-tuning stage. The SPMat schematic describes a large pretraining workflow followed by a smaller downstream task: [Communications Materials — Supervised pretraining for material property prediction](https://www.nature.com/articles/s43246-025-00999-6).
- A general foundation-model comparison figure explicitly contrasts `pretrain → fine-tune` with a supervised workflow that trains on the target task. That supports the three-lane comparison logic: [ICLR 2025 foundation-model workflow comparison](https://openreview.net/attachment?id=JYTQ6ELUVO&name=pdf).
- A materials transfer-learning study uses explicit pathways and fine-tuning strategies, reinforcing that checkpoint lineage and update scope should be named rather than implied by a generic “foundation” box: [Optimal pre-train/fine-tune strategies for accurate material property predictions](https://www.nature.com/articles/s41524-024-01486-1).

## Design decision for MigrationBench

The clarified figure is not a set of three unrelated pipelines. It is a left-to-right composition with two qualitative coordinate-distribution sketches, one shared descriptor/MACE architecture, and three explicit training routes. The broad pretraining distribution and compact in-house distribution are shown as actual 3-D axis mini-panels; the route panel then distinguishes pretraining/use-as-is, scratch training, and fine-tuning. The cyan canvas, green cards, mixed arrow colors, and repeated indistinguishable `MACE` boxes from the rough draft are therefore replaced by the previous reproduction's white-background vector/LaTeX language.

The figure intentionally does not claim parameter freezing, data counts, model-size differences, performance ranking, or quantitative overlap. The coordinate clouds are labeled as qualitative and not to scale. `Foundation / pretraining data` is kept visually distinct from the resulting `Foundation checkpoint` so the fine-tuning lineage is explicit.

## Clean-figure revision

The latest revision adopts the compact graphical-abstract grammar of the supplied [CatBench figure](https://www.sciencedirect.com/science/article/pii/S2666386425005673): short labels, visible scientific glyphs, arrows, and minimal prose. This is a density correction, not a semantic simplification: all four original vector modules remain mandatory, while the explanatory sentences are removed from the artboard and the route semantics are carried by labels and visual lineage.
