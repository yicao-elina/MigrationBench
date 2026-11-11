# MigrationBench figure palette reference

This is the shared source of truth for colors and common plot encodings in the MigrationBench figures. Machine-readable values are in [`MIGRATIONBENCH_FIGURE_PALETTE.json`](MIGRATIONBENCH_FIGURE_PALETTE.json). The existing P03–P10 bundle README and manifests document the palette as applied to those exported figures.

## Model and reference colors

| Meaning | Color |
|---|---|
| Foundation family | `#8FC7AB` |
| Fine-tuning family | `#29A7B6` |
| Scratch family | `#327FB1` |
| DFT / reference | `#303840` |
| MACE-MP checkpoint | `#2F6B4F` |
| MACE-OMAT-0 checkpoint | `#5F9E73` |
| MACE-MH checkpoint | `#8FC7AB` |

## Path-class heading accents

These accents are reserved for column headings that name the three migration regimes. They use legible dark text shades from the shared pink family; model curves retain their own model-class colors.

| Path class | Heading accent |
|---|---|
| In-gap diffusion | `#B65471` |
| Gap→QL | `#C94381` |
| Within-QL | `#74297E` |

## Campaign task-type colors

Use this pink family for Base and the two perturbation task arms throughout campaign figures:

| Task type | Color |
|---|---|
| Base | `#FABAB9` |
| Perturb A | `#F26EA6` |
| Perturb B | `#74297E` |

## Shared encodings

- Training seeds 123, 234, and 345 use circle, triangle, and square markers.
- Foundation checkpoints use diamonds; the DFT reference uses an open diamond.
- Fixed-path curves are solid; self-relaxed curves are dashed. DFT reference curves are dashed.
- For temperature comparisons, 600 K uses open markers and Multi-T uses filled markers.
- Keep SVG labels as text (`svg.fonttype=none`) and use PDF font type 42 for editable/selectable vector text.
- Preferred font order is Arial, Helvetica, then DejaVu Sans.

The larger-font four-dopant fixed-path SI re-render is an implementation example for the path-class heading accents.
