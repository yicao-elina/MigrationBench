# Canva figure palette update index — 2026-11-02

This additive revision applies the approved visual grammar to figures that have a reproducible local source. The latest Cr NEB deliverable is the R05 source with separate `Scratch-600K` and `Scratch-MultiT` groups; the earlier `CANVA-P06_cr_three_path_neb_palette_updated.*` is retained as an obsolete R02 aggregate for provenance and is not the final NEB output.

## Delivered figures

| Canva figure | Current source/content match | Updated output | Status |
|---|---|---|---|
| CANVA-P03 | Local train/test parity and continuous-density variants | `figures/CANVA-P03_parity_2x4_palette_updated.*`; `figures/CANVA-P03/CANVA-P03_hexbin_density_palette_updated.*` | Updated; strategy colors and blue density scale |
| CANVA-P04 | Local frame-level RMSE source | `figures/CANVA-P04_frame_rmse_palette_updated.*` | Updated; strategy colors, open/filled temperature markers |
| CANVA-P05 | Local MD-15 A–F source; six-panel layout matching the supplied MD reference | `figures/CANVA-P05/md_six_panel_palette_updated.*` | Updated; Foundation/FT/Scratch colors, open 600 K, filled Multi-T, cleared A–C panel letters, visible pair hatches across all D groups, and 11.6 pt legends |
| CANVA-P06 | `make_cr_three_path_2x3_R05.py`; five model groups plus DFT | `figures/CANVA-P06_neb_cr_r05_palette_updated.*` | Final NEB update; Scratch-600K and Scratch-MultiT remain separate |
| CANVA-P07 | Local Foundation class 3×3 source | `figures/CANVA-P07_foundation_class_3x3_palette_updated.*` | Updated; MACE-MP/OMAT-0/MH same green family |
| CANVA-P08 | Local Foundation class 4×3 source | `figures/CANVA-P08_foundation_class_4x3_palette_updated.*` | Updated; MACE-MP/OMAT-0/MH same green family |
| CANVA-P09 | Local Cr interlayer sliding grouped source | `figures/CANVA-P09_cr_interlayer_sliding_palette_updated.*` | Updated; Scratch uses blue, temperature uses open/filled |
| CANVA-P10 | Local Foundation interlayer sliding source | `figures/CANVA-P10_foundation_interlayer_sliding_palette_updated.*` | Updated; MP `#2F6B4F`, OMAT-0 `#5F9E73`, MH `#8FC7AB` |

For every `.*` entry above, PDF, SVG, and PNG are available unless the figure is explicitly listed as a two-format density export.

## Not claimed as locally reproduced in this revision

The exact local plotting source was not established for the Canva pages corresponding to the RMSE boxplot/temperature-resolved composition and the later evaluation/SHAP or pointer-only pages. Those pages are not relabeled as updated here. The source Canva PDF remains the reference for those unresolved pages.

## Shared visual encoding

- Foundation: `#8FC7AB`; Fine-tuning: `#29A7B6`; Scratch: `#327FB1`.
- 600 K: open marker; Multi-T: filled marker.
- Foundation checkpoint figures: MACE-MP `#2F6B4F`, MACE-OMAT-0 `#5F9E73`, MACE-MH `#8FC7AB`.
- DFT reference: `#303840`, dashed line, open diamond.
- Fixed path: solid line; self-relaxed: dashed line.

The source hashes and output hashes are recorded in `NEB_PALETTE_MANIFEST.json`, `FOUNDATION_CLASS_PALETTE_MANIFEST.json`, and `MD_SIX_PANEL_MANIFEST.json`.
