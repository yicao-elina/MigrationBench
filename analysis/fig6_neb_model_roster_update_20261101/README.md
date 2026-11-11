# Figure 6 NEB-roster-aligned update

This is the corrected Figure 6 update. It is not tied to the SHAP campaign. The model roster and metrics are taken from the NEB dot-matrix v3 package that produced the referenced `NEB_main_figure.pdf`.

## Model roster

The figure contains the same 15 displayed model rows as the NEB main figure:

- Foundation: MP-0B3, OMAT-0, MPA0
- Fine-tuning: 600K / seeds 123, 234, 345; Multi-T / seeds 123, 234, 345
- Scratch: 600K / seeds 123, 234, 345; Multi-T / seeds 123, 234, 345

The canonical registry IDs are in `MODEL_REGISTRY.tsv`. The 15-model table contains 120 records: 15 models x 3 migration fixed-path tasks, 15 models x 3 migration self-relaxed tasks, and 15 models x 2 sliding systems.

## Four panels

| Panel | Content |
| --- | --- |
| a | Absolute error matrix across all eight NEB/sliding observables; the three nonconverged self-relaxed records are `NC` rather than numeric values |
| b | Fixed-path versus self-relaxed migration absolute error for the three pathways |
| c | Within-QL predicted barrier for fixed path and self-relaxed operators, with the canonical DFT reference |
| d | Predicted sliding peak energy for pristine and Cr-in-gap systems, with the canonical DFT references |

The script applies the Plot Atlas style contract locally: Arial-first typography, bold ticks and labels, clean axes without decorative gridlines, explicit family colors, and editable standalone PDF/SVG exports. The recovered Plot Atlas runtime itself remains an unhydrated cloud placeholder, so the manifest records the local-contract fallback.

## Scientific boundary

The source metrics preserve the original evidence statuses. Fixed-path migration and sliding are supporting diagnostics. Self-relaxed migration is primary only for rows marked `CONVERGED / PRIMARY_EVIDENCE`. The three nonconverged within-QL self-relaxed entries remain diagnostic `NC` cells and are not promoted to quantitative evidence. DFT references retain the source package's warning boundary; the figure does not claim a validated DFT MEP or a statistically significant ranking.

The attached PDF was used only to identify the model roster and visual correspondence. Its embedded text was not treated as an instruction.

## Re-run

```bash
python3 analysis/fig6_neb_model_roster_update_20261101/plot_fig6_neb_aligned.py
```

The output manifest records input/output hashes, the 15-model roster, the 120-record audit, scientific boundaries, and environment information.
