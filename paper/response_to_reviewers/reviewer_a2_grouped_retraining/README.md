# Reviewer 1 A2: grouped retraining and overlap audit

This bundle contains the reviewer-facing reconstruction for A2 and the
linked C10/A3/A5 evidence. The figure is rendered from the compact grouped
seed table and the two machine-readable overlap audits; no raw checkpoint or
trajectory is modified.

## Evidence chain

| claim | source | processing | status |
|---|---|---|---|
| trajectory-grouped split and 25-image audit | `data/overlap_full_leakage_audit.json`, `data/overlap_scratch5_leakage_audit.json` | `plot_a2_grouped_retraining_audit.py` | `VERIFIED` under declared exact/SOAP/RMSD metrics |
| grouped RMSE and pathway 1--4 barrier errors | `data/grouped_seed_comparison.csv`, `data/grouped_neb_barrier_summary.csv` | `plot_a2_grouped_retraining_audit.py` | `VERIFIED` as a three-seed computational comparison; DFT anchor remains provisional |
| historical 0.16 eV value | archived frame-level result | retained only as a labeled comparator | `EXCLUDE_FROM_PUBLICATION` as the current grouped result |
| figure | `figures/a2_grouped_retraining_audit.*` | `plot_a2_grouped_retraining_audit.py` | `RECONSTRUCTION` from the listed compact inputs |
| higher-distortion paths | rows retained in `grouped_neb_barrier_summary.csv` | no pooled aggregation | `PROVENANCE_PARTIAL` (validation gap) because path class/reference identity is not jointly closed |

The overlap pass is bounded to the declared fine-tuning train files and does
not make a claim about the unavailable pretraining corpus. SOAP is available
for all audited target images. The Cr-local RMSD minimum is computed only for
species-composition-compatible local environments (two comparable candidates
for the full grouped split and one for Scratch-5%).

## Rebuild

From the repository root:

```text
MIGRATIONBENCH_PLOT_ATLAS=/path/to/plot_library \
PYTHONPATH=/path/to/plot_library/src python \
  paper/response_to_reviewers/reviewer_a2_grouped_retraining/plot_a2_grouped_retraining_audit.py
```

The script writes PNG, PDF, SVG, and the normalized-distance summary CSV.
The Plot Atlas `bar_with_points` and `grouped_bar` primitives are used for the
two panels. The dashed line in panel (a) is the historical frame-level value;
the dashed line at one in panel (b) is the declared near-duplicate gate. The
figure uses the FIG-004 JHU MACE model palette and keeps methodological notes
in the caption rather than in small text inside the plot.

## Paper placement

The main text uses the grouped values in Methods Sec.~2.1 and Results
Sec.~3.3.3. The complete table, overlap qualification, and the audit figure
are placed in the SI ``MACE Model Performance Evaluation by RMSE`` section
and the ``In-Gap Diffusion`` subsection. The reviewer response contains the
same figure as a direct answer to A2.
