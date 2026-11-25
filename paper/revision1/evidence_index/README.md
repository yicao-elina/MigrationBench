# Revision 1 evidence index: Reviewer 1 A1--A2

This directory is the navigation layer for the A1 and A2 evidence releases. It
does not replace the raw or compact evidence packages; it records which version
is current, which artifact is manuscript-facing, and which result remains
descriptive, provisional, or excluded.

## Release boundary

The release is intentionally additive. It does not include the unrelated
Discovery DFT campaign, HPC outputs, or the 1.7 GB local NEB audit directory.
Small compact inputs required by the manuscript-facing figures are copied into
the A1/A2 response packages. Historical descriptor inputs used only for the A2
task-distribution diagnostic remain external and are referenced by SHA256 and
`MIGRATIONBENCH_HISTORICAL_ROOT`.

## Current versions

| Question | Current manuscript-facing artifact | Latest retained candidate | Boundary |
|---|---|---|---|
| A1: Fig. 3d versus Fig. S3 | `reviewer_a1_fig3_figs3_update/fig3d_figS3_protocol_side_by_side_audited.pdf` (`A1-PROTOCOL-AUDIT-V2`) | `reviewer_a1_fig3_figs3_update/fig3d_figS3_pes_operator_schematic_v2.pdf` (`A1-PES-OPERATOR-SCHEMATIC-V2`) | The historical 0.410 eV record is excluded; the v2 PES figure is a schematic and does not assert a DFT re-score. |
| A2: grouped split and overlap | `reviewer_a2_grouped_retraining/figures/a2_grouped_retraining_audit.pdf` | same | Three-seed grouped comparison is reported for the in-gap path; the DFT anchor remains provisional. |
| A2: task distribution | `reviewer_a2_distribution_shift_task_ladder_20261018/figures/fig_a2_distance_audits.pdf` | `fig_a2_task_distribution_map.pdf` | Descriptive historical map; not a calibrated scalar OOD ladder and not a replacement for direct overlap auditing. |

## Reproduce the release checks

From the repository root:

```bash
python paper/revision1/evidence_index/verify_release.py
```

The fully self-contained figure rebuilds are documented in the A1 and A2
package READMEs. They require the external Plot Atlas only as a plotting
dependency, not as an unrecorded scientific input. The A2 task-distribution
map additionally requires the historical root specified by
`MIGRATIONBENCH_HISTORICAL_ROOT`.

## Status vocabulary

- `VERIFIED`: the listed compact input and calculation/metric audit are present.
- `PROVISIONAL_REFERENCE`: numerically usable but the independent QE convergence gate is open.
- `RECONSTRUCTION`: a deterministic figure regenerated from listed compact inputs.
- `DESCRIPTIVE`: useful diagnostic, not a calibrated benchmark or ranking.
- `VALIDATION_GAP`: the requested evidence is not closed.
- `EXCLUDE_FROM_PUBLICATION`: preserved for provenance and failure analysis, not for ranking.

The claim matrix, version table, and exclusion registry are tab-separated so
they can be filtered without parsing prose. The final release manifest records
the A1 and A2 commit IDs after the two evidence commits are created.
