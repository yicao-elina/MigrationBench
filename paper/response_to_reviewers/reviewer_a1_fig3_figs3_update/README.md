# Reviewer A1 Fig. 3d / Fig. S3 Update

This folder contains replacement text for the critical reviewer point about the apparent
inconsistency between main-text Fig. 3d and SI Fig. S3.

Files:

- `response_letter_A1_replacement.md`: replace the A1 section of
  `Revision1/response_to_reviewers/response_letter.md`.
- `proposed_text_edits_E3_E4_SI_replacement.md`: replace current E3/E4 blocks in
  `Revision1/response_to_reviewers/proposed_text_edits.md`, and add SI text/caption edits.
- `fig3d_figS3_protocol_side_by_side.csv`: compact table of the two protocols and current values.
- `fig3d_figS3_protocol_side_by_side_audited.pdf`: current manuscript-facing audit figure
  (`A1-PROTOCOL-AUDIT-V2`), included by the revised SI.
- `fig3d_figS3_pes_operator_schematic_v2.pdf`: latest explanatory schematic candidate
  (`A1-PES-OPERATOR-SCHEMATIC-V2`); it is a reduced-coordinate reconstruction, not a quantitative
  PES or a completed DFT re-score, and is not silently substituted for the active SI figure.
- `data/self_relaxed_foundation_path_61_1-4.extxyz`: compact five-image structural input used by
  the v2 schematic and atom renders.

Core correction:

- Fig. 3d = fixed-geometry single-point energy evaluation on DFT-relaxed NEB images.
- The historical Fig. S3 record was intended as a self-consistent MLIP NEB, but the audit shows
  that it is not the same physical system/path as Fig. 3d and is not quantitatively valid because
  of inconsistent calculators and corrupted final-band energy serialization.
- Therefore 1.024 eV / +0.688 eV is the current fixed-path result for the 61-atom path 1-4,
  whereas the submitted 0.41 eV record is retained only as a provenance failure audit and is
  excluded from model ranking. A valid same-system self-consistent comparison remains open.

Current real numbers used:

- DFT path 1-4 current audited barrier: 0.336050 eV.
- Current audit status for 1-4: `UNCONVERGED` but numerically stationary, so keep convergence caveat
  unless the final restartable QE run confirms/replaces it.
- Fixed-geometry barriers on path 1-4:
  - Foundation: 1.024296 eV, signed error +0.688246 eV.
  - Scratch: 4.537677 eV, signed error +4.201627 eV.
  - FT-600K: 0.496792 eV, signed error +0.160742 eV.
  - FT-MultiT: 0.823040 eV, signed error +0.486990 eV.
- Historical SI Fig. S3 submitted value: Foundation self-consistent MLIP NEB barrier 0.41 eV,
  but `EXCLUDE_FROM_PUBLICATION` after audit. The historical record uses Cr2Sb32Te48 (82 atoms,
  12 images); uniform rescoring gives forward 0.000 eV, reverse 0.379 eV, and whole-band span
  1.037 eV, not 0.410 eV.

Rebuild the manuscript-facing audit figure from the repository root with the Plot Atlas on the
Python path:

```bash
PYTHONPATH=/path/to/plot_library/src python \
  paper/response_to_reviewers/reviewer_a1_fig3_figs3_update/plot_a1_protocol_audit_v2.py
```

The v2 operator schematic uses the same repository inputs and additionally requires the Plot Atlas:

```bash
MIGRATIONBENCH_PLOT_ATLAS=/path/to/plot_library \
PYTHONPATH=/path/to/plot_library/src python \
  paper/response_to_reviewers/reviewer_a1_fig3_figs3_update/plot_pes_operator_schematic_v2.py
```
