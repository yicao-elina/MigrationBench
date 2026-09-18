# Reviewer A1 Fig. 3d / Fig. S3 Update

This folder contains replacement text for the critical reviewer point about the apparent
inconsistency between main-text Fig. 3d and SI Fig. S3.

Files:

- `response_letter_A1_replacement.md`: replace the A1 section of
  `Revision1/response_to_reviewers/response_letter.md`.
- `proposed_text_edits_E3_E4_SI_replacement.md`: replace current E3/E4 blocks in
  `Revision1/response_to_reviewers/proposed_text_edits.md`, and add SI text/caption edits.
- `fig3d_figS3_protocol_side_by_side.csv`: compact table of the two protocols and current values.

Core correction:

- Fig. 3d = fixed-geometry single-point energy evaluation on DFT-relaxed NEB images.
- Fig. S3 = self-consistent MLFF NEB where the model relaxes its own path from endpoints.
- Therefore 1.024 eV / +0.688 eV Foundation fixed-path result and 0.41 eV Foundation self-NEB
  result are different measurement operators, not directly contradictory.

Current real numbers used:

- DFT path 1-4 current audited barrier: 0.336050 eV.
- Current audit status for 1-4: `UNCONVERGED` but numerically stationary, so keep convergence caveat
  unless the final restartable QE run confirms/replaces it.
- Fixed-geometry barriers on path 1-4:
  - Foundation: 1.024296 eV, signed error +0.688246 eV.
  - Scratch: 4.537677 eV, signed error +4.201627 eV.
  - FT-600K: 0.496792 eV, signed error +0.160742 eV.
  - FT-MultiT: 0.823040 eV, signed error +0.486990 eV.
- SI Fig. S3 submitted value: Foundation self-consistent MLFF NEB barrier 0.41 eV.

Suggested sync command from this Codex workspace to Revision1:

```bash
rsync -av /Users/alina/Documents/Codex/2026-09-08/new-chat/outputs/reviewer_a1_fig3_figs3_update/ \
  /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/Revision1/response_to_reviewers/reviewer_a1_fig3_figs3_update/
```
