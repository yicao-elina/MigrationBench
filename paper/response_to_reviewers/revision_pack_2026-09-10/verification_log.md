# Verification Log

Date: 2026-09-10

## Completed checks

1. `response_to_reviewers_revised_full.md` contains one section for each reviewer item:
   A1, A2, A3, A4, A5, A6, A7, C8, C9, C10, C11, C12, and C13.

2. `pending_values_register.csv` contains 13 placeholders:
   `final-QE-1-4-barrier`, `SI-FigS3-audit`, `DFT-on-MLFF-path`,
   `SOAP-RMSD-overlap-audit`, `multiseed-kinetic-barrier-errors`,
   `second-dopant-cross-validation`, `scratch-5pct-control-rmse-and-barriers`,
   `MACE-CrCr-perturbation-test`, `final-code-release-hash-for-zero-force-fix`,
   `corrected-FT600K-MD-transport`, `grouped-split-retrained-rmse`,
   `final-SHAP-code-release-hash`, and `final-empty-stub-audit`.

3. `apply_reviewer_markers.py` compiles with:

   ```bash
   python3 -m py_compile outputs/reviewer_response_revision_pack/apply_reviewer_markers.py
   ```

4. The marker insertion script was tested on
   `outputs/reviewer_response_revision_pack/smoke_fixture/`.

   First dry-run result:

   - `sn-article-final.tex`: 15 inserted/changed markers including preamble.
   - `sn-article-SI.tex`: 6 inserted/changed markers including preamble.

   Apply result:

   - wrote both fixture files.
   - created `.bak-reviewer-markers` backups.

   Second dry-run result:

   - `sn-article-final.tex`: 0 changes.
   - `sn-article-SI.tex`: 0 changes.

   This verifies that the script is idempotent on the covered anchors.

5. ASCII check passed for the generated revision pack and A1-specific update folder:

   ```bash
   LC_ALL=C rg -n "[^[:ascii:]]" outputs/reviewer_response_revision_pack outputs/reviewer_a1_fig3_figs3_update
   ```

   No non-ASCII matches were returned.

## Not completed in this Codex environment

The script was not applied to the actual OneDrive/Overleaf source directory because the current
Codex sandbox can read many files there but cannot reliably stream the full
`sn-article-final.tex` file. A dry-run on the true manuscript directory was interrupted after
30 seconds while reading the file.

Run this from the user's normal local terminal to perform the actual source marking:

```bash
python3 /Users/alina/Documents/Codex/2026-09-08/new-chat/outputs/reviewer_response_revision_pack/apply_reviewer_markers.py \
  /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission \
  --apply
```

The command will create backups next to the source files:

- `sn-article-final.tex.bak-reviewer-markers`
- `sn-article-SI.tex.bak-reviewer-markers`
