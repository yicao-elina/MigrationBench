# Reviewer Response Revision Pack

Generated in the Codex workspace because the current sandbox can read but not write directly into
the OneDrive Revision1 manuscript folder.

Files:

- `response_to_reviewers_revised_full.md`: full revised working response letter.
- `pending_values_register.csv`: all placeholders, what they mean, and how each should be closed.
- `latex_reviewer_annotation_plan.md`: exact source-comment tags for Overleaf/manuscript/SI review.
- `reviewer_markers_for_overleaf.tex`: optional no-op LaTeX marker macro and pasteable
  `\reviewermap{}` annotations.
- `overleaf_search_replace_checklist.md`: practical anchor-by-anchor checklist for applying the
  reviewer-linked edits in Overleaf.
- `sn_article_annotation_patch_style.diff`: patch-style reviewer marker map for the main text and SI.
- `reviewer_response_status_board.md`: coauthor-facing readiness board and safe-to-cite values.
- `apply_reviewer_markers.py`: dry-run/apply script that inserts reviewer markers into local
  `sn-article-final.tex` and `sn-article-SI.tex` using stable text anchors.
- `verification_log.md`: checks performed, including marker-script smoke test and current
  environment limitation for direct OneDrive application.
- `coauthor_handoff_note.md`: short advisor/coauthor handoff note with review order and caveats.

Recommended sync command:

```bash
rsync -av /Users/alina/Documents/Codex/2026-09-08/new-chat/outputs/reviewer_response_revision_pack/ \
  /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission/Revision1/response_to_reviewers/revision_pack_2026-09-10/
```

The earlier A1-specific files are in:

```text
/Users/alina/Documents/Codex/2026-09-08/new-chat/outputs/reviewer_a1_fig3_figs3_update/
```

Marker insertion dry-run:

```bash
python3 /Users/alina/Documents/Codex/2026-09-08/new-chat/outputs/reviewer_response_revision_pack/apply_reviewer_markers.py \
  /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission
```

Apply in place with backups:

```bash
python3 /Users/alina/Documents/Codex/2026-09-08/new-chat/outputs/reviewer_response_revision_pack/apply_reviewer_markers.py \
  /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission \
  --apply
```
