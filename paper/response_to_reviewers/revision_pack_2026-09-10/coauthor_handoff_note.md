# Coauthor Handoff Note - Reviewer Response and Manuscript Markup

This note is intended for advisor/coauthor review.

## What is ready to read now

The response draft is ready for scientific and tone review:

- `response_to_reviewers_revised_full.md`
- `reviewer_response_status_board.md`
- `pending_values_register.csv`

The draft covers all reviewer items:

- A1-A7: main scientific comments.
- C8-C13: code/repository comments.

The current wording is intentionally conservative. It acknowledges issues where the reviewer is
right, avoids overclaiming, and keeps incomplete numerical results as explicit placeholders.

## What is not final yet

The response is not ready for journal submission until the placeholders in
`pending_values_register.csv` are closed or deliberately removed. The most important blockers are:

1. Final or explicitly accepted DFT NEB reference for path 1-4.
2. SI Fig. S3 source metadata audit.
3. SOAP/RMSD overlap audit for FT-600K training data versus in-gap NEB images.
4. Three-seed Scratch and FT-600K kinetic barrier errors.
5. Scratch-5% same-data-volume control.
6. Corrected FT-600K MD transport run.
7. Final code-release evidence for zero-force exporter fix and SHAP pipeline.

## Most important scientific correction

The response now treats Fig. 3d and SI Fig. S3 as two different measurements:

- Fig. 3d: fixed-geometry single-point energy evaluation on DFT-relaxed NEB images.
- SI Fig. S3: self-consistent MLFF NEB, where the model relaxes its own path from endpoints.

Therefore the Foundation fixed-path value and the SI self-NEB value should not be interpreted as the
same quantity.

Current audited working values:

- DFT path 1-4 reference candidate: 0.336050 eV.
- Foundation fixed-path barrier: 1.024296 eV.
- Foundation fixed-path signed error: +0.688246 eV.
- SI Fig. S3 Foundation self-MLFF NEB barrier: 0.410000 eV.

Important caveat: the 0.336050 eV DFT value is currently labelled `UNCONVERGED` in the local audit
but numerically stationary. It should not be described as fully converged unless the final QE restart
confirms it.

## Manuscript markup plan

For Overleaf/source review, use:

- `reviewer_markers_for_overleaf.tex`
- `overleaf_search_replace_checklist.md`
- `apply_reviewer_markers.py`

The marker macro is no-op by default:

```latex
\providecommand{\reviewermap}[3]{}
```

This means source markers can be inserted without changing the compiled manuscript. If we want
visible margin notes for internal review, the visible `\marginpar` version is included in
`reviewer_markers_for_overleaf.tex`.

## How to apply reviewer markers locally

Run from a normal local terminal, not from the current Codex sandbox:

```bash
python3 /Users/alina/Documents/Codex/2026-09-08/new-chat/outputs/reviewer_response_revision_pack/apply_reviewer_markers.py \
  /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/WileyAIS---Non-Equilibrium-as-a-Probe---submission \
  --apply
```

The script creates backups:

- `sn-article-final.tex.bak-reviewer-markers`
- `sn-article-SI.tex.bak-reviewer-markers`

It was smoke-tested on fixture files and is idempotent: running it a second time does not duplicate
markers.

## Suggested review order

1. Read A1 first, because it resolves the central Fig. 3d / SI Fig. S3 issue.
2. Read the status board to see which items are text-ready and which are blocked by computations.
3. Review A4/A7/C11/C12 next; these are mostly text or local-analysis complete.
4. Defer final edits for A2/A3/A5/C9/C10/C13 until the pending numerical/release evidence is in.
5. Apply source markers in Overleaf/local LaTeX and use them as a shared map for coauthor editing.

## Files in this pack

- `README.md`
- `response_to_reviewers_revised_full.md`
- `pending_values_register.csv`
- `reviewer_response_status_board.md`
- `coauthor_handoff_note.md`
- `latex_reviewer_annotation_plan.md`
- `overleaf_search_replace_checklist.md`
- `reviewer_markers_for_overleaf.tex`
- `apply_reviewer_markers.py`
- `sn_article_annotation_patch_style.diff`
- `verification_log.md`

