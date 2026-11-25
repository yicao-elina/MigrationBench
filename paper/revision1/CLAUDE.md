# Figure Registry — Agent Rules (paper/revision1)

This directory holds the long-term figure asset system for the Wiley revision
manuscript (`sn-article-final.tex` / `sn-article-SI.tex`). Read this before
touching any figure, caption, or Results/Discussion text tied to a figure.

## Hard rule: never bulk-edit

- `figure_pages.tex` / `figure_pages.pdf` are a **frozen v1 review artifact**
  (211 figures in one 5160-line file). Do not regenerate, edit, or re-derive
  text from it. It exists only for historical comparison.
- The **editable source of truth** is `figure_registry/figures/FIG-NNN/` — one
  directory per figure, each split into small files: `manifest.yaml`,
  `page.tex`, `figure.tex`, `caption.tex`, `results.tex`, `discussion.tex`,
  `reviewer.tex`, `evidence.tex`, `evidence_auto.tex`, `article.tex`,
  `gap.tex`, `asset.pdf`.
- `figure_book.tex` / `figure_book.pdf` are **generated** by `\input`-ing every
  `figures/FIG-NNN/page.tex`. Never hand-edit `figure_book.tex`; rebuild it
  with `figurectl.py build-book` only when you actually need the whole book.

## When asked to change a figure

Given a request like "change FIG-002's caption" or "update the diffusion
figure" or "point this figure at REV-C9":

1. If you don't already have the exact `figure_id`, resolve it first:
   `python3 figurectl.py query "<search text>"` (searches id/title/group/
   article_target/reviewer_ids — do not open directories to search by hand).
2. `python3 figurectl.py show FIG-NNN` — read current state, check
   `lock_status` before touching anything.
3. If `lock_status` is `locked` or `promoted`: say so explicitly before
   editing. Editing a locked/promoted figure is allowed but is an
   **intentional validation failure** until a human reviews the diff and you
   run `update-hashes`. Do not silently "fix" that failure by reverting your
   edit — surface it and wait for confirmation.
4. Edit **only the smallest relevant file** inside
   `figure_registry/figures/FIG-NNN/` (e.g. `caption.tex` for a caption change,
   not `page.tex` or `manifest.yaml`).
5. `python3 figurectl.py validate FIG-NNN`
6. `python3 figurectl.py render FIG-NNN` — single-figure preview only, never
   the full book.
7. `python3 figurectl.py update-hashes FIG-NNN`
8. Leave `lock_status` unchanged unless the user explicitly asks to
   promote/lock/retire this figure — then
   `python3 figurectl.py set-lock FIG-NNN <draft|reviewed|locked|promoted|retired>`.
9. Do **not** run `figurectl.py build-book` (full rebuild) unless the user
   asks for the compiled book, or you've changed several figures in one
   session and need to check the whole thing compiles.

## What NOT to do

- Do not open or scan all 211 `figure_registry/figures/FIG-*/` directories to
  locate one figure — use `query`/`show` with a known or searched ID.
- Do not regenerate `figure_pages.tex` (legacy v1, frozen, not a generator
  target anymore).
- Do not treat a figure's `reviewer_status: DONE` as evidence that the
  underlying scientific gap is closed. Figure-level coverage and the
  project's scientific red lines (A1/A3/A5/C9 convergence/multi-seed gates)
  are tracked separately — see the Obsidian vault note
  `MigrationBench-Reviewer-Response-Tracker` for the authoritative scientific
  status. A `DONE` figure only means the figure supports the already-narrowed
  reviewer response text.

## Reference

`figure_registry/README.md` and `figure_registry/SCHEMA.md` document the full
schema and the lock lifecycle (`draft -> reviewed -> locked -> promoted ->
retired`), enforced by `figurectl.py validate`.
