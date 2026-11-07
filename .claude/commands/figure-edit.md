---
description: Targeted single-figure edit in the MigrationBench figure_registry — never scans all 211 figures
---

Edit figure **$1** in the MigrationBench figure registry
(`paper/revision1/figure_registry/figures/$1/`). Requested change: $2

If `$1` does not look like a `FIG-NNN` id, first run
`cd paper/revision1 && python3 figurectl.py query "$1"` to resolve it, then
restart from step 1 below with the resolved id.

Follow this exact sequence, scoped to figure `$1` only — do not open or
rescan any other `figure_registry/figures/FIG-*` directory, and do not touch
`figure_pages.tex` (frozen v1) or hand-edit `figure_book.tex` (generated):

1. `cd paper/revision1 && python3 figurectl.py show $1` — read current state,
   note `lock_status`.
2. If `lock_status` is `locked` or `promoted`: tell the user before editing.
   The change will intentionally fail validation until they review it and you
   run `update-hashes` — do not treat that failure as something to silently
   revert.
3. Edit only the smallest relevant file under
   `figure_registry/figures/$1/` (`caption.tex`, `results.tex`,
   `discussion.tex`, `reviewer.tex`, `evidence.tex`, `figure.tex`, or
   `page.tex`) to make this change: $2
4. `python3 figurectl.py validate $1`
5. `python3 figurectl.py render $1` — single-figure preview only.
6. `python3 figurectl.py update-hashes $1`
7. Leave `lock_status` unchanged unless the user explicitly asked to
   promote/lock/retire this figure — then
   `python3 figurectl.py set-lock $1 <state>`.
8. Do not run `figurectl.py build-book` unless the user explicitly asks for
   the compiled book.
9. Report what changed, which single file was edited, and the resulting
   `lock_status`.
