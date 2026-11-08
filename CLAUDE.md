# migrationbench-paper-repro

Entry point for agent sessions in this repo. Read nested `CLAUDE.md` files
when working inside the directory they live in.

- `paper/revision1/CLAUDE.md` — **figure registry editing rules.** Read this
  before touching any figure, caption, or Results/Discussion text tied to a
  figure in the Wiley revision manuscript. Core rule: use `figurectl.py`
  (`show` / `query` / `validate` / `render` / `update-hashes` / `set-lock`)
  scoped to a single `FIG-NNN`; never bulk-scan or regenerate the 211-figure
  `figure_pages.tex` (frozen v1) or hand-edit the generated `figure_book.tex`.
  A `/figure-edit FIG-NNN <change>` slash command is also available
  (`.claude/commands/figure-edit.md`) for the common "change one figure" case.
