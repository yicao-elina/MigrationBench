# Overleaf/GitHub manuscript synchronization

The authoritative revision manuscript snapshot is `paper/revision1/` in this
repository. It is an exact mirror of the Overleaf clone's `revision1/`
directory, including the coauthor-edited manuscript, SI, response letter,
tables, figures, processed inputs, and audit script.

Synchronization record (2026-09-29):

| Remote | Commit | Verification |
|---|---|---|
| Overleaf `https://git.overleaf.com/693b515e50f0071cb03b505c` | `0b215fe` | fast-forward from `c2c4db8`; `revision1/{sn-article-final,sn-article-SI,response_to_reviewers}.tex` compiled cleanly with local `pdflatex` before push |
| GitHub `https://github.com/yicao-elina/MigrationBench.git` | `33ffe23` on branch `fix/reviewer-response-a2-c10-c13-writeback` (PR #5) | not yet merged to `main`; PR must be reviewed/merged separately from the Overleaf push |

The Overleaf source was fetched before the mirror import. The earlier online
commit `e10d82a` was an ancestor of the local Overleaf branch, so the C13
changes were fast-forwarded and pushed without overwriting an independent
coauthor branch. A subsequent minimal caption fix was compiled successfully
and pushed as `c2c4db8`. The flat `paper/` files remain for legacy reproduction
commands; new manuscript edits must be applied to `paper/revision1/` first.

A second synchronization (same day) fixed two stale `\reviewermap` tags
(`REV-C8`, `REV-A6/REV-C13`) and replaced the A2/C10/A3/A5 `[PLACEHOLDER: ...]`
markers with the actual 2026-09-29 grouped-split closeout numbers. This was
pushed to GitHub as commit `33ffe23` (PR #5, not yet merged) and mirrored to
Overleaf as commit `0b215fe` (fast-forward, no divergence). Both remotes were
compiled locally before pushing. Because the GitHub side is an unmerged PR
branch while Overleaf `main` already has the content, `origin/main` on GitHub
and Overleaf `main` will disagree on these three files until PR #5 is merged
— resolve by merging the PR, not by re-pushing Overleaf.

For the next synchronization, fetch both remotes, compare `HEAD...origin/main`,
inspect changed files, and only then merge or copy. Never replace an online
revision wholesale from an older local checkout.
