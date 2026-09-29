# Overleaf/GitHub manuscript synchronization

The authoritative revision manuscript snapshot is `paper/revision1/` in this
repository. It is an exact mirror of the Overleaf clone's `revision1/`
directory, including the coauthor-edited manuscript, SI, response letter,
tables, figures, processed inputs, and audit script.

Synchronization record (2026-09-29):

| Remote | Commit | Verification |
|---|---|---|
| Overleaf `https://git.overleaf.com/693b515e50f0071cb03b505c` | `3bf53db091463d09d6bf339d2e9b35c5ba563e97` | local `main` = `origin/main`; no uncommitted changes |
| GitHub `https://github.com/yicao-elina/MigrationBench.git` | C13 release commit recorded after mirror import | `git fetch origin --prune` and explicit remote comparison required before every update |

The Overleaf source was fetched before the mirror import. The earlier online
commit `e10d82a` was an ancestor of the local Overleaf branch, so the C13
changes were fast-forwarded and pushed without overwriting an independent
coauthor branch. The flat `paper/` files remain for legacy reproduction
commands; new manuscript edits must be applied to `paper/revision1/` first.

For the next synchronization, fetch both remotes, compare `HEAD...origin/main`,
inspect changed files, and only then merge or copy. Never replace an online
revision wholesale from an older local checkout.
