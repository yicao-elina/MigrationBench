# C13 empty-stub audit

Audit date: 2026-09-29. Scope: tracked release files (`git ls-files`), excluding
the local `.venv`, caches, generated LaTeX auxiliaries, and cluster scratch.

## Repository alignment

Before this repair, local `main` was at `8e367781` and tracked
`origin/main` at the same commit. Fetching GitHub showed that the remote default
branch had advanced to `6451ea5a` (PR merges for Skipjack/NEB controls), so the
local checkout was one remote commit lineage behind. The C13 repair is based on
the local canonical paper-reproduction checkout; it does not overwrite or
silently merge the unrelated remote changes.

## Current release

The only tracked zero-byte file is:

```
data/raw_compact/rockfish/mb_qehandoff2_sm_s44_30850445/slurm-30850445.err
```

It is an empty scheduler stderr artifact, not an advertised source file. The
advertised package is populated (`src/migrationbench`, 100+ source files), and
the prior `scripts/` source stubs are no longer present. `paper/manuscript.tex`
is not a current release path; the maintained manuscript sources are
`paper/sn-article-final.tex` and `paper/sn-article-SI.tex`.

## Historical finding and disposition

The pre-reproduction commit `43877c2` contained a zero-byte
`scripts/postprocessing/shap_analysis.py` and other placeholder script trees.
They were removed during the canonical repository migration. This release now
provides `scripts/shap_analysis.py`, an explicit-input surrogate workflow, and
`requirements-shap.txt`; it does not silently claim access to omitted cluster
checkpoints or the original 450-row prediction CSV.

## Verification commands

```bash
git ls-files -z | xargs -0 -n1 sh -c 'test ! -s "$0" && echo "$0"'
python scripts/shap_analysis.py --demo --output /tmp/migrationbench-shap-smoke
```

The remaining empty stderr file is retained for provenance and is explicitly
classified above rather than presented as executable code.
