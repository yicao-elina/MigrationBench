# C13 empty-stub audit

Audit date: 2026-09-29. Scope: tracked release files (`git ls-files`), excluding
the local `.venv`, caches, generated LaTeX auxiliaries, and cluster scratch.

## Repository alignment

GitHub was fetched before the C13 import and compared against `origin/main`.
Overleaf was also fetched before editing; its newer `e10d82a` commit was an
ancestor of the local manuscript branch. The final GitHub and Overleaf tips are
recorded in `docs/audit/overleaf_sync.md`.

## Current release

The only tracked zero-byte file is:

```
data/raw_compact/rockfish/mb_qehandoff2_sm_s44_30850445/slurm-30850445.err
```

It is an empty scheduler stderr artifact, not an advertised source file. The
advertised package is populated (`src/migrationbench`, 100+ source files), and
the prior `scripts/` source stubs are no longer present. The authoritative
coauthor-edited manuscript sources are mirrored under `paper/revision1/` from
Overleaf; the flat `paper/` files are retained for legacy build compatibility.

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
PYTHONPATH=scripts python scripts/shap_analysis.py \
  --xyz data/raw_compact/revision1/shap/sampled_trajectory_Cr2_temp.xyz \
  --predictions data/raw_compact/revision1/shap/sampled_trajectory_Cr2_temp_predictions_with_binding1.csv \
  --output /tmp/migrationbench-shap-smoke
```

The remaining empty stderr file is retained for provenance and is explicitly
classified above rather than presented as executable code. The X-FORCE input
hashes and the independently observed preprocessing discrepancy are recorded
in `docs/audit/shap_source_lineage.md`; they are not silently suppressed.
