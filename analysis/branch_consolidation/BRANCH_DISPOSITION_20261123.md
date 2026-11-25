# Branch consolidation disposition — 2026-11-23

This record is for the Revision 1 consolidation candidate on branch codex/revision1-main-consolidation-20261123. Repository refs were fetched from https://github.com/yicao-elina/MigrationBench after the submitted release merge.

## Included branch histories

- release/revision1-submitted-20261110 at f924ca6c is merged into the candidate. The submitted paper source remains canonical.
- PR #10 head codex/document-mlip-639-selection-provenance at 51cb06f7 is merged. Its 639-candidate evidence bundle is retained. The three conflicting Overleaf manuscript/source-ledger files are preserved as a hash-manifested candidate snapshot under paper/revision1_candidates/pr10_overleaf_candidate_20261123/source; paper/revision1 remains the submitted version.
- PR #12 head paper/cr-neb-evidence-workflow-20261016 at 0505125b is merged. Its dated monitoring records are retained, with an added note that the submit scripts are historical and must not be rerun.
- PR #13 head codex/dft-mace47-evidence-package-20261023 at 2a40d7a1 is merged. The stale PENDING_COMPUTE status for the completed 12-by-47 static source-geometry calculation is reconciled against the package scheduler and quality records; its diagnostic-only scientific boundary remains explicit.
- PR #20 head integrate/final-paper-20261106 at 65fa4fc0 and the submitted release descendant are ancestors of this candidate. Other open PR branches observed in the fetched refs are ancestors of that consolidated history or the three explicit PR integrations above.

## Superseded revert branch

The only fetched branch tip not yet ancestral to the candidate was revert-16-codex/paper-figures-repro-release at 360c0acf. It contains one unassociated revert commit with no pull request; the commit removes 47 paths from the PR #16 paper-figure release, including its figure provenance package and registry. The candidate already contains PR #16 through the stacked paper-figure history and later PR #17–#19 work. The revert was therefore reviewed as an unmerged trial and is incorporated as history only; its tree deletion is not applied. The merge commit message records this choice.

## External evidence boundary

Rockfish raw data, checkpoint/model weights, and restart cleanup receipts remain in the organized archive and cataloged by path/hash. The Git candidate contains compact evidence, scripts, manifests, paper source, and archive pointers. This record does not imply paper acceptance or resolve scientific limitations already recorded in each package.
