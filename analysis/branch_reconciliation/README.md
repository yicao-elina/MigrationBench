# Git branch payload reconciliation

`PR20_SUPERSEDED_RELEASE_BRANCH_FILE_MAP_20261125.tsv` maps every path in the older `release/revision1-submitted-20261110` tree that is absent from PR #20 candidate `de702829`.

- All source paths are anchored to Git commit `f924ca6c11236c4b75469e290a0abff884c56c02`; that commit is already an ancestor of PR #20, so the content remains in Git history even though some paths are not in the candidate tree.
- Each mapping includes the original Git blob, exact size and SHA256, and a destination. Exact blob matches in the PR #20 tree or previously verified Rockfish archives point there directly.
- The remaining unique payloads are content-addressed in `rockfish:/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_branch_retirement_20261125/blobs/` and have a local/remote SHA256 receipt.
- The two raw QE `neb.out` outputs, six MLIP NEB attempt logs, Slurm records and old source sidecars remain available; no historical manuscript version is chosen as current by this map.

The table can be joined to `release/revision1_submitted_20261110.imports.EXTERNAL_STORE_MANIFEST.tsv` in the Revision 1 asset registry by source path. `analysis/branch_reconciliation/branch_payload_lookup.sql` gives example queries.
