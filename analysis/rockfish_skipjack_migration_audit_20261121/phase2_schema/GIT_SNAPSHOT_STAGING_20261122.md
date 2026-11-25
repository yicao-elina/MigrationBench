# Git snapshot staging plan (2026-11-22)

This is an additive, review-first staging importer for the saved read-only Git reports about `/data/pclancy3/yi`. It has **not been run** and has not opened or queried `provenance_catalog.sqlite`. The catalog remains untouched by this staging work.

## Sources and expected observations

The importer reads these immutable local report snapshots from `../phase2_data/`:

- `git_reconciliation_review_20261122.jsonl` and `.tsv`: 25 repositories, 295 cached remote-tracking refs, 24 configured remote observations (48 fetch/push URL lines; 24 repositories with a remote; one repository has no remote).
- `git_reconciliation_review_20261122_branches.tsv`: 30 local branch refs.
- `lammps_repo26_git_snapshot_20261122.json`: one supplemental repository snapshot for `/data/pclancy3/ycao73_home/flare/lammps-flare/lammps`, a shallow upstream LAMMPS `develop` clone at `b620c52b58f1b5810f3c78e23ab34283f3a2d700`. Its cached `origin/develop` matches HEAD. The importer records 1 modified tracked file and 406 untracked source paths, marks the upstream base third-party with user customization, and leaves current remote tip unknown because no live query is represented.
- `first_party_git_branch_graph_20261122.jsonl`: 16 first-party repositories, 21 later local branch-ref observations and 4 named branch-pair comparisons.

The five source files are registered with SHA-256 checksums. `as_of_utc` is derived from the source report's filesystem modification time, so it is a capture-time proxy rather than an independently recorded time for each Git command. The later graph report becomes a separate status observation for its 16 repositories. Expected totals after extension: 26 repository snapshots, 31 local branch refs, 297 cached remote-tracking refs, and 25 configured remote observations. The 16 later first-party graph snapshots remain separate.

## Schema additions

`004_git_snapshot_observations.sql` adds tables for repository status snapshots, remote URL/fetch-state observations, ref comparison context, and local branch-pair graph comparisons. It supplements the base `git_repository` and `git_ref_observation` tables. `common_base_ref_or_oid` preserves the audit report's raw value because identical branches sometimes record a ref name there, while non-identical comparisons carry a commit OID.

The script records dirty tracked, staged, unstaged, untracked, and ignored counts where the source provides them. For LAMMPS, the dirty source manifest contains 388 full-file hashes and 19 symlink targets; the importer derives an SHA-256 over each symlink target string and labels it as link-payload identity. It retains cached tracking-ref OIDs as local metadata and does not interpret them as current server tips.

## Remote freshness policy

The source reports do not contain authoritative, completed fetch outcomes or a verified `ls-remote` result. For all 25 configured remotes, the staged fetch state is `NOT_RECORDED`, the current tip is `UNKNOWN_NOT_VERIFIED`, and `current_tip_sha` is null. This is deliberate: no remote tip is inferred from a local cached tracking ref. The sole repository without configured remotes is recorded as `NO_REMOTE_CONFIGURED` / `NOT_APPLICABLE`.

## Running policy

By default, `stage_git_reconciliation_snapshots.py` only reads and validates its report inputs, prints intended counts, and does not open the catalog. Applying requires an explicit `--apply`; that mode is intended only after the parent importer finishes and reviews the staging SQL. It uses `INSERT OR IGNORE`/stable keys for repeatable appends, checks the report hashes again before writing, and runs a foreign-key check within its write transaction. No `git fetch`, remote query, ref update, or Rockfish command is performed.

No test or dry-run invocation was made for this staging task; the code and SQL received manual review only. Reviewers should inspect the SQL and importer before applying it to the catalog.
