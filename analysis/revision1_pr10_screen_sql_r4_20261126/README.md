# R4 PR #10 screening provenance snapshot

This is an additive SQLite snapshot and schema for the checked-in PR #10
failure-attribution screening capsule. It is separate from the R3 claim-matrix
and Rockfish snapshot directories.

The importer reads source bytes from a pinned Git tree (`origin/main` by
default), records each source file's SHA-256, Git blob ID and size, and stores
each CSV/TSV data row as JSON with its 1-based data-record ordinal (the header
is not counted). Duplicate CSV headings are preserved with `__duplicate_N`
suffixes in the stored row JSON. It also imports normalized MD runs and frames, canonical
candidates and execution aliases, registered checkpoints, static model
evaluations, selection batches, claims, and panel-level figure evidence links.

Build from the repository root:

```sh
python3 analysis/revision1_pr10_screen_sql_r4_20261126/import_pr10_screen.py \
  --source-ref 19708d74d14103832e4f86b51b8b5d8875175cc3
```

The importer uses `git show` against this immutable source commit, so unrelated
uncommitted working-copy changes are not imported. `--output` can direct the
build to another path. The default output is `snapshot.sqlite` in this folder.

## Interpretation boundaries

- The canonical source trajectory model and the checkpoint evaluated on a
  candidate are represented separately. Candidate evaluation is a static MLIP
  single-point result, not a DFT reference result.
- The database records the source campaign's manifest status and limitations.
  The 120-atom prepared QE structures are reconstructions, and the source
  campaign has no completed DFT result.
- Figure panel (a) is schematic. Panels (b)–(d) summarize paired static MLIP
  evaluations. The force disagreement statistic is descriptive and is not
  calibrated uncertainty or causal evidence.
- External trajectories, original checkpoints, QE outputs and full scheduler
  trees are not copied into this SQLite file. The database preserves their
  declared URIs and available hashes; current retention is not inferred.
- No DUAL-X performance gain or model-matrix equivalence is asserted. This
  snapshot describes the MigrationBench Cr–Sb–Te campaign and three registered
  checkpoints only.

## Contents

- `schema.sql`: versioned relational schema and trace views.
- `import_pr10_screen.py`: deterministic importer pinned to a Git ref.
- `queries.sql`: candidate, claim, figure, matrix-count and DFT-status queries.
- `snapshot.sqlite`: generated database.
- `snapshot_manifest.json`: resolved source commit, row counts and integrity
  receipt.
- `ROCKFISH_ARCHIVE_RECEIPT_R4.json`: Rockfish path and byte/hash confirmation.

The R4 SQLite is also archived on Rockfish at
`/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_pr10_screen_snapshots/20261126_r4/snapshot.sqlite`
(17,055,744 bytes; SHA-256
`35eccb46790ff1d576d8598bade50cff4e1051e1e4b858bb19f8d1a5fa8540ec`). The
archive receipt and `SHA256SUMS` are stored beside it.

The importer checks SQLite `quick_check`, foreign keys, and key campaign counts
(9 MD runs, 639 candidates, 3 checkpoints, 1,917 static evaluations, four
figure panels, and all source claims) before writing the snapshot manifest.
It also checks one successful row per candidate/checkpoint cell and verifies
that all three recorded evaluation input hashes match the candidate's recorded
prepared-structure hash where the latter is available. Claim references that
cannot be resolved uniquely remain explicit unresolved or ambiguous rows. The
current import has 2 exact-path references, 12 unique-basename references, and
9 unresolved/external references. The manifest hashes the database, schema,
importer, and query file.
