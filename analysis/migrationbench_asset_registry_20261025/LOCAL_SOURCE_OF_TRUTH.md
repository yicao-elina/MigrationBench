# Local source-of-truth policy

The authoritative working copy of the MigrationBench registry is this
directory:

`/Users/alina/Project/migrationbench-paper-repro/analysis/migrationbench_asset_registry_20261025`

## Operating rule

1. Write and validate model names, task coverage, chart links, aliases, and
   provenance locally first. SQLite is the primary write store; the local UI
   writes controlled status changes there.
2. Run `python3 sync_registry_mirror.py --check` before staging.
3. After review, run `python3 sync_registry_mirror.py --stage`. It first
   exports the current SQLite state to `registry_data.json`, then creates an
   immutable timestamped snapshot under the dedicated Google Drive folder.
4. Update the native Google Sheet only from the validated local snapshot. The
   Sheet is a collaboration/readout surface, not the primary write target.

The filesystem sync does not overwrite the native Google Sheet. It keeps a
portable SQLite database, workbook, JSON/CSV exports, schema, query examples,
and quality report in Drive so the files remain recoverable even when a
mounted Drive source is temporarily unavailable.

`build_registry.py` is a recovery/rebuild tool for reconstructing SQLite from
the portable JSON snapshot. Routine edits should happen in SQLite (through
the local UI or a reviewed SQL change), followed by `--stage`; do not edit a
stale JSON file and then rebuild over newer database work.

## Why this boundary exists

The paper repository has extensive unrelated uncommitted research changes.
The registry is therefore isolated in its own directory and can be committed
or reviewed independently. Raw trajectories and model checkpoints remain
outside this mirror; they are referenced by immutable locators and provenance
records rather than copied blindly.

## Native Google Sheet

Current collaboration mirror:

<https://docs.google.com/spreadsheets/d/11USdbpky-S3LtiPInVlndrHwWNO9XDfd57_snm1TPr0/edit>

The Sheet update should be treated as an explicit review step after a local
registry build. Do not use a blind two-way sync: that would allow a stale or
partially loaded Drive file to overwrite a validated local record.
