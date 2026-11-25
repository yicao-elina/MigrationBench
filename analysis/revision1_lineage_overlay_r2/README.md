# Revision 1 lineage overlay r2

This directory contains the versioned importer and SQL lookup queries for the additive Revision 1 lineage overlay archived on Rockfish. The canonical Git repository carries the import logic and query interface; the SQLite payload and mapped HPC evidence remain outside Git in the personal data archive.

## Archived snapshot

- Archive location: `<ROCKFISH_PERSONAL_ARCHIVE>/revision1_lineage_overlays/20261125_r2/`
- SQLite: `migrationbench_r1_lineage_overlay_20261125_r2.sqlite`
- Size: 8,740,864 bytes; mode `0600`
- SHA-256: `7a8dc7505ea7e8d5646ad79074f8db03e2755bcc630167ac85fc2bb4f12389a6`
- Parent r1 overlay SHA-256: `e1dedf0e4e7aba93c41450e8426a16e9fedc167c4ad8cc4ad52f7e6667ef1678`
- Integrity: `quick_check=ok`; foreign-key violations `0`

The r2 snapshot has 1,971 lineage artifacts, 13 MD attempt observations, 24 campaign records, 226 manuscript result cells, 894 result-artifact links, and 37 run-artifact links. It registers 1,866 hash-identified input/run metadata files, six located legacy QE NEB outputs, 37 MD continuation files, three Lmax1 G3 campaign summaries, and eight archived audit sidecars.

## Evidence boundaries

- The 226 manuscript result statuses from r1 are unchanged: 201 `PARTIAL`, 24 `CONFLICT`, and 1 `MISSING`.
- The six legacy QE outputs map to twelve Table S5 forward/reverse cells by recovered path ID. Those links stay `PARTIAL` because no expected SHA was recorded in the earlier crosswalk and no exact producer attempt was established.
- Manifest SHA matches are recorded as candidate evidence only; they do not establish which attempt generated a result. The database contains no `PRODUCED_BY` result/run links.
- MD scheduler and run statuses remain `UNKNOWN`; observations from logs and the 1 ns output chronology are retained in notes. Scientific acceptance is `PROVENANCE_PARTIAL`.
- Lmax1 entries are campaign summaries with their diagnostic-only/candidate-only boundaries. The import does not synthesize per-row runs or promote results to publication status.
- The central provenance catalog r11 and r1 overlay are immutable parents. This is a separate additive overlay revision.

## Rebuild

Obtain the archived r2 bundle, including its `source/` and `maps/` directories. Run:

```bash
python3 import_revision1_lineage_overlay_r2.py \
  --bundle-dir "$R2_BUNDLE" \
  --rockfish-yi-root "$ROCKFISH_YI_ROOT"
```

The importer refuses to overwrite the destination and validates SQLite integrity before writing its receipt. With the archived source and maps, the rebuilt SQLite SHA-256 matches the snapshot above.

## Queries

Run [`queries.sql`](queries.sql) against the SQLite database in read-only mode. QL07–QL12 cover manifest inventory, Table S5 legacy outputs, MD attempts/artifacts, Lmax1 campaign boundaries, and the absence of deterministic producer links.
