# WFC and mixing restart residual review — 2026-11-24

This evidence records a metadata-only refresh and exact-name QE restart check
under the accessible personal roots `/scratch16/pclancy3/yi` and
`/data/pclancy3/yi`. The scan visited 114,907 directories with zero traversal
errors. `revision1_migrationbench_runs` contains no basename with `wfc` or
`mix`; none of the 3,817 WFC and 481 mixing restart files reported in the
2026-11-22 threshold scan remained.

The exact QE-style scan found eight zero-byte `.mix1`–`.mix4` placeholders in
two Cr-in-gap directories and no nonempty restart payloads. After an all-user
`squeue` check and exact identity checks for each path, all eight placeholders
were deleted. The postcheck found zero remaining candidate paths. The receipt
records zero logical bytes freed and is retained as
`delete_empty_mix_receipt.json`. Its queue statement is a summary only; the
raw queue output was not retained. No personal quota-after read or allocated
byte measurement is available.

The summary JSON is a point-in-time pre-delete capture: its eight matches were
observed at 06:08:38Z. The deletion receipt is timestamped 06:24:04Z. A separate
read-only exact-path check at 08:58:26Z found all eight paths absent with zero
errors; see `postdelete_path_check_20261124.json`. The catalog records the
receipt and postcheck as separate evidence sources, preserves the earlier
`HOLD` rows as historical decisions, and records the follow-up batch as
`DELETE/VERIFIED`.

The broad-token inventory still has 3,715 WFC/mix name matches across the two
accessible roots. These include `projwfc`/PDOS output and software/library
paths, so the token alone is not an exact QE restart match. They were retained.
The complete metadata-only observations are in
`wfc_mix_residual_inventory_20261124.jsonl`; the small TSV/JSON files summarize
exact candidates and post-delete verification. Payloads were not read or
hashed. This is not a complete `/data` or `/scratch4` mount inventory, and no
personal quota-after value was available.
