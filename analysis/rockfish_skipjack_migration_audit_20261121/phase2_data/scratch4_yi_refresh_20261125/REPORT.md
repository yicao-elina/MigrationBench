# Fresh metadata inventory: `/scratch4/pclancy3/yi`

Captured **2026-11-25 10:42:33 UTC** on Rockfish `login02`. This metadata-only walk traversed exactly `/scratch4/pclancy3/yi`, did not follow symlinks, did not read file contents or compute hashes, and made no changes. It completed in 0.38 seconds with zero traversal/stat errors.

## Counts and bytes

| Measure | Result |
|---|---:|
| Regular files | 2,875 |
| Directories below root | 58 |
| Symlinks / other entries | 0 / 0 |
| Descendant entries | 2,933 |
| Logical file bytes | 382,958,312 |
| Allocated bytes (path sum) | 403,117,056 |
| Regular-file owners | uid 3901 / gid 3001 for all 2,875 files |
| In-tree hardlink groups | 0 |
| Traversal/stat errors | 0 |

No files are at least 100 MiB. The complete large-file list is `large_files_ge_100MiB.tsv` (header only). The top 100 file paths by logical size are in `top_100_files.tsv`.

The case-insensitive filename scan found **zero** names containing `wfc` or `mix`; see the empty `wfc_mix_filename_candidates.tsv`. This is only a filename result, not a content scan.

## Current job/ownership boundary

All regular-file metadata under this exact root reports uid 3901/gid 3001. A live owner-scoped Slurm workdir check could not be performed because `squeue` is absent from the Rockfish login PATH (`command not found`, observed 2026-11-25); no current job overlap is asserted. This inventory clears no cleanup action.

## Evidence

- `summary.json`: compact counts, ownership, byte totals and artifact file references.
- `inventory.json`: full summary, top 100, WFC/mix name candidates, and error list.
- `top_100_files.tsv`: detailed metadata for the 100 largest file paths.
- `large_files_ge_100MiB.tsv`: files at least 100 MiB; none found.
- `wfc_mix_filename_candidates.tsv`: filename matches; none found.
