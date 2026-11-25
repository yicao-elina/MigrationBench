# `/scratch4/pclancy3/yi` R02 cache/build/temp review

**Inventory window:** 2026-11-22T19:12:37+00:00 to 2026-11-22T19:12:38+00:00; metadata-only R02 scan.

## Decision

No user-owned cache, build, temporary, editor cache, compiled object, or bytecode files were found in R02. No `wfc<number>.dat` or mixing restart files remain. There are no deletion candidates from this review.

The complete R02 inventory contains 2878 regular files, 55 directories, zero symlinks, zero special files, and zero scan errors; it totals 597580272 logical bytes and 617763328 allocated bytes. The user-owned file count is 2878. The WSe2 cleanup reduced this tree from R01's 2,886 files / 5,642,682,016 logical bytes to R02's 2,878 files / 597,580,272 logical bytes.

Seven `charge-density.dat` files remain (four Cr–Sb₂Te₃ phonon runs and three WSe₂ relaxations), totaling 364,602,168 logical / 364,642,304 allocated bytes. These are retained QE outputs. The biggest is the 95,980,840-byte WSe₂ run-2 charge density; no remaining file is larger than 100 MB.

The phonon NAC scratch directory has 576 zero-length `pwscf.{bar,com,dvkb,dwf,ebar,prd}<n>` sidecars. All allocate zero blocks, so deleting them would not free storage; their lifecycle is not item-level mapped. Four other zero-byte files include an empty `combined_band_dos.dat`, `NAC/matdyn`, `NAC/BORN`, and `dmat.sb2te3`; preserve them until their calculation/output context is reviewed. No cleanup is recommended for any zero-byte file.

The historical path-overlap preflight recorded no active workdir under `/scratch4/pclancy3/yi`. Its queue export is excluded from this scope-limited package; the 2026-11-25 inventory is a later metadata-only refresh and does not clear any cleanup action.

## Method and evidence

- This review reads the R02 path ledger and metadata only; no file contents were read and no Rockfish paths were changed.
- Generated-file classification used path components for common cache/build/temp locations and exact suffixes for bytecode, compiled objects/libraries, editor swaps, backups, and temporary files.
- `scratch4_inventory_R02_20261122.jsonl` is the complete input snapshot; `scratch4_yi_r02_cache_decisions_20261122.tsv` gives category-level decisions and the seven retained charge-density paths.
- This review makes no assertion about current scheduler state. Recheck the exact target paths with a working Slurm client before any future cleanup.
