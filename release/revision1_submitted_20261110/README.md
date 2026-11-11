# MigrationBench — Revision 1 as submitted (2026-11-10)

This folder is the entry point for finding, checking and regenerating every
figure and table in the Revision 1 submission to Wiley. Start here at the next
revision.

## What was submitted

| Part | Source in this repo | Built from commit |
|---|---|---|
| Response letter (Tables R1–R11) | `paper/revision1/response_to_reviewers.tex` | `334260c` |
| Main text (Figs. 1–7) | `paper/revision1/sn-article-final.tex` | `334260c` |
| Supplementary Information (Figs. S1–S5, Tables S1–S11) | `paper/revision1/sn-article-SI.tex` | `334260c` |
| Graphical abstract, hi-res figure uploads | Canva exports (see `items/main/`) | — |
| Submitted PDF as received from Wiley (63 pp.) | `submitted/MigrationBench_R1_submitted_20261110.pdf` | — |

The TeX sources are byte-identical to the Overleaf export of 2026-11-09
(`MigrationBench_overleaf (2).zip`). The recompiled text matches the submitted
PDF (word-sequence similarity 0.9996 for the main text). The `overleaf` git
remote in this repo is a different, stale Overleaf project and is not the source.

Build the PDFs:

```bash
cd paper/revision1 && export TEXINPUTS=.:..//: BSTINPUTS=.:..: && latexmk -pdf sn-article-final.tex sn-article-SI.tex response_to_reviewers.tex
```

## Where to look

| File | Use it to |
|---|---|
| `RESULTS_INDEX.tsv` | Look up any figure panel or table: final asset + sha256, generator script, plotted data, upstream data, raw location, models, calculation type, status. **This is the maintenance list.** |
| `items/<main,si,response>/<item>/PROVENANCE.md` | Read the same information per item; item-specific sources sit next to it in `source/` / `upstream/`. |
| `PIPELINE.md` | Follow the end-to-end chain DFT/AIMD → split → training → evaluation (pointwise, MD, NEB, sliding) → features → SHAP → figures. |
| `MODEL_REGISTRY.tsv` | Find every checkpoint (used and unused), its sha256, config, training data and location. |
| `STORAGE_LOCATIONS.tsv` | Find every raw-data location outside git (Rockfish, local archive, iCloud), with the migration target. |
| `ROCKFISH_MIGRATION_PLAN.tsv`, `tools/rockfish_to_skipjack_rsync.sh` | Move Rockfish data to Skipjack (not yet executed; see below). |
| `KNOWN_ISSUES.md` | Read the inconsistencies found in the submitted version, to fix at the next revision. |
| `imports/` | See which outside-repo files were imported (`IMPORT_MANIFEST.tsv`), which heavy binaries live in the local archive store (`EXTERNAL_STORE_MANIFEST.tsv`) and which were still iCloud-only (`ICLOUD_DATALESS.tsv`). |

The same tables are in the SQL registry
`analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite`
(tables `rev1_results_index`, `rev1_storage_locations`, `rev1_model_registry`,
`rev1_imported_files`, `rev1_external_store`; migration
`migrations/002_revision1_submission_trace.sql`). Example:

```bash
sqlite3 -header -column analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite "select item_id, panel, status, generator from rev1_results_index where item_id='Fig4'"
```

## Status legend

| Status | Meaning |
|---|---|
| `FULL` | Script and data in the repo regenerate the item. |
| `FULL_DATA_MANUAL_LAYOUT` | Every data panel regenerates; the final page was relabelled or composed in Canva. |
| `FULL_VALUES_HAND_ROUNDED` | The generator reproduces the values; the submitted table was rounded by hand. `tools/verify_table_rounding.py` proves they agree. |
| `MANUAL_SCHEMATIC` | Drawing with no data content (Canva). |
| `FULL_CAPTION_MISMATCH`, `FULL_VALUES_STALE_RELEASE`, `FULL_VALUES_WORDING` | Reproducible, with an inconsistency listed in `KNOWN_ISSUES.md`. |
| `PARTIAL`, `MISSING_SOURCE` | A piece is still missing (see the row note). |

## Checks

```bash
python3 release/revision1_submitted_20261110/tools/build_index.py      # rebuild index, item pages, SQL tables
python3 release/revision1_submitted_20261110/tools/verify_trace.py     # every cited path exists, asset sha256s match
python3 -m pytest tests/test_dft_neb_reference.py                      # Fig. 4 DFT reference = QE neb.out
```

Regenerate the SI tables, then compare them with the submitted (hand-rounded) ones,
in the reference environment `release/revision1_camera_ready/requirements-current.txt`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python release/revision1_submitted_20261110/items/si/_shared_generators/audit_current_figures.py
```

```bash
python3 release/revision1_submitted_20261110/tools/verify_table_rounding.py --submitted paper/revision1/tables --regenerated release/revision1_submitted_20261110/items/si/_shared_generators/rerun_output/tables
```

## Data outside git

- **Rockfish** (read-only survey 2026-11-10): Revision 1 campaigns are in
  `/scratch16/pclancy3/yi` (about 5.3 TiB, of which about 150 GB is worth keeping once QE
  wavefunctions and per-epoch checkpoints are dropped); training pools, AIMD and
  legacy references are in `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3` (about 40 GB). The scratch16
  purge policy was not verified, so treat it as at risk. All 12 trained paper
  models were re-hashed on Rockfish and match the recorded sha256.
- **Skipjack**: target for the migration and backup; not verified (needs
  interactive login). Run `tools/rockfish_to_skipjack_rsync.sh --dry-run` on
  Rockfish first.
- **Local archive store** `/Users/alina/Project/migrationbench-archive/` (about 8.5 GB), to
  back up to Skipjack together with the Rockfish data:
  - `repo_mirror/`: heavy binaries and raw run outputs at their repo-relative
    paths (`imports/EXTERNAL_STORE_MANIFEST.tsv`, `imports/ARCHIVED_UNTRACKED_RAW.tsv`);
  - `downloads_20261110/`: everything MigrationBench-related that was in `~/Downloads`
    (final Overleaf zip, Canva backups, Figure 1 kit, drafts, public MACE checkpoints),
    per-file sha256 in `imports/DOWNLOADS_ARCHIVE.tsv`.
- **ChatGPT workspace** `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench` (11 GB, iCloud,
  partly evicted): everything the trace needs is imported. Keep it as a cold copy until it
  has been archived in full; deleting it also deletes the iCloud-only files.
- **Legacy repo** `/Users/alina/Project/26MIgrationBench` (2.6 GB, no remote, uncommitted
  changes): the only file this release needs (`legacy_61` QE NEB) is imported.
