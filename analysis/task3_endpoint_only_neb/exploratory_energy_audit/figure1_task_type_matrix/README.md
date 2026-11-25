# Figure 1 task-type matrix: Cr pathway reclassification + Cu/Li/Na/Cr composite

Generated 2026-10-03. Pure analysis/figure-building on the external repos; the
Obsidian vault was only read for context, never written.

## 1. What this directory contains

| File | What it is |
|---|---|
| `cr_pathway_reclassification.csv` | Old numeric Cr pathway ID -> new {QLs, gap-diff, gap2QL} category, with confidence level and the geometric evidence used. |
| `cr_unit_geometry_sanity_checks.csv` | SHA256 + barrier cross-checks proving the QE source data is byte-identical across the two repos that reference it, and that the re-extracted QE barriers match the independently-stored table to 6 decimal places. |
| `cr_foundation_path_image_profiles.csv` | Per-image, per-model relative energies (eV) for the 5 Cr paths that have foundation-model data (1-2, 1-3, 1-4, 1-5, 1-7), plus the matching QE path-table trace. |
| `cr_foundation_path_barriers.csv` | Per-path, per-model barrier-like value (`max(E)-E[0]`) vs. the QE reference, with signed/absolute error. |
| `figure1_task_type_matrix.png` / `.pdf` | The draft composite "Figure 1" matrix: 3 schematic placeholders + 8 existing Cu/Li/Na panels (unchanged, reused from `../corrected_dft_foundation_models/`) + 5 new Cr panels. |

Nothing in `../corrected_dft_foundation_models/` was modified.

## 2. Where the Cr source data came from

Three previously-undocumented local locations turned out to hold real Cr-Sb2Te3
NEB data (none of this was in the two repos named in the original task — it
required following a `source_file` provenance pointer inside
`migrationbench-paper-repro/data/published/neb_barrier_atlas/barrier_atlas.json`):

1. **`/Users/alina/Project/26MIgrationBench/data_raw/historical_neb/legacy_61/{1-2..1-8}/`**
   — the actual QE NEB geometry+output for seven 61-atom Cr-Sb2Te3 paths
   (`pw_1.in`, `sb2te3.xyz` with 5 images x 61 atoms, `neb.out`). This is the
   geometry data used for the reclassification in step 3 below. No `1-1`
   directory exists here or anywhere else searched.
2. **`/Users/alina/Project/26MIgrationBench/data_raw/historical_neb/layer_81/neb_{1..5}/`**
   — a chained 81-atom-cell NEB series (10 images each). Found but **not**
   confidently classified (see caveats).
3. **`/Users/alina/Documents/Codex/foundation_grouped_benchmark_20261001/`**
   — a complete, already-run local foundation-model campaign: 10 MACE
   variants (`agnesi_medium`, `agnesi_stress_{large,medium,small}`, `mh0`,
   `mh1`, `mp_0b3_medium`, `mpa0_medium`, `omat0_{medium,small}`) evaluated as
   single points on paths 1-2, 1-3, 1-4, 1-5, 1-7 via
   `evaluate_neb_foundation.py` (uses `ase.io.read(xyz, index=':')` on the
   literal saved path file — the same correct approach used to fix the
   Cu/Li/Na endpoint-interpolation bug). Results already copied into
   `migrationbench-paper-repro/data/processed/foundation_grouped/neb/*.json`
   (50/50 files present). No foundation-model data exists locally for 1-6,
   1-8, or any 81-atom path.

The vault's `MigrationBench-Reviewer-Response-Tracker.md` pointer to
`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/...` on Rockfish is indeed not
locally mounted — but it didn't need to be, because the exact same campaign's
outputs (and the underlying QE files) are mirrored locally in the three
locations above. The foundation-model JSONs themselves even retain the
original remote model path string
(`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/.../mace-mpa-0-medium.model`) as a
field, confirming where the run actually executed.

## 3. Reclassification method and result

All 7 of the 61-atom legacy paths (1-2 through 1-8) start from the **same**
Cr interstitial coordinate (x=2.107, y=3.650, z=15.517 A in the path's own
cell), which sits at the midpoint of a van-der-Waals gap (z in roughly
13.9-17.1 A, flanked by two quintuple layers, each a clean Te-Sb-Te-Sb-Te
5-layer stack ~6.9-7.0 A apart from the next). This single shared starting
site is itself consistent with the old project's binary "in-gap" label for
the common Cr interstitial.

For each path, the migrating Cr atom's (x,y,z) was read from every one of the
5 (or 10, for 81-atom) saved NEB images, and classified by: (a) whether its
z-coordinate stays inside a detected large (>2.5 A) inter-layer gap across all
images, vs. crosses into a region with tight (~1.4-1.5 A) nearest-neighbor
spacing characteristic of being squeezed inside a QL's dense Te/Sb layers; and
(b) the lateral (x,y) displacement, as a secondary check for "diffuse within
gap" character. Full per-path evidence is in `cr_pathway_reclassification.csv`.

Result:

| Old ID | New category | Confidence |
|---|---|---|
| 1-1 | insufficient data — not found | n/a |
| 1-2 | gap2QL | high |
| 1-3 | gap2QL | high |
| 1-4 | gap2QL | high (shallower crossing than 1-2/1-3) |
| 1-5 | gap2QL | medium (crosses two QLs + two gaps; not a clean single crossing) |
| 1-6 | gap-diff | high |
| 1-7 | gap2QL | high — matches the PI's explicit mapping (1-7 = gap2QL) |
| 1-8 | gap-diff | low (lattice looks disordered/poorly converged in later images) |
| 81_neb_1..5 | insufficient data | n/a — different cell, heuristic not re-validated for it; conflicts with old doc's informal "deep penetration" tag, so not asserted |

**No old Cr path in this set maps to plain "QLs"** (migration fully inside one
QL's interior, never touching the gap) — all 7 share the same in-gap starting
site, so the taxonomy split here is only between gap-diff (stays in the gap)
and gap2QL (exits the gap into a QL). A genuine Cr "QLs" example, if one
exists, must come from a different source pathway not found in this search.

## 4. Was Cr data integrated into the comparison figure? Yes, partially.

Integrated: **1-2, 1-3, 1-4, 1-5, 1-7** (all gap2QL), each with QE + all 10
local foundation models, geometry-aligned image-by-image.

Unit/geometry sanity checks performed (see `cr_unit_geometry_sanity_checks.csv`):
- The `neb.out` files used both for pathway classification
  (`26MIgrationBench/.../legacy_61/{path}/neb.out`) and for the campaign
  that fed the foundation-model single-point runs
  (`foundation_grouped_benchmark_20261001/data/dft_paths/{path}-neb.out`)
  are **SHA256-identical** for all 5 paths — same underlying QE run.
- `neb.out`'s "image energy (eV)" table is QE's native eV output (distinct
  from the `.pathN` Hartree bug fixed for Cu/Li/Na). Re-extracting the
  forward barrier from this table reproduces
  `migrationbench-paper-repro/data/processed/dft_neb_barriers.csv` to 6
  decimal places for all 5 paths (e.g. 1-4: 0.336050 eV both ways; 1-7:
  0.582586 eV both ways) — this is the required two-independent-file
  cross-check.
- Foundation-model energies are already-labeled eV and are of plausible MACE
  magnitude (roughly -3 to -4 eV/atom for a 61-atom cell); if they were
  mistakenly Hartree-scale the per-atom value would be an implausible ~-100
  eV/atom, so this also supports eV being correct.

**Residual caveat (not fully closed):** the foundation-model evaluation
script reads a `.extxyz` file (`$BASE/data/{path_id}.extxyz` on the remote
cluster) whose byte contents are **not** present in any locally mirrored
copy — only the resulting per-image JSON energies were synced back. I could
not hash-compare that `.extxyz` directly against the `sb2te3.xyz` used for
geometry classification. The evidence that they are the same path frames is
strong (identical path IDs, identical n_images=5 matching the QE path exactly
rather than an arbitrary interpolated count, identical underlying `neb.out`
SHA256, and a correct read-literal-frames evaluation script) but it is
circumstantial rather than a byte-for-byte proof. Flagging this explicitly
per the instruction not to assume alignment from naming alone.

**Not integrated:**
- 1-6, 1-8 (gap-diff): QE geometry/energy exists, but no foundation-model
  JSON was found for either — the foundation_grouped campaign only covers
  1-2, 1-3, 1-4, 1-5, 1-7. These are left out of the comparison figure
  entirely (no fabricated numbers).
- 81_neb_1..5: both QE and a full set of foundation-model runs exist
  locally (`foundation_grouped_benchmark_20261001/data/expanded_81_paths/`,
  `.../results/neb/*81*`), but since the pathway classification for this
  81-atom cell is itself unresolved (section 3), they were not assigned a
  pathway-type column and were left out of the matrix to avoid putting real
  numbers under a guessed label.
- The QE reference for all 5 integrated Cr paths remains the same kind of
  **screened, non-final** path-table value as Cu/Li/Na: every retained
  61-atom `neb.out` lacks an explicit `JOB DONE`/`NEB converged` marker
  (confirmed by `audit_historical_grouped_foundation_benchmark.py`, run
  2026-10-03 in the Wiley-MigrationBench repo), so these are NOT converged
  DFT migration barriers — same caveat as the Cu/Li/Na QE curves.

## 5. The composite figure

`figure1_task_type_matrix.png`/`.pdf`: 16-panel grid (4x4), same visual style
(Plot-Atlas `timecourse`, black QE trace + 9-10 colored model traces) as
`../corrected_dft_foundation_models/figure_corrected_dft_foundation_path_profiles.png`.

- Panels a-c: clearly labeled dashed-border placeholders for the 3 schematic
  migration-type diagrams (QLs / gap-diff / gap2QL) the PI said she will add
  herself. No schematic content was invented.
- Panels d-k: the existing, unmodified Cu/Li/Na corrected comparison (8
  dopant x pathway-type panels), reused as-is.
- Panels l-p: the 5 new Cr gap2QL panels (paths 1-2, 1-3, 1-4, 1-5, 1-7).

Qualitative read: paths 1-2, 1-3, and 1-5 (the deeper QL-penetration
trajectories) show the same kind of catastrophic foundation-model
over-prediction (roughly 20-35 eV vs. a QE reference of a few eV) seen for
the Cu QLs path; paths 1-4 and 1-7 (shallower gap/QL-boundary crossings) show
a much more moderate, order-of-magnitude-correct divergence (~1-2 eV vs.
~0.3-0.6 eV QE), closer to the Li QLs / gap-diff behavior. This is a
descriptive observation from this run, not a claim fit for the manuscript
without further review.

## 6. Honest status summary

- Reclassification: done with evidence for 1-2 through 1-8 (7/7 found paths);
  1-1 and the 81-atom series explicitly marked insufficient-data rather than
  guessed.
- Cr integration into the comparison figure: done for the 5 gap2QL paths
  with available foundation-model data; blocked (missing foundation-model
  runs) for 1-6/1-8/81-atom paths, stated plainly rather than fabricated.
- No MACE models were installed or run by this task — all foundation-model
  numbers were already-computed JSON files discovered on disk and reused.
