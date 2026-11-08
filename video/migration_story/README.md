# MigrationBench story video (Remotion)

Lean, reproducible source for the ~102 s multi-scale pitch video
("Meet Cr" → six probes → model × probe scorecard). Every number on screen is
read from `public/data/*.json`; colours live in `src/theme.ts`, scene timing in
`src/story/StoryVideo.tsx` (see [`TUNING.md`](TUNING.md) for which file owns what).

> **Status of the evidence.** This is a pitch/communication artifact, not a
> paper figure. Source charts carry registry statuses such as `NEEDS_AUDIT` /
> `PARTIAL`, and the Cr NEB reference remains `PROVENANCE_PARTIAL` with SCF
> warnings. The video shows these statuses in its closing frames; do not cite it
> as final evidence. Lattice jitter (dynamics chapter) and layer offsets
> (sliding chapter) are schematic and labelled as such on screen.

## Contents

| Path | What |
|---|---|
| `release/migration_story.mp4` | The rendered video (1920×1080, 30 fps, 102 s, ~18 MB) |
| `src/` | Remotion project (composition id `MigrationStory`) |
| `public/data/` | Frozen data the video renders from (`story.json`, `cr_neb_profiles.json`, `geometry.json`), with per-source sha256 + registry status |
| `scripts/export_*.py` | Regenerate `public/data/` from the asset registry (optional, see below) |

## Re-render

Requires Node `^20.19 || ^22.13 || >=24` and Chrome (Remotion downloads one on first run).

```bash
cd video/migration_story
npm ci
npm run lint                      # eslint + tsc
npm run dev                       # Remotion Studio, scrub the timeline
npx remotion render src/index.ts MigrationStory out/migration_story.mp4 --codec=h264 --crf=16
npx remotion still  src/index.ts MigrationStory out/check.png --frame=1770   # single-frame check
```

Frame = seconds × 30. Output goes to `out/` (git-ignored); copy the final file to
`release/` when it should replace the committed one.

## Refresh data from the registry (optional)

`public/data/` is committed, so rendering needs nothing else. To rebuild it, run
from this directory with the full analysis tree present (default `--root` is the
repository root; the registry is
`analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite`):

```bash
python3 scripts/export_neb_profiles.py   # Cr fixed-path NEB profiles
python3 scripts/export_story_data.py     # static / MD / sliding probes
python3 scripts/export_geometry.py       # 81-atom NEB image coordinates
```

The exporters open the registry read-only and fail if a linked data file is
missing. Provenance paths in the committed JSON were rewritten to be
repository-relative; sha256 fingerprints are unchanged.

## Scorecard grading rules (as rendered)

- NEB columns (in-gap, within-QL): predicted / DFT peak on the fixed path;
  |ratio − 1| < 10 % lightest, < 30 % middle, otherwise darkest.
- Gap-to-QL column: largest |ΔE − ΔE(DFT)| over the path images ÷ the 0.096 eV
  DFT drop, same 10 % / 30 % cut-offs. DFT has no barrier on this path, so peak
  height alone is not a valid grade (implemented as `entryCell` in
  `src/story/scenes/ClosingScenes.tsx`).
- Sliding: profile RMSE vs DFT (`SLIDING_RMSE_TOL_EV` = 0.5 eV).
- Static RMSE has no DFT-relative grade.
- From-scratch uses different checkpoints per probe (static/sliding: S600;
  MD/NEB: all-data seed 123); marked with an asterisk on screen.

Thresholds are the video author's choice and can be changed to match the paper.
