# Tuning MigrationStory: where to change what

Rule of thumb: **change one layer at a time, and only in the file that owns it.**
Numbers come from the data layer, colours from the theme, timing from
`StoryVideo.tsx`, and wording from the scene files. If you edit something in
the wrong layer, it stops agreeing with the rest of the video.

## 1. Fast feedback loop

```bash
cd video/migration_story && npm run dev
```

- Studio hot-reloads when you save. Scrub to the scene in the timeline; each
  scene is a named `Series.Sequence`, such as "Probe: within_gap".
- To check one frame at full resolution without re-rendering the video, bundle
  once and then render single stills from that bundle:

```bash
npx remotion bundle --out-dir build && npx remotion still build MigrationStory out/check.png --frame=1770
```

- Frame = seconds × 30. A scene's start time is the sum of the `SECONDS` values
  before it in `src/story/StoryVideo.tsx`.
- Render the full video only once the stills look right:

```bash
npx remotion render build MigrationStory out/migration_story.mp4 --codec=h264 --crf=16
```

## 2. Which file owns what

| You want to change… | Edit | Don't |
|---|---|---|
| Any colour (atoms, models, probes, evidence, stage) | `src/theme.ts` only | hard-code hex values in a scene |
| Model encoding (shape / open / filled) | `SERIES_STYLE` in `src/theme.ts` | invent a new colour per chart |
| Scene length / order | `SECONDS` and the `<Series>` in `src/story/StoryVideo.tsx` | change `durationInFrames` inside a scene |
| When an element appears inside a scene | the `ramp(frame, fps, startSec, durSec)` calls in that scene | use CSS animations or transitions (Remotion can't render them deterministically) |
| Headline, verdict, caveat text | `CHAPTERS` in `scenes/ChapterScene.tsx`; other scenes' JSX | write a number that is not in `public/data/*.json` |
| Plot axes / y-range | `Y_DOMAIN` in `src/data.ts`; tick and axis constants in `EnergyPanel.tsx` / `charts.tsx` | clip data silently: off-scale peaks must keep their "▲ X eV" label |
| Data values | the Python exporters in `scripts/`, then re-export | edit the JSON files by hand |
| Atom look (size, bonds, glow, trail) | `ATOM` radii in `theme.ts`; `HOST_BOND_A`, `DEPTH_FADE` and the glow filters in `story/Lattice.tsx` | change the coordinates (they are real NEB images) |
| Background mood | `STAGE` tokens, `BEAM_OPACITY` and `DUST_COUNT` in `story/Stage.tsx` | add per-scene backgrounds (the stage is shared on purpose, so cuts never flash) |
| Grading thresholds on the scorecard | `GOOD_TOL`, `WARN_TOL`, `ENTRY_TOL_EV`, `SLIDING_RMSE_TOL_EV` in `ClosingScenes.tsx` | change a threshold without updating the footnote that discloses it |

## 3. Visual rules (Global Visual Standard)

- **Colour carries one meaning.** Strategy colour = zero-shot / fine-tuned /
  scratch. Thermal regime is shown by fill (open = 600 K, filled = multi-T),
  never by a new hue. Seed is shown by marker shape. Evidence states (failed,
  pending) are never reused to mean "bad accuracy".
- **Magnitudes are sequential.** The scorecard uses one light-to-dark ramp. Don't
  switch back to traffic-light green, amber and red.
- **Data lives on paper, story lives on stage.** Every quantitative plot sits on
  a `PaperCard` (light background, 2 px axes, outward ticks, no gridlines, bold
  lowercase panel letter). Narrative text sits on the dark `Stage`.
- **Type.** Use the serif (`serifFamily`) for headlines and verdicts. Use Inter
  for labels, axes and captions. In a 1080p frame, keep axis text at 22 px or
  larger and body text at 32 px or larger.
- **One hero.** Only Cr glows. If you add emphasis to anything else, it competes
  with the hero.
- **No journal branding.** Don't add journal logos or mastheads. The style is
  "journal-quality", not "published in X".

## 4. Common edits, step by step

**Make a chapter breathe longer:** raise `SECONDS.chapter` (this applies to all
six probe chapters). To lengthen only one chapter, give that probe its own
duration in the `PROBE_ORDER.map` inside `StoryVideo.tsx`. Then move
`VERDICT_SEC` so the verdict still lands after the curves finish drawing.

**Reword a verdict:** edit `CHAPTERS[probe].verdict`. First recompute the number
from `public/data/story.json` or `cr_neb_profiles.json`. If the claim compares
models with DFT, re-read the matching `note` caveat.

**Try a different accent colour:** change the hex in `theme.ts`, run one still
per affected scene, then compare them side by side:

```bash
ffmpeg -pattern_type glob -i 'out/story/f*.png' -vf "scale=640:360,tile=3x4" -frames:v 1 out/story_contact.png
```

**Add a new data series:** add it in the exporter, re-run the export, and add a
`SERIES_STYLE` entry that follows the shape and fill rules above. The legend and
markers pick it up automatically.

## 5. Before you share a render

- Watch the whole video once at full screen, not just stills.
- Check each on-screen number against the JSON, and check that each caveat is
  still visible: SCF warning, `FINITE_DIAGNOSTIC`, `NEEDS_AUDIT`, and the
  non-pooled from-scratch lineages.
- Check the licensing. Remotion is free only for individuals and companies of
  up to 3 people; for anything larger, check with TRI Legal about a company
  license.
