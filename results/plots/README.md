# Canonical figure bundles

Publication figures are organized here as named, date-stamped bundles rather
than as loose files in analysis worktrees:

```text
<figure>_<date>_<revision>/
├── figures/       # PDF, SVG, PNG
├── scripts/       # standalone renderer
├── data/          # plotted tables and audits
├── source/        # copied inputs and remote pointers
└── provenance/    # manifests, hashes, and render records
```

## Current release

`cr_three_path_fixed_vs_self_scratch600k_20261101_R05/` is the first
canonical, user-certified migration in this directory. It contains the
standalone Plot Atlas renderer, editable SVG, PDF/PNG outputs, the exact
plotted-profile summary and coverage audit, materialized campaign/model/path
manifests, endpoint records, and provenance records. Provider-backed source
tables and large trajectories that are not materialized locally remain
explicit hash/path pointers; the original analysis-root R05 commit retains
the tracked compact source tables.

The R05 renderer was rerun during release preparation and passed the release
checks: one-page output, matching PDF/PNG/SVG hashes, 1,270 loaded profile
rows, 90 Scratch-600K fixed-path rows, 70 primary self-relaxed rows, and 9
Scratch-600K endpoint manifests. The two nonconverged self-relaxed cases are
explicitly diagnostic-only and are not silently promoted into the main values.
The current checkout's File Provider eviction is recorded in the bundle
manifest and registry as `pointer-only`, not hidden as a local file.

## Historical audit

`FIGURE_BUNDLE_AUDIT_20261101.tsv` records the figure-related commits and
release candidates found during the 2026-11-01 audit. The historical commits
are mostly coherent analysis packages, not narrow figure-only commits, so
they are indexed with their scientific boundaries instead of being blindly
cherry-picked into the main results tree.

The two managed Codex worktrees checked in this audit were clean: zero tracked
diffs and zero untracked files. The main checkout itself has a large existing
untracked analysis surface (11,324 files overall; 367 under the Cr grouped
analysis root). Those files were not staged, deleted, or otherwise modified.

For older releases, migrate only after selecting the exact user-certified
artifact and reconstructing a self-contained source/data/provenance bundle.
The audit keeps the original commit, branch, and scientific status visible so
that a later migration can be additive and reviewable.
