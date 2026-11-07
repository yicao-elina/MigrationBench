# Data and visual acceptance criteria

These gates apply to every panel and every final export. A reviewer must record
`PASS`, `FAIL`, or `N/A` with a source path, manifest row, and short reason.

## A. Provenance and reproducibility

- [ ] Every plotted series has an absolute source path, source SHA256, table
  schema/columns, extraction command, and plot-script commit.
- [ ] Model family, checkpoint identifier/hash, seed, split, protocol,
  temperature, task family, and scientific status are explicit wherever
  applicable.
- [ ] The figure manifest lists every raw/processed input and the hashes of all
  generated PNG/PDF/SVG outputs.
- [ ] External Rockfish, Downloads, or checkpoint data are recorded as pointers;
  no untracked copy silently becomes a second authority.
- [ ] The plotted row count equals the manifest-expected row count, and duplicate
  geometry/model/image keys are either zero or explicitly justified.
- [ ] A clean rerun from the recorded command reproduces the same values within
  the stated floating-point tolerance and the same labels/legend order.

## B. Numerical/data gates

- [ ] All required values are finite after the documented mask; rejected rows,
  NaNs, infinities, missing images, and duplicate keys are counted.
- [ ] Units and denominators are frozen and visible in the panel/caption:
  per-atom versus total energy, eV versus meV, and Å versus Å⁻¹ are not inferred
  from column names alone.
- [ ] For parity, energy/force summary values recompute from the per-frame/raw
  arrays, with no train/test leakage and no mixed seed or split.
- [ ] For fixed paths, all compared methods use the same ten image indices,
  same coordinates, same endpoint identity, and same reference rows.
- [ ] For NEB generated paths, all ten images have finite coordinates, energy,
  forces, endpoint identity, convergence status, final maximum force, residuals,
  and iteration count. Endpoint-only or reconstruction controls cannot pass.
- [ ] For MD, each curve terminates at the recorded `last_time_ps`; failed runs
  are censored/observed-horizon records, not completed 1 ns trajectories. No
  diffusion coefficient or ranking is reported unless the campaign coverage and
  validity gate passes.
- [ ] For sliding, fixed-image and self-relaxed/generated paths remain separate;
  displacement coordinate, image count, reference status, model identities, and
  class/checkpoint counts are explicit. Sample SD across model identities is not
  relabelled as seed uncertainty.
- [ ] For configuration-universe/QE diagnostics, the join is geometry-keyed and
  the high-error threshold, held-out validation path, calibration, and selection
  rule are frozen before the result is inspected.
- [ ] A warning state such as `COMPLETED_BUT_SCF_WARNING`, `PRECANONICAL`,
  `SCIENTIFIC_FAILURE`, or `ENDPOINT_ONLY_PROXY` is retained in the figure and
  cannot be silently promoted to `PASS`.

## C. Visual and composition gates

- [ ] Final placement uses Arial or Helvetica; no fallback glyph substitution is
  visible in labels, units, legends, or captions.
- [ ] At final placement, axis labels are at least 11 pt, tick labels at least
  9 pt, and panel labels/annotations remain legible without zooming.
- [ ] All axes have full black spines (~1.05 pt), outward major/minor ticks, no
  decorative grid, and consistent tick padding.
- [ ] Colors are semantic and stable across figures: model family, reference,
  diagnostic/warning, and blocked state do not change meaning between panels.
- [ ] Profile grammar is consistent: reference as dashed/open diamond, model as
  solid/open circle, mean line about 2 pt, and pointwise SD band at alpha about
  0.20. Error bars state whether they are SD, CI, or model spread.
- [ ] Legends do not cover data, warning text, or error bars. Long labels are
  wrapped or shortened consistently rather than clipped.
- [ ] No title, axis label, annotation, tick, legend, or caption is clipped at
  the PDF/SVG page boundary. A raster preview is rendered and visually checked.
- [ ] PDFs are non-empty and parseable; SVGs contain vector paths/text; PNGs have
  the expected dimensions and a white background unless transparency is
  explicitly required.
- [ ] Composite/canvas boards preserve the individual figure aspect ratios,
  reading order, and warning/blocked labels. A placeholder is visually marked
  and cannot be mistaken for data.

## D. Independent reviewer sign-off

The reviewer must inspect source tables and rendered PNG/PDF/SVG independently
of the plot author. For each panel, record:

`figure | panel | data gate | visual gate | evidence path | result | blocker | reviewer/date`

Release rules:

- `PUBLICATION_PASS`: all applicable A–C checks pass, no unresolved warning or
  proxy status conflicts with the claim, and an independent reviewer signs off.
- `PROVISIONAL`: data-backed and reproducible but a stated non-scientific gate
  remains open.
- `DIAGNOSTIC`: useful for audit/interpretation but not a publication claim.
- `BLOCKED`: missing evidence or an invalid comparison; keep the placeholder and
  do not fill it with an inferred/synthetic plot.
