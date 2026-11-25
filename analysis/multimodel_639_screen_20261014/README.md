# Twelve-model static screening of the 639 failure-attribution candidates

This additive campaign evaluates the same 639 saved, fully periodic
2050-atom configurations with the nine checkpoints that complement the
existing three-checkpoint from-scratch campaign:

- FT-600K at training seeds 123, 234, and 345;
- FT-MultiT at training seeds 123, 234, and 345;
- three foundation representatives: the legacy compiled OMAT checkpoint,
  MPA-0 medium, and MP-0b3 medium.

The existing from-scratch results are retained as an immutable comparison
branch. The unit of comparison is one candidate structure evaluated by every
registered checkpoint. No geometry relaxation or MD continuation is performed.
Total energies are retained for provenance but are not ranked across model
families; the primary observables are force magnitudes, force tails, geometry
validity, and paired force-vector disagreement.

Candidate provenance is unchanged: 639 rows pooled from three trajectory
sources (`original`, `reproduce-1`, `reproduce-2`) and three training seeds,
giving nine source run keys. Candidate group labels are inherited from the
source trajectory and are not redefined by this static evaluation.

## Scientific gates

The campaign is complete only when all 32 chunks exist, each candidate/model
pair appears exactly once, all model hashes match the registry, input hashes
are recorded, and energy/force/geometry statuses are finite or explicitly
retained in the failure ledger. Completion is a numerical and provenance
gate; it does not establish DFT correctness or a causal explanation of the
MD instability.

The legacy compiled OMAT checkpoint is named explicitly as such. It must not
be silently relabelled as the public MATPES-PBE OMAT-FT foundation model.

## Paper and evidence handoff

- `obsidian/` is the navigable evidence package for sampling, model identity,
  distribution shift, force divergence, figures, reproduction and gaps.
- `overleaf_handoff/` contains copy-ready Nature-style Results/Discussion,
  SI Methods and figure-caption TeX. It is a handoff only; no Overleaf commit
  is claimed until the package is synchronized and compiled there.
- `CLAIM_EVIDENCE_MATRIX.tsv` maps paper-facing claims to manifests, tables,
  scripts and job evidence.
- `PAPER_RELEASE_MANIFEST.json` records the release hashes and will be amended
  with the final Git commit after review.
- `RESULTS_AND_DISCUSSION.md` contains the full interpretation and explicit
  DFT/causality boundaries.

## Family-grouped figure release

`figures/fig_multimodel_639_overview_family_grouped.{png,pdf,svg}` is the
updated four-panel Plot Atlas reconstruction; the older
`figures/fig_multimodel_639_overview.{png,pdf,svg}` name is maintained as an
identical compatibility alias. Panel c retains the full global SOAP-PCA scale
and adds a local inset for the training/reference/candidate cloud. In panels a and b, colour is
the primary family encoding and the three within-family arms are separated by
their ordered labels/hatches. Panels c and d retain candidate-stage colours,
because those panels describe configuration stage and descriptor/force-spread
distributions rather than model family.

The exact mapping is recorded in `MODEL_GROUPING.tsv` and summarized in
`MODEL_GROUPING.md`. FT-600K and FT-MultiT are true three-training-seed
comparisons; the Foundation entries are three checkpoint variants, and the
From Scratch entries contain two Lmax2 branches plus a distinct baseline
branch.

The strict same-parameter Lmax2 control is an additive campaign, not a silent
reinterpretation of this historical 12-model matrix. Its frozen
configuration, acceptance gates, and Rockfish job record are in
`../strict_lmax2_control_20261015/`. The historical seed123 branch remains
excluded from strict seed-to-seed pooling until the new configuration-matched
seed123 and seed345 models pass all gates.


**Rockfish status update (2026-11-22):** the two configuration-matched training
tasks failed in the shared launcher before training (`RF_CONDA_SITE` unset).
No strict checkpoints or static evaluation exist yet. The earlier pending
status in the campaign's submission note is historical; see
`../strict_lmax2_control_20261015/ROCKFISH_RECHECK_20261122.md`.
