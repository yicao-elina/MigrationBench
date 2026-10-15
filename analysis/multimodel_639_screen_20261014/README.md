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
