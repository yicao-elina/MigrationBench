# Revision1 NEB/Data Pipeline Design

## Immediate Audit Verdict

The manuscript should treat the current NEB evidence as mixed quality rather than uniformly solid.
The defensible local DFT reference is path 1-4 at 0.336 eV, because the value is reproducible from
`sb2te3.dat` and stationary to much less than 0.005 eV near the end of the available run. However,
the local `neb.out` is still not a clean `JOB DONE` record, so the final revision should either
finish the job or explicitly label it as a stationary-but-not-formally-complete reference.

Paths 1-2, 1-3, 1-5, and 1-7 are not manuscript-solid as local DFT references. Their present local
barriers come from unconverged or diverged NEB states. Any deep-penetration conclusion must therefore
come from converged rockfish reruns or new DFT NEB jobs initialized from a cleaner MLFF-preconditioned
path.

Fig. 3d and Fig. S3 must be separated as two protocols:

- fixed-geometry single-point energies on DFT NEB images;
- self-consistent MLFF NEB, where the model relaxes its own path.

The 0.41 eV Fig. S3 Foundation claim is not locally reproducible because the local
`neb_final_path.xyz` stores constant placeholder energies. It is quarantined until the raw optimizer
trajectory or final per-image energies are retrieved from rockfish.

## Recommended Compute Strategy

Use a two-stage NEB strategy for every pathway:

1. MLFF preconditioning:
   run `scripts/migrationbench/run_mlff_neb.py` with the MACE foundation model or the relevant
   fine-tuned checkpoint. Use 7-12 images, IDPP interpolation, FIRE, `fmax <= 0.05 eV/A`, and reject
   paths with energy spikes larger than 5 eV relative to endpoints.
2. DFT refinement:
   convert the MLFF-relaxed images into a QE `neb.x` input using
   `scripts/migrationbench/qe_neb_from_images.py`. Start with `opt_scheme='broyden'`,
   `CI_scheme='auto'`, `minimum_image=.true.`, `ds=0.2`, `k_min=0.1`, `k_max=0.3`,
   and `path_thr=0.03 eV/A`; rerun final checks at `path_thr=0.02 eV/A` for values that enter the
   paper.

This follows QE's supported `FIRST_IMAGE / INTERMEDIATE_IMAGE / LAST_IMAGE` input format and keeps
the MLFF proposal separate from the DFT reference.

## Data Asset Rules

All artifacts enter the dataset through four tables declared in
`configs/migrationbench_hf_schema.yaml`:

- `configurations`: structures with hashes and source paths;
- `calculations`: one row per method/image with energy, real forces if present, convergence status,
  raw-log URI, and raw-log hash;
- `neb_paths`: ordered pathway-level records and canonical barrier metadata;
- `derived_barriers`: protocol-specific barriers and manuscript inclusion gates.

Quarantined records are preserved, not deleted. This is important: reviewer-facing transparency is
better when suspect rows are visible with `manuscript_allowed=false` and an explicit exclusion reason.

## Data Leakage Control

The reported zero-force leak is controlled in two layers:

1. Exporters must never fabricate `forces:R:3`. If QE forces are unavailable, forces are absent and
   the record is eval-only.
2. Training splits must be group-aware by `path_name`, `trajectory`, or `group`; NEB image metadata
   must remain with the frame, and any NEB-adjacent AIMD segment is excluded or audited before FT-600K
   claims are made.

With those rules, there is no evidence of current data leakage in the reported FT-600K/FT-MultiT
models from local files alone, but the claim still needs rockfish log evidence showing the exact
training file paths and nonzero-force sampled frames.

## Manuscript Language Until JB-5/JB-2 Finish

Use the conservative version:

> We distinguish fixed-geometry evaluation on DFT NEB images from self-consistent MLFF NEB relaxation.
> For the in-gap 1-4 pathway, all fixed-geometry errors are now computed against a single DFT reference,
> 0.336 eV. The previous 0.34 eV and approximately 0.3 eV phrasing referred to rounded or different
> pathway contexts and has been standardized. Self-consistent MLFF NEB results are reported separately
> and are not used to rank fixed-geometry barrier accuracy.

Do not call the Foundation model "exceptional" on Fig. S3 until the 0.41 eV value is recomputed from
raw per-image energies.
