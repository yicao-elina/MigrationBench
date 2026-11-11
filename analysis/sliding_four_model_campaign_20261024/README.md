# Fig. 4 interlayer sliding: fixed-path campaign

This directory is the additive Stage 1 package for the MigrationBench Fig. 4
interlayer-sliding update. It covers only the fixed-path observable:

* M0: input and DFT-reference identity checks;
* M1: A0 legacy pipeline self-check on the recovered 10-frame paths;
* M2: Tier 1 evaluation of the 15 current checkpoints on pristine and
  Cr-in-gap Sb2Te3 (300 image energies and forces);
* M3: Tier 2 MIC-aware dense-path diagnostic and the two publication-facing
  fixed-path panels.

The directory is intentionally separate from the historical A0 outputs and
from the later self-relaxed NEB campaign. The old legacy Foundation checkpoint
is retained only as an A0 reproduction control and is excluded from the new
plots.

## Reproduction sequence on Rockfish

The Slurm scripts use the Python executable, cluster paths, model paths, and
protocol values from `config/campaign.yaml`. The campaign root is supplied by
the submit command; no input or model path is embedded in the evaluator.

1. Copy this directory to the configured campaign root and copy the recovered
   DFT inputs and legacy prediction CSVs into `inputs/`.
2. Generate `jobs/a0_fixed_jobs.tsv`, `jobs/fixed_jobs.tsv`, and
   `jobs/dense_jobs.tsv` with the job-table helper.
3. Run the read-only input validation and one smoke row.
4. Submit the eight A0 rows. Do not submit the current matrix unless the A0
   profile comparison passes the configured 1 meV tolerance.
5. Submit the 30 current Tier 1 rows. Collect only after every row has a
   terminal manifest and finite profile.
6. Submit the 30 Tier 2 rows after the Tier 1 gate passes.
7. Run `scripts/collect.py`, then `scripts/plot_fig4_fixed.py` from a clean
   local analysis environment. The plot script writes separate pristine and
   Cr-in-gap PDFs/PNGs under `results/figures/`.

The fixed-path values are relative energies on retained DFT/QE geometries.
They are not self-relaxed MACE NEB barriers. The DFT reference is retained as
`COMPLETED_BUT_SCF_WARNING` and includes SOC/PBE/DFT-D3 in the source input;
that status is carried into the manifests and discussion.

## Validation gates

`collect.py` stops with a nonzero exit code for missing or duplicate matrix
rows, non-finite values, wrong atom/image counts, model hash mismatches, failed
A0 comparisons, or incomplete Tier 2 coverage. A failed model is preserved as
a failure row and cannot be silently dropped from the expected-versus-actual
table.

The updated panels retain the established colors: black DFT, dark blue
FT-600K, light blue FT-MultiT, red Foundation checkpoint variants, and orange
Scratch subfamilies. Scratch-600K and Scratch-MultiT use the same orange role
with filled/open diamonds so they are not merged without explanation.

