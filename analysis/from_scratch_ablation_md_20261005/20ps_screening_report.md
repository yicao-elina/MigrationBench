# MigrationBench from-scratch ablation MD — 20 ps screening audit

Campaign root: `/scratch16/pclancy3/yi/mace_from_scratch_ablation_MD_20261005`.
The common input is the read-only source
`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/0-foundation-mace_mp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz`
with SHA256
`bb42c493fb0c997af3957a89dfde03e3e5a67ebcfd37f5eced8f722dad546161`.
The old four-model root was not used as an input, output, checkpoint, or
restart source.

The formal scratch matrix contains 14 auditable models. `ablation_correlation_3`
seed123 is excluded because the source directory has overlapping legacy
`data_600K`/foundation and later `data/` scratch records under overlapping
artifact names (`PROVENANCE_CONFLICT`).

| outcome | count | interpretation |
|---|---:|---|
| exact 20,000-step complete | 3 | screening-pass candidates only |
| early scientific instability | 11 | failure onset retained; no Dxy reported |
| launcher/resource failure | 13 first-attempt array tasks | corrected with additive `r01` retry; no scientific output from first attempt |

The three complete runs were baseline seed345 and lmax2 seeds123/234. They
had 101 saved frames, finite values, fixed 2050-atom fully periodic identity,
temperature ranges of 568–751 K, and maximum sampled forces of 2.76–15.53
eV/Å. Candidate hop counts were 0, 0, and 1. The 20 ps MSD/Dxy values are
screening diagnostics only: the window is too short for a formal diffusivity
claim and no Arrhenius or replica-converged transport claim is made.

The selected 1 ns follow-up is Slurm array `31630588_[0-2]` with three
independent output directories: baseline seed345 and lmax2 seeds123/234.
The 1 ns results remain pending until exact steps, trajectory completeness,
finite energy/forces/temperature, PBC geometry, hop statistics, block/replica
uncertainty, and window convergence are audited. HFACF/running heat-flux
integral is unavailable unless heat flux is recorded.

Current state is `exploratory/screening`; formal transport results are
`blocked/pending` until the 1 ns scientific audit passes. The failure model is
`unstable — no Dxy`. A second-pass unified post-processing release now exists
under `remote_unified_20ps/`: all 14 formal trajectories have finite pressure
frames and recorded stress, so repository-convention HFACF/running-κ files are
available as diagnostics. They remain non-converged screening quantities and
are not promoted to transport claims.
