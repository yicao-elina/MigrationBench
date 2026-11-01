# Scratch-600K canonical migration fill campaign

This is an isolated fill campaign for the missing `Scratch-600K` arm in the
canonical Cr@Sb2Te3 migration matrix. It does not modify or overwrite the
20261019 parent campaign.

Scope is deliberately narrow:

- one model family: `Scratch-600K-Lmax1`;
- training seeds 123, 234, and 345, with the exact registered checkpoint hash;
- the three canonical 81-atom, 10-image paths: in-gap diffusion, gap-to-QL,
  and within-QL;
- nine fixed-path single-point jobs and nine endpoint-only IDPP self-relaxed
  jobs, all with NEB seed 42.

The existing Scratch-600K self files from the interlayer-sliding campaign are
not reused: that campaign lacks the gap-to-QL/within-QL inputs and its bulk
record is 80 atoms, whereas this strict comparison is the canonical 81-atom
matrix. Fixed-path and self-relaxed observables remain separate.

## Rockfish execution

The bootstrap script is run on Rockfish after this directory is copied to the
remote checkout:

```bash
python bootstrap_scratch_600k_campaign.py
sbatch /scratch16/pclancy3/yi/scratch_600k_migration_20261101/submit_model_load_smoke.slurm
# inspect all three smoke records; only then submit the two science arrays
sbatch /scratch16/pclancy3/yi/scratch_600k_migration_20261101/submit_fixed_array.slurm
sbatch /scratch16/pclancy3/yi/scratch_600k_migration_20261101/submit_idpp_array.slurm
```

Scientific acceptance requires finite profiles and the runner's convergence
gates. `COMPLETED` or finite output alone is not sufficient. The DFT reference
is retained with status `COMPLETED_BUT_SCF_WARNING`, so these outputs remain
diagnostic/supporting evidence until that reference gate is repaired.

## 2026-11-01 execution result

The isolated campaign was submitted on Rockfish with model-load smoke job
`32791406`, fixed-path array `32791432`, and IDPP self-relaxed array
`32791433`. All three checkpoint smoke tests passed with registry-matching
SHA256 values. All nine fixed-path jobs produced finite 10-image profiles and
are retained as `FINITE_DIAGNOSTIC / SUPPORTING_EVIDENCE` because of the DFT
reference warning.

The nine self-relaxed jobs all produced endpoint-only manifests. Seven passed
the target-model convergence gate and are `CONVERGED / PRIMARY_EVIDENCE`; two
are retained but excluded from primary summary values as
`NONCONVERGED / DIAGNOSTIC_ONLY`:

- seed 123 × within-QL, residual 0.2748 eV/A;
- seed 234 × in-gap, residual 0.3613 eV/A.

The local processed tables contain all 18 rows and preserve these statuses.
The large trajectory and iteration-history extxyz files remain on Rockfish;
their manifest paths and SHA256 values are preserved, while
`OMITTED_OR_POINTER_ONLY.tsv` records the intentional local omission. The
inherited fixed-path runner writes its historical parent campaign identifier
inside the raw manifest schema; the new campaign identity is recorded in the
root manifest, job tables, and processed result matrix `campaign_id` column.
