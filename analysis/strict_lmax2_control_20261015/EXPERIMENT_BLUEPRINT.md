# Strict Lmax2 control: experiment blueprint and acceptance gates

## Scientific objective

Construct a genuine same-parameter, different-training-seed control for the
from-scratch Lmax2 model. The control tests whether the force response seen in
the 639 paired candidate configurations is attributable to the Lmax2
parameterization or is dominated by training-seed variation.

## Frozen design

The auditable template is the surviving `ablation_lmax_2_seed_234` launch and
its successful training log, rather than the historical filename label. The
new runs use the same data root, MACE version/environment, model family,
architecture, loss, optimizer, schedule, batch sizes, epoch count, checkpoint
policy, and compilation path. Only the training seed and output directory
change.

| field | frozen value |
|---|---|
| data | `data/train.xyz`, `data/valid.xyz`, `data/test.xyz` |
| data counts | 980 / 122 / 124 configurations |
| model | MACE, 2 layers |
| channels / angular order | 32 channels, `max_L=2` |
| correlation | 2 (body order 3) |
| cutoff | 6.0 Angstrom |
| radial / basis | 8 radial, 5 basis functions |
| hidden irreps | `32x0e+32x1o+32x2e` |
| batch / epochs | 32 / 1000 |
| force weight | 10.0 |
| EMA / SWA | EMA 0.99; SWA from epoch 800 |
| optimizer | ADAM; learning rate 0.01; weight decay 5e-7 |
| Stage Two | after epoch 800; energy weight 1000; force weight 100; learning rate 0.001 |
| environment | MACE 0.3.12, CUDA 11.8, ASE 3.23.0b1, `mace` conda environment |
| new seeds | 123 and 345 |

The existing seed234 model remains a member only after its launch-log and
model-hash checks pass. The historical model currently labelled
`ablation_lmax_2_seed123` is retained as `LEGACY_HETEROGENEOUS` because its
launch log records Multi-T data and a different architecture/Stage-Two setup.

## Three-level acceptance

### A. Training and model identity

- Both new Slurm tasks exit zero and write a terminal training-complete marker.
- Each output contains the expected checkpoint and compiled model.
- New model files are finite, loadable by the same MACE environment, and
  element-compatible with Cr/Sb/Te.
- The launch manifest records seed, command, environment, input hashes, output
  hashes, job ID, and timestamps.
- No existing historical directory is overwritten.

### B. 639-candidate static evaluation

- The strict trio is evaluated on the same 639 candidate structures.
- Each candidate has one finite successful result per strict model, giving
  exactly 1,917 strict rows with no duplicate `(candidate_id, model_id)` keys.
- Candidate ordering, source-stage labels, PBC, and model input paths are
  unchanged from the current 639 campaign.
- The strict trio is compared using paired forces and force tails; absolute
  cross-family energies are not ranked.

### C. Scientific and publication gates

- Stable/pre-onset/precursor/threshold summaries are finite and reproducible.
- Seed-to-seed summaries are computed only within the strict trio.
- The historical heterogeneous branch remains labelled and is not pooled into
  the strict variance estimate.
- Any claim about DFT correctness or causality remains `PROVENANCE_PARTIAL`
  until matched DFT/local-fragment evidence exists.
- The updated family-grouped figure removes panel-b vertical separators while
  retaining the 100 eV Angstrom^-1 horizontal gate.

## Analysis plan after completion

1. Validate the new model manifest and static-result coverage.
2. Merge the strict trio with the existing 639 matrix using canonical model IDs.
3. Recompute stage medians, q95 tails, paired force ranges, and within-Lmax2
   seed dispersion.
4. Re-render the four-panel figure with family colour, arm hatch, larger text,
   and no vertical grey lines in panel b.
5. Write a bounded Results/SI paragraph distinguishing parameterization,
   training-seed variability, and DFT-correctness uncertainty.
