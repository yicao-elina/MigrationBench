# Model provenance

## Roster

The full paired comparison contains three existing from-scratch checkpoints, six fine-tuned checkpoints and three foundation representatives:

| family | checkpoints |
|---|---|
| from-scratch | `ablation_lmax_2` seed 123; `ablation_lmax_2` seed 234; `baseline` seed 345 |
| FT-600K | seeds 123, 234, 345 |
| FT-MultiT | seeds 123, 234, 345 |
| foundation representatives | legacy compiled OMAT; MPA-0 medium; MP-0b3 medium |

The three foundation models are model representatives, not seed replicas. The OMAT model is explicitly labelled **legacy compiled OMAT**. It must not be relabelled as the public MATPES-PBE OMAT-FT checkpoint without a separate provenance record.

## Hashes and identity

The canonical registry is `../MODEL_REGISTRY.tsv`; the nine remote model file checks are in `../MODEL_HASH_AUDIT.tsv`. The exact absolute Rockfish paths and SHA256 values are retained there. Model binaries are not committed to Git.

## Environment

- Rockfish environment: `/data/pclancy3/yi/.conda/envs/mace`
- Python 3.9.22
- MACE 0.3.12
- ASE 3.25.0
- PyTorch 2.7.0+cu118
- CUDA device: one GPU per task
- Static protocol: no geometry relaxation and no MD continuation
