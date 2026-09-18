# Zero-Force And Data-Leakage Verdict

Date: 2026-09-09

## Verdict

CLEAN for the reported FT-600K, FT-MultiT, and scratch training runs checked here.

I found no evidence that fabricated zero-force NEB frames were used in the reported training. The training logs point to `data_600K` and `data_Multi_T`, not to NEB proposal/evaluation files, and the force-column audit found no NEB provenance tags in those splits.

## Evidence

### Training Log Lineage

Reported FT-600K seed-123:

- Log: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_run-123.log`
- Training file loaded: `data_600K/train.xyz` with 1888 configs, 1888 energies, 1888 forces, 1888 stresses.
- Validation file loaded: `data_600K/valid.xyz` with 236 configs.
- Test file loaded: `data_600K/test.xyz` with 237 configs.

Reported FT-600K seed-234 and seed-345 logs also load the same `data_600K/{train,valid,test}.xyz` splits.

Reported FT-MultiT seed-123:

- Log: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804_run-123.log`
- Training file loaded: `data_Multi_T/train.xyz` with 3012 configs, 3012 energies, 3012 forces, 3012 stresses.
- Validation file loaded: `data_Multi_T/valid.xyz` with 376 configs.
- Test file loaded: `data_Multi_T/test.xyz` with 377 configs.

Reported scratch seed-123:

- Log: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_run-123.log`
- Training file loaded: `data_600K/train.xyz` with 1888 configs, 1888 energies, 1888 forces, 1888 stresses.

### Force-Column Audit

Raw CSV: `data_processed/cluster/training_provenance/zero_force_training_audit.csv`

| Split | Frames | Frames With Force Columns | All-Zero Force Frames | NEB Tags |
|---|---:|---:|---:|---|
| `data_600K/train.xyz` | 1891 | 1891 | 3 | false |
| `data_600K/valid.xyz` | 236 | 236 | 0 | false |
| `data_600K/test.xyz` | 237 | 237 | 0 | false |
| `data_Multi_T/train.xyz` | 3015 | 3015 | 3 | false |
| `data_Multi_T/valid.xyz` | 376 | 376 | 0 | false |
| `data_Multi_T/test.xyz` | 377 | 377 | 0 | false |

The three all-zero-force frames in each training split are isolated atoms:

- Cr isolated atom
- Sb isolated atom
- Te isolated atom

Evidence file: `data_processed/cluster/training_provenance/isolated_atom_zero_force_frames.txt`

These frames have `config_type=IsolatedAtom`, one atom, a 20 A cubic cell, and no NEB tags. They are consistent with isolated-atom/E0 references, not fabricated NEB labels.

## Data-Leakage Controls Added

The new pipeline makes the training/evaluation boundary explicit:

- `record_type=dft_training_frame`: allowed for training only if real forces exist.
- `record_type=dft_neb_reference`: evaluation/reference data, not training.
- `record_type=mlff_neb_proposal`: MLFF-generated images, not DFT labels.
- `path_id`, `source_run_id`, `split`, `calculator_label`, and `convergence_status` are required metadata.

The dataset exporter no longer fabricates zero forces when force labels are unavailable.

## Response-Letter Draft

We audited the training lineage on Rockfish for the reported FT-600K, FT-MultiT, and scratch MACE runs. The MACE training logs show that the reported models loaded `data_600K/{train,valid,test}.xyz` or `data_Multi_T/{train,valid,test}.xyz`, each with matching energy/force/stress labels, and not the NEB proposal/evaluation files. A direct extxyz force-column audit found force columns in all checked frames and no NEB provenance tags. The only all-zero-force frames were three one-atom `config_type=IsolatedAtom` records for Cr, Sb, and Te, used as isolated-atom references rather than NEB images. We therefore find no evidence that fabricated zero-force NEB frames entered the reported training.
